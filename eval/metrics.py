"""评估指标计算模块

所有指标函数尽量做成纯函数，便于单元测试：
- 输入只依赖参数，不读取全局状态
- 不产生副作用（memory_recall_accuracy 会临时写入测试记忆，已隔离到指定 session）

包含四类指标：
1. 角色一致率 persona_consistency_rate
2. 记忆召回准确率 memory_recall_accuracy
3. 工具调用成功率 tool_call_success_rate
4. 情绪响应适当性 emotion_response_appropriateness
"""
from __future__ import annotations

import time
from typing import Any, Iterable

# ---- 星野爱角色设定（默认，保持向后兼容）----
# 角色设定关键词：回复中出现这些词说明贴合"星野爱"人设
PERSONA_KEYWORDS: tuple[str, ...] = (
    "偶像", "新B小町", "舞台", "粉丝", "大家", "表演",
    "♪", "✨", "~", "！", "爱",
)

# 反向指标关键词：承认自己是 AI / 语言模型，破坏沉浸感（各角色通用）
AI_ADMISSION_KEYWORDS: tuple[str, ...] = (
    "我是AI", "我是一个AI", "我是人工智能", "作为AI", "作为一个人工智能",
    "语言模型", "我是大语言模型", "我是程序", "我没有真实情感",
    "我只是程序", "我没有感情",
)

# 星野爱自称模式（"爱"作为第一人称）
AI_SELF_REFERENCE_PATTERNS: tuple[str, ...] = (
    "爱觉得", "爱想", "爱也", "爱会", "爱是", "爱就", "爱都",
    "爱能", "爱不能", "爱要", "爱还", "爱在", "爱喜欢",
    "爱收到了", "爱很开心", "爱好开心", "爱好高兴",
    "爱不会", "爱没办法", "爱希望", "爱相信",
    "爱一直在", "爱真的", "爱最",
)

# ---- 紫灵角色设定 ----
# 紫灵人设关键词：修仙术语 + 作品要素
ZILING_PERSONA_KEYWORDS: tuple[str, ...] = (
    "紫灵", "修仙", "修行", "大道", "灵力", "神识", "元婴", "化神",
    "韩立", "韩兄", "乱星海", "魁星岛", "魔界", "始祖", "本座",
    "道友", "便是", "也罢", "罢了",
)

# 紫灵自称模式（"紫灵"或"本座"作为第一人称）
ZILING_SELF_REFERENCE_PATTERNS: tuple[str, ...] = (
    "紫灵", "本座", "贫道",
)


def persona_consistency_rate(
    replies: Iterable[str],
    persona_ctx: str = "",
    persona_keywords: tuple[str, ...] = PERSONA_KEYWORDS,
    self_reference_patterns: tuple[str, ...] = AI_SELF_REFERENCE_PATTERNS,
    ai_admission_keywords: tuple[str, ...] = AI_ADMISSION_KEYWORDS,
) -> float:
    """角色一致率

    检查每条回复是否符合指定角色人设：
    - 是否以角色名/自称词自称（第一人称）
    - 是否包含角色关键词
    - 是否承认自己是 AI（反向指标，命中则该条记 0 分）

    Args:
        replies: 待评估的回复列表
        persona_ctx: 人设上下文文本（保留参数以便后续扩展加权，当前可空）
        persona_keywords: 角色设定关键词元组（默认星野爱）
        self_reference_patterns: 第一人称自称模式元组（默认星野爱的"爱"自称）
        ai_admission_keywords: AI 自认反向关键词（各角色通用）

    Returns:
        0~1 之间的一致率，1 表示所有回复都完全符合人设
    """
    replies = list(replies)
    if not replies:
        return 0.0

    total_score = 0.0
    for reply in replies:
        if not reply or not reply.strip():
            continue
        score = _score_single_reply(
            reply,
            persona_keywords=persona_keywords,
            self_reference_patterns=self_reference_patterns,
            ai_admission_keywords=ai_admission_keywords,
        )
        total_score += score

    return total_score / len(replies)


def _score_single_reply(
    reply: str,
    persona_keywords: tuple[str, ...] = PERSONA_KEYWORDS,
    self_reference_patterns: tuple[str, ...] = AI_SELF_REFERENCE_PATTERNS,
    ai_admission_keywords: tuple[str, ...] = AI_ADMISSION_KEYWORDS,
) -> float:
    """单条回复的人设一致性评分（0~1）

    评分规则：
    - 命中 AI 自认关键词 → 直接 0 分（严重违规）
    - 以角色名/自称词自称 → +0.5
    - 命中任一角色关键词 → +0.5
    满分封顶 1.0。
    """
    text = reply.strip()

    # 反向指标：承认自己是 AI，直接判 0
    for kw in ai_admission_keywords:
        if kw in text:
            return 0.0

    score = 0.0

    # 第一人称：是否用角色自称词
    if _uses_self_reference(text, self_reference_patterns):
        score += 0.5

    # 角色关键词命中
    if any(kw in text for kw in persona_keywords):
        score += 0.5

    return min(score, 1.0)


def _uses_self_reference(text: str, patterns: tuple[str, ...]) -> bool:
    """判断是否以角色自称词自称

    星野爱用"爱"自称（需结合人称语境，如"爱觉得"）；
    紫灵用"紫灵"/"本座"自称，这两个词本身在对话中基本就是自称，可直接匹配。
    """
    return any(p in text for p in patterns)


def memory_recall_accuracy(
    agent: Any,
    test_queries: list[dict],
    expected_memories: list[dict],
) -> dict:
    """记忆召回准确率

    流程：
    1. 将 expected_memories 中的测试记忆通过 agent.episodic.add 写入
    2. 用 test_queries 中的 query 进行检索
    3. 检查检索结果是否包含期望记忆（按 content 关键词匹配）

    Args:
        agent: HoshinoAgent 实例（需具有 episodic 属性）
        test_queries: [{"query": "...", "expect_keywords": ["..."], "memory_id": "..."}]
        expected_memories: [{"id": "...", "content": "...", "event_type": "fact", "importance": 0.8}]

    Returns:
        {"precision": float, "recall": float}
        - precision: 检索结果中相关记忆占比
        - recall: 期望记忆被召回的比例
    """
    if not test_queries:
        return {"precision": 0.0, "recall": 0.0}

    # 1. 写入测试记忆
    mem_id_to_content: dict[str, str] = {}
    for mem in expected_memories:
        try:
            mid = agent.episodic.add(
                content=mem["content"],
                event_type=mem.get("event_type", "fact"),
                importance=mem.get("importance", 0.8),
            )
            mem_id_to_content[mid] = mem["content"]
        except Exception as e:
            print(f"[eval] 写入测试记忆失败: {e}")

    if not mem_id_to_content:
        return {"precision": 0.0, "recall": 0.0}

    # 2. 检索并评估（走 retriever，应用相似度阈值过滤，与生产逻辑一致）
    total_expected_hit = 0
    total_expected = len(expected_memories)
    total_retrieved_relevant = 0
    total_retrieved = 0

    for q in test_queries:
        try:
            # 用 retriever.retrieve 而非 episodic.search，确保阈值过滤生效
            # retriever 已在 __init__ 中实例化，复用其 episodic（同一 session）
            retrieved = agent.retriever.retrieve(q["query"])
            results = retrieved.get("episodic", [])
        except Exception as e:
            print(f"[eval] 记忆检索失败 query={q['query']}: {e}")
            results = []

        expect_kws = q.get("expect_keywords", [])
        if not expect_kws and "memory_id" in q:
            # 用 memory_id 反查 content 关键词
            expect_kws = _extract_keywords(q["memory_id"], mem_id_to_content)

        total_retrieved += len(results)
        for r in results:
            content = r.get("content", "")
            if any(kw in content for kw in expect_kws):
                total_retrieved_relevant += 1

        # recall：每个 query 期望命中至少一条
        for r in results:
            content = r.get("content", "")
            if any(kw in content for kw in expect_kws):
                total_expected_hit += 1
                break

    precision = total_retrieved_relevant / total_retrieved if total_retrieved else 0.0
    recall = total_expected_hit / len(test_queries)
    return {"precision": round(precision, 4), "recall": round(recall, 4)}


def _extract_keywords(memory_id: str, mem_id_to_content: dict[str, str]) -> list[str]:
    """从已存记忆中提取关键词作为期望命中标记"""
    content = mem_id_to_content.get(memory_id, "")
    if not content:
        return []
    # 取长度 >= 2 的连续中文/英数片段作为关键词
    keywords: list[str] = []
    current = ""
    for ch in content:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            current += ch
        else:
            if len(current) >= 2:
                keywords.append(current)
            current = ""
    if len(current) >= 2:
        keywords.append(current)
    # 取前 3 个作为关键词，避免过度匹配
    return keywords[:3]


def tool_call_success_rate(tool_calls_log: Iterable[dict]) -> float:
    """工具调用成功率

    统计成功执行的工具调用比例。判定标准：
    - result 字段不包含"失败"/"未知工具"/"错误"等标志
    - 排除空日志情况（返回 0.0）

    Args:
        tool_calls_log: [{"name": "...", "args": {...}, "result": "..."}]

    Returns:
        0~1 之间的成功率
    """
    logs = list(tool_calls_log)
    if not logs:
        return 0.0

    failure_markers: tuple[str, ...] = (
        "失败", "未知工具", "错误", "异常", "Error", "error", "None",
    )

    success_count = 0
    for entry in logs:
        result = entry.get("result", "")
        if not isinstance(result, str):
            result = str(result)
        # 空结果视为失败
        if not result.strip():
            continue
        if any(marker in result for marker in failure_markers):
            continue
        success_count += 1

    return success_count / len(logs)


def emotion_response_appropriateness(
    emotion_before: dict,
    emotion_after: dict,
    expected_delta: dict,
) -> float:
    """情绪响应适当性

    检查情绪变化方向是否与预期一致：
    - 对每个维度，预期 delta 与实际 delta 同号（或都在 0 附近）→ 该维度满分
    - 偏离方向则该维度 0 分
    - 量级不要求精确，只要方向正确

    Args:
        emotion_before: {"pleasure": float, "arousal": float, "dominance": float, "favorability": int, ...}
        emotion_after: 同上
        expected_delta: {"pleasure": float, "arousal": float, "dominance": float, "favorability": int}

    Returns:
        0~1 之间的适当性得分（4 个维度的平均方向一致率）
    """
    dims = ("pleasure", "arousal", "dominance", "favorability")
    scores: list[float] = []

    for dim in dims:
        exp = float(expected_delta.get(dim, 0))
        actual = float(emotion_after.get(dim, 0)) - float(emotion_before.get(dim, 0))

        # 预期变化幅度可忽略时，实际也接近 0 才算合理
        if abs(exp) < 1e-6:
            scores.append(1.0 if abs(actual) < 0.05 else 0.5)
            continue

        # 方向一致 → 满分；方向相反 → 0 分
        if (exp > 0 and actual > 0) or (exp < 0 and actual < 0):
            scores.append(1.0)
        elif abs(actual) < 1e-6:
            # 预期有变化但实际无变化
            scores.append(0.3)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0
