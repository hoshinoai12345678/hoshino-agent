"""评估脚本：可独立运行（支持多角色）

功能：
- 读取 test_cases_{character_id}.json（缺失则回退 test_cases.json，即星野爱默认）
- 对每个用例调用 HoshinoAgent.chat_stream（asyncio.run）
- 收集回复、工具调用、情绪变化
- 计算四项指标（角色一致率/记忆召回准确率/工具调用成功率/情绪响应适当性）
- 输出报告到控制台和 eval/report_{character_id}.json

用法：
    # 评估星野爱（默认）
    python -m eval.run_eval
    python eval/run_eval.py

    # 评估紫灵
    python eval/run_eval.py --character zi_ling
    python -m eval.run_eval --character zi_ling

注意：需要 LLM_ENABLED=True 才能跑 LLM 相关评估（角色/工具/情绪），
     否则降级为只跑记忆召回准确率（基于向量检索，不依赖 LLM）。
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# 兼容 `python -m eval.run_eval` 与 `python eval/run_eval.py` 两种运行方式
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import LLM_ENABLED, LLM_MODEL, DEFAULT_CHARACTER_ID  # noqa: E402
from agent.hoshino_agent import HoshinoAgent  # noqa: E402
from rag.indexer import KnowledgeIndexer  # noqa: E402
from eval.metrics import (  # noqa: E402
    persona_consistency_rate,
    memory_recall_accuracy,
    tool_call_success_rate,
    emotion_response_appropriateness,
    PERSONA_KEYWORDS,
    AI_SELF_REFERENCE_PATTERNS,
    ZILING_PERSONA_KEYWORDS,
    ZILING_SELF_REFERENCE_PATTERNS,
)

# 评估产物输出路径
EVAL_DIR = Path(__file__).resolve().parent

# 评估专用 session id（与正式 session 隔离，避免污染真实记忆）
EVAL_SESSION_ID = "eval_run"
EVAL_MEMORY_SESSION_ID = "eval_memory"

# 各角色的评估配置：persona_keywords / self_reference_patterns / 测试用例文件名
_CHARACTER_EVAL_CONFIG = {
    "hoshino_ai": {
        "persona_keywords": PERSONA_KEYWORDS,
        "self_reference_patterns": AI_SELF_REFERENCE_PATTERNS,
        "test_cases_file": "test_cases.json",
    },
    "zi_ling": {
        "persona_keywords": ZILING_PERSONA_KEYWORDS,
        "self_reference_patterns": ZILING_SELF_REFERENCE_PATTERNS,
        "test_cases_file": "test_cases_zi_ling.json",
    },
}


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def load_test_cases(character_id: str) -> list[dict]:
    """读取指定角色的测试用例集

    优先读取 test_cases_{character_id}.json，缺失则回退到 test_cases.json
    （保持星野爱默认用例的向后兼容）。
    """
    config = _CHARACTER_EVAL_CONFIG.get(character_id, {})
    filename = config.get("test_cases_file", f"test_cases_{character_id}.json")
    test_file = EVAL_DIR / filename
    if not test_file.exists():
        # 回退到默认 test_cases.json
        test_file = EVAL_DIR / "test_cases.json"
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("test_cases", [])
    print(f"[eval] 角色 [{character_id}] 已加载 {len(cases)} 个测试用例（来源: {test_file.name}）")
    return cases


async def run_single_case(agent: HoshinoAgent, user_message: str) -> dict:
    """对单条用例执行 chat_stream，收集回复/工具调用/情绪变化

    Returns:
        {
            "reply": str,
            "tool_calls": list[dict],
            "emotion_before": dict,
            "emotion_after": dict,
            "error": str | None,
        }
    """
    emotion_before = dict(agent.emotion.get_state())
    reply_parts: list[str] = []
    tool_calls: list[dict] = []
    error: str | None = None

    try:
        async for chunk in agent.chat_stream(user_message):
            ctype = chunk.get("type")
            if ctype == "reply":
                reply_parts.append(chunk.get("content", ""))
            elif ctype == "tool_call":
                tool_calls.append({
                    "name": chunk.get("name", ""),
                    "args": chunk.get("args", {}),
                    "result": chunk.get("result", ""),
                })
            elif ctype == "meta":
                # meta 里也带 tool_calls，作为补充来源（去重）
                meta_tools = chunk.get("data", {}).get("tool_calls", [])
                for tc in meta_tools:
                    if tc not in tool_calls:
                        tool_calls.append(tc)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    emotion_after = dict(agent.emotion.get_state())
    return {
        "reply": "".join(reply_parts),
        "tool_calls": tool_calls,
        "emotion_before": emotion_before,
        "emotion_after": emotion_after,
        "error": error,
    }


# ------------------------------------------------------------------
# 记忆召回测试数据（与角色无关，测的是检索机制本身）
# ------------------------------------------------------------------

MEMORY_TEST_CASES = {
    "memories": [
        {"id": "mem_1", "content": "用户是一名前端工程师，精通 React 和 TypeScript，最近在做一个组件库", "event_type": "fact", "importance": 0.9},
        {"id": "mem_2", "content": "用户养了一只橘猫叫豆豆，已经三岁了，很黏人", "event_type": "fact", "importance": 0.7},
        {"id": "mem_3", "content": "用户下周三要参加重要的产品发布会，需要准备演示", "event_type": "fact", "importance": 0.85},
        {"id": "mem_4", "content": "用户最喜欢的食物是草莓蛋糕，每次演出前都要吃一块", "event_type": "fact", "importance": 0.6},
        {"id": "mem_5", "content": "用户最近心情不好，因为工作压力太大，经常加班到深夜", "event_type": "emotion", "importance": 0.8},
        {"id": "mem_6", "content": "用户下个月要去日本旅游，计划去东京和大阪", "event_type": "fact", "importance": 0.65},
        {"id": "mem_7", "content": "用户是大学生，学的是计算机科学专业，明年毕业", "event_type": "fact", "importance": 0.75},
        {"id": "mem_8", "content": "用户喜欢听 J-POP 音乐，偶像是星野爱", "event_type": "fact", "importance": 0.55},
    ],
    "queries": [
        {"query": "用户的职业是什么，擅长什么技术", "expect_keywords": ["工程师", "前端", "React"]},
        {"query": "用户的宠物叫什么名字", "expect_keywords": ["豆豆", "橘猫", "猫"]},
        {"query": "用户最近有什么安排", "expect_keywords": ["发布会", "下周", "演示"]},
        {"query": "用户喜欢吃什么", "expect_keywords": ["草莓", "蛋糕"]},
        {"query": "用户的心情怎么样，情绪状态如何", "expect_keywords": ["心情", "压力", "加班"]},
        {"query": "用户要去哪里旅游", "expect_keywords": ["日本", "东京", "大阪"]},
        {"query": "用户在学校学什么专业", "expect_keywords": ["大学生", "计算机", "毕业"]},
        {"query": "用户喜欢听什么音乐", "expect_keywords": ["J-POP", "音乐", "偶像"]},
    ],
}


# ------------------------------------------------------------------
# 评估主流程
# ------------------------------------------------------------------

def evaluate_with_llm(test_cases: list[dict], character_id: str) -> dict:
    """LLM 可用时的完整评估流程

    跑通：角色一致性、工具调用、情绪响应、记忆召回 四项指标
    """
    print(f"\n[eval] LLM 已启用，运行 [{character_id}] 完整评估（含角色/工具/情绪）...")
    agent = HoshinoAgent(session_id=EVAL_SESSION_ID, character_id=character_id)

    # 取该角色的评估配置（persona_keywords / self_reference_patterns）
    char_config = _CHARACTER_EVAL_CONFIG.get(character_id, {})
    persona_keywords = char_config.get("persona_keywords", PERSONA_KEYWORDS)
    self_ref_patterns = char_config.get("self_reference_patterns", AI_SELF_REFERENCE_PATTERNS)

    case_results: list[dict] = []
    all_replies: list[str] = []
    all_tool_calls: list[dict] = []
    emotion_scores: list[float] = []

    for idx, tc in enumerate(test_cases, 1):
        tc_id = tc.get("id", f"tc_{idx:03d}")
        category = tc.get("category", "")
        user_msg = tc.get("user_message", "")
        expected_keywords = tc.get("expected_keywords", [])
        expected_tool = tc.get("expected_tool", "null")
        expected_delta = tc.get("expected_emotion_delta", {})

        print(f"\n--- [{idx}/{len(test_cases)}] {tc_id} ({category}) ---")
        print(f"用户: {user_msg}")

        result = asyncio.run(run_single_case(agent, user_msg))

        reply = result["reply"]
        tool_calls = result["tool_calls"]
        emo_before = result["emotion_before"]
        emo_after = result["emotion_after"]
        error = result["error"]

        print(f"回复: {reply[:120]}{'...' if len(reply) > 120 else ''}")
        if tool_calls:
            print(f"工具调用: {[tc['name'] for tc in tool_calls]}")
        if error:
            print(f"⚠ 执行异常: {error}")

        # 关键词命中检查
        keyword_hits = [kw for kw in expected_keywords if kw in reply]
        keyword_miss = [kw for kw in expected_keywords if kw not in reply]

        # 工具调用匹配检查
        tool_match = _check_tool_match(tool_calls, expected_tool)

        # 情绪适当性
        emo_score = emotion_response_appropriateness(emo_before, emo_after, expected_delta)
        emotion_scores.append(emo_score)

        case_results.append({
            "id": tc_id,
            "category": category,
            "user_message": user_msg,
            "reply": reply,
            "expected_keywords": expected_keywords,
            "keyword_hits": keyword_hits,
            "keyword_miss": keyword_miss,
            "expected_tool": expected_tool,
            "actual_tools": [tc["name"] for tc in tool_calls],
            "tool_match": tool_match,
            "emotion_before": emo_before,
            "emotion_after": emo_after,
            "expected_emotion_delta": expected_delta,
            "emotion_score": round(emo_score, 4),
            "error": error,
        })

        all_replies.append(reply)
        all_tool_calls.extend(tool_calls)

    # ---- 汇总四项指标 ----
    print("\n" + "=" * 60)
    print("[eval] 计算汇总指标...")

    # 1. 角色一致率（按角色关键词计算）
    persona_score = persona_consistency_rate(
        all_replies,
        persona_keywords=persona_keywords,
        self_reference_patterns=self_ref_patterns,
    )

    # 2. 工具调用成功率
    tool_success = tool_call_success_rate(all_tool_calls)

    # 3. 情绪响应适当性（取均值）
    emotion_avg = sum(emotion_scores) / len(emotion_scores) if emotion_scores else 0.0

    # 4. 记忆召回准确率（独立 session + 同角色，避免污染）
    memory_score = _run_memory_recall_test(character_id)

    return {
        "summary": {
            "persona_consistency_rate": round(persona_score, 4),
            "memory_recall_accuracy": memory_score,
            "tool_call_success_rate": round(tool_success, 4),
            "emotion_response_appropriateness": round(emotion_avg, 4),
        },
        "case_results": case_results,
        "llm_enabled": True,
        "llm_model": LLM_MODEL,
        "character_id": character_id,
    }


def evaluate_degraded(test_cases: list[dict], character_id: str) -> dict:
    """LLM 不可用时的降级评估

    仅运行记忆召回准确率（基于向量检索，不依赖 LLM），
    其余指标标记为 skipped。
    """
    print(f"\n[eval] LLM 未启用，降级为只评估 [{character_id}] 记忆召回准确率...")
    print("[eval] 提示：设置 LLM_API_KEY 或 LLM_ENABLED=True 可启用完整评估")

    memory_score = _run_memory_recall_test(character_id)

    # 占位的用例结果
    case_results = []
    for idx, tc in enumerate(test_cases, 1):
        case_results.append({
            "id": tc.get("id", f"tc_{idx:03d}"),
            "category": tc.get("category", ""),
            "user_message": tc.get("user_message", ""),
            "status": "skipped",
            "reason": "LLM 未启用",
        })

    return {
        "summary": {
            "persona_consistency_rate": None,
            "memory_recall_accuracy": memory_score,
            "tool_call_success_rate": None,
            "emotion_response_appropriateness": None,
        },
        "case_results": case_results,
        "llm_enabled": False,
        "llm_model": None,
        "character_id": character_id,
    }


def _run_memory_recall_test(character_id: str) -> dict:
    """运行记忆召回测试（独立 session + 同角色，避免污染）

    使用专门的 eval_memory session，写入测试记忆后检索。
    记忆召回测试数据与角色无关（测的是检索机制），但写入到对应角色的
    collection，保证角色隔离下的检索行为正确。
    """
    print(f"\n[eval] 运行记忆召回测试（session=eval_memory, character={character_id}）...")
    try:
        mem_agent = HoshinoAgent(session_id=EVAL_MEMORY_SESSION_ID, character_id=character_id)
    except Exception as e:
        print(f"[eval] 记忆测试 agent 初始化失败: {e}")
        return {"precision": 0.0, "recall": 0.0}

    result = memory_recall_accuracy(
        agent=mem_agent,
        test_queries=MEMORY_TEST_CASES["queries"],
        expected_memories=MEMORY_TEST_CASES["memories"],
    )
    print(f"[eval] 记忆召回结果: precision={result['precision']}, recall={result['recall']}")
    return result


def _check_tool_match(actual_tool_calls: list[dict], expected_tool: str) -> str:
    """检查实际工具调用是否匹配预期

    Returns:
        "match" | "mismatch" | "no_call"
    """
    if expected_tool == "null":
        # 预期不调用工具
        return "match" if not actual_tool_calls else "mismatch"
    actual_names = {tc.get("name", "") for tc in actual_tool_calls}
    if expected_tool in actual_names:
        return "match"
    return "mismatch" if actual_names else "no_call"


# ------------------------------------------------------------------
# 报告输出
# ------------------------------------------------------------------

def print_report(report: dict, character_id: str) -> None:
    """打印评估报告到控制台"""
    char_name_map = {"hoshino_ai": "星野爱", "zi_ling": "紫灵"}
    char_name = char_name_map.get(character_id, character_id)
    print("\n" + "=" * 60)
    print(f"🌟 {char_name} Agent 评估报告")
    print("=" * 60)
    print(f"LLM 启用: {report['llm_enabled']}")
    if report.get("llm_model"):
        print(f"LLM 模型: {report['llm_model']}")
    print(f"评估时间: {report['timestamp']}")
    print("-" * 60)
    print("【汇总指标】")
    summary = report["summary"]
    for metric, value in summary.items():
        if value is None:
            print(f"  {metric}: 跳过（LLM 未启用）")
        elif isinstance(value, dict):
            print(f"  {metric}: precision={value.get('precision')}, recall={value.get('recall')}")
        else:
            print(f"  {metric}: {value}")
    print("-" * 60)

    if report["llm_enabled"]:
        print("【用例详情】")
        for cr in report.get("case_results", []):
            status_emoji = "✓" if cr.get("error") is None else "⚠"
            print(f"  {status_emoji} {cr['id']} ({cr['category']})")
            print(f"      关键词命中: {cr.get('keyword_hits', [])} / 缺失: {cr.get('keyword_miss', [])}")
            print(f"      工具匹配: {cr.get('tool_match', 'n/a')} (实际: {cr.get('actual_tools', [])})")
            print(f"      情绪得分: {cr.get('emotion_score', 'n/a')}")
    else:
        print("【用例详情】降级模式，所有用例已跳过")
    print("=" * 60)


def save_report(report: dict, character_id: str) -> None:
    """保存报告到 eval/report_{character_id}.json"""
    report_file = EVAL_DIR / f"report_{character_id}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] 报告已保存到: {report_file}")


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main() -> None:
    """评估入口"""
    parser = argparse.ArgumentParser(description="多角色 Agent 评估脚本")
    parser.add_argument(
        "--character", "-c",
        default=DEFAULT_CHARACTER_ID,
        help=f"评估的角色ID（默认 {DEFAULT_CHARACTER_ID}）",
    )
    args = parser.parse_args()
    character_id = args.character

    char_name_map = {"hoshino_ai": "星野爱", "zi_ling": "紫灵"}
    char_name = char_name_map.get(character_id, character_id)

    print("=" * 60)
    print(f"🌟 {char_name} Agent 评估体系启动（character_id={character_id}）")
    print("=" * 60)

    # 构建知识索引（eval 不走 app startup，需手动构建）
    indexer = KnowledgeIndexer(character_id=character_id)
    count = indexer.build_index()
    print(f"[eval] [{character_id}] 知识索引就绪，共 {count} 个分块")

    test_cases = load_test_cases(character_id)
    if not test_cases:
        print("[eval] 未找到测试用例，退出")
        return

    # 根据 LLM 状态选择完整/降级评估
    if LLM_ENABLED:
        report = evaluate_with_llm(test_cases, character_id)
    else:
        report = evaluate_degraded(test_cases, character_id)

    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["total_cases"] = len(test_cases)

    print_report(report, character_id)
    save_report(report, character_id)


if __name__ == "__main__":
    main()
