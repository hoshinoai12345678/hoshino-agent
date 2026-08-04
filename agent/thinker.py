"""思考者 Agent（Multi-agent 架构）

在 ReAct loop 前置一个思考者：
- 理解用户意图和情绪
- 决策是否需要工具调用及调用顺序
- 生成回应策略

这是 multi-agent 架构的体现：Thinker 负责"理解+决策"，ReAct loop 负责"执行"。
短消息（<10 字符）跳过，节省 token。
"""
from typing import Optional

from openai import AsyncOpenAI

from config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL, LLM_ENABLED
from core.logger import get_logger
from core.utils import extract_json

logger = get_logger(__name__)

# 短消息阈值：<10 字符跳过 LLM（如"你好""在吗"等极短问候），
# ≥10 字符的正常伴聊消息（如"今天好累啊""我喜欢草莓"）会触发思考
_SHORT_MSG_THRESHOLD = 10

# 多任务信号关键词（命中任一可能需要任务拆解）
_COMPLEX_SIGNALS = (
    "并且", "然后", "接着", "同时", "之后", "帮我", "分析",
    "对比", "比较", "总结", "规划", "计划", "步骤",
)


class Thinker:
    """思考者 Agent：ReAct 前的语义理解与决策"""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        if LLM_ENABLED:
            try:
                self._client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
            except Exception as e:
                logger.error(f"Thinker LLM 初始化失败: {e}")

    async def think(self, user_message: str, retrieval_ctx: str,
                    emotion_ctx: str, semantic_ctx: str) -> dict:
        """思考并返回结构化指导

        Returns:
            {
                "intent": str,           # 用户意图
                "user_emotion": str,     # 用户情绪判断
                "needs_tools": bool,     # 是否需要工具
                "needs_plan": bool,      # 是否需要多任务拆解
                "tool_plan": list[dict], # 工具调用计划（多任务时为步骤清单）
                "strategy": str,         # 回应策略
                "has_complex_signal": bool,  # 是否命中多任务关键词（辅助标记）
            }
            或 {} (跳过/失败时)
        """
        # 短消息跳过，省 token
        if len(user_message) < _SHORT_MSG_THRESHOLD:
            return {}

        # 检测多任务信号关键词，作为 LLM 判断 needs_plan 的辅助提示
        has_complex_signal = any(s in user_message for s in _COMPLEX_SIGNALS)

        if not self._client:
            # LLM 不可用：降级返回角色化独白，保证思考链总有内容
            return self._fallback_thought(user_message, has_complex_signal)

        try:
            prompt = self._build_prompt(user_message, retrieval_ctx, emotion_ctx, semantic_ctx, has_complex_signal)
            resp = await self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是星野爱的内心思考助手，分析用户请求并输出结构化思考。只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                # DeepSeek json_object 模式偶发空返回（预估超 max_tokens 时直接空）
                # 打日志方便排查，并降级返回简短独白保证 dev_mode 思考链总有内容
                finish = resp.choices[0].finish_reason if resp.choices else "unknown"
                logger.warning(f"Thinker LLM 返回空 content（finish_reason={finish}），降级生成独白")
                return self._fallback_thought(user_message, has_complex_signal)

            result = extract_json(content)
            if result is None:
                logger.warning(f"Thinker JSON 解析失败: {content[:100]}")
                return self._fallback_thought(user_message, has_complex_signal)

            # 规范化 tool_plan
            tool_plan = result.get("tool_plan", [])
            valid_plan = []
            for step in (tool_plan if isinstance(tool_plan, list) else []):
                if isinstance(step, dict) and "action" in step:
                    valid_plan.append({
                        "action": step.get("action", "reply"),
                        "target": str(step.get("target", "")),
                        "reason": str(step.get("reason", "")),
                    })

            return {
                "intent": str(result.get("intent", "")),
                "user_emotion": str(result.get("user_emotion", "")),
                "needs_tools": bool(result.get("needs_tools", False)),
                "needs_plan": bool(result.get("needs_plan", False)),
                "tool_plan": valid_plan,
                "strategy": str(result.get("strategy", "")),
                "has_complex_signal": has_complex_signal,
                "inner_monologue": str(result.get("inner_monologue", "")),
            }
        except Exception as e:
            logger.error(f"Thinker 失败: {e}")
            return self._fallback_thought(user_message, has_complex_signal)

    @staticmethod
    def _fallback_thought(user_message: str, has_complex_signal: bool) -> dict:
        """LLM 失败/空返回/JSON 解析失败的降级：用规则生成简短角色化独白

        保证 dev_mode 思考链总有内容展示，system prompt 总有上下文注入。
        独白质量不如 LLM 生成，但比空值好——ReAct 主要靠 tools 定义和 user_message，
        独白只是辅助上下文，简短独白不会误导 ReAct。
        """
        snippet = user_message[:30] + ("..." if len(user_message) > 30 else "")
        monologue = f"爱听到你说「{snippet}」~让爱想想怎么回应才好呢♪"
        if has_complex_signal:
            monologue += " 这件事有好几步，爱要一个一个来~"
        return {
            "intent": "",
            "user_emotion": "",
            "needs_tools": False,
            "needs_plan": False,
            "tool_plan": [],
            "strategy": "",
            "has_complex_signal": has_complex_signal,
            "inner_monologue": monologue,
        }

    @staticmethod
    def _build_prompt(user_message: str, retrieval_ctx: str,
                      emotion_ctx: str, semantic_ctx: str,
                      has_complex_signal: bool = False) -> str:
        signal_hint = f"（已检测到多任务关键词，请重点判断是否需要拆解）" if has_complex_signal else ""
        return f"""请分析用户请求，输出结构化思考。

用户消息：「{user_message}」{signal_hint}

当前情绪状态：
{emotion_ctx}

用户画像：
{semantic_ctx}

检索到的相关信息：
{retrieval_ctx}

【多任务判断标准】
- needs_plan=true：请求包含多个子任务、需要组合多种工具、需要先检索再回答
  例："我叫小枫，记住我喜欢草莓，然后告诉我爱酱的口头禅" → 拆为 [save_memory, save_memory, search_knowledge, reply]
- needs_plan=false：单轮问答、简单问候、闲聊、直接回复

【可用 action】
- search_memory: 检索情景记忆
- search_knowledge: 检索角色知识库
- save_memory: 保存用户信息
- update_emotion: 更新情绪
- reply: 直接回复用户

输出JSON：
{{
  "intent": "用户真实意图（一句话）",
  "user_emotion": "用户当前情绪（如：好奇/开心/悲伤/中性）",
  "needs_tools": true/false,
  "needs_plan": true/false,
  "tool_plan": [
    {{"action": "save_memory", "target": "...", "reason": "..."}}
  ],
  "strategy": "回应策略（如何回复才符合角色且让用户舒适）",
  "inner_monologue": "用星野爱第一人称写的内心独白（自称'爱'，活泼偶像风格，带情感和符号）。自然表达：对用户这句话的反应、当前感受、打算怎么回应、是否需要查记忆/知识库。多任务时自然提到'先...再...'。像真实内心活动，不要用机械标签或列表格式。"
}}

只输出JSON："""

    def to_prompt_context(self, thought: dict) -> str:
        """把思考结果转为角色化内心独白

        作为 system prompt 注入 + dev_mode 思考链展示（同一份文本，本末一致）。
        优先用 LLM 生成的 inner_monologue（星野爱第一人称）；
        降级时用 strategy 拼简短独白，保证总有内容。
        """
        if not thought:
            return ""

        # 优先：LLM 生成的角色化内心独白
        monologue = thought.get("inner_monologue", "")
        if monologue:
            return monologue

        # 降级：LLM 没返回 inner_monologue 时，用 intent + strategy 拼一个
        strategy = thought.get("strategy", "")
        if not strategy:
            return ""

        intent = thought.get("intent", "")
        if intent:
            return f"爱在想：{intent}。{strategy}"
        return f"爱心里想：{strategy}"
