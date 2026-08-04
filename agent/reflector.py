"""反思模块

回复生成后自检：是否需要更新记忆/情绪/画像。
使用 JSON 模式（response_format=json_object）+ 强化提取，保证结构化输出可靠。
注：DeepSeek thinking mode 不支持强制 tool_choice，故不用 Function Calling。
"""
from typing import Optional

from openai import AsyncOpenAI

from config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL, LLM_ENABLED
from agent.prompts import build_reflection_prompt
from core.logger import get_logger
from core.utils import extract_json

logger = get_logger(__name__)


class Reflector:
    """反思引擎（基于 JSON 模式 + 强化提取）"""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        if LLM_ENABLED:
            try:
                self._client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
            except Exception as e:
                logger.error(f"LLM 初始化失败: {e}")

    async def reflect(self, user_message: str, assistant_reply: str,
                      emotion_ctx: str) -> dict:
        """反思对话，返回需要执行的更新

        Returns:
            {
                "user_revealed": str | None,
                "emotion_change": dict,
                "should_remember": bool,
                "persona_consistent": bool,
            }
        """
        if not self._client:
            return self._default_result()

        try:
            prompt = build_reflection_prompt(user_message, assistant_reply, emotion_ctx)
            resp = await self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是反思助手，分析对话并输出结构化结果。只输出JSON，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                return self._default_result()
            content = content.strip()

            result = extract_json(content)
            if result is None:
                logger.error(f"JSON 解析失败，原始内容: {content[:100]}")
                return self._default_result()

            return {
                "user_revealed": result.get("user_revealed") or None,
                "user_profile": result.get("user_profile", []) or [],
                "emotion_change": result.get("emotion_change", {
                    "pleasure": 0, "arousal": 0, "dominance": 0, "favorability": 0
                }),
                "should_remember": bool(result.get("should_remember", False)),
                "persona_consistent": bool(result.get("persona_consistent", True)),
            }
        except Exception as e:
            logger.error(f"反思失败: {e}")
            return self._default_result()

    @staticmethod
    def _default_result() -> dict:
        return {
            "user_revealed": None,
            "user_profile": [],
            "emotion_change": {"pleasure": 0, "arousal": 0, "dominance": 0, "favorability": 0},
            "should_remember": False,
            "persona_consistent": True,
        }
