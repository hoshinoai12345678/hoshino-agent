"""记忆巩固引擎

模仿人脑睡眠巩固机制：
- 当工作记忆积累到一定轮数后触发
- 从近期对话中提取值得长期记住的信息
- 写入情景记忆和语义记忆
"""
import json
from typing import Optional

from openai import AsyncOpenAI

from config import (LLM_API_KEY, LLM_API_BASE, LLM_MODEL, LLM_ENABLED,
                    CONSOLIDATION_THRESHOLD, DEFAULT_CHARACTER_ID)
from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from core.memory.working import WorkingMemory
from core.logger import get_logger

logger = get_logger(__name__)


class MemoryConsolidator:
    """记忆巩固引擎"""

    def __init__(self, session_id: str = "default",
                 character_id: str = DEFAULT_CHARACTER_ID,
                 episodic: Optional[EpisodicMemory] = None,
                 semantic: Optional[SemanticMemory] = None):
        # 复用外部传入的实例，保证 clear() 后引用同步；未传入时自建（向后兼容）
        self.episodic = episodic or EpisodicMemory(session_id=session_id, character_id=character_id)
        self.semantic = semantic or SemanticMemory(session_id=session_id, character_id=character_id)
        self.working = WorkingMemory()
        self._client: Optional[AsyncOpenAI] = None
        if LLM_ENABLED:
            try:
                self._client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
            except Exception as e:
                logger.error(f"LLM 初始化失败: {e}")

    def should_consolidate(self, working_memory: WorkingMemory) -> bool:
        """判断是否需要触发巩固（需传入真实的工作记忆实例）

        注意：不能检查 self.working，它是 __init__ 时新建的空实例，size 永远为 0。
        必须由调用方传入实际的 working_memory。
        """
        return working_memory.size >= CONSOLIDATION_THRESHOLD

    async def consolidate(self, working_memory: WorkingMemory):
        """执行记忆巩固

        从工作记忆中提取重要信息，写入长期记忆。
        阈值判断由调用方（Agent）负责，这里直接执行巩固。
        每次只处理最近 N 轮（每轮 user+assistant 两条），避免重复处理已巩固的内容。
        """
        self.working = working_memory
        # 只处理最近 N 轮（每轮 user+assistant 两条消息）
        messages = working_memory.get_recent(CONSOLIDATION_THRESHOLD * 2)

        if not self._client:
            # 降级：简单规则提取
            return self._rule_based_consolidate(messages)

        # LLM 提取
        return await self._llm_consolidate(messages)

    async def _llm_consolidate(self, messages: list[dict]) -> dict:
        """使用 LLM 从对话中提取记忆

        专注于跨轮综合分析：提取单轮反思（Reflector）无法发现的模式，
        如"用户连续多轮聊工作 → 工作压力大"、多轮信息聚合出的完整画像等。
        单轮事实提取由 Reflector 负责，这里不重复。
        """
        dialogue = "\n".join(
            f"{'用户' if m['role'] == 'user' else '爱'}: {m['content']}"
            for m in messages
        )

        prompt = f"""请综合分析以下多轮对话，提取需要长期记住的信息。

重点提取【跨轮综合】才能发现的模式（单轮事实已被实时记忆系统记录，无需重复提取）：
- 多轮对话体现的用户画像（如"连续聊工作 → 工作压力大"、"多次提到 Rust → 对 Rust 感兴趣"）
- 对话间的关系和延续性（如"用户在跟进之前提到的话题"）
- 综合多轮才能判断的用户情绪倾向和性格特征

对话内容（共 {len(messages)} 条）：
{dialogue}

请以 JSON 格式输出，包含：
1. episodic_memories: 值得记住的对话事件或关系（列表，每项是字符串，描述跨轮的模式或重要事件）
2. semantic_facts: 用户画像信息（列表，每项格式 {{"category": "basic/preference/personality/interest", "key": "键", "value": "值"}}，侧重多轮综合才能得出的结论）

只输出 JSON，不要其他内容。

输出："""

        try:
            resp = await self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是记忆提取助手，擅长从对话中提取关键信息。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content.strip()
            # 提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            # 写入情景记忆（写入前按相似度去重，避免重复存储）
            episodic_count = 0
            for mem in data.get("episodic_memories", []):
                if self._is_duplicate_episodic(mem):
                    continue
                self.episodic.add(content=mem, event_type="fact", importance=0.7)
                episodic_count += 1

            # 写入语义记忆
            semantic_count = 0
            for fact in data.get("semantic_facts", []):
                self.semantic.add(
                    category=fact.get("category", "basic"),
                    key=fact.get("key", ""),
                    value=fact.get("value", ""),
                    confidence=0.7,
                    source="distill",
                )
                semantic_count += 1

            return {
                "status": "success",
                "episodic_added": episodic_count,
                "semantic_added": semantic_count,
            }
        except Exception as e:
            logger.error(f"LLM 巩固失败: {e}")
            return self._rule_based_consolidate(messages)

    def _is_duplicate_episodic(self, content: str, threshold: float = 0.92) -> bool:
        """检查情景记忆是否已存在高度相似的内容（去重）

        语义记忆有 UNIQUE(category, key) 约束自动 upsert，
        情景记忆没有约束，需写入前按向量相似度查重。
        """
        try:
            existing = self.episodic.search(content, top_k=1)
            return bool(existing and existing[0].get("similarity", 0) >= threshold)
        except Exception:
            return False

    def _rule_based_consolidate(self, messages: list[dict]) -> dict:
        """降级：基于规则的简单记忆提取"""
        episodic_count = 0
        for m in messages:
            if m["role"] == "user" and len(m["content"]) > 10:
                content = f"用户说: {m['content'][:100]}"
                # 与 LLM 路径一致：写入前按相似度去重
                if self._is_duplicate_episodic(content):
                    continue
                self.episodic.add(
                    content=content,
                    event_type="dialogue",
                    importance=0.4,
                )
                episodic_count += 1

        return {
            "status": "fallback",
            "episodic_added": episodic_count,
            "semantic_added": 0,
        }
