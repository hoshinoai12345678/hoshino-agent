"""混合检索器

整合角色知识库、情景记忆、语义记忆，提供统一的检索接口。
"""
from typing import Optional

from config import KNOWLEDGE_TOP_K, EPISODIC_TOP_K
from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from core.utils import embed, embed_query, cosine_similarity
from rag.indexer import KnowledgeIndexer

# 相似度阈值：rerank 后低于此值的检索结果视为不相关，过滤掉以提升 precision
# BGE-small-zh 归一化向量的 cosine 相似度经验值：0.5+ 强相关，0.35~0.5 弱相关，<0.35 噪声
KNOWLEDGE_SIM_THRESHOLD = 0.35
EPISODIC_SIM_THRESHOLD = 0.35


class HybridRetriever:
    """混合检索器：知识 + 记忆"""

    def __init__(self, session_id: str = "default",
                 episodic: Optional[EpisodicMemory] = None,
                 semantic: Optional[SemanticMemory] = None):
        self.knowledge_indexer = KnowledgeIndexer()  # 角色知识全局共享
        # 复用外部传入的实例，保证 clear() 后引用同步；未传入时自建（向后兼容）
        self.episodic = episodic or EpisodicMemory(session_id=session_id)
        self.semantic = semantic or SemanticMemory(session_id=session_id)

    def retrieve(self, query: str) -> dict:
        """混合检索（去重 + rerank + 阈值过滤）

        Returns:
            {
                "knowledge": [...],   # 角色知识（rerank + 阈值过滤后）
                "episodic": [...],    # 情景记忆（去重 + rerank + 阈值过滤后）
                "semantic": [...],    # 语义记忆
            }
        """
        # 1. 检索角色知识
        knowledge = self.knowledge_indexer.search(query, top_k=KNOWLEDGE_TOP_K)

        # 2. 检索情景记忆 + 去重
        episodic = self._dedup(self.episodic.search(query, top_k=EPISODIC_TOP_K))

        # 3. 检索语义记忆（用户画像）
        semantic = self.semantic.search(query)

        # 4. Rerank：用精确 cosine 相似度重排序（ChromaDB HNSW 是近似检索）
        knowledge = self._rerank(query, knowledge)
        episodic = self._rerank(query, episodic)

        # 5. 阈值过滤：rerank 后丢弃弱相关结果，提升 precision
        knowledge = [k for k in knowledge if k.get("rerank_score", 0) >= KNOWLEDGE_SIM_THRESHOLD]
        episodic = [m for m in episodic if m.get("rerank_score", 0) >= EPISODIC_SIM_THRESHOLD]

        return {
            "knowledge": knowledge,
            "episodic": episodic,
            "semantic": semantic,
        }

    @staticmethod
    def _dedup(items: list[dict], threshold: float = 0.95) -> list[dict]:
        """按内容向量相似度去重"""
        if not items:
            return []
        result = []
        vectors = [embed(item.get("content", "")) for item in items]
        for i, item in enumerate(items):
            is_dup = False
            for j in range(i):
                if cosine_similarity(vectors[i], vectors[j]) > threshold:
                    is_dup = True
                    break
            if not is_dup:
                result.append(item)
        return result

    @staticmethod
    def _rerank(query: str, items: list[dict]) -> list[dict]:
        """用 query-document 精确 cosine 相似度重排序"""
        if not items:
            return items
        query_vec = embed_query(query)
        for item in items:
            doc_vec = embed(item.get("content", ""))
            item["rerank_score"] = cosine_similarity(query_vec, doc_vec)
        items.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return items

    def to_prompt_context(self, query: str) -> str:
        """生成检索结果上下文，注入 Prompt"""
        results = self.retrieve(query)

        lines = []

        # 角色知识
        if results["knowledge"]:
            lines.append("【相关角色知识】")
            for k in results["knowledge"]:
                lines.append(f"- {k['content'][:200]}")

        # 情景记忆
        if results["episodic"]:
            lines.append("\n【相关历史记忆】")
            for m in results["episodic"]:
                lines.append(f"- {m['content'][:200]}（相似度:{m['similarity']:.2f}）")

        # 语义记忆
        if results["semantic"]:
            lines.append("\n【已知用户信息】")
            for s in results["semantic"][:5]:
                lines.append(f"- {s['key']}: {s['value']}")

        return "\n".join(lines) if lines else "（无相关检索结果）"
