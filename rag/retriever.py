"""混合检索器

整合角色知识库、情景记忆、语义记忆，提供统一的检索接口。

检索流程（两阶段检索）：
1. 召回阶段：bi-encoder（BGE-small-zh）向量检索 top_k + 去重
2. 精排阶段：cross-encoder（BGE-reranker）对 query-doc 对精确打分重排序
   - cross-encoder 不可用时降级为 bi-encoder 余弦相似度重排
3. 阈值过滤：丢弃弱相关结果
"""
from typing import Optional

from config import (KNOWLEDGE_TOP_K, EPISODIC_TOP_K,
                    ENABLE_CROSS_ENCODER_RERANK, CROSS_ENCODER_THRESHOLD)
from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from core.utils import embed, embed_query, cosine_similarity, rerank_cross_encoder
from rag.indexer import KnowledgeIndexer
from core.logger import get_logger

logger = get_logger(__name__)

# bi-encoder 余弦相似度阈值（降级模式用）
# BGE-small-zh 归一化向量的 cosine 相似度经验值：0.5+ 强相关，0.35~0.5 弱相关，<0.35 噪声
BI_ENCODER_SIM_THRESHOLD = 0.35


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

        两阶段检索：bi-encoder 召回 → cross-encoder 精排 → 阈值过滤

        Returns:
            {
                "knowledge": [...],   # 角色知识（rerank + 阈值过滤后）
                "episodic": [...],    # 情景记忆（去重 + rerank + 阈值过滤后）
                "semantic": [...],    # 语义记忆
            }
        """
        # 1. 召回阶段：bi-encoder（BGE-small-zh）向量检索 top_k
        knowledge = self.knowledge_indexer.search(query, top_k=KNOWLEDGE_TOP_K)
        episodic = self._dedup(self.episodic.search(query, top_k=EPISODIC_TOP_K))
        semantic = self.semantic.search(query)

        # 2. 精排阶段：cross-encoder（BGE-reranker）对 query-doc 对打分重排序
        knowledge = self._rerank(query, knowledge)
        episodic = self._rerank(query, episodic)

        # 3. 阈值过滤：rerank 后丢弃弱相关结果
        threshold = CROSS_ENCODER_THRESHOLD if self._use_cross_encoder() else BI_ENCODER_SIM_THRESHOLD
        knowledge = [k for k in knowledge if k.get("rerank_score", 0) >= threshold]
        episodic = [m for m in episodic if m.get("rerank_score", 0) >= threshold]

        return {
            "knowledge": knowledge,
            "episodic": episodic,
            "semantic": semantic,
        }

    @staticmethod
    def _use_cross_encoder() -> bool:
        """是否启用 cross-encoder 精排（配置开关 + 模型可用）"""
        if not ENABLE_CROSS_ENCODER_RERANK:
            return False
        from core.utils import _get_reranker
        return _get_reranker() is not None

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
        """精排重排序

        优先用 cross-encoder（BGE-reranker）对 query-doc 对精确打分，
        模型不可用时降级为 bi-encoder 余弦相似度重排。
        """
        if not items:
            return items

        if HybridRetriever._use_cross_encoder():
            # cross-encoder 精排：query 和 doc 拼一起送入模型，输出相关性分数
            docs = [item.get("content", "") for item in items]
            scores = rerank_cross_encoder(query, docs)
            for item, score in zip(items, scores):
                item["rerank_score"] = score
            logger.debug(f"cross-encoder rerank，分数: {scores}")
        else:
            # 降级：bi-encoder 余弦相似度重排
            query_vec = embed_query(query)
            for item in items:
                doc_vec = embed(item.get("content", ""))
                item["rerank_score"] = cosine_similarity(query_vec, doc_vec)
            logger.debug("降级为 bi-encoder 余弦相似度 rerank")

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
