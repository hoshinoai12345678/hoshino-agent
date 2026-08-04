"""情景记忆（长期记忆 - 向量检索）

存储具体对话事件，支持语义检索。
使用 ChromaDB 作为向量数据库，记录带时间戳的对话片段。
"""
import time
import uuid

from config import VECTOR_DB_PATH, EPISODIC_TOP_K, DECAY_HALF_LIFE_DAYS
from core.utils import embed, embed_query
from core.chroma_client import get_collection, get_client
from core.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "episodic_memory"


class EpisodicMemory:
    """情景记忆：存储和检索对话事件

    每个 session 拥有独立的 collection（episodic_memory_{session_id}），
    实现多用户数据隔离。ChromaDB client 在类级别共享。
    """

    def __init__(self, session_id: str = "default"):
        # 校验 session_id，防止 collection 名注入
        if not session_id or not all(c.isalnum() or c in "_-" for c in session_id):
            raise ValueError(f"非法 session_id: {session_id}（仅允许字母数字下划线横线）")
        self._session_id = session_id
        self._collection = get_collection(
            name=f"{COLLECTION_NAME}_{self._session_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, content: str, event_type: str = "dialogue",
            importance: float = 0.5, metadata: dict | None = None) -> str:
        """存储一条情景记忆

        Args:
            content: 记忆内容（如"用户说自己是程序员"）
            event_type: 事件类型（dialogue/fact/emotion）
            importance: 重要性 0~1
            metadata: 额外元数据
        Returns:
            memory_id
        """
        # ChromaDB 边界类型守卫：documents 必须 str，LLM 偶尔传 dict/list 触发 multimodal 错误
        if not isinstance(content, str):
            if isinstance(content, dict):
                content = "，".join(f"{k}:{v}" for k, v in content.items())
            elif isinstance(content, list):
                content = "，".join(str(x) for x in content)
            else:
                content = str(content)
        # 用 uuid 后缀避免毫秒时间戳冲突（consolidator 批量写入时同一毫秒内 id 重复）
        memory_id = f"ep_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        vector = embed(content)
        ts = time.time()
        meta = {
            "event_type": event_type,
            "importance": importance,
            "timestamp": ts,
            **(metadata or {}),
        }
        self._collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[vector],
            metadatas=[meta],
        )
        return memory_id

    def search(self, query: str, top_k: int = EPISODIC_TOP_K) -> list[dict]:
        """语义检索相关记忆（带时间衰减加权）

        Args:
            query: 查询文本
            top_k: 返回条数
        Returns:
            记忆列表，按相关性排序
        """
        query_vec = embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k * 2,  # 多取一些，衰减后重新排序
        )

        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return []

        memories = []
        now = time.time()
        half_life = DECAY_HALF_LIFE_DAYS * 86400  # 转为秒

        for i, mem_id in enumerate(results["ids"][0]):
            content = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = max(0.0, 1.0 - distance)

            # 时间衰减：越久远的记忆权重越低
            ts = meta.get("timestamp", now)
            age = now - ts
            decay_factor = 0.5 ** (age / half_life) if half_life > 0 else 1.0

            # 重要性加权
            importance = meta.get("importance", 0.5)

            # 综合得分
            score = similarity * (0.5 + 0.3 * decay_factor + 0.2 * importance)

            memories.append({
                "id": mem_id,
                "content": content,
                "event_type": meta.get("event_type", "dialogue"),
                "importance": importance,
                "timestamp": ts,
                "similarity": similarity,
                "decay_factor": decay_factor,
                "score": score,
            })

        # 按综合得分排序
        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories[:top_k]

    def get_recent(self, limit: int = 10, offset: int = 0) -> list[dict]:
        """获取最近的记忆（按时间倒序，支持分页）

        Args:
            limit: 返回条数
            offset: 偏移量（从第几条开始，按时间倒序计）
        """
        # ChromaDB get() 不保证按 timestamp 排序，需取出后排序再切片
        # 个人伴聊场景记忆量有限，全取排序可接受；如数据量大可改用 where 过滤 timestamp
        results = self._collection.get()
        if not results or not results.get("ids"):
            return []

        memories = []
        for i, mem_id in enumerate(results["ids"]):
            memories.append({
                "id": mem_id,
                "content": results["documents"][i],
                "event_type": results["metadatas"][i].get("event_type", "dialogue"),
                "timestamp": results["metadatas"][i].get("timestamp", 0),
            })
        memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return memories[offset:offset + limit]

    def count(self) -> int:
        """记忆总数"""
        return self._collection.count()

    def clear(self) -> int:
        """清空当前 session 的全部情景记忆

        通过删除 collection 重建实现。不影响其他 session 和知识库。

        Returns:
            被清除的记忆条数
        """
        try:
            deleted = self._collection.count()
        except Exception:
            deleted = 0
        client = get_client()
        try:
            client.delete_collection(name=f"{COLLECTION_NAME}_{self._session_id}")
        except Exception as e:
            logger.warning(f"删除 collection 失败（可能已不存在）: {e}")
            deleted = 0
        # 重建空 collection，保证后续 add/search 可用
        self._collection = client.get_or_create_collection(
            name=f"{COLLECTION_NAME}_{self._session_id}",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"已清空 session={self._session_id} 的情景记忆，共 {deleted} 条")
        return deleted
