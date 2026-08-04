"""ChromaDB 客户端共享模块

复用 PersistentClient → EphemeralClient 降级逻辑，避免 episodic.py 和 indexer.py 重复实现。
Windows 下 PersistentClient 可能因文件权限/锁定报"拒绝访问 (os error 5)"，
此时降级为 EphemeralClient（内存模式），保证服务可用但数据不持久化。
"""
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import VECTOR_DB_PATH
from core.logger import get_logger

logger = get_logger(__name__)

# 类级别共享单例（全进程一个 client，多个 collection 共存）
_client: Optional[chromadb.api.ClientAPI] = None


def get_client() -> chromadb.api.ClientAPI:
    """获取 ChromaDB 客户端单例

    首次调用初始化 PersistentClient，失败则降级为 EphemeralClient。
    """
    global _client
    if _client is not None:
        return _client

    try:
        _client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        logger.info(f"PersistentClient 已初始化: {VECTOR_DB_PATH}")
    except Exception as e:
        logger.warning(
            f"PersistentClient 初始化失败，降级为 EphemeralClient（内存模式）。原因: {e}"
        )
        _client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def get_collection(name: str, metadata: dict | None = None) -> chromadb.api.Collection:
    """获取或创建 collection

    Args:
        name: collection 名称（如 episodic_memory_session_xxx）
        metadata: collection 元数据（如 {"hnsw:space": "cosine"}）
    Returns:
        Collection 对象
    """
    client = get_client()
    if metadata is None:
        metadata = {"hnsw:space": "cosine"}
    return client.get_or_create_collection(name=name, metadata=metadata)
