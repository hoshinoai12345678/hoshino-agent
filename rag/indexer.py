"""知识索引构建

将 data/knowledge/ 下的角色知识文档分块并向量化，存入 ChromaDB。
"""
import json
import time
from pathlib import Path

from config import KNOWLEDGE_DIR
from core.utils import embed_query, embed_batch
from core.chroma_client import get_client, get_collection
from core.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "persona_knowledge"
# manifest 文件：记录每个知识文件的 mtime，用于判断是否需要重建索引
_MANIFEST_FILE = Path(KNOWLEDGE_DIR) / ".index_manifest.json"


class KnowledgeIndexer:
    """角色知识索引器"""

    def __init__(self):
        self._collection = get_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, force: bool = False) -> int:
        """构建知识索引（增量构建：内容未变更时跳过，避免每次启动都重建）

        Args:
            force: 强制重建（忽略 manifest 检测）
        Returns:
            当前索引的分块总数
        """
        knowledge_path = Path(KNOWLEDGE_DIR)
        if not knowledge_path.exists():
            logger.warning(f"知识目录不存在: {KNOWLEDGE_DIR}")
            return 0

        # 1. 扫描当前知识文件，计算 mtime 指纹
        md_files = sorted(knowledge_path.glob("*.md"))
        if not md_files:
            logger.info("知识目录为空，索引 0 个分块")
            return 0

        current_manifest = {f.name: f.stat().st_mtime for f in md_files}

        # 2. 非 force 模式：对比 manifest，未变更则跳过重建
        if not force:
            cached_manifest = self._load_manifest()
            if cached_manifest == current_manifest and self._collection.count() > 0:
                count = self._collection.count()
                logger.info(f"知识文件未变更，跳过重建（现有 {count} 个分块）")
                return count

        # 3. 内容变更或首次构建：删除旧 collection + 全量重建
        # 用 client.delete_collection 彻底删除，避免 collection.delete() 残留数据
        logger.info("检测到知识文件变更（或首次构建），开始重建索引...")
        client = get_client()
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception as e:
            logger.debug(f"删除旧 collection（可能不存在）: {e}")
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # 4. 收集所有分块
        all_chunks: list[dict] = []
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            chunks = self._chunk(content, str(md_file.stem))
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.info("知识索引构建完成，共 0 个分块")
            self._save_manifest(current_manifest)
            return 0

        # 5. 批量嵌入（一次 BGE 推理，比逐条快很多）
        texts = [c["text"] for c in all_chunks]
        vectors = embed_batch(texts)

        # 6. 批量写入
        ids = []
        metadatas = []
        ts_ms = int(time.time() * 1000)
        for i, chunk in enumerate(all_chunks):
            chunk_id = f"kg_{chunk['source']}_{chunk['section']}_{ts_ms}_{i}_{hash(chunk['text']) % 10000}"
            ids.append(chunk_id)
            metadatas.append({"source": chunk["source"], "section": chunk["section"]})

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )

        # 7. 保存 manifest，下次启动可跳过重建
        self._save_manifest(current_manifest)

        count = len(all_chunks)
        logger.info(f"知识索引构建完成，共 {count} 个分块")
        return count

    @staticmethod
    def _load_manifest() -> dict:
        """加载上次的 manifest（文件不存在返回空 dict）"""
        try:
            if _MANIFEST_FILE.exists():
                return json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"读取 manifest 失败，将触发重建: {e}")
        return {}

    @staticmethod
    def _save_manifest(manifest: dict) -> None:
        """保存当前 manifest"""
        try:
            _MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
            _MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"保存 manifest 失败（不影响本次构建）: {e}")

    def _chunk(self, content: str, source: str) -> list[dict]:
        """按标题分块"""
        chunks = []
        current_section = "default"
        current_lines: list[str] = []

        for line in content.split("\n"):
            if line.startswith("#"):
                # 保存上一块
                if current_lines:
                    text = "\n".join(current_lines).strip()
                    if text:
                        chunks.append({"text": text, "source": source, "section": current_section})
                current_section = line.lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # 最后一块
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                chunks.append({"text": text, "source": source, "section": current_section})

        return chunks

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """检索相关知识"""
        query_vec = embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
        )

        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return []

        memories = []
        for i, mem_id in enumerate(results["ids"][0]):
            memories.append({
                "id": mem_id,
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "section": results["metadatas"][0][i].get("section", ""),
                "similarity": max(0.0, 1.0 - results["distances"][0][i]),
            })
        return memories

    def count(self) -> int:
        return self._collection.count()
