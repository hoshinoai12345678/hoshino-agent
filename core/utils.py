"""工具函数：向量嵌入与相似度计算

使用 BGE-small-zh（北京智源研究院开源中文 embedding 模型）生成 512 维语义向量。
模型首次加载需下载（约 100MB），之后缓存在本地。
若 sentence-transformers 未安装或模型加载失败，降级为哈希嵌入（256 维）。
"""
import hashlib
import json
import math
import re
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)

# ---- BGE 模型懒加载（全局单例，避免重复加载）----
_MODEL = None
_MODEL_LOAD_ERROR: Optional[str] = None

# BGE-small-zh 输出 512 维；降级哈希嵌入用 256 维
EMBEDDING_DIM = 512
FALLBACK_DIM = 256

# BGE 检索需要加 query 前缀（官方推荐）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _get_model():
    """懒加载 BGE-small-zh 模型（失败返回 None，触发降级）"""
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None or _MODEL_LOAD_ERROR is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        logger.info("BGE-small-zh 模型加载成功")
    except Exception as e:
        _MODEL_LOAD_ERROR = str(e)
        logger.error(f"BGE 模型加载失败，降级为哈希嵌入。原因: {e}")
    return _MODEL


def embed(text: str) -> list[float]:
    """将文本转为语义向量

    优先用 BGE-small-zh（512 维真语义向量），
    模型不可用时降级为哈希嵌入（256 维）。

    Args:
        text: 输入文本
    Returns:
        归一化向量（512 或 256 维）
    """
    if not text:
        return [0.0] * EMBEDDING_DIM

    model = _get_model()
    if model is not None:
        # BGE 推理：normalize=True 直接返回归一化向量
        vec = model.encode(text, normalize_embeddings=True)
        # 转为 float 列表（JSON 可序列化）
        return [float(x) for x in vec]

    # 降级：哈希嵌入
    return _hash_embed(text)


def _hash_embed(text: str) -> list[float]:
    """哈希嵌入（降级方案，256 维）"""
    vec = [0.0] * FALLBACK_DIM
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % FALLBACK_DIM
        sign = 1.0 if int(h[8:16], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    vec = [0.0 if (v != v or abs(v) == float("inf")) else v for v in vec]
    return vec


def embed_query(query: str) -> list[float]:
    """检索查询的向量嵌入（BGE 推荐加前缀提升检索质量）"""
    return embed(QUERY_PREFIX + query)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """批量嵌入（BGE 模型支持批量推理，比逐条快很多）

    用于知识索引构建等批量场景。
    """
    if not texts:
        return []
    model = _get_model()
    if model is not None:
        vecs = model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]
    return [_hash_embed(t) for t in texts]


def _tokenize(text: str) -> list[str]:
    """简单分词（降级哈希嵌入用）"""
    if not text:
        return []
    en_tokens = re.findall(r"[a-zA-Z]+", text.lower())
    zh_tokens = re.findall(r"[\u4e00-\u9fff]", text)
    return en_tokens + zh_tokens


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度（支持不同维度，降级场景下 a/b 维度可能不同）"""
    if not a or not b:
        return 0.0
    # 维度不一致时取较短长度（降级兼容）
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a[:min_len], b[:min_len]))
    norm_a = math.sqrt(sum(x * x for x in a[:min_len]))
    norm_b = math.sqrt(sum(y * y for y in b[:min_len]))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_json(content: str) -> dict | None:
    """从 LLM 输出中提取 JSON（多重兜底）

    顺序：直接解析 → ```json 代码块 → ``` 代码块 → 花括号截取。
    用于 Thinker / Reflector 等需要结构化输出的模块。
    """
    if not content:
        return None
    content = content.strip()

    # 1. 直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. 从 ```json ... ``` 或 ``` ... ``` 中提取
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 3. 找到第一个 { 和最后一个 } 之间截取
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
