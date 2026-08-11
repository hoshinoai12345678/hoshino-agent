"""工具函数：向量嵌入与相似度计算

使用 BGE-small-zh（北京智源研究院开源中文 embedding 模型）生成 512 维语义向量。
模型首次加载需下载（约 100MB），之后缓存在本地。
若 sentence-transformers 未安装或模型加载失败，降级为哈希嵌入（256 维）。
"""
import hashlib
import json
import math
import os
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
        # 优先使用本地路径 + local_files_only=True，跳过联网检查（省 9 秒）
        local_path = r"E:\ai-agent\hf_cache\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620"
        if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "model.safetensors")):
            _MODEL = SentenceTransformer(local_path, local_files_only=True)
            logger.info(f"使用本地 BGE-small-zh 模型: {local_path}")
        else:
            _MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            logger.info("BGE-small-zh 模型加载成功（在线）")
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


# ---- BGE Reranker 懒加载（cross-encoder 精排，全局单例）----
_RERANKER = None
_RERANKER_LOAD_ERROR: Optional[str] = None


def _get_reranker():
    """懒加载 BGE-reranker-base（cross-encoder 模型）

    cross-encoder vs bi-encoder：
    - bi-encoder（BGE-small-zh）：query 和 doc 分别编码，算余弦相似度。快但粗。
    - cross-encoder（BGE-reranker）：query 和 doc 拼一起送入模型，直接输出相关性分数。慢但精。

    检索流程：bi-encoder 召回 top_k → cross-encoder 精排（本项目用法）。
    """
    global _RERANKER, _RERANKER_LOAD_ERROR
    if _RERANKER is not None or _RERANKER_LOAD_ERROR is not None:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
        # 优先使用本地路径，避免在线下载
        local_path = r"E:\ai-agent\hf_cache\hub\models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70"
        if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "model.safetensors")):
            model_source = local_path
            logger.info(f"使用本地 BGE-reranker-base 模型: {local_path}")
            _RERANKER = CrossEncoder(model_source, model_kwargs={"local_files_only": True})
        else:
            model_source = "BAAI/bge-reranker-base"
            logger.info("本地未找到 BGE-reranker-base，尝试在线下载")
            _RERANKER = CrossEncoder(model_source)
        logger.info("BGE-reranker-base 模型加载成功（cross-encoder 精排）")
    except Exception as e:
        _RERANKER_LOAD_ERROR = str(e)
        logger.warning(f"BGE-reranker 加载失败，降级为余弦相似度重排。原因: {e}")
    return _RERANKER


def rerank_cross_encoder(query: str, documents: list[str]) -> list[float]:
    """用 cross-encoder 对 query-documents 对打分

    Args:
        query: 查询文本
        documents: 文档文本列表
    Returns:
        相关性分数列表（分数越高越相关，cross-encoder 分数可为负数）
    """
    reranker = _get_reranker()
    if reranker is None or not documents:
        return [0.0] * len(documents)
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    # predict 返回 numpy 数组，统一转成 list[float]
    return [float(s) for s in scores]


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


# ---- Token 计数（tiktoken）----
# tiktoken 对 OpenAI 模型精确，对 DeepSeek 等兼容模型近似估算（cl100k_base 编码）
_ENCODER = None


def _get_encoder():
    """懒加载 tiktoken 编码器"""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
        logger.info("tiktoken 编码器加载成功（cl100k_base）")
    except Exception as e:
        logger.warning(f"tiktoken 加载失败，token 计数将退化为字符数估算。原因: {e}")
    return _ENCODER


def count_tokens(text: str) -> int:
    """计算文本的 token 数

    优先用 tiktoken 精确计算（cl100k_base 编码），
    不可用时退化为字符数 / 2 粗略估算（中文约 1 字 ≈ 1-2 token）。

    Args:
        text: 输入文本
    Returns:
        token 数
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    # 降级：字符数 / 2 粗略估算
    return len(text) // 2


def count_messages_tokens(messages: list[dict]) -> int:
    """计算 messages 数组的总 token 数

    Args:
        messages: [{"role": "system", "content": "..."}, ...]
    Returns:
        总 token 数（含 role 标记开销，每条约 +4 token）
    """
    total = 0
    for msg in messages:
        # 每条 message 约 4 token 的结构开销（role + delimiters）
        total += 4
        total += count_tokens(msg.get("content", ""))
    total += 2  # primer overhead
    return total
