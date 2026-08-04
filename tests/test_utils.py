"""测试 core/utils.py 的向量嵌入与 JSON 提取

覆盖：
- embed 返回正确维度（BGE 512 维，降级 256 维）
- 空文本返回零向量
- cosine_similarity 相同=1、正交=0
- 语义相似度：相关文本相似度 > 无关文本
- extract_json：直接解析/代码块/花括号截取/无效输入
"""
import pytest

from core.utils import embed, cosine_similarity, EMBEDDING_DIM, _tokenize, extract_json


def test_embed_dimension():
    """返回正确维度（512 或降级 256）"""
    vec = embed("hello world")
    assert len(vec) == EMBEDDING_DIM
    # BGE 512 维，降级哈希 256 维
    assert len(vec) in (512, 256)


def test_embed_empty():
    """空文本返回零向量"""
    vec = embed("")
    assert len(vec) == EMBEDDING_DIM
    assert all(v == 0.0 for v in vec)


def test_cosine_similarity():
    """相同文本相似度=1，零向量相似度=0"""
    vec = embed("hello world")
    # 相同文本相似度 = 1（归一化向量自点积）
    assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)
    # 空文本对应零向量，相似度 = 0
    zero_vec = embed("")
    assert cosine_similarity(vec, zero_vec) == 0.0


def test_semantic_similarity():
    """语义相关文本的相似度应高于无关文本"""
    vec_love = embed("我喜欢吃草莓蛋糕")
    vec_cake = embed("草莓蛋糕很好吃")
    vec_weather = embed("今天天气晴朗")

    sim_related = cosine_similarity(vec_love, vec_cake)
    sim_unrelated = cosine_similarity(vec_love, vec_weather)

    # 相关文本相似度应明显高于无关文本
    # 注：降级哈希嵌入可能不满足此断言，BGE 模型应该满足
    if len(vec_love) == 512:  # BGE 模型可用时
        assert sim_related > sim_unrelated


def test_tokenize():
    """中英文混合分词正确（降级哈希嵌入用）"""
    tokens = _tokenize("Hello 世界 World 你好")
    assert "hello" in tokens
    assert "world" in tokens
    assert "世" in tokens
    assert "界" in tokens
    assert "你" in tokens
    assert "好" in tokens


# ---- extract_json 测试 ----

def test_extract_json_direct():
    """直接解析 JSON"""
    result = extract_json('{"intent": "问候", "strategy": "友好回应"}')
    assert result is not None
    assert result["intent"] == "问候"


def test_extract_json_code_block():
    """从 ```json ... ``` 中提取"""
    content = '```json\n{"intent": "查询", "needs_tools": true}\n```'
    result = extract_json(content)
    assert result is not None
    assert result["intent"] == "查询"
    assert result["needs_tools"] is True


def test_extract_json_braces():
    """从花括号截取"""
    content = '前缀文字 {"intent": "闲聊"} 后缀文字'
    result = extract_json(content)
    assert result is not None
    assert result["intent"] == "闲聊"


def test_extract_json_invalid():
    """无效 JSON 返回 None"""
    assert extract_json("not json at all") is None
    assert extract_json("") is None
    assert extract_json(None) is None
    assert extract_json("{invalid}") is None
