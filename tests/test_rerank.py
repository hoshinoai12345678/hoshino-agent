"""测试 rag/retriever.py 的 _dedup 和 _rerank 静态方法

覆盖：
- _dedup 移除相似度 > 0.95 的近重复项
- _rerank 按 rerank_score 降序排列
- 空列表输入返回空
"""
from rag.retriever import HybridRetriever


def test_dedup_removes_near_duplicates():
    """相似度>0.95 的去重"""
    items = [
        {"content": "今天天气真好"},
        {"content": "今天天气真好"},  # 完全相同，相似度=1.0 > 0.95，应去重
        {"content": "我喜欢吃苹果"},  # 不同内容，保留
    ]
    result = HybridRetriever._dedup(items, threshold=0.95)
    assert len(result) == 2


def test_rerank_sorts_by_similarity():
    """rerank 后按 rerank_score 降序"""
    query = "我喜欢吃苹果"
    items = [
        {"content": "今天天气真好"},
        {"content": "我喜欢吃苹果"},  # 与 query 完全相同，相似度最高
    ]
    result = HybridRetriever._rerank(query, items)
    # 第一个应是与 query 完全相同的高分项
    assert result[0]["content"] == "我喜欢吃苹果"
    # 验证按 rerank_score 降序排列
    scores = [item["rerank_score"] for item in result]
    assert scores == sorted(scores, reverse=True)


def test_dedup_empty():
    """空列表返回空"""
    assert HybridRetriever._dedup([], threshold=0.95) == []


def test_rerank_empty():
    """空列表返回空"""
    assert HybridRetriever._rerank("query", []) == []
