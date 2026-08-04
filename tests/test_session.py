"""测试 session 隔离

覆盖：
- 不同 session_id 的 EpisodicMemory 使用不同 collection
- 不同 session_id 的 SemanticMemory 使用不同 table
- 非法字符的 session_id 抛 ValueError
"""
import pytest

from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory


def test_episodic_session_isolation():
    """不同 session_id 的 EpisodicMemory 用不同 collection"""
    em1 = EpisodicMemory(session_id="test_a")
    em2 = EpisodicMemory(session_id="test_b")
    assert em1._collection.name != em2._collection.name
    assert "test_a" in em1._collection.name
    assert "test_b" in em2._collection.name


def test_semantic_session_isolation():
    """不同 session_id 的 SemanticMemory 用不同 table"""
    sm1 = SemanticMemory(session_id="test_a")
    sm2 = SemanticMemory(session_id="test_b")
    assert sm1._table != sm2._table
    assert "test_a" in sm1._table
    assert "test_b" in sm2._table


def test_invalid_session_id():
    """非法字符的 session_id 抛 ValueError"""
    # 含点号
    with pytest.raises(ValueError):
        EpisodicMemory(session_id="test.user")
    # 含空格
    with pytest.raises(ValueError):
        SemanticMemory(session_id="test user")
    # 含斜杠
    with pytest.raises(ValueError):
        EpisodicMemory(session_id="test/path")
