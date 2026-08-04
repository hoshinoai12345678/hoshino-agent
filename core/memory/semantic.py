"""语义记忆（长期记忆 - 用户画像）

存储从对话中蒸馏出的稳定用户特征。
使用 SQLite 存储结构化画像数据。
"""
import sqlite3
import time
from typing import Optional

from config import MEMORY_DB_PATH
from core.logger import get_logger

logger = get_logger(__name__)


class SemanticMemory:
    """语义记忆：用户画像蒸馏

    每个 session 拥有独立的表（user_profile_{session_id}），
    实现多用户数据隔离。SQLite 连接在类级别共享。
    """

    _conn: Optional[sqlite3.Connection] = None

    def __init__(self, session_id: str = "default"):
        # 校验 session_id，防止表名注入
        if not session_id or not all(c.isalnum() or c in "_-" for c in session_id):
            raise ValueError(f"非法 session_id: {session_id}（仅允许字母数字下划线横线）")
        self._session_id = session_id
        self._table = f"user_profile_{session_id}"
        self._ensure_db()

    def _ensure_db(self):
        """初始化 SQLite（连接单例，表按 session 分离）"""
        if SemanticMemory._conn is None:
            SemanticMemory._conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
            SemanticMemory._conn.row_factory = sqlite3.Row
        # 表名已通过 session_id 校验，安全拼接
        SemanticMemory._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,      -- 类别：basic/preference/personality/interest
                key TEXT NOT NULL,           -- 键：如"职业"、"喜欢的食物"
                value TEXT NOT NULL,         -- 值
                confidence REAL DEFAULT 0.5, -- 置信度 0~1
                source TEXT DEFAULT 'distill', -- 来源：distill/manual
                created_at REAL,
                updated_at REAL,
                UNIQUE(category, key)
            )
        """)
        SemanticMemory._conn.commit()

    def add(self, category: str, key: str, value: str,
            confidence: float = 0.5, source: str = "distill") -> bool:
        """添加或更新画像条目"""
        now = time.time()
        try:
            self._conn.execute(f"""
                INSERT INTO {self._table} (category, key, value, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, key) DO UPDATE SET
                    value=excluded.value,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
            """, (category, key, value, confidence, source, now, now))
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"添加画像失败: {e}")
            return False

    def get_all(self) -> list[dict]:
        """获取全部画像"""
        rows = self._conn.execute(
            f"SELECT * FROM {self._table} ORDER BY category, updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def search(self, keyword: str) -> list[dict]:
        """关键词搜索画像"""
        rows = self._conn.execute(
            f"SELECT * FROM {self._table} WHERE value LIKE ? OR key LIKE ? ORDER BY updated_at DESC",
            (f"%{keyword}%", f"%{keyword}%")
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self._table}").fetchone()
        return row["cnt"] if row else 0

    def clear(self) -> int:
        """清空当前 session 的全部语义记忆（用户画像）

        通过 DROP 表实现。不影响其他 session。

        Returns:
            被清除的画像条数
        """
        try:
            row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self._table}").fetchone()
            deleted = row["cnt"] if row else 0
        except Exception:
            deleted = 0
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS {self._table}")
            self._conn.commit()
        except Exception as e:
            logger.error(f"清空语义记忆失败: {e}")
            return 0
        # 重建空表，保证后续 add 可用
        self._ensure_db()
        logger.info(f"已清空 session={self._session_id} 的语义记忆，共 {deleted} 条")
        return deleted

    def to_prompt_context(self) -> str:
        """生成语义记忆上下文（用户画像）"""
        profile = self.get_all()
        if not profile:
            return "（暂无用户画像，还不了解用户）"

        # 按类别分组
        groups: dict[str, list[dict]] = {}
        for item in profile:
            groups.setdefault(item["category"], []).append(item)

        category_names = {
            "basic": "基本信息",
            "preference": "偏好",
            "personality": "性格",
            "interest": "兴趣",
        }

        lines = []
        for cat, items in groups.items():
            cat_name = category_names.get(cat, cat)
            entries = "、".join(f"{i['key']}:{i['value']}" for i in items)
            lines.append(f"【{cat_name}】{entries}")
        return "\n".join(lines)
