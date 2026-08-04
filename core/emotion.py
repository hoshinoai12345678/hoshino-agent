"""PAD 情绪状态机

基于心理学 PAD 三维情绪模型：
- Pleasure（愉悦度）：-1~+1
- Arousal（唤醒度）：-1~+1
- Dominance（支配度）：-1~+1

情绪会随对话内容变化，每轮对话后向基线衰减。
P/A/D 相对短期（衰减快），好感度长期积累，均持久化到 SQLite（按 session 隔离）。
"""
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Optional

from config import EMOTION_DECAY_RATE, MEMORY_DB_PATH
from core.persona import Persona
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmotionState:
    """情绪状态"""
    pleasure: float = 0.0  # -1~+1
    arousal: float = 0.0   # -1~+1
    dominance: float = 0.0  # -1~+1
    favorability: int = 0  # -100~+100
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionState":
        return cls(
            pleasure=data.get("pleasure", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.0),
            favorability=data.get("favorability", 0),
            updated_at=data.get("updated_at", 0.0),
        )


class EmotionEngine:
    """情绪状态机

    P/A/D 短期波动 + 好感度长期积累，均持久化到 SQLite（按 session 隔离）。
    """

    _conn: Optional[sqlite3.Connection] = None

    def __init__(self, session_id: str = "default"):
        # 校验 session_id，防止表名注入（与 SemanticMemory 一致）
        if not session_id or not all(c.isalnum() or c in "_-" for c in session_id):
            raise ValueError(f"非法 session_id: {session_id}（仅允许字母数字下划线横线）")
        self._session_id = session_id
        self._table = f"emotion_state_{session_id}"
        self._ensure_db()

        persona = Persona()
        baseline = persona.emotion_baseline
        self._baseline = EmotionState(
            pleasure=baseline.get("pleasure", 0.3),
            arousal=baseline.get("arousal", 0.5),
            dominance=baseline.get("dominance", 0.2),
            favorability=persona.initial_favorability,
            updated_at=time.time(),
        )
        # 先尝试从 SQLite 加载持久化状态，没有才用 baseline
        loaded = self._load_state()
        self._state = loaded if loaded is not None else EmotionState.from_dict(self._baseline.to_dict())

    def _ensure_db(self):
        """初始化 SQLite（连接单例，表按 session 分离）"""
        if EmotionEngine._conn is None:
            EmotionEngine._conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
            EmotionEngine._conn.row_factory = sqlite3.Row
        # 表名已通过 session_id 校验，安全拼接
        EmotionEngine._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                pleasure REAL NOT NULL,
                arousal REAL NOT NULL,
                dominance REAL NOT NULL,
                favorability INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        EmotionEngine._conn.commit()

    def _load_state(self) -> Optional[EmotionState]:
        """从 SQLite 加载持久化的情绪状态（无记录返回 None，用 baseline）"""
        try:
            row = EmotionEngine._conn.execute(
                f"SELECT * FROM {self._table} WHERE id = 1"
            ).fetchone()
            if row is None:
                return None
            return EmotionState(
                pleasure=row["pleasure"],
                arousal=row["arousal"],
                dominance=row["dominance"],
                favorability=row["favorability"],
                updated_at=row["updated_at"],
            )
        except Exception as e:
            logger.warning(f"加载情绪状态失败，使用 baseline: {e}")
            return None

    def _save_state(self):
        """把当前状态写回 SQLite（单行 upsert，id 固定为 1）"""
        try:
            EmotionEngine._conn.execute(f"""
                INSERT INTO {self._table} (id, pleasure, arousal, dominance, favorability, updated_at)
                VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    pleasure=excluded.pleasure,
                    arousal=excluded.arousal,
                    dominance=excluded.dominance,
                    favorability=excluded.favorability,
                    updated_at=excluded.updated_at
            """, (
                self._state.pleasure, self._state.arousal, self._state.dominance,
                self._state.favorability, self._state.updated_at,
            ))
            EmotionEngine._conn.commit()
        except Exception as e:
            logger.error(f"保存情绪状态失败: {e}")

    @property
    def state(self) -> EmotionState:
        return self._state

    def get_state(self) -> dict:
        """获取当前情绪状态（前端展示用）"""
        return {
            **self._state.to_dict(),
            "emotion_label": self._emotion_label(),
            "favorability_level": self._favorability_level(),
        }

    def update(self, pleasure_delta: float = 0, arousal_delta: float = 0,
               dominance_delta: float = 0, favorability_delta: int = 0):
        """更新情绪状态并持久化"""
        self._state.pleasure = self._clamp(self._state.pleasure + pleasure_delta, -1, 1)
        self._state.arousal = self._clamp(self._state.arousal + arousal_delta, -1, 1)
        self._state.dominance = self._clamp(self._state.dominance + dominance_delta, -1, 1)
        self._state.favorability = self._clamp_int(self._state.favorability + favorability_delta, -100, 100)
        self._state.updated_at = time.time()
        self._save_state()

    def decay(self):
        """每轮对话后情绪向基线衰减（不写库，下次 update 时一并持久化）"""
        rate = EMOTION_DECAY_RATE
        self._state.pleasure = self._state.pleasure + (self._baseline.pleasure - self._state.pleasure) * rate
        self._state.arousal = self._state.arousal + (self._baseline.arousal - self._state.arousal) * rate
        self._state.dominance = self._state.dominance + (self._baseline.dominance - self._state.dominance) * rate
        # 好感度不衰减（长期积累）

    def reset(self):
        """重置为基线并清空持久化记录（用户主动 /reset 时调用）"""
        self._state = EmotionState.from_dict(self._baseline.to_dict())
        try:
            EmotionEngine._conn.execute(f"DELETE FROM {self._table} WHERE id = 1")
            EmotionEngine._conn.commit()
        except Exception as e:
            logger.error(f"清空情绪持久化记录失败: {e}")

    def to_prompt_context(self) -> str:
        """生成情绪上下文，注入提示词"""
        label = self._emotion_label()
        fav_level = self._favorability_level()
        fav_desc = self._favorability_description()
        return (
            f"【当前情绪状态】{label}（愉悦度:{self._state.pleasure:.2f}，"
            f"唤醒度:{self._state.arousal:.2f}，支配度:{self._state.dominance:.2f}）\n"
            f"【对用户好感度】{self._state.favorability}（{fav_level}）\n"
            f"【情绪对回复的影响】{fav_desc}"
        )

    def _emotion_label(self) -> str:
        """根据 PAD 三维值推断情绪标签

        P=愉悦度, A=唤醒度, D=支配度。
        D 高=自信掌控, D 低=顺从被动，与 P/A 组合产生更精细的情绪标签。
        """
        p, a, d = self._state.pleasure, self._state.arousal, self._state.dominance
        if p > 0.3 and a > 0.3:
            return "自信兴奋" if d > 0.2 else "羞涩兴奋"
        if p > 0.3 and a < -0.2:
            return "满足放松" if d > 0 else "安心依赖"
        if p > 0.3:
            return "愉快自信" if d > 0.2 else "愉快温顺"
        if p < -0.3 and a > 0.3:
            return "焦躁强硬" if d > 0.2 else "焦虑不安"
        if p < -0.3 and a < -0.2:
            return "低落消沉" if d < 0 else "委屈被动"
        if p < -0.3:
            return "不悦倔强" if d > 0.2 else "委屈"
        if a > 0.5:
            return "激动"
        return "平淡"

    def _favorability_level(self) -> str:
        fav = self._state.favorability
        if fav >= 70:
            return "挚爱"
        if fav >= 40:
            return "亲密"
        if fav >= 10:
            return "友好"
        if fav >= -10:
            return "普通"
        if fav >= -40:
            return "疏远"
        return "冷漠"

    def _favorability_description(self) -> str:
        """好感度对回复风格的影响描述"""
        fav = self._state.favorability
        if fav >= 70:
            return "对用户非常亲近，会主动分享内心想法，语气温柔甜蜜"
        if fav >= 40:
            return "对用户信任，会流露更多真实情感，偶尔撒娇"
        if fav >= 10:
            return "保持友好的偶像风格，适度热情"
        if fav >= -10:
            return "保持礼貌但有距离感，回归专业偶像模式"
        if fav >= -40:
            return "略显冷淡，语气客气但疏远"
        return "非常冷淡，只做最低限度的回应"

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _clamp_int(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))
