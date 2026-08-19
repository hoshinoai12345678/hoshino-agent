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

from config import EMOTION_DECAY_RATE, MEMORY_DB_PATH, DEFAULT_CHARACTER_ID
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

    P/A/D 短期波动 + 好感度长期积累，均持久化到 SQLite。
    按 character_id + session_id 双维度隔离：
    表名 = emotion_state_{character_id}_{session_id}
    """

    _conn: Optional[sqlite3.Connection] = None

    def __init__(self, session_id: str = "default",
                 character_id: str = DEFAULT_CHARACTER_ID):
        # 校验 character_id 和 session_id，防止表名注入
        for name, val in [("character_id", character_id), ("session_id", session_id)]:
            if not val or not all(c.isalnum() or c in "_-" for c in val):
                raise ValueError(f"非法 {name}: {val}（仅允许字母数字下划线横线）")
        self._session_id = session_id
        self._character_id = character_id
        self._table = f"emotion_state_{character_id}_{session_id}"
        self._ensure_db()

        persona = Persona(character_id=character_id)
        baseline = persona.emotion_baseline
        self._baseline = EmotionState(
            pleasure=baseline.get("pleasure", 0.0),
            arousal=baseline.get("arousal", 0.0),
            dominance=baseline.get("dominance", 0.0),
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
        P/A 各分四档（强负/弱负/弱正/强正），D 分两档（顺从/掌控），
        形成 4×4×2=32 格精细化标签，避免雷同。
        阈值用 0.05/0.35 避开常见 baseline 边界，防止抖动跳变。
        """
        p, a, d = self._state.pleasure, self._state.arousal, self._state.dominance

        def band(v: float) -> int:
            """四档：0=强负, 1=弱负, 2=弱正, 3=强正"""
            if v < -0.3:
                return 0
            if v < 0.05:
                return 1
            if v < 0.35:
                return 2
            return 3

        pb, ab = band(p), band(a)
        high_d = d >= 0.1  # D≥0.1 掌控自信, D<0.1 顺从依赖

        # (P档, A档) → (D低:顺从依赖, D高:掌控自信)
        _TABLE: dict[tuple[int, int], tuple[str, str]] = {
            (3, 3): ("雀跃依恋", "自信兴奋"),   # 强正P 激昂A
            (3, 2): ("欣喜温顺", "愉悦自信"),   # 强正P 活跃A
            (3, 1): ("满足依赖", "惬意从容"),   # 强正P 平缓A
            (3, 0): ("慵懒安适", "恬淡自若"),   # 强正P 沉静A
            (2, 3): ("激动期待", "兴致勃勃"),   # 弱正P 激昂A
            (2, 2): ("欢喜亲近", "轻松愉快"),   # 弱正P 活跃A
            (2, 1): ("平和安稳", "从容淡定"),   # 弱正P 平缓A
            (2, 0): ("安静闲适", "沉静内敛"),   # 弱正P 沉静A
            (1, 3): ("慌张无措", "烦躁警惕"),   # 弱负P 激昂A
            (1, 2): ("微恼疏离", "冷淡戒备"),   # 弱负P 活跃A
            (1, 1): ("怅然若失", "郁郁寡欢"),   # 弱负P 平缓A
            (1, 0): ("倦怠无力", "消沉自闭"),   # 弱负P 沉静A
            (0, 3): ("惊慌恐惧", "愠怒强硬"),   # 强负P 激昂A
            (0, 2): ("焦虑不安", "愠怒抵触"),   # 强负P 活跃A
            (0, 1): ("悲伤低落", "郁结难解"),   # 强负P 平缓A
            (0, 0): ("哀痛绝望", "心如死灰"),   # 强负P 沉静A
        }
        return _TABLE[(pb, ab)][0 if not high_d else 1]

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
