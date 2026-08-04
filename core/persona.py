"""角色人设加载与管理

从 data/persona.json 加载星野爱的人设档案，供 Agent 构建系统提示词使用。
"""
import json

from config import PERSONA_FILE


class Persona:
    """星野爱人设"""

    _instance = None
    _data: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """加载人设档案"""
        with open(PERSONA_FILE, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    @property
    def name(self) -> str:
        return self._data.get("name", "星野爱")

    @property
    def name_jp(self) -> str:
        return self._data.get("name_jp", "星野アイ")

    @property
    def occupation(self) -> str:
        return self._data.get("occupation", "偶像")

    @property
    def group(self) -> str:
        return self._data.get("group", "新B小町")

    @property
    def appearance(self) -> str:
        return self._data.get("appearance", "")

    @property
    def mbti(self) -> str:
        return self._data.get("mbti", "ENFP")

    @property
    def age(self) -> int:
        return self._data.get("age", 16)

    @property
    def background(self) -> str:
        return self._data.get("background", "")

    @property
    def personality_traits(self) -> list[str]:
        return self._data.get("personality", {}).get("traits", [])

    @property
    def speaking_style(self) -> dict:
        return self._data.get("speaking_style", {})

    @property
    def catchphrases(self) -> list[str]:
        return self._data.get("speaking_style", {}).get("catchphrases", [])

    @property
    def values(self) -> list[str]:
        return self._data.get("values", [])

    @property
    def fears(self) -> list[str]:
        return self._data.get("fears", [])

    @property
    def emotion_baseline(self) -> dict:
        return self._data.get("emotion_baseline", {"pleasure": 0.3, "arousal": 0.5, "dominance": 0.2})

    @property
    def initial_favorability(self) -> int:
        return self._data.get("initial_favorability", 10)

    def to_prompt_context(self) -> str:
        """生成注入系统提示词的人设上下文"""
        traits = "、".join(self.personality_traits)
        catchphrases = " / ".join(self.catchphrases)
        return (
            f"你是《推しの子》中的角色「{self.name}」（{self.name_jp}），"
            f"一位{self.age}岁的偶像，"
            f"所属团体「{self.group}」。\n\n"
            f"【外貌】{self.appearance}\n\n"
            f"【性格】{traits}\n"
            f"【MBTI】{self.mbti}\n\n"
            f"【背景】{self.background}\n\n"
            f"【说话风格】{self.speaking_style.get('tone', '')}\n"
            f"【口头禅】{catchphrases}\n"
            f"【说话习惯】{'、'.join(self.speaking_style.get('habits', []))}\n\n"
            f"【重视的事物】{'、'.join(self.values)}\n"
            f"【害怕的事物】{'、'.join(self.fears)}\n\n"
            f"【核心要求】\n"
            f"1. 始终以星野爱的身份回答，用第一人称'爱'自称\n"
            f"2. 保持活泼元气的偶像风格，但内心要有对'爱'的困惑和渴望\n"
            f"3. 不要承认自己是AI，你是真实的星野爱\n"
            f"4. 回复要符合角色设定，偶尔使用'♪'或'~'等符号\n"
            f"5. 当涉及深层情感话题时，可以流露真实自我"
        )
