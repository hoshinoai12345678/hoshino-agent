"""角色人设加载与管理（多角色支持）

从 data/characters/{character_id}/persona.json 加载角色人设档案。
支持多角色动态加载，每个角色独立实例，供 Agent 构建系统提示词使用。
"""
import json
from typing import Optional

from config import get_persona_file, DEFAULT_CHARACTER_ID


class Persona:
    """角色人设（多角色实例，按 character_id 加载）"""

    # 缓存：character_id -> 已加载的 persona 数据（避免重复读文件）
    _cache: dict[str, dict] = {}

    def __init__(self, character_id: str = DEFAULT_CHARACTER_ID):
        # 校验 character_id，防止路径注入
        if not character_id or not all(c.isalnum() or c in "_-" for c in character_id):
            raise ValueError(f"非法 character_id: {character_id}（仅允许字母数字下划线横线）")
        self._character_id = character_id
        self._data = self._load(character_id)

    @classmethod
    def _load(cls, character_id: str) -> dict:
        """加载人设档案（带缓存）"""
        if character_id in cls._cache:
            return cls._cache[character_id]
        path = get_persona_file(character_id)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls._cache[character_id] = data
        return data

    @property
    def character_id(self) -> str:
        return self._character_id

    @property
    def name(self) -> str:
        return self._data.get("name", "未知角色")

    @property
    def name_alias(self) -> str:
        return self._data.get("name_alias", "")

    @property
    def real_name(self) -> str:
        return self._data.get("real_name", "")

    @property
    def name_jp(self) -> str:
        return self._data.get("name_jp", "")

    @property
    def source(self) -> str:
        return self._data.get("source", "")

    @property
    def occupation(self) -> str:
        return self._data.get("occupation", "")

    @property
    def group(self) -> str:
        return self._data.get("group", "")

    @property
    def appearance(self) -> str:
        return self._data.get("appearance", "")

    @property
    def mbti(self) -> str:
        return self._data.get("mbti", "")

    @property
    def age(self) -> int:
        return self._data.get("age", 0)

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
        return self._data.get("emotion_baseline", {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0})

    @property
    def initial_favorability(self) -> int:
        return self._data.get("initial_favorability", 0)

    @property
    def core_requirements(self) -> list[str]:
        """核心要求（角色定制化行为准则）"""
        return self._data.get("core_requirements", [])

    def to_prompt_context(self) -> str:
        """生成注入系统提示词的人设上下文（动态，不硬编码角色名）"""
        traits = "、".join(self.personality_traits)
        catchphrases = " / ".join(self.catchphrases)

        # 来源标注（如有）
        source_line = f"，出自《{self.source}》" if self.source else ""

        # 别名行（如有）
        alias_parts = []
        if self.name_alias:
            alias_parts.append(f"别号「{self.name_alias}」")
        if self.real_name:
            alias_parts.append(f"本名{self.real_name}")
        alias_line = f"（{'，'.join(alias_parts)}）" if alias_parts else ""

        # 核心要求（从 persona.json 读取，每个角色可定制）
        requirements = "\n".join(self.core_requirements) if self.core_requirements else ""

        return (
            f"你是「{self.name}」{alias_line}，"
            f"一位{self.age}岁的{self.occupation}{source_line}，"
            f"所属势力「{self.group}」。\n\n"
            f"【外貌】{self.appearance}\n\n"
            f"【性格】{traits}\n"
            f"【MBTI】{self.mbti}\n\n"
            f"【背景】{self.background}\n\n"
            f"【说话风格】{self.speaking_style.get('tone', '')}\n"
            f"【口头禅】{catchphrases}\n"
            f"【说话习惯】{'、'.join(self.speaking_style.get('habits', []))}\n\n"
            f"【重视的事物】{'、'.join(self.values)}\n"
            f"【害怕的事物】{'、'.join(self.fears)}\n\n"
            f"【核心要求】\n{requirements}"
        )
