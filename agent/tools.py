"""Agent 工具集（DeepSeek Function Calling 格式）

工具定义 + 执行器。Agent 自主决定何时调用哪个工具。
工具定义按角色定制（search_knowledge 描述使用角色名）。
"""
import time
from typing import Optional

from config import DEFAULT_CHARACTER_ID
from core.emotion import EmotionEngine
from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from core.persona import Persona
from rag.indexer import KnowledgeIndexer


def get_tool_definitions(character_id: str = DEFAULT_CHARACTER_ID) -> list[dict]:
    """生成工具定义（search_knowledge 描述使用角色名）"""
    try:
        persona = Persona(character_id=character_id)
        char_name = persona.name
    except Exception:
        char_name = "角色"

    return [
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": "当用户透露重要信息（如姓名、职业、喜好、情绪状态等）时调用，将信息存入长期记忆。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "要记住的内容，如'用户是程序员'或'用户今天心情不好'",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "emotion", "dialogue"],
                            "description": "记忆类型：fact=事实, emotion=情绪, dialogue=对话",
                        },
                        "importance": {
                            "type": "number",
                            "description": "重要性0-1，默认0.5",
                        },
                    },
                    "required": ["content", "memory_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "检索与当前话题相关的历史记忆。当需要回忆之前聊过的内容时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "检索关键词或话题",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": f"检索角色（{char_name}）的背景知识。当用户问到关于角色设定、背景、人际关系等问题时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"查询内容，如'{char_name}的背景'或'相关人物关系'",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_emotion",
                "description": "更新当前情绪状态和好感度。当对话内容影响了情绪时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pleasure_delta": {
                            "type": "number",
                            "description": "愉悦度变化量(-1~1)，正向=更开心",
                        },
                        "arousal_delta": {
                            "type": "number",
                            "description": "唤醒度变化量(-1~1)，正向=更激动",
                        },
                        "dominance_delta": {
                            "type": "number",
                            "description": "支配度变化量(-1~1)",
                        },
                        "favorability_delta": {
                            "type": "integer",
                            "description": "好感度变化量(-20~20)",
                        },
                        "reason": {
                            "type": "string",
                            "description": "情绪变化原因",
                        },
                    },
                    "required": ["favorability_delta", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前日期和时间。当对话涉及时间相关话题时调用。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


# 向后兼容：默认角色的工具定义（部分旧代码可能直接 import TOOL_DEFINITIONS）
TOOL_DEFINITIONS = get_tool_definitions()


class ToolExecutor:
    """工具执行器（按 character_id 隔离知识库和记忆）"""

    def __init__(self, emotion: EmotionEngine, session_id: str = "default",
                 character_id: str = DEFAULT_CHARACTER_ID,
                 episodic: Optional[EpisodicMemory] = None,
                 semantic: Optional[SemanticMemory] = None):
        self.emotion = emotion
        self._character_id = character_id
        # 复用外部传入的实例，保证 clear() 后引用同步；未传入时自建（向后兼容）
        self.episodic = episodic or EpisodicMemory(session_id=session_id, character_id=character_id)
        self.semantic = semantic or SemanticMemory(session_id=session_id, character_id=character_id)
        self.knowledge = KnowledgeIndexer(character_id=character_id)

    def execute(self, name: str, arguments: dict) -> str:
        """执行工具调用，返回结果字符串"""
        try:
            if name == "save_memory":
                return self._save_memory(arguments)
            elif name == "search_memory":
                return self._search_memory(arguments)
            elif name == "search_knowledge":
                return self._search_knowledge(arguments)
            elif name == "update_emotion":
                return self._update_emotion(arguments)
            elif name == "get_current_time":
                return self._get_current_time()
            else:
                return f"未知工具: {name}"
        except Exception as e:
            return f"工具执行失败: {e}"

    def _save_memory(self, args: dict) -> str:
        content = args.get("content", "")
        # LLM 偶尔会把结构化信息作为 dict 传入，转为可读字符串
        if isinstance(content, dict):
            content = "，".join(f"{k}:{v}" for k, v in content.items())
        elif not isinstance(content, str):
            content = str(content)
        mem_type = args.get("memory_type", "fact")
        importance = args.get("importance", 0.5)
        self.episodic.add(content=content, event_type=mem_type, importance=importance)
        return f"已记住: {content}"

    def _search_memory(self, args: dict) -> str:
        query = args.get("query", "")
        results = self.episodic.search(query, top_k=3)
        if not results:
            return "未找到相关记忆"
        lines = [f"- {r['content']}（相似度:{r['similarity']:.2f}）" for r in results]
        return "\n".join(lines)

    def _search_knowledge(self, args: dict) -> str:
        query = args.get("query", "")
        results = self.knowledge.search(query, top_k=3)
        if not results:
            return "未找到相关知识"
        lines = [f"- {r['content'][:200]}" for r in results]
        return "\n".join(lines)

    def _update_emotion(self, args: dict) -> str:
        self.emotion.update(
            pleasure_delta=args.get("pleasure_delta", 0),
            arousal_delta=args.get("arousal_delta", 0),
            dominance_delta=args.get("dominance_delta", 0),
            favorability_delta=args.get("favorability_delta", 0),
        )
        reason = args.get("reason", "")
        state = self.emotion.get_state()
        return f"情绪已更新（原因: {reason}）。当前好感度: {state['favorability']}（{state['favorability_level']}）"

    def _get_current_time(self) -> str:
        now = time.localtime()
        return time.strftime("%Y年%m月%d日 %H:%M:%S %A", now)
