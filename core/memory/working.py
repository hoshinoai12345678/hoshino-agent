"""工作记忆（短期记忆）

维护最近 N 轮对话，作为当前上下文。
存储在内存中，进程重启后清空。
"""
from collections import deque
from time import time

from config import WORKING_MEMORY_SIZE


class WorkingMemory:
    """工作记忆：最近对话轮次"""

    def __init__(self, max_size: int = WORKING_MEMORY_SIZE):
        self._messages = deque(maxlen=max_size)

    def add(self, role: str, content: str, metadata: dict | None = None):
        """添加一条消息"""
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": time(),
            **(metadata or {}),
        })

    def add_user(self, content: str):
        self.add("user", content)

    def add_assistant(self, content: str):
        self.add("assistant", content)

    def get_messages(self) -> list[dict]:
        """获取工作记忆消息列表（用于构建 Prompt）"""
        return list(self._messages)

    def get_recent(self, n: int = 5) -> list[dict]:
        """获取最近 n 条"""
        msgs = list(self._messages)
        return msgs[-n:] if len(msgs) > n else msgs

    def clear(self):
        self._messages.clear()

    def to_prompt_context(self) -> str:
        """生成工作记忆上下文"""
        msgs = self.get_recent(8)
        if not msgs:
            return "（暂无历史对话）"
        lines = []
        for m in msgs:
            role = "用户" if m["role"] == "user" else "爱"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._messages)
