"""工作记忆（短期记忆）

维护最近 N 轮对话，作为当前上下文。
存储在内存中，进程重启后清空。
双重控制：deque maxlen 按轮数截断 + token 上限按 token 数截断。
"""
from collections import deque
from time import time

from config import WORKING_MEMORY_SIZE, MAX_CONTEXT_TOKENS
from core.utils import count_tokens


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

    def get_messages_within_token_limit(self, max_tokens: int = MAX_CONTEXT_TOKENS) -> list[dict]:
        """从最近的消息往前取，直到 token 数超过上限

        与 deque maxlen 双重控制：
        - deque maxlen 按轮数截断（粗粒度，防无限增长）
        - token 上限按 token 数截断（细粒度，防超长消息爆上下文）

        Returns:
            token 数未超限的消息列表（按时间正序）
        """
        msgs = list(self._messages)
        result = []
        total = 0
        for msg in reversed(msgs):  # 从最近往前
            msg_tokens = count_tokens(msg.get("content", ""))
            if total + msg_tokens > max_tokens and result:
                break  # 超限且已有消息，停止
            result.insert(0, msg)  # 插到头部保持正序
            total += msg_tokens
        return result

    def clear(self):
        self._messages.clear()

    def to_prompt_context(self) -> str:
        """生成工作记忆上下文（受 token 上限控制）"""
        msgs = self.get_messages_within_token_limit()
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
