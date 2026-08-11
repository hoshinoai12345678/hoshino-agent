"""LangChain 版工具定义

用 @tool 装饰器定义工具，对比手写版的 OpenAI Function Calling JSON 格式。
复用手写版的 EpisodicMemory / KnowledgeIndexer，保证工具行为一致。
"""
from langchain_core.tools import tool

from core.memory.episodic import EpisodicMemory
from rag.indexer import KnowledgeIndexer


def create_tools(episodic: EpisodicMemory):
    """创建 LangChain 工具集（闭包绑定 memory 实例）

    对比手写版（agent/tools.py）：
    - 手写版：ToolExecutor 类 + TOOL_DEFINITIONS JSON 数组，工具定义和执行分离
    - LangChain：@tool 装饰器自动从函数签名+docstring 生成 schema，定义和执行合一

    Args:
        episodic: 情景记忆实例（按 session 隔离）
    Returns:
        list[BaseTool]
    """
    knowledge = KnowledgeIndexer()  # 角色知识全局共享

    @tool
    def save_memory(content: str, memory_type: str = "fact", importance: float = 0.5) -> str:
        """当用户透露重要信息（如姓名、职业、喜好、情绪状态等）时调用，将信息存入长期记忆。

        Args:
            content: 要记住的内容，如"用户是程序员"或"用户今天心情不好"
            memory_type: 记忆类型：fact=事实, emotion=情绪, dialogue=对话
            importance: 重要性0-1，默认0.5
        """
        episodic.add(content=content, event_type=memory_type, importance=importance)
        return f"已记住: {content}"

    @tool
    def search_memory(query: str) -> str:
        """检索与当前话题相关的历史记忆。当需要回忆之前聊过的内容时调用。

        Args:
            query: 检索关键词或话题
        """
        results = episodic.search(query, top_k=3)
        if not results:
            return "未找到相关记忆"
        return "\n".join(f"- {r['content']}（相似度:{r['similarity']:.2f}）" for r in results)

    @tool
    def search_knowledge(query: str) -> str:
        """检索角色（星野爱）的背景知识。当用户问到关于角色设定、背景、人际关系等问题时调用。

        Args:
            query: 查询内容，如"星野爱的家庭"或"新B小町"
        """
        results = knowledge.search(query, top_k=3)
        if not results:
            return "未找到相关知识"
        return "\n".join(f"- {r['content'][:200]}" for r in results)

    @tool
    def get_current_time() -> str:
        """获取当前日期和时间。当对话涉及时间相关话题时调用。"""
        import time
        return time.strftime("%Y年%m月%d日 %H:%M:%S %A", time.localtime())

    return [save_memory, search_memory, search_knowledge, get_current_time]
