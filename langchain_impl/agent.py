"""LangChain 版 Agent（简化对比实现）

用 LangChain 的 create_openai_tools_agent + AgentExecutor 实现 ReAct 循环。
对比手写版（agent/hoshino_agent.py）的 for 循环实现。

差异说明：
1. 工具调用：LangChain 的 AgentExecutor 自动管理 messages 拼装和 tool_calls 解析，
   手写版需要自己 append assistant_msg + tool result 到 messages 数组。
2. 流式输出：LangChain 的 astream 不支持带类型的 chunk（thinking/reply/tool_call），
   手写版用 async generator 灵活控制每种 chunk 类型。
3. 定制受限：LangChain 难以实现 Thinker 前置思考、Reflector 后置反思、
   Consolidator 每 5 轮触发等定制流程，手写版可精确控制每一步。
4. 可观测性：手写版能直接 yield tool_call chunk 到前端展示，LangChain 要用回调。
"""
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

from config import (LLM_API_KEY, LLM_API_BASE, LLM_MODEL,
                    LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_REACT_ITERATIONS)
from core.persona import Persona
from core.memory.episodic import EpisodicMemory
from langchain_impl.tools import create_tools
from core.logger import get_logger

logger = get_logger(__name__)


class LangChainAgent:
    """LangChain 版简化 Agent（对比手写版 HoshinoAgent）

    仅实现 ReAct + 工具调用 + 基本对话，不含：
    - Thinker 前置思考（LangChain 的 AgentExecutor 不支持前置流程）
    - Reflector 后置反思（LangChain 无"后置处理"抽象）
    - PAD 情绪模型（需定制注入，LangChain Memory 抽象太粗）
    - Consolidator 每 5 轮触发（LangChain 不感知"轮数"）
    - SSE 流式带类型输出（LangChain astream 只输出文本）
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.persona = Persona()
        self.episodic = EpisodicMemory(session_id=session_id)

        # LangChain LLM 客户端（对比手写版的 AsyncOpenAI）
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        # 创建工具（闭包绑定 memory 实例）
        tools = create_tools(self.episodic)

        # 构建 prompt（对比手写版的 build_system_prompt）
        persona_ctx = self.persona.to_prompt_context()
        prompt = ChatPromptTemplate.from_messages([
            ("system", persona_ctx + "\n\n你是星野爱，不是AI助手。永远以星野爱的身份回应。"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # 创建 Agent + Executor
        # 对比手写版：for iteration in range(MAX_REACT_ITERATIONS) + 手动拼 messages
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=MAX_REACT_ITERATIONS,  # 对比手写版的 range(MAX_REACT_ITERATIONS)
            verbose=True,
            return_intermediate_steps=True,  # 返回工具调用记录（对比手写版的 tool_calls_log）
        )

    async def chat(self, user_message: str) -> dict:
        """非流式对话（对比手写版的 chat_stream）

        返回格式与手写版对齐，但不含 thinking/emotion/meta 等定制 chunk。

        Returns:
            {"reply": str, "tool_calls": list}
        """
        try:
            result = await self.executor.ainvoke({"input": user_message})
        except Exception as e:
            logger.error(f"LangChain Agent 执行失败: {e}")
            return {"reply": f"出错了: {e}", "tool_calls": []}

        # 提取工具调用记录（对比手写版的 tool_calls_log）
        tool_calls = []
        for step in result.get("intermediate_steps", []):
            action, observation = step
            tool_calls.append({
                "name": action.tool,
                "args": action.tool_input,
                "result": observation,
            })

        return {
            "reply": result.get("output", ""),
            "tool_calls": tool_calls,
        }


async def demo():
    """对比演示入口：用 LangChain 版 Agent 跑一轮对话"""
    import os
    if not LLM_API_KEY:
        print("请先设置环境变量 deepseek_apikey")
        return

    agent = LangChainAgent(session_id="langchain_demo")

    # 测试简单对话
    print("=== 测试1：简单对话 ===")
    result = await agent.chat("你好呀")
    print(f"回复: {result['reply']}")

    # 测试工具调用
    print("\n=== 测试2：记忆工具调用 ===")
    result = await agent.chat("我叫小枫，记住我喜欢草莓")
    print(f"回复: {result['reply']}")
    print(f"工具调用: {result['tool_calls']}")

    # 测试知识检索
    print("\n=== 测试3：知识检索 ===")
    result = await agent.chat("告诉我爱酱的口头禅")
    print(f"回复: {result['reply']}")
    print(f"工具调用: {result['tool_calls']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
