"""星野爱 Agent 主循环

整合人设、情绪、记忆、RAG、ReAct、反思的完整 Agent 闭环。

工作流程：
1. 检索阶段：RAG 检索相关记忆和知识
2. 思考阶段：生成内心独白（开发者模式可见）
3. 行动阶段：ReAct Loop（LLM 自主调用工具）
4. 生成阶段：流式输出最终回复
5. 反思阶段：自检并更新记忆/情绪
"""
import asyncio
import json
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI

from config import (LLM_API_KEY, LLM_API_BASE, LLM_MODEL, LLM_ENABLED,
                    LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_REACT_ITERATIONS,
                    ENABLE_REFLECTION, ENABLE_DEVMODE, CONSOLIDATION_THRESHOLD)
from core.persona import Persona
from core.emotion import EmotionEngine
from core.memory.working import WorkingMemory
from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from core.memory.consolidator import MemoryConsolidator
from rag.retriever import HybridRetriever
from agent.prompts import build_system_prompt
from agent.tools import TOOL_DEFINITIONS, ToolExecutor
from agent.reflector import Reflector
from agent.thinker import Thinker
from core.logger import get_logger

logger = get_logger(__name__)


class HoshinoAgent:
    """星野爱 Agent"""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        # 核心模块（记忆/情绪按 session 隔离，人设/工作记忆天然实例隔离）
        self.persona = Persona()
        self.emotion = EmotionEngine(session_id=session_id)
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(session_id=session_id)
        self.semantic = SemanticMemory(session_id=session_id)
        # 各组件复用同一份记忆实例，确保 clear() 后引用同步（避免双实例导致失效引用）
        self.retriever = HybridRetriever(
            session_id=session_id, episodic=self.episodic, semantic=self.semantic
        )
        self.consolidator = MemoryConsolidator(
            session_id=session_id, episodic=self.episodic, semantic=self.semantic
        )
        self.reflector = Reflector()
        self.thinker = Thinker()

        # 工具执行器
        self.tool_executor = ToolExecutor(
            self.emotion, session_id=session_id,
            episodic=self.episodic, semantic=self.semantic,
        )

        # LLM 客户端（异步，避免阻塞事件循环）
        self._client: Optional[AsyncOpenAI] = None
        if LLM_ENABLED:
            try:
                self._client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
                logger.info(f"LLM 已连接: {LLM_MODEL}")
            except Exception as e:
                logger.error(f"LLM 初始化失败: {e}")

        # 对话轮数计数器（用于取模触发巩固，避免工作记忆溢出后 size 取模失效）
        self._chat_round = 0

    async def chat(self, user_message: str, dev_mode: bool = ENABLE_DEVMODE) -> dict:
        """非流式对话（返回完整结果）"""
        result = {"reply": "", "thinking": "", "tool_calls": [], "emotion": {}, "memory_ops": []}

        async for chunk in self.chat_stream(user_message, dev_mode=dev_mode):
            if chunk["type"] == "reply":
                result["reply"] += chunk["content"]
            elif chunk["type"] == "thinking":
                result["thinking"] += chunk["content"]
            elif chunk["type"] == "tool_call":
                result["tool_calls"].append(chunk)
            elif chunk["type"] == "meta":
                result.update(chunk["data"])

        return result

    async def chat_stream(self, user_message: str, dev_mode: bool = ENABLE_DEVMODE) -> AsyncGenerator[dict, None]:
        """流式对话（SSE 用）

        Args:
            user_message: 用户消息
            dev_mode: 是否开启开发者模式（输出思考链）。默认取 config.ENABLE_DEVMODE，
                      由前端开关实时覆盖。

        Yields:
            {"type": "thinking", "content": "..."}  Thinker 思考链（仅 dev_mode 开启时展示）
            {"type": "tool_call", "name": "...", "result": "..."}  工具调用
            {"type": "reply", "content": "..."}  回复片段
            {"type": "meta", "data": {...}}  元数据（情绪/记忆等）
            {"type": "done"}  完成
        """
        # 1. 添加用户消息到工作记忆
        self.working.add_user(user_message)

        # 2. 检索阶段：RAG 检索
        # cross-encoder 精排是 CPU 密集同步操作，放到线程池避免阻塞事件循环
        retrieval_ctx = await asyncio.to_thread(self.retriever.to_prompt_context, user_message)
        emotion_ctx = self.emotion.to_prompt_context()
        semantic_ctx = self.semantic.to_prompt_context()
        working_ctx = self.working.to_prompt_context()
        persona_ctx = self.persona.to_prompt_context()

        # 3. 思考阶段：Thinker Agent 进行语义理解 + 工具决策（multi-agent）
        # 短消息跳过（节省 token）；dev_mode 下额外展示思考链（可观测性）
        thought = await self.thinker.think(
            user_message, retrieval_ctx, emotion_ctx, semantic_ctx
        )
        think_ctx = self.thinker.to_prompt_context(thought)
        if dev_mode and think_ctx:
            yield {"type": "thinking", "content": think_ctx}

        # 4. 构建系统提示词（注入 Thinker 的思考结果指导 ReAct loop）
        system_prompt = build_system_prompt(
            persona_ctx, emotion_ctx, retrieval_ctx, working_ctx, semantic_ctx
        )
        if think_ctx:
            system_prompt = f"{system_prompt}\n\n{think_ctx}"

        # 5. 行动 + 生成阶段：ReAct Loop
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        full_reply = ""
        tool_calls_log = []

        if self._client:
            # ReAct Loop：LLM 可能调用工具，最多迭代 MAX_REACT_ITERATIONS 次
            # 非流式调用以判断 tool_calls，最终回复统一用 stream=True 增量输出
            for iteration in range(MAX_REACT_ITERATIONS):
                resp = await self._client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                msg = resp.choices[0].message

                # 如果没有工具调用，跳出循环进行流式输出
                if not msg.tool_calls:
                    break

                # 执行工具调用
                # 注意：model_dump 后 content 可能为 None 或非 str，需清洗为 str
                assistant_msg = msg.model_dump(exclude_none=True)
                if "content" in assistant_msg and not isinstance(assistant_msg["content"], str):
                    assistant_msg["content"] = str(assistant_msg["content"])
                messages.append(assistant_msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = self.tool_executor.execute(name, args)
                    # 确保 result 是字符串（API 要求 tool 消息 content 为 str）
                    if not isinstance(result, str):
                        result = str(result)
                    tool_calls_log.append({"name": name, "args": args, "result": result})
                    yield {"type": "tool_call", "name": name, "args": args, "result": result}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            # 生成最终回复：优先用 ReAct 最后一轮已有的 content（避免再调一次流式触发空返回）
            final_content = msg.content if msg and isinstance(msg.content, str) and msg.content.strip() else ""
            if final_content and not msg.tool_calls:
                # ReAct 最后一轮已生成完整文本回复，直接用它
                full_reply = final_content
                yield {"type": "reply", "content": full_reply}
            else:
                # 流式输出最终回复（stream=True 增量 yield）
                # 不传 tools：最终回复阶段 LLM 只需生成文本，传 tools 会诱导它再调工具导致空输出
                stream = await self._client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_reply += delta
                        yield {"type": "reply", "content": delta}

                # 流式输出为空：暴露错误，不降级
                if not full_reply:
                    if final_content:
                        logger.info("流式输出为空，回退使用 ReAct 最后一轮 content")
                        full_reply = final_content
                        yield {"type": "reply", "content": full_reply}
                    else:
                        raise RuntimeError("LLM 流式输出为空且无历史 content（检查 max_tokens / API 状态）")
        else:
            # 无 LLM：直接报错，不降级
            raise RuntimeError("LLM 未启用（LLM_ENABLED=False 或 deepseek_apikey 未配置）")

        # 6. 添加助手回复到工作记忆
        self.working.add_assistant(full_reply)

        # 7. 反思阶段
        memory_ops = []
        if ENABLE_REFLECTION and self._client:
            reflection = await self.reflector.reflect(user_message, full_reply, emotion_ctx)

            # 应用情绪变化
            ec = reflection.get("emotion_change", {})
            if any(v != 0 for v in ec.values()):
                self.emotion.update(
                    pleasure_delta=ec.get("pleasure", 0),
                    arousal_delta=ec.get("arousal", 0),
                    dominance_delta=ec.get("dominance", 0),
                    favorability_delta=ec.get("favorability", 0),
                )
                memory_ops.append({"op": "emotion_update", "detail": ec})

            # 记住用户透露的信息（写入前按相似度去重，避免重复存储）
            revealed = reflection.get("user_revealed")
            if revealed:
                if not self.consolidator._is_duplicate_episodic(revealed):
                    self.episodic.add(content=revealed, event_type="fact", importance=0.8)
                    memory_ops.append({"op": "save_memory", "content": revealed})

            # 写入用户画像（语义记忆，UNIQUE 约束自动 upsert 去重）
            # 让画像面板实时更新，不必等 5 轮巩固
            for fact in reflection.get("user_profile", []):
                key = fact.get("key", "")
                value = fact.get("value", "")
                if key and value:
                    self.semantic.add(
                        category=fact.get("category", "basic"),
                        key=key,
                        value=value,
                        confidence=0.6,
                        source="reflect",
                    )
                    memory_ops.append({"op": "save_profile", "key": key, "value": value})

        # 8. 情绪衰减
        self.emotion.decay()

        # 9. 记忆巩固（每 N 轮触发，用独立计数器避免工作记忆溢出后取模失效）
        self._chat_round += 1
        if self._chat_round % CONSOLIDATION_THRESHOLD == 0:
            try:
                result = await self.consolidator.consolidate(self.working)
                if result.get("status") in ("success", "fallback"):
                    memory_ops.append({"op": "consolidate", "status": "done"})
                    logger.info(
                        f"记忆巩固完成：episodic +{result.get('episodic_added', 0)}，"
                        f"semantic +{result.get('semantic_added', 0)}"
                    )
            except Exception as e:
                logger.error(f"记忆巩固失败: {e}")

        # 10. 输出元数据
        yield {
            "type": "meta",
            "data": {
                "emotion": self.emotion.get_state(),
                "tool_calls": tool_calls_log,
                "memory_ops": memory_ops,
                "working_memory_size": self.working.size,
                "episodic_count": self.episodic.count(),
                "semantic_count": self.semantic.count(),
            },
        }

        yield {"type": "done"}

    def get_state(self) -> dict:
        """获取 Agent 当前状态（前端展示用）"""
        return {
            "emotion": self.emotion.get_state(),
            "working_memory_size": self.working.size,
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
        }

    def reset(self):
        """重置 Agent 状态"""
        self.working.clear()
        self.emotion.reset()
        self._chat_round = 0
        logger.info("状态已重置")
