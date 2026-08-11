# LangChain 对比实现

本项目同时提供**手写版**（`agent/`）和 **LangChain 版**（`langchain_impl/`）两种 Agent 实现，用于技术选型对比。

## 两个版本对比

| 维度 | 手写版（`agent/hoshino_agent.py`） | LangChain 版（`langchain_impl/agent.py`） |
|------|-------------------------------------|-------------------------------------------|
| LLM 客户端 | `AsyncOpenAI` 直接调用 | `ChatOpenAI`（langchain-openai） |
| 工具定义 | `TOOL_DEFINITIONS` JSON 数组 + `ToolExecutor` 类 | `@tool` 装饰器（定义和执行合一） |
| ReAct 循环 | `for iteration in range(3)` 手动拼 messages | `AgentExecutor` 自动管理 |
| 流式输出 | `async generator` + SSE 带类型 chunk | `astream` 只输出文本 |
| Thinker 前置思考 | ✅ 支持（注入 system prompt） | ❌ AgentExecutor 不支持前置流程 |
| Reflector 后置反思 | ✅ 支持（每轮更新记忆/情绪） | ❌ 无"后置处理"抽象 |
| PAD 情绪模型 | ✅ 支持（注入 system prompt） | ❌ Memory 抽象太粗 |
| Consolidator 每 5 轮 | ✅ 支持（独立计数器） | ❌ 不感知"轮数" |
| session 隔离 | ✅ 四层隔离 | ⚠️ 需自己管理实例池 |
| SSE chunk 类型 | ✅ thinking/reply/tool_call/meta | ❌ 只有文本 |

## 为什么主实现选手写版

### 1. 定制流程 LangChain 不支持

项目核心的 `Thinker → ReAct → Reflector → Consolidator` 流水线是定制的，LangChain 的 `AgentExecutor` 是"输入→Agent→输出"的单一模型，不支持前置/后置处理节点。

### 2. 流式输出 LangChain 不灵活

手写版用 `async generator` 可以按 chunk 类型（thinking/reply/tool_call/meta）分别 yield，前端按类型渲染不同 UI 组件。LangChain 的 `astream` 只输出文本流，不支持带类型的 chunk。

### 3. 三层记忆 LangChain 抽象太粗

项目的三层记忆（工作记忆 deque / 情景记忆 ChromaDB / 语义记忆 SQLite）各有不同检索方式，LangChain 的 `ConversationBufferMemory` 只是单一缓冲区，套不上。

### 4. 手写能讲清原理

面试问"你的 ReAct 怎么实现的"，手写版能指着代码讲每一步：messages 拼装、tool_calls 解析、循环控制。LangChain 版只能说"AgentExecutor 帮我做了"。

## LangChain 版的价值

LangChain 版作为**对比实证**存在，证明：
1. **会 LangChain** — 不是不会，是评估后选择不用
2. **有判断力** — 知道什么时候该用、什么时候不该用
3. **了解框架能力边界** — 知道 AgentExecutor 支持什么、不支持什么

## 运行对比

```bash
# 手写版（主实现，完整功能）
python app.py

# LangChain 版（简化对比，仅 ReAct + 工具调用）
python -m langchain_impl.agent
```

## 什么时候该用 LangChain

- 快速搭原型验证想法
- 简单 RAG（文档加载 + 分块 + 检索 + 生成）
- 团队都熟 LangChain，要协作
- 不需要精细控制流程的标准化场景

本项目属于"定制重、要讲原理"的场景，手写更合适。
