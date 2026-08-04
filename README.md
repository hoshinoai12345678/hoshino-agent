# 星野爱 Agent

基于《推しの子》角色"星野爱"构建的虚拟角色拟人 Agent，集成 ReAct 推理、三层记忆系统、PAD 三维情绪状态机和 RAG 检索增强。

## 核心特性

- **Thinker + ReAct 双层决策**：Thinker Agent 先做语义理解（用户意图/情绪/工具决策/回应策略）+ 多任务识别（复杂请求拆解为步骤清单），输出结构化指导注入 ReAct loop，体现 multi-agent 的 think→execute 分工
- **PAD 三维情绪模型**：区别于单一好感度，用愉悦度(P)/唤醒度(A)/支配度(D)三维度刻画角色情绪，影响回复风格
- **三层记忆架构 + 双通道**：工作记忆（短期上下文）+ 情景记忆（向量检索）+ 语义记忆（用户画像）。Reflector 每轮实时写入单轮事实，Consolidator 每 5 轮做跨轮综合分析，职责不重复
- **ReAct + Function Calling**：LLM 自主决策调用工具（记忆存储/知识检索/情绪更新），实现闭环推理
- **BGE-small-zh 语义嵌入**：512 维真语义向量，记忆召回率 R=100%（相比哈希嵌入 33% → 100%）
- **Session 隔离**：每个用户独立 Agent 实例，记忆/情绪/画像按 session 分离，支持多用户并发
- **多通道接入**：Web（FastAPI SSE）+ Telegram Bot，复用同一套 agent 内核，通过 session_id 前缀隔离
- **全异步架构**：AsyncOpenAI + async/await，不阻塞事件循环
- **RAG 检索增强**：向量检索 + 去重 + BGE 精确 rerank，补偿 HNSW 近似检索的精度损失
- **统一日志系统**：logging 模块统一配置，环境变量控级别，替代散落的 print
- **可量化评估**：内置评估框架，35 个用例覆盖角色一致性/记忆召回/工具调用/情绪响应/边界场景

## 评估结果

基于 35 个测试用例的真实评估数据（DeepSeek LLM + BGE-small-zh 嵌入 + Thinker + 改进 reflector）：

| 指标 | 数值 | 说明 |
|------|------|------|
| 角色一致率 | **95.7%** | 回复是否以"爱"自称、包含角色关键词、不承认是 AI |
| 工具调用成功率 | **100%** | ReAct 工具调用（记忆存储/知识检索/情绪更新）成功执行比例 |
| 情绪响应适当性 | **70.9%** | 情绪变化方向与预期的吻合度（P/A/D/好感度四维） |
| 记忆召回 | **R=100%** | 写入测试记忆后，相关 query 的召回率 |

测试用例覆盖五大场景：角色一致性(8) / 记忆存储(7) / 知识检索(7) / 情绪响应(7) / 边界场景(6)

运行评估：`python eval/run_eval.py`

## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（设置环境变量 `deepseek_apikey`）

### 安装

```bash
pip install -r requirements.txt
```

### 配置

设置 DeepSeek API Key（系统环境变量）：

```powershell
# Windows 系统级（需重启终端生效）
setx deepseek_apikey "sk-your-key-here"
```

### 启动

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

访问 http://127.0.0.1:8001

### 运行测试

```bash
python -m pytest tests/ -v
```

## 项目结构

```
hoshino-agent/
├── agent/                  # Agent 核心
│   ├── hoshino_agent.py    # ReAct 主循环 + 流式输出
│   ├── prompts.py          # 提示词构建
│   ├── thinker.py          # 思考者 Agent（语义理解+工具决策，multi-agent）
│   ├── reflector.py        # 反思引擎（每轮：单轮事实提取+情绪更新+画像实时写入）
│   └── tools.py            # Function Calling 工具定义与执行
├── core/
│   ├── persona.py          # 角色人设（从 persona.json 加载）
│   ├── emotion.py          # PAD 三维情绪状态机
│   ├── utils.py            # BGE-small-zh 语义向量嵌入 + 余弦相似度
│   ├── logger.py           # 统一日志配置（logging 模块）
│   └── memory/
│       ├── working.py      # 工作记忆（短期对话上下文）
│       ├── episodic.py     # 情景记忆（ChromaDB 向量检索，按 session 隔离）
│       ├── semantic.py     # 语义记忆（SQLite 用户画像，按 session 隔离）
│       └── consolidator.py # 记忆巩固引擎（每5轮跨轮综合分析，工作记忆 → 长期记忆）
├── rag/
│   ├── indexer.py          # 角色知识库索引
│   └── retriever.py        # 混合检索 + 去重 + rerank
├── api/
│   └── chat.py             # FastAPI SSE 流式接口 + session 管理
├── channel/                # 对外通道适配层（复用同一套 agent 内核）
│   └── telegram_bot.py     # Telegram Bot 通道（支持代理/命令/多用户）
├── eval/
│   ├── metrics.py          # 评估指标（纯函数，可单元测试）
│   ├── test_cases.json     # 35 个测试用例（5 类场景）
│   └── run_eval.py         # 评估脚本
├── tests/                  # 单元测试（27 个，覆盖情绪/向量/rerank/session/thinker）
├── static/                 # 前端页面
├── data/
│   ├── persona.json        # 角色人设配置
│   └── knowledge/          # 角色知识库（背景/性格/语录/关系）
├── Dockerfile              # 容器化部署
├── .github/workflows/ci.yml # GitHub Actions CI
└── app.py                  # FastAPI 应用入口
```

## 技术栈

| 层 | 技术 |
|----|------|
| LLM | DeepSeek（OpenAI 兼容协议，异步调用） |
| 框架 | FastAPI + SSE 流式响应 |
| 通道 | Web（FastAPI）+ Telegram Bot（python-telegram-bot，long polling） |
| 向量数据库 | ChromaDB（PersistentClient，降级 EphemeralClient） |
| 关系数据库 | SQLite（用户画像，按 session 分表） |
| 嵌入模型 | BGE-small-zh-v1.5（512 维语义向量，本地推理，降级哈希嵌入） |
| 日志 | Python logging（统一配置，环境变量控级别） |
| 测试 | pytest（27 个单元测试） |
| CI/CD | GitHub Actions |
| 容器化 | Docker |

## API 接口

### Web 通道（FastAPI）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat?session_id=xxx` | 流式聊天（SSE） |
| GET | `/api/state?session_id=xxx` | 获取 Agent 状态（情绪/好感度/记忆数） |
| GET | `/api/history?session_id=xxx` | 获取工作记忆中的对话历史 |
| POST | `/api/reset?session_id=xxx` | 重置 Agent（清短期记忆和情绪，保留长期记忆） |
| POST | `/api/forget?session_id=xxx` | 忘记用户（清长期记忆，保留知识库和工作记忆） |
| GET | `/api/memories?session_id=xxx` | 获取历史记忆和用户画像 |

### Telegram 通道

```bash
# 1. 从 @BotFather 获取 Bot Token
# 2. 设置环境变量
set TG_BOT_TOKEN=your_token
# 国内需配代理
set HTTPS_PROXY=http://127.0.0.1:7890

# 3. 启动
python -m channel.telegram_bot
```

TG 命令：`/start` `/reset` `/forget` `/state`，普通消息直接聊天。

## 工程亮点

### 异步架构
所有 LLM 调用使用 `AsyncOpenAI` + `await`，避免同步阻塞事件循环。ReAct 循环结束后用 `stream=True` 增量输出，实现真正的流式响应。

### BGE 语义嵌入
采用北京智源研究院开源的 BGE-small-zh-v1.5 模型生成 512 维语义向量，相比哈希嵌入大幅提升记忆召回率（R: 33% → 100%）。检索查询自动添加 BGE 推荐前缀，进一步优化检索质量。模型不可用时自动降级为哈希嵌入。

### Session 隔离
- EpisodicMemory：每个 session 独立 ChromaDB collection
- SemanticMemory：每个 session 独立 SQLite 表
- Agent 实例：通过 `asyncio.Lock` 保护的字典管理，协程安全

### RAG 增强
- **去重**：按向量相似度 >0.95 去除重复检索结果
- **Rerank**：用 BGE 语义向量计算精确余弦相似度重排序，补偿 ChromaDB HNSW 近似检索的精度损失

### 统一日志
通过 `core/logger.py` 统一配置 logging 模块，所有模块使用 `get_logger(__name__)` 获取 logger，支持 `LOG_LEVEL` 环境变量控制输出级别。

### 容错降级
- ChromaDB PersistentClient 失败时降级为 EphemeralClient（内存模式）
- BGE 模型加载失败时降级为哈希嵌入（256 维）
- Reflector JSON 解析失败时多重提取（直接解析 → 代码块提取 → 花括号截取）
- 主回复路径不降级：LLM 异常时直接 `raise RuntimeError`，经 SSE error 事件返回前端，便于排查而非用假回复掩盖问题

## 设计决策

记录关键架构选型背后的权衡，说明"为什么这么做"而非"做了什么"。

### 为什么选 BGE-small-zh 而非 OpenAI Embedding

- **离线可用**：BGE-small-zh 本地推理，不依赖外部 API，部署成本低，无网络延迟
- **中文优化**：针对中文语义优化，在角色对话场景（中文为主）召回率优于通用模型
- **维度可控**：512 维，在精度和存储/检索成本间平衡；OpenAI text-embedding-3-large 3072 维对个人项目过重
- **代价**：首次加载需下载模型（约 100MB），通过 HF 镜像缓解；牺牲了多语言能力（本项目不需要）
- **验证**：替换 MD5 哈希嵌入后，记忆召回率从 33% 提升到 100%

### 为什么用 PAD 三维情绪而非简单标签

- **表达力**：单一"好感度"无法区分"开心但平静"和"开心且兴奋"；PAD 三维可表达 27 种情绪状态
- **影响回复风格**：高唤醒度 → 更活泼多用 ♪/~；低唤醒度 → 更沉稳；高支配度 → 更主动引导话题
- **代价**：评估复杂度上升（4 维方向一致性判定）；LLM 生成情绪变化的稳定性需通过提示工程约束
- **验证**：情绪适当性 66.7% → 通过 reflector 提示词加入量级规则后提升至 75%+

### 为什么所有组件复用 Agent 的记忆实例

- **背景**：早期 Retriever/ToolExecutor/Consolidator 各自 `new EpisodicMemory(session_id)`，虽然底层指向同一 collection，但 Python 层持有不同的 `_collection` 引用对象
- **故障**：`agent.episodic.clear()` 删除并重建自己的引用后，其他组件的引用仍指向已删除的旧 collection，后续检索抛异常，导致"点忘记我后再发消息无输出"
- **修复**：Agent 初始化时创建唯一实例，通过构造参数注入各组件，保证 `clear()` 后引用全局同步
- **教训**：跨组件共享有状态资源时，必须统一实例化入口，避免"逻辑上同一份数据，物理上多个引用"的陷阱

### 为什么 Thinker 用角色化内心独白而非机械模板

- **角色一致性**：Thinker 输出 `inner_monologue` 字段——用星野爱第一人称语气生成的内心独白（如"哇～有人从2024年就喜欢爱到现在呢！爱真的好开心呀~"），同一份文本既用于 dev_mode 思考链展示，又注入 system prompt 指导 ReAct，思考链本身就是角色个性的体现
- **替代机械模板**：早期用"【执行计划】1. search_memory... 2. reply..."的技术模板展示思考链，不符合角色语气；改为 LLM 生成角色化独白，多任务时自然提到"先...再..."，单任务时流露情绪和打算
- **成本考量**：每次 thinking 多一次 LLM 调用（max_tokens 2048），对极短问候是浪费
- **预筛策略**：消息长度 <10 字时直接跳过 LLM 调用（如"你好""在吗"），≥10 字的正常伴聊消息会触发思考
- **multi-agent 体现**：Thinker 负责"理解+决策"，ReAct 负责"执行"，形成 think→execute 分工

### 为什么用 DeepSeek 而非 OpenAI

- **成本**：DeepSeek API 价格约为 GPT-4 的 1/10，适合个人项目持续运行
- **中文能力**：中文对话质量与 GPT-4 接近，角色扮演场景表现稳定
- **OpenAI 兼容协议**：可直接用 `openai` SDK，切换模型只需改 `base_url` 和 `model`，无迁移成本
- **代价**：thinking mode 不支持 `tool_choice` 强制，故 Reflector 用 `response_format=json_object` 而非 Function Calling

## License

MIT
