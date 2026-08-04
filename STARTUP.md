# 星野爱 · 拟人化 AI 伴聊 — 启动文档

> 虚拟角色拟人 Agent 项目，基于 FastAPI + ChromaDB + DeepSeek（OpenAI 兼容协议）实现。
> 本文档覆盖环境准备、启动方式、接口验证与常见故障排查。

---

## 一、环境要求

| 依赖 | 最低版本 | 备注 |
|------|---------|------|
| Python | 3.10+ | 用到 `str | None` 等 PEP604 语法 |
| pip | 任意 | 用于安装依赖 |

操作系统：Windows / macOS / Linux 均可（本项目在 Windows 上验证）。

---

## 二、安装依赖

在项目根目录 `e:\ai-agent\hoshino-ai` 下执行：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

```
fastapi>=0.110
uvicorn[standard]>=0.27
chromadb>=0.4.22
openai>=1.12
pydantic>=2.6
python-multipart>=0.0.9
sse-starlette>=1.6
```

---

## 三、配置（可选）

所有配置通过环境变量覆盖，默认值定义在 [config.py](file:///e:/ai-agent/hoshino-ai/config.py)。常用项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | 内置 demo key | DeepSeek API Key，**生产环境务必替换** |
| `LLM_API_BASE` | `https://api.deepseek.com/v1` | LLM 接口地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名 |
| `WEB_PORT` | `8000` | 配置中的端口（实际启动以命令为准） |
| `ENABLE_DEVMODE` | `false` | 开发者模式，开启后前端展示思考链 |
| `ENABLE_REFLECTION` | `true` | 是否启用反思机制 |
| `VECTOR_DB_PATH` | `data/vector_db` | 向量库存储路径 |
| `MEMORY_DB_PATH` | `data/memory.db` | 语义记忆 SQLite 路径 |
| `HIDE_API_DOCS` | `0` | 公网部署设 `1`，关闭 `/docs` `/redoc` `/openapi.json` 防止接口结构泄露 |
| `COOKIE_SECURE` | `0` | HTTPS 部署设 `1`，登录 Cookie 仅通过 HTTPS 传输 |
| `AUTH_SECRET_KEY` | 自动生成并落盘 | Cookie 签名密钥；未设置时自动生成 64 位随机值存到 `data/.auth_secret`，生产环境建议通过环境变量显式指定 |

> Tip：Windows PowerShell 设置环境变量用 `$env:LLM_API_KEY="你的key"`；CMD 用 `set LLM_API_KEY=你的key`。

---

## 四、启动命令

### 1. 默认启动（8000 端口，开发者测试用）

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

- `--reload`：代码变更自动重启，开发阶段推荐
- 启动成功标志：日志出现 `Application startup complete.`
- 启动时会自动构建角色知识索引，控制台输出 `[startup] 角色知识索引就绪，共 N 个分块`

### 2. 用户端启动（8001 端口，生产/正式使用）

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

- 生产环境去掉 `--reload`，避免文件监听开销
- 如需对外访问，把 `--host` 改为 `0.0.0.0`（注意网络安全）

### 3. 后台运行（Windows PowerShell）

```powershell
Start-Process python -ArgumentList "-m","uvicorn","app:app","--host","127.0.0.1","--port","8001" -WorkingDirectory "e:\ai-agent\hoshino-ai"
```

### 4. 公网穿透（临时分享给他人试用）

服务默认绑 `127.0.0.1`，不直接对外。要给别人试用，**推荐用内网穿透**（自带 HTTPS，比改 `--host 0.0.0.0` 更安全），用完关掉隧道链接立即失效。

**前置：先设生产环境变量**（关文档、强制 HTTPS Cookie）：

```powershell
$env:HIDE_API_DOCS="1"
$env:COOKIE_SECURE="1"
```

**步骤 1：启动服务**（仍绑 127.0.0.1，不开公网端口）

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

**步骤 2：另开一个终端，启动穿透隧道**

```bash
# 方案 A：cloudflared（推荐，免费、免注册、自带 HTTPS）
# 安装：winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://127.0.0.1:8001

# 方案 B：ngrok（需注册）
# 安装：https://ngrok.com/download
ngrok http 8001
```

隧道启动后会输出一个公网链接，例如：

```
https://xxx-yyy-zzz.trycloudflare.com
```

把这个链接发给对方即可，对方通过浏览器访问会看到登录页，注册账号后开始聊天。

**步骤 3：停止**

- 隧道终端：`Ctrl+C`，链接立即失效
- 服务终端：`Ctrl+C`

> **安全提示**：
> - `data/.auth_secret`（自动生成的 Cookie 签名密钥）不要提交到 git，建议加进 `.gitignore`
> - 穿透链接发给可信的人即可，虽已关 API 文档，但任何人拿到链接都能注册账号占用 LLM 额度
> - 长期公网部署建议改用反向代理 + 真实域名 + Let's Encrypt 证书，而非临时穿透

### 5. 停止服务

- 前台运行：终端按 `Ctrl+C`
- 后台运行：`Stop-Process -Name python -Force`（会杀掉所有 python，谨慎使用）
  或根据端口查找：`Get-NetTCPConnection -LocalPort 8001 | Select-Object OwningProcess`，然后 `Stop-Process -Id <PID>`

---

## 五、访问入口

| 地址 | 说明 |
|------|------|
| http://127.0.0.1:8000 | 开发者测试入口（8000） |
| http://127.0.0.1:8001 | 用户正式入口（8001） |
| http://127.0.0.1:8000/health | 健康检查，返回 `{"status":"ok","service":"hoshino-ai"}` |

---

## 六、API 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | SSE 流式聊天，body：`{"message": "你好"}` |
| GET | `/api/state` | 获取 Agent 当前状态（情绪、好感度等） |
| POST | `/api/reset` | 重置 Agent，清空工作记忆与情绪 |
| GET | `/api/memories?limit=20` | 获取情景记忆与语义记忆列表 |

### 接口验证示例

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

发送一条消息（SSE 流，需支持 SSE 的客户端或浏览器 EventSource）：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"你好\"}"
```

---

## 七、常见故障排查

### 1. ChromaDB 报 "拒绝访问 (os error 5)"

**现象**：Windows 下启动报 `PersistentClient` 初始化失败，`os error 5`。

**已内置降级处理**：[core/memory/episodic.py](file:///e:/ai-agent/hoshino-ai/core/memory/episodic.py#L27) 和 [rag/indexer.py](file:///e:/ai-agent/hoshino-ai/rag/indexer.py#L28) 的 `_ensure_client` 已用 try-except 包裹，失败时自动降级为 `EphemeralClient`（内存模式），服务仍可启动。

**降级影响**：向量数据仅存于内存，**进程重启后丢失**。

**恢复持久化**：
1. 检查 `data/vector_db` 目录是否存在且当前用户有写权限
2. 关闭可能锁定 `.sqlite3` 文件的进程（旧 uvicorn、IDE 索引、其他 Python 进程）
3. 必要时删除 `data/vector_db` 目录后重启，让 ChromaDB 重建

### 2. 端口被占用

```bash
# 查看占用进程
Get-NetTCPConnection -LocalPort 8001
# 按 PID 终止
Stop-Process -Id <PID> -Force
```

### 3. LLM 调用失败

- 检查 `LLM_API_KEY` 是否有效
- 检查网络能否访问 `https://api.deepseek.com`
- 查看控制台是否打印异常堆栈

### 4. 知识索引为 0

- 确认 `data/knowledge/` 目录下存在 `.md` 文件
- 当前已有：`background.md`、`personality.md`、`quotes.md`、`relationships.md`

---

## 八、目录结构

```
hoshino-ai/
├── app.py                # FastAPI 入口
├── config.py             # 全局配置
├── requirements.txt
├── STARTUP.md            # 本文档
├── agent/                # Agent 核心（ReAct + 反思）
├── api/                  # FastAPI 路由
├── core/                 # 角色、情绪、记忆系统
│   └── memory/           # 工作记忆 / 情景记忆 / 语义记忆
├── rag/                  # 知识索引与检索
├── data/
│   ├── knowledge/        # 角色知识源文档（.md）
│   ├── memory.db         # 语义记忆 SQLite
│   └── persona.json      # 角色人设
└── static/               # 前端页面
```
