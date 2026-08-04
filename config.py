"""全局配置管理

环境变量覆盖默认值，LLM 使用 DeepSeek（OpenAI 兼容协议）。
"""
import os
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

# HuggingFace 镜像（国内网络必需，必须在导入 sentence-transformers 前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# HuggingFace 模型缓存目录（BGE-small-zh 等模型存于此）
# 默认放在项目同级目录，可通过环境变量覆盖
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", str(BASE_DIR.parent / "hf_cache"))
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

# ---- 数据存储 ----
DATA_DIR = BASE_DIR / "data"
PERSONA_FILE = str(DATA_DIR / "persona.json")
KNOWLEDGE_DIR = str(DATA_DIR / "knowledge")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", str(DATA_DIR / "memory.db"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "vector_db"))

# ---- 记忆系统 ----
WORKING_MEMORY_SIZE = int(os.getenv("WORKING_MEMORY_SIZE", "20"))  # 工作记忆保留近K轮
EPISODIC_TOP_K = int(os.getenv("EPISODIC_TOP_K", "3"))  # 情景记忆检索条数（收紧以提升 precision）
KNOWLEDGE_TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "3"))  # 知识检索条数
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "7"))  # 记忆衰减半衰期
CONSOLIDATION_THRESHOLD = int(os.getenv("CONSOLIDATION_THRESHOLD", "5"))  # 触发巩固的对话轮数

# ---- 情绪系统 ----
EMOTION_DECAY_RATE = float(os.getenv("EMOTION_DECAY_RATE", "0.05"))  # 每轮情绪向基线衰减
INITIAL_FAVORABILITY = int(os.getenv("INITIAL_FAVORABILITY", "10"))  # 初始好感度

# ---- LLM 配置（DeepSeek，OpenAI 兼容协议）----
# API Key 必须从环境变量 deepseek_apikey 读取，禁止硬编码
LLM_API_KEY = os.getenv("deepseek_apikey")
LLM_API_BASE = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-v4-flash"

LLM_ENABLED = bool(LLM_API_KEY)
if not LLM_ENABLED:
    logger.warning("未设置环境变量 deepseek_apikey，LLM 功能将不可用")

# LLM 生成参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ---- Agent 配置 ----
MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", "3"))  # ReAct 最大迭代次数
ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "true").lower() == "true"
ENABLE_DEVMODE = os.getenv("ENABLE_DEVMODE", "false").lower() == "true"  # 开发者模式（展示思考链）

# ---- Web 服务 ----
# 注意：此配置项仅供参考，实际启动端口由 uvicorn 命令行参数 --port 决定
# 开发者测试用 8000，用户正式用 8001（见 STARTUP.md）
WEB_PORT = int(os.getenv("WEB_PORT", "8001"))

# ---- 默认会话 ----
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "default")

# ---- 用户认证 ----
# 用户数据文件（JSON），存储用户名和密码哈希
USERS_FILE = os.getenv("USERS_FILE", str(DATA_DIR / "users.json"))
# Cookie 签名密钥：优先读环境变量；未设置时从 data/.auth_secret 持久化读取，
# 文件不存在则生成随机密钥并落盘（保证重启后登录态不失效，避免默认占位值）
_AUTH_SECRET_FILE = DATA_DIR / ".auth_secret"


def _resolve_auth_secret_key() -> str:
    env_val = os.getenv("AUTH_SECRET_KEY")
    if env_val:
        return env_val
    try:
        if _AUTH_SECRET_FILE.exists():
            cached = _AUTH_SECRET_FILE.read_text(encoding="utf-8").strip()
            if cached:
                return cached
    except Exception as e:
        logger.warning(f"读取 AUTH_SECRET_KEY 缓存失败: {e}")
    # 生成 32 字节随机密钥并落盘
    import secrets as _secrets
    new_key = _secrets.token_hex(32)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _AUTH_SECRET_FILE.write_text(new_key, encoding="utf-8")
        logger.info("已自动生成 AUTH_SECRET_KEY 并落盘到 data/.auth_secret")
    except Exception as e:
        logger.warning(f"AUTH_SECRET_KEY 落盘失败，本次用临时密钥: {e}")
    return new_key


AUTH_SECRET_KEY = _resolve_auth_secret_key()
# Cookie 有效期（秒），默认 7 天
AUTH_COOKIE_MAX_AGE = int(os.getenv("AUTH_COOKIE_MAX_AGE", "604800"))
# Cookie 名称
AUTH_COOKIE_NAME = "hoshino_session"

# ---- Telegram Bot 配置 ----
# Bot Token（从 @BotFather 获取），必须通过环境变量配置，禁止硬编码
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
# 代理地址（国内访问 Telegram API 必需，FlClash 默认 7890）
# 环境变量 HTTPS_PROXY 优先级更高
TG_HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or "http://127.0.0.1:7890"
