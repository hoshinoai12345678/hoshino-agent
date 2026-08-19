"""FastAPI 应用入口（多角色支持）

启动时遍历所有角色构建知识索引，挂载静态文件，注册 API 路由。
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import BASE_DIR, ENABLE_DEVMODE, list_available_characters, DEFAULT_CHARACTER_ID
from rag.indexer import KnowledgeIndexer
from api.chat import router as chat_router
from api.auth import router as auth_router
from core.logger import get_logger

logger = get_logger(__name__)

# 生产部署关闭 /docs /redoc /openapi.json，避免暴露接口结构
# 通过环境变量控制：生产环境设 HIDE_API_DOCS=1
_hide_docs = os.getenv("HIDE_API_DOCS", "0").lower() == "1"
app = FastAPI(
    title="拟人化AI伴聊 · 多角色 Agent",
    version="2.0",
    docs_url=None if _hide_docs else "/docs",
    redoc_url=None if _hide_docs else "/redoc",
    openapi_url=None if _hide_docs else "/openapi.json",
)

# 注册 API 路由
app.include_router(auth_router)
app.include_router(chat_router)

# 静态文件
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup():
    """启动时构建所有角色知识索引 + 预加载模型（避免首次请求卡顿）"""
    character_ids = list_available_characters()
    if not character_ids:
        logger.warning("未发现任何角色目录，请检查 data/characters/ 结构")
        return

    total_chunks = 0
    for cid in character_ids:
        try:
            indexer = KnowledgeIndexer(character_id=cid)
            count = indexer.build_index()
            total_chunks += count
            logger.info(f"角色 [{cid}] 知识索引就绪，{count} 个分块")
        except Exception as e:
            logger.error(f"角色 [{cid}] 知识索引构建失败: {e}")
    logger.info(f"全部角色索引完成，共 {len(character_ids)} 个角色，{total_chunks} 个分块")

    # 预加载模型到内存（懒加载触发，首次请求不再等模型加载）
    # 模型全局共享，不需要按角色加载
    import core.utils as _utils
    _utils._get_model()      # BGE-small-zh embedding
    _utils._get_reranker()   # BGE-reranker-base cross-encoder
    _utils._get_encoder()    # tiktoken
    logger.info("模型预加载完成")

    logger.info(f"默认角色: {DEFAULT_CHARACTER_ID}，开发者模式: {'开启' if ENABLE_DEVMODE else '关闭'}")


@app.get("/")
async def index():
    """主页"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "hoshino-agent",
        "characters": list_available_characters(),
        "default_character": DEFAULT_CHARACTER_ID,
    }
