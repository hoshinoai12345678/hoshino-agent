"""FastAPI 应用入口

启动时构建知识索引，挂载静态文件，注册 API 路由。
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import BASE_DIR, ENABLE_DEVMODE
from rag.indexer import KnowledgeIndexer
from api.chat import router as chat_router
from api.auth import router as auth_router
from core.logger import get_logger

logger = get_logger(__name__)

# 生产部署关闭 /docs /redoc /openapi.json，避免暴露接口结构
# 通过环境变量控制：生产环境设 HIDE_API_DOCS=1
_hide_docs = os.getenv("HIDE_API_DOCS", "0").lower() == "1"
app = FastAPI(
    title="星野爱 · 拟人化AI伴聊",
    version="1.0",
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
    """启动时构建知识索引 + 预加载模型（避免首次请求卡顿）"""
    indexer = KnowledgeIndexer()
    count = indexer.build_index()
    logger.info(f"角色知识索引就绪，共 {count} 个分块")

    # 预加载模型到内存（懒加载触发，首次请求不再等模型加载）
    import core.utils as _utils
    _utils._get_model()      # BGE-small-zh embedding
    _utils._get_reranker()   # BGE-reranker-base cross-encoder
    _utils._get_encoder()    # tiktoken
    logger.info("模型预加载完成")

    logger.info(f"开发者模式: {'开启' if ENABLE_DEVMODE else '关闭'}")


@app.get("/")
async def index():
    """主页"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "hoshino-ai"}
