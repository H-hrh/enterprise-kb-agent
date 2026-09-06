"""FastAPI 应用入口。

启动方式（在项目根目录）：
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
然后浏览器打开：http://localhost:8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import services  # noqa: F401  导入即完成组件初始化与索引重建
from app.api import routes_chat, routes_docs
from app.config import settings
from app.schemas import HealthResponse

# 日志配置：让 uvicorn 和我们自己的日志统一格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="企业知识库智能体", version="1.0.0")

# ---------- 注册 API 路由 ----------
app.include_router(routes_docs.router)
app.include_router(routes_chat.router)


@app.get("/api/health", response_model=HealthResponse, tags=["系统"])
def health() -> HealthResponse:
    """健康检查：服务是否存活、Key 是否配置、知识库规模。"""
    return HealthResponse(
        status="ok",
        api_key_configured=settings.has_api_key,
        document_count=len(services.pipeline.list_documents()),
        chunk_count=services.vector_store.count(),
    )


# ---------- 静态前端（放在 API 路由之后注册，避免覆盖 /api）----------
WEB_DIR = Path(__file__).resolve().parent / "web"

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """首页：聊天网页。"""
    return FileResponse(WEB_DIR / "index.html")
