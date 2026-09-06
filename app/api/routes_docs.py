"""文档管理接口：上传 / 列表 / 删除。"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.ingestion.parsers import ParseError
from app.schemas import DocumentInfo
from app.services import pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["文档管理"])


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)) -> dict:
    """上传一个文档文件（PDF/DOCX/MD/TXT），解析切分后入库。"""
    # 大小预检：Starlette 会记录上传文件大小，超限直接拒绝，不必读入内存
    if file.size and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400, detail=f"文件过大：单文件上限为 {settings.max_upload_mb}MB"
        )

    content = await file.read()
    logger.info("收到上传：%s（%d 字节）", file.filename, len(content))

    try:
        info = await pipeline.ingest(file.filename or "未命名文件", content)
    except ParseError as e:
        # 文件格式 / 解析类问题 → 400（用户能看懂的错误）
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        # 模型 API 调用失败 → 502（上游服务问题）
        raise HTTPException(status_code=502, detail=str(e)) from e

    return info


@router.get("", response_model=list[DocumentInfo])
def list_documents() -> list[dict]:
    """获取已上传文档列表。"""
    return pipeline.list_documents()


@router.get("/{doc_id}/chunks")
def get_document_chunks(doc_id: str) -> dict:
    """获取某文档的全部片段（前端点击文档卡片时预览原文用）。"""
    chunks = pipeline.get_document_chunks(doc_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail="文档不存在或已被删除")
    return {"doc_id": doc_id, "chunks": chunks}


@router.delete("/{doc_id}")
def delete_document(doc_id: str) -> dict:
    """删除指定文档及其全部片段。"""
    ok = pipeline.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在或已被删除")
    return {"ok": True}
