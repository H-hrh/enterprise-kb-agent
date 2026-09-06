"""文档入库流水线。

完整链路：
    上传文件字节 → 校验类型/大小 → 保存原件 → 解析文本
    → 切分片段 → 批量向量化 → 写入 Chroma → 刷新检索索引

文档列表不单独维护一张表：直接从向量库每条片段的 metadata 聚合得到，
少一个数据源就少一份不一致的风险。
"""

import logging
import time
import uuid
from pathlib import Path

from app.config import UPLOAD_DIR, settings
from app.core import embeddings
from app.ingestion.parsers import ParseError, parse_file
from app.ingestion.splitter import split_text
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """文档入库 / 列表 / 删除。"""

    def __init__(self, vector_store: VectorStore, retriever: Retriever) -> None:
        self.vector_store = vector_store
        self.retriever = retriever

    async def ingest(self, filename: str, content: bytes) -> dict:
        """处理一个上传的文件，返回文档信息。

        :param filename: 原始文件名（用于展示和判断扩展名）
        :param content: 文件二进制内容
        """
        # ---------- 1. 校验 ----------
        ext = Path(filename).suffix.lower()
        if ext not in settings.allowed_extensions:
            raise ParseError(f"不支持的文件类型：{ext}（仅支持 .pdf / .docx / .md / .txt）")
        if len(content) > settings.max_upload_bytes:
            raise ParseError(f"文件过大：单文件上限为 {settings.max_upload_mb}MB")

        # ---------- 2. 保存原件（文件名用 doc_id，避免中文/重名问题）----------
        doc_id = uuid.uuid4().hex
        stored_path = UPLOAD_DIR / f"{doc_id}{ext}"
        stored_path.write_bytes(content)
        logger.info("收到文件《%s》，已保存为 %s（%d 字节）", filename, stored_path.name, len(content))

        try:
            # ---------- 3. 解析 → 切分 ----------
            text = parse_file(stored_path)
            chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                raise ParseError("文档内容为空或切分后没有有效片段")
            logger.info("《%s》解析切分完成，共 %d 个片段", filename, len(chunks))

            # ---------- 4. 向量化 ----------
            vectors = await embeddings.embed_texts(chunks)

            # ---------- 5. 写入向量库 ----------
            created_at = int(time.time())
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "doc_id": doc_id,
                    "file_name": filename,
                    "index": i,
                    "created_at": created_at,
                }
                for i in range(len(chunks))
            ]
            self.vector_store.add(
                ids=ids,
                texts=chunks,
                embeddings=vectors,
                metadatas=metadatas,
            )
        except Exception:
            # 入库失败：清理已保存的原件，避免残留脏数据
            stored_path.unlink(missing_ok=True)
            raise

        # ---------- 6. 刷新检索索引（BM25 + 缓存）----------
        self.retriever.refresh()

        return {
            "doc_id": doc_id,
            "file_name": filename,
            "chunk_count": len(chunks),
            "created_at": created_at,
        }

    def list_documents(self) -> list[dict]:
        """汇总文档列表：按 doc_id 分组统计片段数。"""
        agg: dict[str, dict] = {}
        for item in self.vector_store.get_all():
            meta = item["metadata"]
            doc_id = meta.get("doc_id", "")
            if doc_id not in agg:
                agg[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": meta.get("file_name", "未知文档"),
                    "chunk_count": 0,
                    "created_at": meta.get("created_at", 0),
                }
            agg[doc_id]["chunk_count"] += 1
        # 新上传的排前面
        return sorted(agg.values(), key=lambda doc: doc["created_at"], reverse=True)

    def delete_document(self, doc_id: str) -> bool:
        """删除文档（向量片段 + 原件 + 刷新索引）。返回是否删除了内容。"""
        all_chunks = self.vector_store.get_all()
        target = [item for item in all_chunks if item["metadata"].get("doc_id") == doc_id]
        if not target:
            return False

        # 删除原件（原件命名为 {doc_id}{扩展名}）
        ext = Path(target[0]["metadata"].get("file_name", "")).suffix
        (UPLOAD_DIR / f"{doc_id}{ext}").unlink(missing_ok=True)

        # 删除向量片段并刷新索引
        self.vector_store.delete_by_doc(doc_id)
        self.retriever.refresh()
        logger.info("文档 %s 已删除（%d 个片段）", doc_id, len(target))
        return True
