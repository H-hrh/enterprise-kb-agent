"""混合检索编排器：向量检索 + BM25 关键词检索 → 合并去重 → Rerank 精排。

整条检索链路（也是本项目的技术核心之一）：

    用户问题
      ├── 向量检索（语义相似，top 20）──┐
      │                                 ├── 合并去重 ── Rerank 精排 ── top 5 片段
      └── BM25 检索（关键词匹配，top 20）┘

设计要点：
- Retriever 内部维护一份 chunk 缓存（id → 原文/元数据），
  每次上传或删除文档后调用 refresh() 全量重建（同时重建 BM25）；
- BM25 只返回 chunk_id 和分数，原文统一从缓存/向量库取，避免两份数据不一致。
"""

import logging
from dataclasses import dataclass

from app.config import settings
from app.core import embeddings, reranker
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """检索结果片段。"""

    chunk_id: str
    text: str
    doc_id: str
    file_name: str
    score: float  # rerank 相关性分数（0~1，越高越相关）


class Retriever:
    """混合检索器。"""

    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        # chunk_id -> {"text": str, "metadata": dict}
        self._chunks: dict[str, dict] = {}

    def refresh(self) -> None:
        """从向量库拉取全部片段，刷新缓存并重建 BM25 索引。

        服务启动时、每次上传/删除文档后都要调用一次。
        """
        all_chunks = self.vector_store.get_all()
        self._chunks = {item["id"]: item for item in all_chunks}
        self.bm25_store.build(
            [(item["id"], item["text"]) for item in all_chunks]
        )
        logger.info("检索索引刷新完成，当前片段总数：%d", len(self._chunks))

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """对用户问题执行完整的混合检索 + 重排序。

        :return: 按相关性降序排列的片段列表（最多 final_top_k 个）
        """
        if not self._chunks:
            return []

        # ---------- 第 1 路：向量语义检索 ----------
        query_vector = await embeddings.embed_query(query)
        vector_hits = self.vector_store.query(query_vector, settings.vector_top_k)
        vector_ids = [hit["id"] for hit in vector_hits]

        # ---------- 第 2 路：BM25 关键词检索 ----------
        bm25_hits = self.bm25_store.search(query, settings.bm25_top_k)
        bm25_ids = [cid for cid, _ in bm25_hits]

        # ---------- 合并去重（保持向量结果优先的顺序）----------
        candidate_ids: list[str] = list(dict.fromkeys(vector_ids + bm25_ids))
        # 双保险：只保留缓存里确实存在的 id
        candidate_ids = [cid for cid in candidate_ids if cid in self._chunks]

        logger.info(
            "检索召回：向量 %d 条 / BM25 %d 条 / 合并去重后 %d 条",
            len(vector_ids),
            len(bm25_ids),
            len(candidate_ids),
        )

        if not candidate_ids:
            return []

        # ---------- Rerank 精排 ----------
        documents = [self._chunks[cid]["text"] for cid in candidate_ids]
        reranked = await reranker.rerank(query, documents, top_n=settings.final_top_k)

        results: list[RetrievedChunk] = []
        for cand_index, score in reranked:
            cid = candidate_ids[cand_index]
            meta = self._chunks[cid]["metadata"]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=self._chunks[cid]["text"],
                    doc_id=meta.get("doc_id", ""),
                    file_name=meta.get("file_name", "未知文档"),
                    score=score,
                )
            )

        if results:
            logger.info(
                "Rerank 完成：最终 %d 条，最高分 %.4f，来源文档：%s",
                len(results),
                results[0].score,
                {r.file_name for r in results},
            )
        return results
