"""向量数据库封装（ChromaDB 本地持久化模式）。

Chroma 是一款轻量级开源向量数据库：
- 本地模式不需要单独启动数据库服务，数据以文件形式存在 data/chroma 目录；
- 重启服务后数据还在（持久化）；
- 我们自己调用硅基流动 API 生成向量，再把向量连同原文一起写入 Chroma。

每个片段（chunk）在库里对应一条记录：
- id        : 唯一标识，格式 "{doc_id}_{chunk_index}"
- embedding : 1024 维向量
- document  : 片段原文
- metadata  : {"doc_id", "file_name", "index", "created_at"}
"""

import chromadb

from app.config import CHROMA_DIR

COLLECTION_NAME = "knowledge_base"


class VectorStore:
    """ChromaDB 的薄封装，只暴露本项目用到的几个方法。"""

    def __init__(self) -> None:
        # PersistentClient：数据落盘到指定目录（重启不丢）
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # cosine（余弦相似度）是文本向量检索最常用的距离度量
        # 注意：不指定 embedding_function，因为向量由我们自己调 API 生成后传入
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """写入一批片段（已存在相同 id 则覆盖，用 upsert）。"""
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        """向量检索：返回与输入向量最相似的 top_k 个片段。

        :return: [{"id", "text", "metadata", "distance"}, ...]
                 distance 是余弦距离（0~2），越小越相似
        """
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        for cid, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            hits.append({"id": cid, "text": text, "metadata": meta, "distance": float(dist)})
        return hits

    def get_all(self) -> list[dict]:
        """取出全部片段（用于启动时重建 BM25 索引、汇总文档列表）。"""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": cid, "text": text, "metadata": meta}
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        """删除某篇文档的所有片段。"""
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        """库中片段总数。"""
        return self._collection.count()
