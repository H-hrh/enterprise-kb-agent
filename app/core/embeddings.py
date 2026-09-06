"""向量模型客户端：把文本变成向量（Embedding）。

【为什么需要 Embedding？】
大模型本身记不住企业文档。RAG 的做法是：
1. 把文档切成小片段；
2. 用 Embedding 模型把每个片段转成一个 1024 维的浮点数向量
   （语义相近的文本，向量距离也近）；
3. 提问时把问题也转成向量，在向量库里找距离最近的片段。
这样就能按"语义"而不是"关键词一模一样"来检索资料。

bge-m3 是中文效果很好的开源向量模型，硅基流动上免费调用。
"""

from openai import APIError

from app.config import settings
from app.core.llm import get_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量把文本转成向量。

    :param texts: 文本列表
    :return: 与 texts 顺序一致的向量列表（每个是 1024 个浮点数）

    接口单次条数有上限，所以按 embedding_batch_size 自动分批。
    """
    if not texts:
        return []

    client = get_client()
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start : start + settings.embedding_batch_size]
        try:
            resp = await client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
        except APIError as e:
            raise RuntimeError(f"向量模型调用失败（请检查模型 ID 与 Key）：{e}") from e

        # resp.data 里每条带 index，按 index 排序保证顺序不乱
        ordered = sorted(resp.data, key=lambda item: item.index)
        all_vectors.extend(item.embedding for item in ordered)

    return all_vectors


async def embed_query(text: str) -> list[float]:
    """把单条文本（通常是用户问题）转成向量。"""
    vectors = await embed_texts([text])
    return vectors[0]
