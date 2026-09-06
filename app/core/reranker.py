"""重排序客户端（Rerank）。

【为什么还需要 Rerank？】
混合检索（向量 + BM25）会快速召回几十个候选片段，但里面有不少"凑数"的：
- 向量检索擅长语义相似，但可能召回"意思接近但没回答问题"的片段；
- BM25 擅长关键词匹配，但可能命中只是用词相同的片段。

Rerank 模型（bge-reranker-v2-m3）把"问题"和"每个候选片段"成对送入，
直接输出一个相关性分数，比向量相似度更准。
我们用它对候选片段重新排序，只把最相关的 5 个交给大模型回答。
"""

import httpx

from app.config import settings


async def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """对候选片段按与问题的相关性重新打分排序。

    :param query: 用户问题（改写后的独立问题）
    :param documents: 候选片段文本列表
    :param top_n: 只返回前 N 个；None 表示全部返回
    :return: [(片段在 documents 中的下标, 相关性分数), ...]，按分数降序；
             分数范围约 0~1，越高越相关
    """
    if not documents:
        return []

    if not settings.has_api_key:
        raise RuntimeError("未配置 SILICONFLOW_API_KEY，无法调用重排序接口")

    # 硅基流动的 rerank 接口地址：{base_url}/rerank
    url = f"{settings.api_base}/rerank"
    headers = {"Authorization": f"Bearer {settings.siliconflow_api_key}"}
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_n if top_n is not None else len(documents),
        "return_documents": False,  # 只需要分数和下标，不用把原文回传
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"重排序接口调用失败：{e}") from e

    data = resp.json()
    # 返回结构：{"results": [{"index": 3, "relevance_score": 0.98}, ...]}
    return [
        (item["index"], float(item["relevance_score"]))
        for item in data.get("results", [])
    ]
