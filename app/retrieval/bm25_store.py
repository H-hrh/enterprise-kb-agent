"""BM25 关键词检索索引。

【为什么向量检索之外还要 BM25？】
- 向量检索擅长"意思相近"：问"放假几天"能匹配到讲"年假"的片段；
- 但遇到产品型号、人名、专有名词（如"RTX 4090""张三"），
  关键词精确匹配往往更可靠——这正是 BM25 的强项。
两者结合 = 混合检索（Hybrid Search），是工业界 RAG 的标配。

【中文分词】
BM25 按"词"匹配，而中文句子词与词之间没有空格，
所以先用 jieba 把句子切成词（"员工年假有几天" → ["员工", "年假", "有", "几天"]），
再交给 rank_bm25 的 BM25Okapi 建索引。

数据量小（演示场景几千个片段以内），每次上传/删除后直接全量重建即可。
"""

import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """中文分词：返回词列表，过滤空白词。"""
    return [word for word in jieba.lcut(text) if word.strip()]


class BM25Store:
    """BM25 索引，与向量库保持同步（重建式更新）。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._index: BM25Okapi | None = None

    def build(self, items: list[tuple[str, str]]) -> None:
        """全量重建索引。

        :param items: [(chunk_id, 片段原文), ...]
        """
        self._ids = [cid for cid, _ in items]
        tokenized_corpus = [_tokenize(text) for _, text in items]
        # 语料为空时不建索引（search 会直接返回空）
        self._index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """关键词检索。

        :return: [(chunk_id, BM25 分数), ...]，按分数降序，过滤掉 0 分（完全没命中）
        """
        if not self._index or not self._ids:
            return []

        scores = self._index.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._ids, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            (cid, float(score))
            for cid, score in ranked[:top_k]
            if score > 0
        ]
