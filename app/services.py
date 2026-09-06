"""服务装配模块（依赖注入的简化版）。

在这里把各个组件实例化并"接好线"，其他地方直接 import 使用：
    向量库 ──┐
             ├── 检索器（混合检索 + Rerank）──┐
    BM25 ───┘                                 ├── 问答链路（Agent）
                                             │
    入库流水线 ──────────────────────────────┘
    会话记忆 ─────────────────────────────────┘

导入本模块即完成初始化（含从持久化数据重建索引），main.py 启动时导入一次。
"""

import logging

from app.agent.chain import QAChain
from app.agent.memory import ConversationMemory
from app.config import ensure_dirs
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25_store import BM25Store
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# 0. 确保 data/ 目录存在
ensure_dirs()

# 1. 底层组件
vector_store = VectorStore()
bm25_store = BM25Store()

# 2. 检索器：向量库 + BM25 + Rerank
retriever = Retriever(vector_store, bm25_store)
# 服务启动时从 Chroma 持久化数据重建 BM25 索引（重启不丢的关键一步）
retriever.refresh()

# 3. 入库流水线
pipeline = IngestionPipeline(vector_store, retriever)

# 4. 对话链路（ReAct Agent：检索器 + 记忆 + 入库流水线[list_documents 工具用]）
memory = ConversationMemory()
qa_chain = QAChain(retriever, memory, pipeline)

logger.info(
    "服务组件初始化完成：已有片段 %d 条",
    vector_store.count(),
)
