"""企业知识库智能体（Enterprise Knowledge Base Agent）。

包结构说明：
- app.config      : 全局配置（从 .env 读取）
- app.core        : 大模型 / 向量 / 重排序客户端、Prompt 模板
- app.ingestion   : 文档解析、切分、入库流水线
- app.retrieval   : 向量库、BM25、混合检索编排
- app.agent       : 会话记忆、问答主链路（问题改写 → 检索 → 生成）
- app.api         : FastAPI 路由
- app.web         : 网页前端（原生 HTML/CSS/JS）
"""
