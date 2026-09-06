"""问答主链路（Agent 的核心编排）。

一次问答的完整流程：

    用户问题 + 会话历史
        │
        ├─ ① 问题改写：有历史时，先让 LLM 把"它的范围？"这类追问
        │              补全为可独立检索的问题
        ├─ ② 混合检索：向量 + BM25 召回 → Rerank 精排（见 retriever）
        ├─ ③ 置信度兜底：最相关片段分数仍低于阈值 → 直接回复"未找到"，绝不编造
        ├─ ④ 流式生成：把资料编号 [1][2]... 喂给大模型，逐字输出带引用的回答
        └─ ⑤ 存入记忆：本轮问答写入会话历史，供下一轮改写用

链路是一个异步生成器，yield 事件 dict，由 API 层转成 SSE 推给前端：
    {"type": "status",  "data": "正在检索知识库..."}     状态提示
    {"type": "sources", "data": [{编号、文件名、片段原文、分数}, ...]}  引用来源
    {"type": "token",   "data": "年假"}                  回答的一个文字片段
    {"type": "done",    "data": {"rewritten_query": ...}}  结束
    {"type": "error",   "data": "错误信息"}               异常
"""

import logging
from collections.abc import AsyncGenerator

from app.agent.memory import ConversationMemory
from app.config import settings
from app.core import llm, prompts
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class QAChain:
    """问答链路：编排"改写 → 检索 → 兜底 → 生成"。"""

    def __init__(self, retriever: Retriever, memory: ConversationMemory) -> None:
        self.retriever = retriever
        self.memory = memory

    async def run(
        self,
        session_id: str,
        question: str,
    ) -> AsyncGenerator[dict, None]:
        """执行一次问答，yield SSE 事件。"""
        question = question.strip()
        if not question:
            yield {"type": "error", "data": "问题不能为空"}
            return

        try:
            # ---------- ⓪ 闲聊/元问题直答（"你是谁"检索必然兜底，直接友好回复）----------
            chitchat = prompts.match_chitchat(question)
            if chitchat:
                yield {"type": "sources", "data": []}
                yield {"type": "token", "data": chitchat}
                self.memory.add_turn(session_id, question, chitchat)
                yield {"type": "done", "data": {"rewritten_query": None}}
                return

            # ---------- ① 问题改写（仅多轮对话时触发）----------
            history = self.memory.get_history(session_id)
            retrieval_query = question  # 默认用原问题检索
            if history:
                try:
                    rewritten = await llm.chat(
                        prompts.build_rewrite_messages(history, question),
                        temperature=0.1,
                    )
                    rewritten = rewritten.strip()
                    if rewritten:
                        logger.info("[会话 %s] 问题改写：%r → %r",
                                    session_id[:8], question, rewritten)
                        retrieval_query = rewritten
                except Exception as e:  # noqa: BLE001
                    # 改写失败不应阻断问答，降级为直接用原问题检索
                    logger.warning("问题改写失败，使用原问题检索：%s", e)

            # ---------- ② 混合检索 + Rerank ----------
            yield {"type": "status", "data": "正在检索知识库..."}
            chunks = await self.retriever.retrieve(retrieval_query)

            # ---------- ③ 置信度兜底 ----------
            if not chunks or chunks[0].score < settings.rerank_score_threshold:
                logger.info(
                    "检索结果置信度不足（最高分 %s），走兜底回答",
                    f"{chunks[0].score:.4f}" if chunks else "无结果",
                )
                answer = prompts.FALLBACK_ANSWER
                yield {"type": "sources", "data": []}
                yield {"type": "token", "data": answer}
                self.memory.add_turn(session_id, question, answer)
                yield {"type": "done", "data": {"rewritten_query": retrieval_query}}
                return

            # ---------- ④ 推送来源 + 流式生成 ----------
            sources = [
                {
                    "index": i,
                    "file_name": chunk.file_name,
                    "chunk_id": chunk.chunk_id,
                    "score": round(chunk.score, 4),
                    "text": chunk.text,
                }
                for i, chunk in enumerate(chunks, start=1)
            ]
            yield {"type": "sources", "data": sources}

            messages = prompts.build_answer_messages(
                retrieval_query,
                [chunk.text for chunk in chunks],
            )

            collected: list[str] = []
            async for token in llm.stream_chat(messages, temperature=0.5):
                collected.append(token)
                yield {"type": "token", "data": token}

            # ---------- ⑤ 保存本轮对话到记忆 ----------
            answer = "".join(collected).strip() or prompts.FALLBACK_ANSWER
            self.memory.add_turn(session_id, question, answer)
            yield {"type": "done", "data": {"rewritten_query": retrieval_query}}

        except Exception as e:  # noqa: BLE001
            # 任何异常都转成 error 事件，不让请求挂死
            logger.exception("问答链路异常")
            yield {"type": "error", "data": f"服务处理失败：{e}"}
