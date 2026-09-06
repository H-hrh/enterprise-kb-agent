"""问答主链路：ReAct Agent 决策循环。

一次问答的完整流程：

    用户问题 + 会话历史
        |
        |-- 0. 闲聊直答：「你是谁」这类寒暄直接回复，不进 Agent 循环
        |
        |-- 1. Agent 循环（最多 agent_max_rounds 轮，每轮一次模型调用）：
        |       模型通过 Function Calling 自主决策：
        |         * 要查资料 -> 调用 search_knowledge_base（可多轮、多角度检索）
        |         * 要看库里有什么 -> 调用 list_documents
        |         -> 工具结果喂回模型，进入下一轮决策
        |       或：直接生成最终回答（带 [n] 引用）-> 循环结束
        |
        |-- 2. 解析回答里实际引用的 [n]，只推送被引用的来源卡片
        |
        `-- 3. 存入会话记忆

与"固定流水线 RAG"的本质区别：什么时候检索、检索几次、要不要回答，
由大模型自主决策，而不是代码写死 if-else。
若模型接口不支持工具调用或循环早期异常，自动降级为固定流水线（_pipeline_fallback）。
"""

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from app.agent.memory import ConversationMemory
from app.config import settings
from app.core import llm, prompts
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------- 工具注册表（OpenAI function calling 格式） ----------
# 模型看到的就是这两份"工具说明书"，由它自己决定何时调用、参数传什么
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在企业知识库中检索与查询词最相关的文档片段，"
                "返回带编号的资料（含来源文件名与相关度分数），编号用于回答中标注引用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词或问题，尽量具体；可换不同关键词多次检索",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "查看知识库中已上传的全部文档列表（文件名与片段数）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class QAChain:
    """把记忆、检索器、大模型串成一条会自己查资料的 Agent 链路。"""

    def __init__(
        self,
        retriever: Retriever,
        memory: ConversationMemory,
        pipeline: IngestionPipeline,  # list_documents 工具需要
    ) -> None:
        self.retriever = retriever
        self.memory = memory
        self.pipeline = pipeline

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    async def run(self, session_id: str, question: str) -> AsyncGenerator[dict, None]:
        """处理一次提问，产出 SSE 事件流（status/token/sources/done/error）。"""
        question = question.strip()
        if not question:
            yield {"type": "error", "data": "问题不能为空"}
            return

        # ---------- 0. 闲聊直答（寒暄问题检索必然兜底，省一次模型调用） ----------
        chitchat = prompts.match_chitchat(question)
        if chitchat:
            yield {"type": "sources", "data": []}
            yield {"type": "token", "data": chitchat}
            self.memory.add_turn(session_id, question, chitchat)
            yield {"type": "done", "data": {"rewritten_query": None}}
            return

        try:
            yield {"type": "status", "data": "正在思考..."}
            history = self.memory.get_history(session_id)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT}
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": question})

            registry: dict[int, dict] = {}  # 资料编号 -> 来源信息（跨多轮检索累积）
            counter = [0]  # 资料全局编号：多次检索连续编号，避免不同轮的 [1] 撞号
            final_text = ""
            streamed_any = False

            for round_no in range(1, settings.agent_max_rounds + 1):
                # ---------- 1a. 本轮流式调用：可能直接回答，也可能请求调工具 ----------
                tool_calls: list[dict] | None = None
                round_text = ""
                async for kind, data in llm.stream_chat_with_tools(messages, TOOLS):
                    if kind == "text":
                        round_text += data
                        streamed_any = True
                        yield {"type": "token", "data": data}
                    else:  # kind == "tools"
                        tool_calls = data

                if not tool_calls:
                    final_text = round_text
                    break  # 模型直接给出最终回答，决策循环结束

                # ---------- 1b. 模型要求调用工具：执行并把结果喂回去 ----------
                messages.append(
                    {
                        "role": "assistant",
                        "content": round_text or None,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": c["name"],
                                    "arguments": json.dumps(
                                        c["arguments"], ensure_ascii=False
                                    ),
                                },
                            }
                            for c in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    name, args = call["name"], call["arguments"]
                    yield {"type": "status", "data": self._tool_status(name, args)}
                    logger.info(
                        "[会话 %.8s] Agent 第 %d 轮调用 %s %s",
                        session_id,
                        round_no,
                        name,
                        args,
                    )
                    try:
                        result = await self._execute_tool(name, args, registry, counter)
                    except Exception:  # 工具失败不能中断会话，把错误交给模型自行调整
                        logger.exception("工具 %s 执行失败", name)
                        result = "工具执行失败，请基于已有资料回答或换一种问法。"
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": result}
                    )

                if round_no == settings.agent_max_rounds:
                    # 达到轮数上限还没收敛：摘掉工具，强制模型直接作答
                    yield {"type": "status", "data": "整理答案..."}
                    async for token in llm.stream_chat(messages, temperature=0.5):
                        final_text += token
                        streamed_any = True
                        yield {"type": "token", "data": token}

            # ---------- 2. 收尾：只推送回答中实际引用 [n] 的来源 ----------
            answer = final_text.strip()
            if not answer:  # 模型没产出正文（极端情况），走兜底话术
                answer = prompts.FALLBACK_ANSWER
                yield {"type": "token", "data": answer}

            cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}
            sources = [
                {
                    "index": n,
                    "file_name": item["file_name"],
                    "chunk_id": "",
                    "score": round(item["score"], 4),
                    "text": item["text"],
                }
                for n, item in sorted(registry.items())
                if n in cited
            ]
            yield {"type": "sources", "data": sources}

            # ---------- 3. 存入记忆 ----------
            self.memory.add_turn(session_id, question, answer)
            yield {"type": "done", "data": {"rewritten_query": None}}

        except Exception as e:
            # Agent 循环异常（如模型接口不支持工具调用）：降级为固定流水线
            logger.exception("Agent 循环异常，降级为固定流水线")
            if not streamed_any:  # 还没流出任何正文，可以无缝换路径
                async for evt in self._pipeline_fallback(session_id, question):
                    yield evt
            else:
                yield {"type": "error", "data": f"服务处理失败，请重试：{e}"}

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    @staticmethod
    def _tool_status(name: str, args: dict) -> str:
        """工具调用期间前端状态行展示的文案。"""
        if name == "search_knowledge_base":
            return f"🔍 检索知识库：{args.get('query', '')}"
        if name == "list_documents":
            return "📂 查看知识库文档列表"
        return "🛠 执行工具中..."

    async def _execute_tool(
        self,
        name: str,
        args: dict,
        registry: dict[int, dict],
        counter: list[int],
    ) -> str:
        """执行一次工具调用，返回要喂给模型的文本结果。"""
        if name == "search_knowledge_base":
            query = str(args.get("query") or "").strip()
            if not query:
                return "错误：query 不能为空。"
            chunks = await self.retriever.retrieve(query)
            if not chunks or chunks[0].score < settings.rerank_score_threshold:
                return (
                    f"未检索到与「{query}」相关的内容。"
                    "可以换个关键词再试一次，或调用 list_documents 看看库里有什么。"
                )
            lines: list[str] = []
            for c in chunks:
                counter[0] += 1  # 全局连续编号，多轮检索也不会撞号
                registry[counter[0]] = {
                    "file_name": c.file_name,
                    "text": c.text,
                    "score": c.score,
                }
                lines.append(
                    f"[{counter[0]}] 来源：《{c.file_name}》 相关度 {c.score:.3f}\n{c.text}"
                )
            return "\n\n".join(lines)

        if name == "list_documents":
            docs = self.pipeline.list_documents()
            if not docs:
                return "知识库目前是空的，还没有上传任何文档。"
            return "知识库中的文档：\n" + "\n".join(
                f"- 《{d['file_name']}》（{d['chunk_count']} 个片段）" for d in docs
            )

        return f"错误：未知工具 {name}。"

    # ------------------------------------------------------------------
    # 降级路径：固定流水线 RAG（模型不支持工具调用时的保底）
    # ------------------------------------------------------------------
    async def _pipeline_fallback(
        self, session_id: str, question: str
    ) -> AsyncGenerator[dict, None]:
        """改写 -> 检索 -> 兜底/流式生成 的确定性流水线。"""
        try:
            history = self.memory.get_history(session_id)

            # ① 问题改写（仅多轮对话时触发）
            rewritten = question
            if history:
                yield {"type": "status", "data": "正在理解你的问题..."}
                raw = await llm.chat(
                    prompts.build_rewrite_messages(history, question),
                    temperature=0.1,
                )
                rewritten = raw.strip().strip('"').strip("「」") or question

            # ② 混合检索 + 重排序
            yield {"type": "status", "data": "正在检索知识库..."}
            chunks: list[RetrievedChunk] = await self.retriever.retrieve(rewritten)
            if not chunks or chunks[0].score < settings.rerank_score_threshold:
                yield {"type": "sources", "data": []}
                yield {"type": "token", "data": prompts.FALLBACK_ANSWER}
                self.memory.add_turn(session_id, question, prompts.FALLBACK_ANSWER)
                yield {"type": "done", "data": {"rewritten_query": rewritten}}
                return

            # ③ 推送来源卡片（先于正文，前端可先渲染引用区）
            yield {
                "type": "sources",
                "data": [
                    {
                        "index": i + 1,
                        "file_name": c.file_name,
                        "chunk_id": c.chunk_id,
                        "score": round(c.score, 4),
                        "text": c.text,
                    }
                    for i, c in enumerate(chunks)
                ],
            }

            # ④ 流式生成
            yield {"type": "status", "data": "正在生成回答..."}
            messages = prompts.build_answer_messages(
                rewritten, [c.text for c in chunks]
            )
            answer = ""
            async for token in llm.stream_chat(messages):
                answer += token
                yield {"type": "token", "data": token}
            answer = answer.strip()
            if not answer:
                answer = prompts.FALLBACK_ANSWER
                yield {"type": "token", "data": answer}
            self.memory.add_turn(session_id, question, answer)
            yield {"type": "done", "data": {"rewritten_query": rewritten}}
        except Exception as e:
            logger.exception("固定流水线也失败了")
            yield {"type": "error", "data": f"服务处理失败：{e}"}
