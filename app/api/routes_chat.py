"""聊天接口：SSE 流式问答。

为什么用 SSE（Server-Sent Events）？
大模型生成回答是一个字一个字"吐"出来的，SSE 是服务器向浏览器单向持续推送
文本流的标准方式，前端就能实现打字机效果，而不是干等几十秒后一次性出现。

格式约定（每行一条消息）：
    data: {"type": "token", "data": "年假"}\n\n
"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest
from app.services import qa_chain

router = APIRouter(prefix="/api/chat", tags=["问答"])


@router.post("/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式问答接口。"""

    async def event_generator():
        # 问答链路 yield 的是事件 dict，这里序列化为 SSE 文本格式
        async for event in qa_chain.run(req.session_id, req.message):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁止 Nginx 缓冲，保证流式实时性
        },
    )
