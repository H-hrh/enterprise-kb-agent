"""大模型客户端：封装硅基流动（SiliconFlow）的对话接口。

硅基流动提供与 OpenAI 完全兼容的接口，所以直接用官方 openai SDK，
只需把 base_url 换成硅基流动的地址即可。

提供两个能力：
- chat()        : 一次性拿到完整回答（用于"问题改写"这类内部步骤）
- stream_chat() : 流式逐字产出回答（用于网页聊天，用户不用等整段话生成完）
"""

from collections.abc import AsyncGenerator

import json

from openai import APIError, AsyncOpenAI

from app.config import settings


def get_client() -> AsyncOpenAI:
    """创建 OpenAI 兼容的异步客户端（指向硅基流动）。

    异步客户端可以在等待网络响应时让出 CPU，适合 FastAPI 这种高并发 Web 服务。
    """
    if not settings.has_api_key:
        raise RuntimeError(
            "未配置 SILICONFLOW_API_KEY：请复制 .env.example 为 .env，"
            "并填入你在 https://cloud.siliconflow.cn 创建的 API Key"
        )
    return AsyncOpenAI(
        api_key=settings.siliconflow_api_key,
        base_url=settings.api_base,
        timeout=60,
    )


async def chat(messages: list[dict[str, str]], temperature: float = 0.3) -> str:
    """一次性对话：发送消息列表，返回完整回答文本。

    :param messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]
    :param temperature: 温度值，越低越稳定（改写场景用低温保证确定性）
    """
    client = get_client()
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
    except APIError as e:
        raise RuntimeError(f"大模型调用失败（请检查 Key 是否正确、模型 ID 是否可用）：{e}") from e
    return resp.choices[0].message.content or ""


async def stream_chat(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """流式对话：逐 token（文字片段）产出回答。

    用 async for 遍历，每拿到一小段文字就 yield 给前端，
    实现"打字机效果"。
    """
    client = get_client()
    try:
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            stream=True,  # 开启流式
        )
        async for chunk in stream:
            # 兼容部分网关在流末尾推送的空 choices 块（仅含 usage 统计）
            if not chunk.choices:
                continue
            # 每个 chunk 里 choices[0].delta.content 是新生成的一小段文字
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except APIError as e:
        raise RuntimeError(f"大模型流式调用失败：{e}") from e


async def stream_chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    temperature: float = 0.3,
) -> AsyncGenerator[tuple[str, object], None]:
    """流式对话 + 工具调用支持：ReAct Agent 决策循环的核心。

    大模型在一次响应里有两种可能：
    - 直接输出回答文字 → 逐段 yield ("text", 文字片段)
    - 要求调用工具     → 流结束后一次性 yield ("tools", 解析好的调用列表)

    注意：流式传输中 tool_calls 的函数名和参数 JSON 是分片到达的，
    要按 index 累积拼接，流结束后才能解析出完整的调用请求。
    """
    client = get_client()
    try:
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            stream=True,
            tools=tools,
        )
        tool_acc: dict[int, dict] = {}  # {index: {"id","name","arguments"}}
        async for chunk in stream:
            if not chunk.choices:
                continue  # 流末尾的空 choices 块（仅含 usage 统计）
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield "text", delta.content
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = tool_acc.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] += tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        if tool_acc:
            calls: list[dict] = []
            for idx in sorted(tool_acc):
                slot = tool_acc[idx]
                try:
                    args = json.loads(slot["arguments"]) if slot["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}  # 参数损坏时传空参，让模型在下轮自行纠正
                calls.append(
                    {
                        "id": slot["id"] or f"call_{idx}",
                        "name": slot["name"],
                        "arguments": args,
                    }
                )
            yield "tools", calls
    except APIError as e:
        raise RuntimeError(f"大模型工具调用失败（模型可能不支持 Function Calling）：{e}") from e
