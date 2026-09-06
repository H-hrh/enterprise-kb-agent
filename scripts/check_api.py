"""接口自检脚本：验证 API Key 与三个模型接口（Embedding / Rerank / Chat）是否可用。

用法（在项目根目录执行）：
    .venv\\Scripts\\python.exe scripts\\check_api.py        （Windows）
    .venv/bin/python scripts/check_api.py                   （macOS/Linux）
"""

import asyncio
import sys
from pathlib import Path

# 把项目根目录加入 Python 路径，这样可以直接 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core import embeddings, llm, reranker  # noqa: E402


async def main() -> None:
    print("=" * 60)
    print("企业知识库智能体 - 接口自检")
    print("=" * 60)

    if not settings.has_api_key:
        print("❌ 未配置 API Key！")
        print("   请复制 .env.example 为 .env，填入 SILICONFLOW_API_KEY 后重试")
        return

    print(f"接口地址 : {settings.api_base}")
    print(f"对话模型 : {settings.llm_model}")
    print(f"向量模型 : {settings.embedding_model}")
    print(f"重排模型 : {settings.rerank_model}")
    print("-" * 60)

    # 1. 测试 Embedding
    print("[1/3] 测试向量模型 Embedding ...")
    try:
        vec = await embeddings.embed_query("企业知识库测试")
        print(f"      ✅ 成功，向量维度 = {len(vec)}（bge-m3 应为 1024）")
    except Exception as e:  # noqa: BLE001
        print(f"      ❌ 失败：{e}")
        return

    # 2. 测试 Rerank
    print("[2/3] 测试重排序模型 Rerank ...")
    try:
        results = await reranker.rerank(
            query="年假有多少天",
            documents=[
                "员工每年享有 5 天带薪年假。",
                "公司食堂中午提供自助餐。",
                "新员工试用期为三个月。",
            ],
            top_n=3,
        )
        top_idx, top_score = results[0]
        print(f"      ✅ 成功，最相关片段下标 = {top_idx}（应为 0），分数 = {top_score:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"      ❌ 失败：{e}")
        return

    # 3. 测试 Chat（流式）
    print("[3/3] 测试对话大模型 Chat（流式输出）...")
    try:
        chunks = 0
        async for token in llm.stream_chat(
            [{"role": "user", "content": "用一句话介绍你自己"}]
        ):
            chunks += 1
            print(token, end="", flush=True)
        print()
        print(f"      ✅ 成功，共收到 {chunks} 个流式片段")
    except Exception as e:  # noqa: BLE001
        print(f"\n      ❌ 失败：{e}")
        return

    print("-" * 60)
    print("🎉 全部接口正常，可以启动服务了！")


if __name__ == "__main__":
    asyncio.run(main())
