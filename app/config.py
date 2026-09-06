"""全局配置模块。

所有可调参数（API Key、模型 ID、切分大小、检索数量等）都集中在这里，
统一从项目根目录的 `.env` 文件读取（参考 .env.example）。

为什么用配置文件而不是写死在代码里？
- API Key 属于敏感信息，不能写进代码提交到 Git；
- 模型 ID、检索参数以后可能调整，改 .env 即可，不用动代码。
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件在 app/config.py，上一级就是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 运行时数据目录（上传的原件、Chroma 向量库文件）
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"


class Settings(BaseSettings):
    """应用配置。字段名不区分大小写，自动从环境变量 / .env 读取。"""

    # pydantic-settings 配置：读取项目根目录下的 .env，忽略多余字段
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 硅基流动 API ----------
    siliconflow_api_key: str = ""           # API Key，从 .env 读取
    api_base: str = "https://api.siliconflow.cn/v1"  # 接口地址（OpenAI 兼容）
    llm_model: str = "deepseek-ai/DeepSeek-V3"       # 对话大模型
    embedding_model: str = "BAAI/bge-m3"             # 向量模型（1024 维）
    rerank_model: str = "BAAI/bge-reranker-v2-m3"    # 重排序模型

    # ---------- 文档切分 ----------
    chunk_size: int = 500       # 每个片段的目标字符数
    chunk_overlap: int = 50     # 相邻片段的重叠字符数（保证上下文不断裂）

    # ---------- 检索 ----------
    vector_top_k: int = 20            # 向量检索返回的候选数
    bm25_top_k: int = 20              # BM25 关键词检索返回的候选数
    final_top_k: int = 5              # 重排序后最终使用的片段数
    rerank_score_threshold: float = 0.3   # 低于此分数认为"没有相关内容"
    embedding_batch_size: int = 16    # 向量接口每批最多发送的文本条数

    # ---------- 对话 ----------
    max_history_turns: int = 10       # 每个会话最多保留的对话轮数（1 轮 = 1 问 1 答）
    agent_max_rounds: int = 6         # Agent 决策循环最大轮数（每轮一次模型调用，防止死循环）

    # ---------- 文件上传 ----------
    max_upload_mb: int = 20           # 单文件大小上限（MB）
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".md", ".txt")

    @property
    def max_upload_bytes(self) -> int:
        """文件大小上限（字节）。"""
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        """是否已配置 API Key。"""
        return bool(self.siliconflow_api_key) and not self.siliconflow_api_key.startswith("sk-在这里")


def ensure_dirs() -> None:
    """确保运行时需要的目录存在（服务启动时调用一次）。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# 全局唯一的配置对象，其他模块直接 `from app.config import settings` 导入使用
settings = Settings()

# 启动时校验切分参数的合法性，防止误配置（overlap >= size 会导致切分死循环）
if not 0 <= settings.chunk_overlap < settings.chunk_size:
    raise ValueError(
        f"配置错误：CHUNK_OVERLAP({settings.chunk_overlap}) 必须小于 CHUNK_SIZE({settings.chunk_size})，"
        "请检查 .env 文件"
    )
