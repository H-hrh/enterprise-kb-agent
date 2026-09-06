"""请求 / 响应数据模型。

pydantic 模型会自动完成：
- 请求体 JSON 校验（字段缺失、类型错误会直接返回 422 错误）；
- 响应数据序列化（对象 → JSON）。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。"""

    session_id: str = Field(default="default", max_length=64, description="会话 ID，前端生成")
    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")


class DocumentInfo(BaseModel):
    """文档信息。"""

    doc_id: str
    file_name: str
    chunk_count: int
    created_at: int


class SourceItem(BaseModel):
    """引用来源片段。"""

    index: int = Field(..., description="引用编号，对应答案中的 [n]")
    file_name: str
    chunk_id: str
    score: float
    text: str


class HealthResponse(BaseModel):
    """健康检查返回。"""

    status: str
    api_key_configured: bool
    document_count: int
    chunk_count: int
