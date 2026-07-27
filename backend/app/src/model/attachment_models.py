"""私有聊天附件持久化模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, Index, SQLModel


class ChatAttachment(SQLModel, table=True):
    __tablename__ = "chat_attachments"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="accounts.id", description="附件所有者")
    conversation_id: UUID = Field(
        foreign_key="conversations.id",
        description="所属会话",
    )
    message_id: Optional[UUID] = Field(
        default=None,
        foreign_key="messages.id",
        description="发送后绑定的用户消息",
    )
    kind: str = Field(max_length=32, description="附件业务类型")
    original_filename: str = Field(max_length=255)
    storage_key: str = Field(max_length=255, unique=True)
    mime_type: str = Field(max_length=100)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(max_length=64)
    status: str = Field(default="uploaded", max_length=32)
    analysis_result: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    analysis_error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_chat_attachments_user", "user_id"),
        Index("idx_chat_attachments_conversation", "conversation_id"),
        Index("idx_chat_attachments_message", "message_id"),
        Index("idx_chat_attachments_sha256", "sha256"),
        {"extend_existing": True},
    )
