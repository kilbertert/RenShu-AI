"""同一会话只能有一个生成/恢复运行，避免重复消息和 checkpoint 污染。"""

import pytest

from app.src.common.config.redis_config import RedisManager
from app.src.response.exception.exceptions import ConflictException
from app.src.service.chat_servcie import ChatService


@pytest.mark.asyncio
async def test_local_lock_fallback_enforces_owner_and_exclusion() -> None:
    manager = RedisManager()

    first = await manager.acquire_lock("chat_run:user:conversation")
    second = await manager.acquire_lock("chat_run:user:conversation")

    assert first is not None and first.startswith("local:")
    assert second is None
    assert not await manager.release_lock(
        "chat_run:user:conversation",
        "local:not-the-owner",
    )
    assert await manager.release_lock("chat_run:user:conversation", first)
    assert await manager.acquire_lock("chat_run:user:conversation") is not None


@pytest.mark.asyncio
async def test_chat_service_rejects_second_run_with_http_409_semantics(monkeypatch) -> None:
    manager = RedisManager()
    monkeypatch.setattr(
        "app.src.service.chat_servcie.redis_manager",
        manager,
    )
    service = ChatService.__new__(ChatService)

    key, lease = await service._acquire_conversation_run_lock("user-1", "conv-1")
    with pytest.raises(ConflictException) as exc_info:
        await service._acquire_conversation_run_lock("user-1", "conv-1")

    assert exc_info.value.http_status == 409
    assert exc_info.value.error_code == "ConversationRunInProgress"
    assert await manager.release_lock(key, lease)
