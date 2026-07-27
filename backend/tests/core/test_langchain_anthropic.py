"""Anthropic 兼容网关消息格式适配测试。"""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.src.core.language_model.langchain_anthropic import (
    _normalize_chat_generation_chunk,
    _normalize_chat_result,
)


def test_normalize_anthropic_content_blocks_to_text() -> None:
    result = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content=[
                        {"type": "thinking", "thinking": "内部分析"},
                        {"type": "text", "text": "最终回答"},
                    ]
                )
            )
        ]
    )

    normalized = _normalize_chat_result(result)
    message = normalized.generations[0].message

    assert message.content == "最终回答"
    assert message.additional_kwargs["thinking_content"] == "内部分析"


def test_normalize_keeps_plain_text_unchanged() -> None:
    result = ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="普通回答"))]
    )

    normalized = _normalize_chat_result(result)

    assert normalized.generations[0].message.content == "普通回答"


def test_normalize_stream_chunks_aggregate_to_plain_text() -> None:
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[{"type": "thinking", "thinking": "内部分析"}]
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(content=[{"type": "text", "text": "最终"}])
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(content=[{"type": "text", "text": "回答"}])
        ),
    ]

    normalized = [_normalize_chat_generation_chunk(chunk) for chunk in chunks]
    combined = normalized[0] + normalized[1] + normalized[2]

    assert combined.message.content == "最终回答"
    assert normalized[0].message.additional_kwargs["thinking_content"] == "内部分析"
