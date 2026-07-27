"""Anthropic (Claude) 聊天模型。"""

from typing import Any

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.outputs import ChatGenerationChunk, ChatResult


def _extract_text_content(content: Any) -> str:
    """把 Anthropic 兼容响应的内容统一转换为项目需要的文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
    return "".join(text_parts)


def _extract_thinking_content(content: Any) -> str:
    """提取网关额外返回的 thinking 内容，避免混入最终回答。"""
    if not isinstance(content, list):
        return ""

    thinking_parts = [
        str(block.get("thinking", ""))
        for block in content
        if isinstance(block, dict)
        and (block.get("type") == "thinking" or "thinking" in block)
    ]
    return "\n".join(part for part in thinking_parts if part).strip()


def _normalize_message(message):
    """归一化 AIMessage/AIMessageChunk，同时保留工具调用等其他字段。"""
    if not isinstance(message.content, list):
        return message

    additional_kwargs = dict(message.additional_kwargs)
    thinking_content = _extract_thinking_content(message.content)
    if thinking_content:
        previous = str(additional_kwargs.get("thinking_content", ""))
        additional_kwargs["thinking_content"] = previous + thinking_content

    return message.model_copy(
        update={
            "content": _extract_text_content(message.content),
            "additional_kwargs": additional_kwargs,
        }
    )


def _normalize_chat_result(result: ChatResult) -> ChatResult:
    """把 Anthropic 内容块归一化为项目现有节点期望的纯文本。

    LongCat 等兼容网关会在未显式开启 thinking 时也返回
    ``[{type: thinking}, {type: text}]``。LangChain 因而把
    ``AIMessage.content`` 保留为列表，而项目现有诊断节点普遍按字符串处理。
    工具调用字段保留不变，思考内容写入 ``additional_kwargs``。
    """
    for generation in result.generations:
        generation.message = _normalize_message(generation.message)

    return result


def _normalize_chat_generation_chunk(
    chunk: ChatGenerationChunk,
) -> ChatGenerationChunk:
    """归一化流式块，确保 LangGraph 聚合后的 ``content`` 仍是字符串。"""
    return chunk.model_copy(update={"message": _normalize_message(chunk.message)})


class Chat(ChatAnthropic):
    """Anthropic Claude聊天模型

    支持模型: claude-3-opus, claude-3-sonnet, claude-3-haiku 等
    """

    def _generate(self, *args, **kwargs) -> ChatResult:
        return _normalize_chat_result(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args, **kwargs) -> ChatResult:
        return _normalize_chat_result(await super()._agenerate(*args, **kwargs))

    def _stream(self, *args, **kwargs):
        for chunk in super()._stream(*args, **kwargs):
            yield _normalize_chat_generation_chunk(chunk)

    async def _astream(self, *args, **kwargs):
        async for chunk in super()._astream(*args, **kwargs):
            yield _normalize_chat_generation_chunk(chunk)
