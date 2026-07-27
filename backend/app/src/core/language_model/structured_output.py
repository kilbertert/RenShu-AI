"""兼容网关的 Pydantic 结构化输出兜底。"""

from __future__ import annotations

import json
from typing import Any, Sequence, TypeVar

from langchain_core.messages import SystemMessage
from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def parse_json_model(content: Any, schema: type[ModelT]) -> ModelT:
    """从纯 JSON 或 Markdown JSON 代码块解析 Pydantic 模型。"""
    text = _content_to_text(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型响应中没有 JSON 对象")
    return schema.model_validate(json.loads(text[start : end + 1]))


async def invoke_structured_with_json_fallback(
    llm: Any,
    schema: type[ModelT],
    messages: Sequence[Any],
) -> ModelT:
    """优先使用 LangChain 结构化输出，失败或返回 None 时改为普通 JSON。"""
    structured_error: Exception | None = None

    provider_name = str(getattr(llm, "provider_name", "") or "").lower()
    model_name = str(
        getattr(llm, "model_name", "")
        or getattr(llm, "model", "")
        or ""
    ).lower()
    is_qwen = (
        provider_name == "qwen"
        or "qwen" in model_name
        or llm.__class__.__module__.endswith("langchain_qwen")
    )
    if is_qwen:
        schema_instruction = SystemMessage(content=(
            "只返回一个符合以下 JSON Schema 的 JSON 对象，不要使用 Markdown：\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        ))
        try:
            response = await llm.bind(
                extra_body={"enable_thinking": False},
                response_format={"type": "json_object"},
            ).ainvoke([schema_instruction, *messages])
            return parse_json_model(response.content, schema)
        except Exception as exc:
            structured_error = exc

    try:
        result = await llm.with_structured_output(schema).ainvoke(messages)
        if result is not None:
            return result if isinstance(result, schema) else schema.model_validate(result)
        structured_error = ValueError("结构化输出返回 None")
    except Exception as exc:
        if structured_error is None:
            structured_error = exc

    try:
        raw_result = await llm.ainvoke(messages)
        return parse_json_model(raw_result.content, schema)
    except Exception as fallback_error:
        raise RuntimeError(
            "结构化输出及普通 JSON 兜底均失败: "
            f"{type(structured_error).__name__} / {type(fallback_error).__name__}"
        ) from fallback_error
