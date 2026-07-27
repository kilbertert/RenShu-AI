"""结构化输出兼容层测试。"""

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.src.core.language_model.structured_output import (
    invoke_structured_with_json_fallback,
    parse_json_model,
)


class ExampleResult(BaseModel):
    name: str
    count: int


class _NoneStructuredRunnable:
    async def ainvoke(self, _messages):
        return None


class _FallbackLLM:
    def with_structured_output(self, _schema):
        return _NoneStructuredRunnable()

    async def ainvoke(self, _messages):
        return AIMessage(
            content='```json\n{"name": "心脾两虚证", "count": 2}\n```'
        )


class _BoundQwen:
    def __init__(self):
        self.ainvoke = AsyncMock(
            return_value=AIMessage(content='{"name": "气血两虚证", "count": 3}')
        )


class _QwenLLM:
    provider_name = "qwen"
    model_name = "qwen3.6-flash"

    def __init__(self):
        self.bound = _BoundQwen()
        self.bind_kwargs = None

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self.bound

    def with_structured_output(self, _schema):
        raise AssertionError("Qwen 应优先使用兼容网关 JSON mode")


def test_parse_markdown_json_model() -> None:
    result = parse_json_model(
        '```json\n{"name": "脾气虚证", "count": 1}\n```',
        ExampleResult,
    )

    assert result == ExampleResult(name="脾气虚证", count=1)


async def test_none_structured_output_uses_json_fallback() -> None:
    result = await invoke_structured_with_json_fallback(
        _FallbackLLM(),
        ExampleResult,
        [],
    )

    assert result == ExampleResult(name="心脾两虚证", count=2)


async def test_qwen_uses_json_mode_with_schema_instruction() -> None:
    llm = _QwenLLM()
    result = await invoke_structured_with_json_fallback(
        llm,
        ExampleResult,
        [],
    )

    assert result == ExampleResult(name="气血两虚证", count=3)
    assert llm.bind_kwargs == {
        "extra_body": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    messages = llm.bound.ainvoke.await_args.args[0]
    assert "JSON Schema" in messages[0].content
