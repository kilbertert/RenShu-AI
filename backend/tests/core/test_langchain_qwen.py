"""Qwen compatible-mode adapter regression tests."""

from langchain_openai import ChatOpenAI

from app.src.core.language_model.langchain_qwen import Chat
from app.src.core.language_model.llm_provider import get_langchain_llm


def test_qwen_adapter_uses_installed_openai_compatible_client() -> None:
    model = Chat(
        model="qwen3.6-flash",
        api_key="test-key",
        base_url="https://example.invalid/compatible-mode/v1",
    )

    assert isinstance(model, ChatOpenAI)


def test_provider_factory_builds_qwen_without_optional_qwq_dependency() -> None:
    model = get_langchain_llm(
        provider_name="qwen",
        model_name="qwen3.6-flash",
        api_key="test-key",
        base_url="https://example.invalid/compatible-mode/v1",
    )

    assert isinstance(model, Chat)
    assert model.extra_body == {"enable_thinking": False}
