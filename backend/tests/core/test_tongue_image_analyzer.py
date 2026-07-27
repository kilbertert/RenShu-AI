"""舌像视觉模型结构化兼容测试。"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.src.agent.tcm_image_analyzer import TongueAnalyzer
from app.src.agent.tcm_states import LLMConfig
from app.src.schema.attachment_schema import TongueAnalysisPayload


class _BoundQwen:
    def __init__(self):
        self.ainvoke = AsyncMock(return_value=AIMessage(content="""{
            "is_tongue_image": false,
            "rejection_reason": "图片中没有人体舌部",
            "tongue_color": null,
            "tongue_shape": null,
            "coating_color": null,
            "coating_quality": null,
            "analysis": "不是舌像",
            "syndrome_hints": "脾虚, 气虚",
            "confidence": "80%",
            "image_quality": "无效"
        }"""))


class _QwenLLM:
    def __init__(self):
        self.bound = _BoundQwen()
        self.bind_kwargs = None

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self.bound


@pytest.mark.asyncio
async def test_qwen_tongue_analysis_uses_json_mode_and_normalizes_nulls() -> None:
    llm = _QwenLLM()
    config = LLMConfig(
        provider_name="qwen",
        model_name="qwen3.6-flash",
        api_key_encrypted="plain-test-key",
        base_url="https://example.invalid/compatible-mode/v1",
    )

    with patch("app.src.agent.tcm_builder.get_llm", return_value=llm):
        result = await TongueAnalyzer().analyze_tongue_image(
            "data:image/png;base64,aW1hZ2U=",
            llm_config=config,
        )

    assert llm.bind_kwargs == {
        "extra_body": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    assert result.tongue_color == ""
    assert result.syndrome_hints == ["脾虚", "气虚"]
    assert result.confidence == 0.8
    assert result.analysis == "不是舌像"
    assert result.is_tongue_image is False
    assert result.is_clinically_usable() is False


def test_linguistic_confidence_is_normalized() -> None:
    assert TongueAnalysisPayload(confidence="低").confidence == 0.35
    assert TongueAnalysisPayload(confidence="中等").confidence == 0.6
    assert TongueAnalysisPayload(confidence="高").confidence == 0.85


def test_only_valid_tongue_with_observable_fields_is_clinically_usable() -> None:
    assert TongueAnalysisPayload(
        is_tongue_image=True,
        tongue_color="淡红",
        coating_color="白",
        confidence=0.82,
        image_quality="good",
    ).is_clinically_usable()
    assert not TongueAnalysisPayload(
        is_tongue_image=True,
        confidence=0.9,
        image_quality="good",
    ).is_clinically_usable()
