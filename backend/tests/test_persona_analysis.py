"""画像分析必须复用项目统一 Provider，不能把 Anthropic 地址拼成 OpenAI 路径。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.src.schema.chat_schema import (
    ModelConfiguration,
    PersonaAnalysisPayload,
    PersonaAnalysisRequest,
)
from app.src.service.chat_servcie import ChatService


class _Result:
    def first(self):
        return None


class _Session:
    async def exec(self, _statement):
        return _Result()


@pytest.mark.asyncio
async def test_persona_analysis_uses_shared_langchain_provider() -> None:
    service = ChatService.__new__(ChatService)
    service.conversation_service = SimpleNamespace(session=_Session())
    service.model_service = SimpleNamespace(
        get_client=AsyncMock(side_effect=AssertionError("不得调用 OpenAI chat.completions"))
    )
    service._get_llm_config_for_agent = AsyncMock(return_value={
        "provider_name": "anthropic",
        "model_name": "LongCat-2.0",
        "api_key": "encrypted-test-key",
        "base_url": "https://api.longcat.chat/anthropic",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_tokens": 512,
    })
    request = PersonaAnalysisRequest(
        user_id="ignored-client-user",
        text="最近乏力心悸",
        current_persona=None,
        conversation_id=None,
        model_configuration=ModelConfiguration(
            provider_id="provider-id",
            model_id="model-id",
            model_name="LongCat-2.0",
        ),
    )
    llm = object()

    with (
        patch("app.src.service.chat_servcie.get_current_user_id", return_value="00000000-0000-0000-0000-000000000001"),
        patch("app.src.service.chat_servcie.get_llm", return_value=llm) as get_llm_mock,
        patch(
            "app.src.service.chat_servcie.invoke_structured_with_json_fallback",
            new=AsyncMock(return_value=PersonaAnalysisPayload(
                healthScore=70,
                chiefComplaint="乏力心悸",
                suspectedDiagnosis="需进一步评估",
                recommendedTreatment="继续问诊并线下复核",
            )),
        ) as structured_mock,
    ):
        result = await ChatService.analyze_persona.__wrapped__(service, request)

    assert result["chiefComplaint"] == "乏力心悸"
    assert result["healthScore"] == 70
    get_llm_mock.assert_called_once()
    structured_mock.assert_awaited_once()
    service.model_service.get_client.assert_not_called()
