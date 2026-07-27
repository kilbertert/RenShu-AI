"""意图路由稳定性测试。"""

from unittest.mock import AsyncMock, patch

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from app.src.agent.components.router.router import analyze_and_route_query
from app.src.agent.intent_recognition.intent_classifier import IntentClassifier
from app.src.agent.intent_recognition.router.rule_router import RuleBasedRouter
from app.src.agent.intent_recognition.schemas import IntentType
from app.src.agent.tcm_states import TCMAgentState
from app.src.schema.attachment_schema import AttachmentContext


class _FailingStructuredRunnable:
    async def ainvoke(self, _messages):
        raise ValueError("structured output unavailable")


class _JsonFallbackLLM:
    def with_structured_output(self, _schema):
        return _FailingStructuredRunnable()

    async def ainvoke(self, _messages):
        return AIMessage(
            content="""```json
{
  "primary_intent": "diagnosis",
  "sub_type": "comprehensive",
  "confidence": 0.95,
  "entities": {"symptoms": ["乏力", "头晕"]}
}
```"""
        )


class _NoneStructuredRunnable:
    async def ainvoke(self, _messages):
        return None


class _NoneThenJsonFallbackLLM(_JsonFallbackLLM):
    def with_structured_output(self, _schema):
        return _NoneStructuredRunnable()


def test_explicit_personal_diagnosis_uses_rule_router() -> None:
    query = "我最近一直乏力头晕，请按中医辨证分析。"

    result = RuleBasedRouter().route(query)

    assert result is not None
    assert result.primary_intent == IntentType.DIAGNOSIS
    assert result.route_source == "rule"


def test_explicit_diagnosis_request_does_not_depend_on_first_person_prefix() -> None:
    query = (
        "昨天受凉后明显怕冷、轻微发热、无汗、头痛身痛、鼻塞流清涕，"
        "舌淡红、苔薄白，脉浮紧。请按中医辨证分析。"
    )

    result = RuleBasedRouter().route(query)

    assert result is not None
    assert result.primary_intent == IntentType.DIAGNOSIS
    assert result.confidence >= 0.95


@pytest.mark.parametrize(
    "query",
    [
        "我最近总是不舒服，帮我看看是什么问题。",
        "大约一个月了，容易疲倦，怕冷，不怎么出汗，食欲一般，大便偏稀。",
        "失眠多梦、心悸、乏力已经一个月了，是什么原因？",
        "昨天受凉后发冷、鼻塞、咳嗽、头身酸痛。",
        "我既怕冷又手足心热，白天自汗、夜里盗汗。",
    ],
)
def test_natural_patient_complaints_use_deterministic_diagnosis_route(query: str) -> None:
    result = RuleBasedRouter().route(query)

    assert result is not None
    assert result.primary_intent == IntentType.DIAGNOSIS
    assert result.confidence >= 0.85


@pytest.mark.parametrize(
    "query",
    [
        "我的电脑最近总是不舒服，运行很慢怎么办？什么问题？",
        "手机发热而且系统很卡，应该怎么处理？",
        "这段代码运行不舒服，总是报错。",
    ],
)
def test_pure_non_medical_context_does_not_route_to_diagnosis(query: str) -> None:
    result = RuleBasedRouter().route(query)

    assert result is not None
    assert result.primary_intent == IntentType.GENERAL
    assert result.sub_type == "non_medical_clarify"


@pytest.mark.parametrize(
    "query",
    [
        "电脑坏了让我连续头痛失眠三天。",
        "手机看久了以后头晕恶心，最近越来越明显。",
        "写代码熬夜后心悸、胸闷、睡不着。",
    ],
)
def test_mixed_non_medical_context_keeps_real_health_complaint(query: str) -> None:
    result = RuleBasedRouter().route(query)

    assert result is not None
    assert result.primary_intent == IntentType.DIAGNOSIS


def test_unknown_formula_name_routes_to_grounded_prescription_lookup() -> None:
    result = RuleBasedRouter().route(
        "请介绍量子补气还魂汤的古籍出处、组成和标准剂量。"
    )

    assert result is not None
    assert result.primary_intent == IntentType.PRESCRIPTION
    assert "量子补气还魂汤" in result.entities.prescriptions


def test_eighteen_incompatibilities_routes_without_llm() -> None:
    result = RuleBasedRouter().route("中药十八反和十九畏具体指什么？")

    assert result is not None
    assert result.primary_intent == IntentType.HERB
    assert result.sub_type == "compatibility"


async def test_classifier_falls_back_to_markdown_json() -> None:
    classifier = IntentClassifier(llm=_JsonFallbackLLM())

    result = await classifier.classify("我最近乏力头晕，请帮我辨证。")

    assert result.primary_intent == IntentType.DIAGNOSIS
    assert result.confidence == 0.95
    assert result.entities.symptoms == ["乏力", "头晕"]


async def test_classifier_falls_back_when_structured_output_returns_none() -> None:
    classifier = IntentClassifier(llm=_NoneThenJsonFallbackLLM())

    result = await classifier.classify("请根据本次会话历史总结我刚才的信息。")

    assert result.primary_intent == IntentType.DIAGNOSIS
    assert result.route_source == "llm"


async def test_router_exception_degrades_to_general_without_terminal_error() -> None:
    state = TCMAgentState(
        messages=[HumanMessage(content="请总结上一轮信息")],
        user_id="user-1",
        conversation_id="conversation-1",
    )

    with (
        patch(
            "app.src.agent.components.router.router._create_intent_classifier",
            return_value=object(),
        ),
        patch(
            "app.src.agent.intent_recognition.router.intent_router.IntentRouter.route",
            new=AsyncMock(side_effect=RuntimeError("temporary failure")),
        ),
    ):
        result = await analyze_and_route_query(state)

    assert "error" not in result
    assert result["router"].query_type == "tcm-chat"


async def test_attachment_only_visual_failure_routes_without_llm_classification() -> None:
    state = TCMAgentState(
        messages=[HumanMessage(content="请结合我上传的舌像进行中医问诊分析。")],
        user_id="user-1",
        conversation_id="conversation-1",
        attachments=[AttachmentContext(
            id="77777777-7777-4777-8777-777777777777",
            kind="tongue_image",
            original_filename="舌像.png",
            mime_type="image/png",
            size_bytes=128,
            status="analysis_failed",
            download_url="/api/v1/attachments/example/content",
            analysis_error="当前模型不支持图片输入",
        ).model_dump(mode="json")],
    )

    with patch(
        "app.src.agent.components.router.router._create_intent_classifier",
        side_effect=AssertionError("纯附件失败路径不应调用分类模型"),
    ):
        result = await analyze_and_route_query(state)

    assert result["router"].query_type == "tcm-image"
    assert result["router"].has_image is True


async def test_report_failure_routes_without_claiming_unread_content() -> None:
    state = TCMAgentState(
        messages=[HumanMessage(content="请帮我解读这份体检报告")],
        user_id="user-1",
        conversation_id="conversation-1",
        attachments=[AttachmentContext(
            id="88888888-8888-4888-8888-888888888888",
            kind="medical_report",
            original_filename="体检报告.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            status="analysis_failed",
            download_url="/api/v1/attachments/example/content",
            analysis_error="PDF 包含嵌入文件",
        ).model_dump(mode="json")],
    )

    with patch(
        "app.src.agent.components.router.router._create_intent_classifier",
        side_effect=AssertionError("报告失败路径不应调用分类模型"),
    ):
        result = await analyze_and_route_query(state)

    assert result["router"].query_type == "tcm-image"


async def test_structured_report_routes_to_diagnosis_without_llm_classification() -> None:
    state = TCMAgentState(
        messages=[HumanMessage(content="请结合报告和症状继续分析")],
        user_id="user-1",
        conversation_id="conversation-1",
        report_analysis={"report_type": "血常规", "summary": "白细胞轻度升高"},
    )

    with patch(
        "app.src.agent.components.router.router._create_intent_classifier",
        side_effect=AssertionError("已有报告结构时不应再次分类"),
    ):
        result = await analyze_and_route_query(state)

    assert result["router"].query_type == "tcm-diagnose"
