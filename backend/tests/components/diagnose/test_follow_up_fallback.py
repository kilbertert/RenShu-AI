"""信息不足时追问决策的确定性降级测试。"""

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from app.src.agent.components.diagnose.nodes.analyze_follow_up import (
    FollowUpDecision,
    NextAction,
    _normalize_resume_value,
    _resume_updates,
    _should_request_tongue,
    analyze_and_follow_up,
)
from app.src.agent.components.diagnose.nodes.collect_info import (
    _fallback_extract_info,
    collect_info,
)
from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo
from app.src.agent.components.diagnose.router import route_collection


@pytest.mark.asyncio
async def test_information_gap_uses_deterministic_follow_up_without_llm():
    state = {
        "messages": [HumanMessage(content="我身体不舒服，请帮我辨证。")],
        "collected_info": {"chief_complaint": "身体不舒服"},
        "follow_up_count": 0,
        "llm_config": None,
    }

    with patch(
        "app.src.agent.components.diagnose.nodes.analyze_follow_up.interrupt",
        return_value="补充回答",
    ):
        result = await analyze_and_follow_up(state)

    assert result["next_action"] == NextAction.ASK_SYMPTOM.value
    assert result["follow_up_question"]
    assert result["messages"][0].content == "补充回答"


@pytest.mark.asyncio
async def test_generic_complaint_collection_does_not_call_llm() -> None:
    state = {
        "messages": [HumanMessage(content="我最近总是不舒服，帮我看看是什么问题。")],
        "collected_info": {},
        "llm_config": None,
    }

    with patch(
        "app.src.agent.components.diagnose.nodes.collect_info.get_llm",
        side_effect=AssertionError("泛化主诉不应调用抽取模型"),
    ):
        result = await collect_info(state)

    assert result["collected_info"]["chief_complaint"]
    assert result["collected_info"]["cold_heat"] is None


@pytest.mark.asyncio
async def test_missing_chief_complaint_uses_deterministic_follow_up():
    state = {
        "messages": [HumanMessage(content="我身体不舒服，请帮我辨证。")],
        "collected_info": {},
        "follow_up_count": 0,
        "llm_config": None,
    }

    with patch(
        "app.src.agent.components.diagnose.nodes.analyze_follow_up.interrupt",
        return_value="补充回答",
    ):
        result = await analyze_and_follow_up(state)

    assert "具体描述" in result["follow_up_question"]
    assert result["messages"][0].content == "补充回答"


def test_tongue_analysis_prevents_duplicate_upload_request() -> None:
    info = CollectedDiagnoseInfo(chief_complaint="腹胀", diet="食欲不振")

    assert not _should_request_tongue(
        info,
        {"tongue_analysis": {"tongue_color": "淡白"}},
    )


def test_textual_tongue_and_pulse_prevent_upload_request() -> None:
    extracted = _fallback_extract_info(
        "心悸多梦、食欲不振、便溏。舌淡、苔薄白，脉细弱。"
    )
    info = CollectedDiagnoseInfo(
        chief_complaint=extracted.chief_complaint,
        diet=extracted.diet,
        tongue=extracted.tongue,
        pulse={"description": extracted.pulse, "source": "text"},
    )

    assert info.tongue
    assert info.tongue["tongue_color"] == "淡"
    assert info.tongue["coating_color"] == "白"
    assert info.pulse == {"description": "细弱", "source": "text"}
    assert not _should_request_tongue(info, {})


def test_negated_tongue_keywords_do_not_trigger_upload_request() -> None:
    info = CollectedDiagnoseInfo(
        chief_complaint="头晕、乏力",
        diet="无口干、无口苦",
        negated_symptoms=["无口干", "无口苦"],
    )

    assert info.get_all_symptoms() == ["头晕", "乏力"]
    assert not _should_request_tongue(info, {})


def test_tongue_request_is_never_repeated_or_forced_after_decline() -> None:
    info = CollectedDiagnoseInfo(
        chief_complaint="食欲不振、腹胀",
        cold_heat="无明显寒热",
        sweat="正常",
        head_body="乏力",
        urine_stool="便溏",
        diet="食欲不振",
        sleep="睡眠正常",
    )

    assert _should_request_tongue(info, {})
    assert not _should_request_tongue(info, {"tongue_request_count": 1})
    assert not _should_request_tongue(info, {"tongue_request_declined": True})


@pytest.mark.asyncio
async def test_sufficient_information_never_interrupts_for_tongue_image() -> None:
    state = {
        "messages": [HumanMessage(content="食欲不振、腹胀、便溏、乏力。")],
        "collected_info": {
            "chief_complaint": "食欲不振、腹胀、便溏、乏力",
            "cold_heat": "无明显寒热",
            "sweat": "正常",
            "head_body": "乏力",
            "urine_stool": "便溏",
            "diet": "食欲不振",
            "sleep": "睡眠正常",
        },
        "follow_up_count": 0,
    }

    with patch(
        "app.src.agent.components.diagnose.nodes.analyze_follow_up.interrupt",
        return_value="暂时无法上传，请按文字继续。",
    ) as interrupt_mock:
        result = await analyze_and_follow_up(state)

    assert result["next_action"] == NextAction.ASSESS_COMPLEXITY.value
    interrupt_mock.assert_not_called()


@pytest.mark.asyncio
async def test_report_aware_follow_up_acknowledges_low_hemoglobin() -> None:
    state = {
        "messages": [HumanMessage(content="血常规提示血红蛋白偏低，最近乏力心悸头晕。")],
        "collected_info": {
            "chief_complaint": "乏力、心悸、头晕",
            "head_body": "乏力、头晕",
            "chest_abdomen": "心悸",
        },
        "report_analysis": {
            "report_type": "血常规",
            "metrics": [{
                "name": "血红蛋白",
                "value": "92",
                "unit": "g/L",
                "reference_range": "115-150",
                "abnormal_flag": "low",
            }],
            "summary": "血红蛋白 92 g/L，低于报告参考范围 115-150",
        },
        "follow_up_count": 0,
    }

    with patch(
        "app.src.agent.components.diagnose.nodes.analyze_follow_up.interrupt",
        return_value="补充回答",
    ):
        result = await analyze_and_follow_up(state)

    assert "血红蛋白 92 g/L" in result["follow_up_question"]
    assert "尽快线下复核" in result["follow_up_question"]
    assert "慢性病" in result["follow_up_question"]


def test_structured_resume_payload_keeps_tongue_analysis() -> None:
    text, tongue, report, llm_config = _normalize_resume_value({
        "text": "",
        "tongue_analysis": {"tongue_color": "淡白"},
    })

    assert text == "我已上传舌像，请结合舌像继续问诊。"
    assert tongue == {"tongue_color": "淡白"}
    assert report is None
    assert llm_config is None


def test_structured_resume_payload_keeps_report_analysis() -> None:
    text, tongue, report, llm_config = _normalize_resume_value({
        "text": "",
        "report_analysis": {"report_type": "血常规"},
    })

    assert text == "我已上传医疗报告，请结合报告继续问诊。"
    assert tongue is None
    assert report == {"report_type": "血常规"}
    assert llm_config is None


def test_resume_payload_replaces_expired_llm_config() -> None:
    updates = _resume_updates(
        {
            "text": "补充回答",
            "llm_config": {
                "provider_name": "anthropic",
                "model_name": "LongCat-2.0",
                "api_key_encrypted": "",
                "base_url": "https://api.longcat.chat/anthropic",
            },
        },
        {"llm_config": None},
    )

    assert updates["llm_config"].base_url == "https://api.longcat.chat/anthropic"


def test_crisis_disclosed_during_resume_exits_diagnosis_flow() -> None:
    updates = _resume_updates(
        "其实我不想活了，已经准备和大家告别。",
        {},
    )

    assert updates["next_action"] == NextAction.INTENT_SWITCH.value
    assert "120" in updates["answer"]
    assert "12356" in updates["answer"]


def test_intent_switch_uses_declared_conditional_edge_key() -> None:
    assert route_collection({"next_action": "intent_switch"}) == "intent_switch"


def test_follow_up_decision_normalizes_qwen_nullable_and_alias_fields() -> None:
    decision = FollowUpDecision.model_validate({
        "action": "继续追问",
        "question": None,
        "reasoning": None,
        "missing_info": "寒热、汗出，睡眠",
    })

    assert decision.action == "ask_symptom"
    assert decision.question == ""
    assert decision.reasoning == ""
    assert decision.missing_info == ["寒热", "汗出", "睡眠"]


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("最近容易疲倦、没精神。", "head_body", "乏力"),
        ("平时不怎么出汗。", "sweat", "少汗"),
        ("胃口一般，饭量还行。", "diet", "食欲一般"),
        ("大便偏稀、不太成形，小便正常。", "urine_stool", "大便偏稀"),
        ("晚上睡不踏实、梦多、容易醒。", "sleep", "睡眠不踏实"),
        ("手脚总是冰凉，偶尔还会发冷。", "cold_heat", "手脚冰凉"),
        ("胸口发闷，偶尔心慌，饭后肚子胀。", "chest_abdomen", "胸闷"),
    ],
)
def test_patient_colloquial_expressions_are_mapped_to_ten_questions(
    text: str,
    field: str,
    expected: str,
) -> None:
    result = _fallback_extract_info(text)

    assert expected in str(getattr(result, field))


def test_colloquial_complete_answer_does_not_repeat_answered_dimensions() -> None:
    extracted = _fallback_extract_info(
        "大约一个月了，容易疲倦，怕冷，不怎么出汗，食欲一般，"
        "睡眠多梦，大便偏稀，小便正常。"
    )
    info = CollectedDiagnoseInfo(
        chief_complaint=extracted.chief_complaint,
        duration=extracted.duration,
        cold_heat=extracted.cold_heat,
        sweat=extracted.sweat,
        head_body=extracted.head_body,
        urine_stool=extracted.urine_stool,
        diet=extracted.diet,
        sleep=extracted.sleep,
        other_symptoms=extracted.other_symptoms,
        negated_symptoms=extracted.negated_symptoms,
    )

    assert "一个月" in info.duration
    assert info.is_sufficient(min_categories=4)
    assert "头身" not in info.get_missing_categories()
    assert "二便" not in info.get_missing_categories()
    assert "饮食" not in info.get_missing_categories()


def test_concrete_medications_allergy_and_menstruation_are_preserved() -> None:
    extracted = _fallback_extract_info(
        "我在吃二甲双胍和氨氯地平，对青霉素过敏。月经周期基本规律，但经量偏少。"
    )

    assert extracted.current_medications == ["二甲双胍", "氨氯地平"]
    assert extracted.allergies == ["青霉素"]
    assert "月经周期基本规律" in extracted.menstruation


def test_combined_normal_answers_and_throat_dryness_are_all_counted() -> None:
    extracted = _fallback_extract_info(
        "近两周乏力、轻微头晕，偶有咽干，体温正常，饮食和睡眠尚可，大小便正常。"
    )

    assert "无明显寒热" in extracted.cold_heat
    assert "饮食正常" in extracted.diet
    assert "咽干" in extracted.diet
    assert "睡眠尚可" in extracted.sleep
    assert "二便正常" in extracted.urine_stool
