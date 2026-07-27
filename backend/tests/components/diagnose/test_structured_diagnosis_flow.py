"""简单/中等辨证统一结构化输出回归测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.src.agent.components.diagnose.handlers import handle_diagnose_query
from app.src.agent.components.diagnose.models import (
    CollectedDiagnoseInfo,
    DiagnosisResult,
)
from app.src.agent.components.diagnose.nodes.simple.simple_diagnosis import simple_diagnosis
from app.src.agent.components.diagnose.nodes.collect_info import _fallback_extract_info
from app.src.agent.components.diagnose.structured_diagnosis import (
    _remove_formula_mentions_from_patient_answer,
    _sanitize_unsupported_seasonal_inference,
    apply_clinical_safety_bounds,
    generate_structured_diagnosis,
)
from app.src.agent.tcm_states import TCMAgentState
from app.src.service.case_service import CaseService


class _StructuredRunnable:
    def __init__(self, value):
        self.value = value

    async def ainvoke(self, _messages):
        return self.value


class _DiagnosisLLM:
    def __init__(self, structured_value=None, raw_content=""):
        self.structured_value = structured_value
        self.raw_content = raw_content

    def with_config(self, **_kwargs):
        return self

    def with_structured_output(self, _schema):
        return _StructuredRunnable(self.structured_value)

    async def ainvoke(self, _messages):
        return AIMessage(content=self.raw_content)


@pytest.mark.asyncio
async def test_structured_diagnosis_accepts_pydantic_result():
    expected = DiagnosisResult(
        syndrome="脾气虚证",
        confidence=0.88,
        syndrome_evidence=["乏力", "食欲不振"],
        patient_answer="**辨证结论**：脾气虚证。",
    )

    result = await generate_structured_diagnosis(
        _DiagnosisLLM(structured_value=expected),
        [],
    )

    assert result.syndrome == "脾气虚证"
    assert result.patient_answer.startswith("**辨证结论**")


@pytest.mark.asyncio
async def test_malformed_json_falls_back_to_legacy_patient_text():
    llm = _DiagnosisLLM(
        structured_value=None,
        raw_content="证型：心脾两虚证\n推荐方剂：归脾汤\n建议规律作息。",
    )

    result = await generate_structured_diagnosis(llm, [])

    assert result.syndrome == "心脾两虚证"
    assert result.prescriptions[0].name == "归脾汤"
    assert "建议规律作息" in result.patient_answer
    assert result.warnings


@pytest.mark.asyncio
async def test_simple_diagnosis_returns_answer_and_structured_payload():
    structured = DiagnosisResult(
        syndrome="风寒束表证",
        confidence=0.82,
        syndrome_evidence=["恶寒", "鼻塞"],
        patient_answer="**辨证结论**：考虑风寒束表证。",
    )
    state = {
        "collected_info": {
            "chief_complaint": "鼻塞怕冷",
            "cold_heat": "恶寒",
            "head_body": "头痛",
        },
        "user_profile": {},
        "llm_config": None,
    }

    with patch(
        "app.src.agent.components.diagnose.nodes.simple.simple_diagnosis.get_llm",
        return_value=_DiagnosisLLM(structured_value=structured),
    ):
        result = await simple_diagnosis(state)

    assert result["answer"] == structured.patient_answer
    assert result["diagnosis_result"]["syndrome"] == "风寒束表证"


@pytest.mark.asyncio
async def test_parent_handler_propagates_syndrome_and_prescriptions():
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "answer": "患者版回答",
        "diagnosis_result": {
            "syndrome": "脾气虚证",
            "confidence": 0.9,
            "syndrome_evidence": ["乏力"],
            "treatment_principle": "益气健脾",
            "prescriptions": [{
                "name": "四君子汤",
                "composition": [{"herb": "人参", "dosage": ""}],
                "source": "《太平惠民和剂局方》",
                "rationale": "益气健脾",
                "cautions": ["需医师辨证"],
            }],
            "should_seek_doctor": False,
        },
        "steps": [],
    })
    state = TCMAgentState(
        messages=[HumanMessage(content="乏力食少")],
        user_id="user",
        conversation_id="conversation",
    )

    with patch(
        "app.src.agent.components.diagnose.handlers.get_diagnose_graph",
        return_value=graph,
    ):
        result = await handle_diagnose_query(
            state,
            {"configurable": {"thread_id": "thread"}},
        )

    assert result["diagnosis_result"]["syndrome"] == "脾气虚证"
    assert result["syndrome_result"].syndrome_name == "脾气虚证"
    assert result["prescriptions"][0].name == "四君子汤"


def test_case_service_prefers_structured_prescriptions_and_secondary_syndromes():
    state = {
        "diagnosis_result": {
            "syndrome": "脾气虚证",
            "syndrome_secondary": ["湿困脾胃证"],
            "prescriptions": [{
                "name": "四君子汤",
                "composition": [{"herb": "人参", "dosage": ""}],
                "usage": "遵医嘱",
                "source": "《太平惠民和剂局方》",
            }],
        },
        "prescriptions": [{"name": "不应优先读取的旧方"}],
    }

    prescriptions = CaseService._extract_prescriptions(state)

    assert prescriptions[0]["name"] == "四君子汤"
    assert "人参" in prescriptions[0]["composition"]
    assert CaseService._extract_secondary_syndromes(state) == ["湿困脾胃证"]


def test_case_service_normalizes_empty_prescription_composition_to_none():
    state = {
        "diagnosis_result": {
            "syndrome": "心脾两虚证",
            "prescriptions": [{"name": "归脾汤", "composition": []}],
        },
    }

    prescriptions = CaseService._extract_prescriptions(state)

    assert prescriptions == [{"name": "归脾汤", "composition": None}]


def test_case_service_does_not_persist_negated_symptoms_as_positive_findings():
    state = {
        "collected_info": {
            "chief_complaint": "头晕、乏力",
            "head_body": "头晕、乏力",
            "cold_heat": "无发热、无怕冷",
            "chest_abdomen": "不胸痛、不心悸",
            "sweat": "无汗",
            "negated_symptoms": ["无发热", "无怕冷", "无胸痛", "无心悸"],
            "other_symptoms": ["头晕", "无发热", "不胸痛"],
        },
    }

    symptoms = CaseService._extract_symptoms(state)
    names = [item["name"] for item in symptoms]

    assert "头晕" in names
    assert "乏力" in names
    assert "无汗" in names
    assert "无发热" not in names
    assert "无怕冷" not in names
    assert "不胸痛" not in names
    assert "不心悸" not in names


def test_case_service_atomizes_chief_complaint_without_duplicate_findings():
    symptoms = CaseService._extract_symptoms({
        "collected_info": {
            "chief_complaint": "乏力、轻微头晕",
            "head_body": "乏力、轻微头晕",
            "other_symptoms": ["乏力", "轻微头晕", "偶有咽干"],
        },
    })

    assert [item["name"] for item in symptoms] == ["乏力", "轻微头晕", "偶有咽干"]


def test_case_payload_keeps_textual_tongue_pulse_and_negated_evidence():
    payload = CaseService._extract_diagnosis_payload({
        "diagnosis_result": {"syndrome": "心脾两虚证"},
        "collected_info": {
            "tongue": {"tongue_color": "淡", "source": "text"},
            "pulse": {"description": "细弱", "source": "text"},
            "negated_symptoms": ["无发热"],
        },
    })

    assert payload["input_evidence"]["tongue"]["source"] == "text"
    assert payload["input_evidence"]["pulse"]["description"] == "细弱"
    assert payload["input_evidence"]["negated_symptoms"] == ["无发热"]


def test_rule_fallback_preserves_ten_questions_when_llm_json_fails():
    result = _fallback_extract_info(
        "我近四个月乏力头晕心悸气短腰痛耳鸣，明显怕冷，无汗，"
        "饮食正常，大便正常但小便偏多，晚上多梦易醒。"
    )

    assert result.duration == "近四个月"
    assert "怕冷" in result.cold_heat
    assert "无汗" in result.sweat
    assert "乏力" in result.head_body
    assert "小便偏多" in result.urine_stool
    assert "多梦" in result.sleep


def test_rule_fallback_extracts_contradiction_chronic_disease_and_polypharmacy():
    result = _fallback_extract_info(
        "我既怕冷又经常手足心热，白天乏力自汗，夜间又盗汗，口干但不想喝水，"
        "大便有时干、有时稀，胸闷心悸，失眠多梦，病程一年。"
        "我有高血压和糖尿病，目前正在服用多种药物。"
    )

    assert result.duration == "一年"
    assert "怕冷" in result.cold_heat and "手足心热" in result.cold_heat
    assert "自汗" in result.sweat and "盗汗" in result.sweat
    assert "不欲饮" in result.diet
    assert "有时干、有时稀" in result.urine_stool
    assert result.medical_history == ["高血压", "糖尿病"]
    assert result.current_medications == ["多种药物（具体名称待补充）"]


def test_qwen_shaped_structured_diagnosis_is_safely_normalized():
    result = DiagnosisResult.model_validate({
        "syndrome": "肾阳亏虚",
        "syndrome_evidence": "怕冷、腰酸、小便偏多",
        "warnings": "线上辨证仅供参考，请线下复核。",
        "confidence": 0.82,
        "citations": [{
            "source": "med_tcm_diagnostic_axis",
            "description": "肾阳亏虚相关证型候选",
            "score": 0.8,
        }],
        "prescriptions": [{
            "name": "金匮肾气丸",
            "cautions": "必须由医师辨证后使用",
        }],
    })

    assert result.syndrome_evidence == ["怕冷、腰酸、小便偏多"]
    assert result.warnings == ["线上辨证仅供参考，请线下复核。"]
    assert result.citations[0].source_type == "knowledge"
    assert result.citations[0].title == "肾阳亏虚相关证型候选"
    assert result.prescriptions[0].cautions == ["必须由医师辨证后使用"]


def test_qwen_sentence_in_should_seek_doctor_is_normalized_to_true():
    result = DiagnosisResult.model_validate({
        "syndrome": "湿热蕴结证",
        "should_seek_doctor": "建议结合白细胞和CRP升高尽快线下就医复核。",
    })

    assert result.should_seek_doctor is True


def test_truncated_json_string_list_recovers_complete_warning_items():
    result = DiagnosisResult.model_validate({
        "warnings": '["第一条完整警告", "第二条完整警告", "第三条被截断',
    })

    assert result.warnings == ["第一条完整警告", "第二条完整警告"]


def test_sparse_positive_symptoms_cannot_produce_formula_recommendation():
    result = DiagnosisResult(
        syndrome="气血亏虚证",
        confidence=0.88,
        prescriptions=[{"name": "归脾汤"}],
        patient_answer="考虑气血亏虚证。",
    )
    info = CollectedDiagnoseInfo(
        chief_complaint="头晕、乏力",
        cold_heat="无发热、无怕冷",
        negated_symptoms=["无发热", "无怕冷"],
    )

    apply_clinical_safety_bounds(result, info)

    assert result.confidence == 0.55
    assert result.prescriptions == []
    assert result.patient_answer.startswith("**信息边界**")


def test_normal_ten_question_answers_are_not_positive_symptoms():
    info = CollectedDiagnoseInfo(
        chief_complaint="偶尔头晕",
        head_body="头晕",
        urine_stool="大便、小便都正常；二便如常",
        diet="饮食和饭量正常",
        sleep="睡眠也正常",
        sweat="无异常",
    )

    assert info.get_all_symptoms() == ["偶尔头晕"]


def test_normal_answers_cannot_inflate_formula_eligibility():
    result = DiagnosisResult(
        syndrome="气血亏虚证",
        confidence=0.88,
        prescriptions=[{"name": "万亿丸"}],
        patient_answer="考虑气血亏虚证。",
    )
    info = CollectedDiagnoseInfo(
        chief_complaint="偶尔头晕",
        urine_stool="大便、小便都正常",
        diet="饮食正常",
        sleep="睡眠正常",
    )

    apply_clinical_safety_bounds(result, info)

    assert result.confidence == 0.55
    assert result.prescriptions == []


def test_sparse_unknown_diagnosis_has_lower_confidence_and_no_false_exclusion():
    result = DiagnosisResult(
        syndrome="未明确",
        confidence=0.9,
        patient_answer=(
            "偶尔头晕，在排除了高血压、颈椎病等器质性病变的前提下，"
            "可先观察。"
        ),
    )

    apply_clinical_safety_bounds(
        result,
        CollectedDiagnoseInfo(chief_complaint="偶尔头晕"),
    )

    assert result.confidence == 0.35
    assert "排除了" not in result.patient_answer
    assert "现有信息不足" in result.patient_answer
    assert "不能替代面诊、检查和处方" in result.patient_answer


def test_formula_sanitizer_keeps_food_soups_but_hides_actual_formula_names():
    text = (
        "可喝姜汤、米汤等温热食物。方剂如麻黄汤需辨证使用，"
        "也不建议服用桂枝汤自行治疗，可食生姜红糖水、葱白豆豉汤等。"
    )

    sanitized = _remove_formula_mentions_from_patient_answer(
        text,
        formula_names=["麻黄汤", "桂枝汤"],
    )

    assert "姜汤" in sanitized
    assert "米汤" in sanitized
    assert "麻黄汤" not in sanitized
    assert "桂枝汤" not in sanitized
    assert "葱白豆豉汤" not in sanitized


def test_formula_sanitizer_does_not_damage_common_words_ending_with_suffix_chars():
    text = "建议散步或练太极，治则为辛温解表、散寒。"

    sanitized = _remove_formula_mentions_from_patient_answer(text)

    assert "散步" in sanitized
    assert "散寒" in sanitized


def test_season_cannot_create_unsupported_syndrome_claim():
    text = (
        "当前正值小暑/大暑节气，外界湿热较重，但体内感受风寒，"
        "形成外寒内湿或寒包火的复杂局面。请注意休息。"
    )

    sanitized = _sanitize_unsupported_seasonal_inference(text)

    assert "寒包火" not in sanitized
    assert "节气仅作为起居调护参考" in sanitized


def test_patient_answer_is_rendered_from_structured_fields_not_free_text():
    result = DiagnosisResult(
        syndrome="风寒感冒证",
        confidence=0.9,
        syndrome_evidence=["恶寒、无汗、脉浮紧支持风寒束表"],
        treatment_principle="辛温解表",
        recommendations=["注意保暖，避免空调直吹", "可饮葱白豆豉汤"],
        warnings=["高热或呼吸急促时及时就医"],
        patient_answer=(
            "当前正值大暑，形成寒包火复杂局面。建议服用麻黄汤。"
        ),
    )

    apply_clinical_safety_bounds(
        result,
        CollectedDiagnoseInfo(
            chief_complaint="明显怕冷、头痛身痛、鼻塞流清涕",
            sweat="无汗",
            cold_heat="轻微发热",
        ),
    )

    assert result.patient_answer.startswith("**辨证结论**")
    assert "寒包火" not in result.patient_answer
    assert "麻黄汤" not in result.patient_answer
    assert "葱白豆豉汤" not in result.patient_answer
    assert "恶寒、无汗、脉浮紧" in result.patient_answer


def test_complex_safe_degradation_caps_confidence_and_removes_formulas():
    result = DiagnosisResult(
        syndrome="阴阳两虚证",
        confidence=0.9,
        syndrome_evidence=["既怕冷又五心烦热"],
        recommendations=["规律作息"],
        prescriptions=[{"name": "七味广枣丸"}],
        patient_answer="建议使用某方。",
    )
    info = CollectedDiagnoseInfo(
        chief_complaint="既怕冷又五心烦热、心悸失眠、腰痛耳鸣",
        cold_heat="既怕冷又五心烦热",
    )

    apply_clinical_safety_bounds(result, info, complexity_level="complex")

    assert result.confidence == 0.65
    assert result.prescriptions == []
    assert "当前不提供具体方剂" in result.patient_answer
    assert result.should_seek_doctor is True


def test_report_and_polypharmacy_are_explicitly_carried_into_complex_safety_answer():
    result = DiagnosisResult(
        syndrome="阴阳两虚证",
        confidence=0.9,
        syndrome_evidence=[
            "既怕冷又手足心热，结合血红蛋白92g/L提示贫血，符合阴阳两虚表现。"
        ],
        recommendations=["规律作息"],
        prescriptions=[{"name": "某方"}],
    )
    info = CollectedDiagnoseInfo(
        chief_complaint="乏力、心悸、头晕、怕冷、手足心热、失眠、多梦",
        cold_heat="怕冷、手足心热",
        medical_history=["高血压", "糖尿病"],
        current_medications=["多种药物（具体名称待补充）"],
    )
    report = {
        "summary": "血红蛋白 92 g/L，低于报告参考范围 115-150",
        "metrics": [{"name": "血红蛋白", "abnormal_flag": "low"}],
        "medical_attention_advice": [
            "携带完整报告就医。",
            "建议完善 MCV、MCH、RDW 和铁蛋白等检查。",
            "由医生结合病史评估贫血原因。",
            "胸痛、呼吸困难或晕厥时立即急诊。",
        ],
        "urgent_warning": False,
    }

    apply_clinical_safety_bounds(
        result,
        info,
        complexity_level="complex",
        report_analysis=report,
    )

    assert result.prescriptions == []
    assert "血红蛋白 92 g/L" in result.patient_answer
    assert "高血压、糖尿病" in result.patient_answer
    assert "多种药物" in result.patient_answer
    assert "不能自行停药" in result.patient_answer
    assert "MCV" in result.patient_answer
    assert "结合血红蛋白" not in result.patient_answer
    assert result.syndrome_evidence == info.get_all_symptoms()[:8]
    assert result.should_seek_doctor is True
