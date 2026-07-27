"""诊断结果写入病例库的数据链回归测试。"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.src.agent.components.diagnose.handlers import handle_diagnose_query
from app.src.agent.tcm_states import TCMAgentState, TongueAnalysisResult
from app.src.schema.attachment_schema import ReportAnalysisPayload
from app.src.service.case_service import CaseService
from app.src.agent.components.diagnose.nodes.save_case import save_case_node


@pytest.mark.asyncio
async def test_diagnose_handler_passes_case_identity_to_subgraph() -> None:
    user_id = str(uuid4())
    conversation_id = str(uuid4())
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"answer": "完成", "steps": []})
    state = TCMAgentState(
        messages=[HumanMessage(content="最近头晕乏力")],
        user_id=user_id,
        conversation_id=conversation_id,
        user_profile={},
        tongue_analysis=TongueAnalysisResult(
            tongue_color="淡白",
            tongue_shape="胖大齿痕",
            coating_color="白",
            coating_quality="滑",
            confidence=0.82,
        ),
        report_analysis=ReportAnalysisPayload(
            report_type="血常规",
            summary="白细胞轻度升高",
        ).model_dump(mode="json"),
    )
    config = {"configurable": {"thread_id": "thread-case-1"}}

    with patch(
        "app.src.agent.components.diagnose.handlers.get_diagnose_graph",
        return_value=graph,
    ):
        await handle_diagnose_query(state, config)

    subgraph_input = graph.ainvoke.await_args.args[0]
    assert subgraph_input["user_id"] == user_id
    assert subgraph_input["conversation_id"] == conversation_id
    assert subgraph_input["thread_id"] == "thread-case-1"
    assert subgraph_input["tongue_analysis"]["tongue_color"] == "淡白"
    assert subgraph_input["report_analysis"]["report_type"] == "血常规"


def test_case_service_extracts_current_diagnose_state_shape() -> None:
    state = {
        "messages": [HumanMessage(content="咳嗽、怕冷三天")],
        "collected_info": {
            "chief_complaint": "咳嗽、怕冷",
            "cold_heat": "恶寒；轻微发热",
            "other_symptoms": ["鼻塞"],
        },
        "complexity": {"level": "moderate", "score": 4},
        "diagnosis_result": {"syndrome": "风寒束表证", "confidence": 0.86},
        "answer": "**证型：风寒束表证**\n建议及时线下就医。",
    }

    assert CaseService._extract_chief_complaint(state) == "咳嗽、怕冷"
    assert CaseService._extract_complexity(state) == "moderate"
    assert CaseService._extract_syndrome(state) == {
        "id": None,
        "name": "风寒束表证",
        "confidence": 0.86,
    }
    assert [item["name"] for item in CaseService._extract_symptoms(state)] == [
        "咳嗽",
        "怕冷",
        "恶寒",
        "轻微发热",
        "鼻塞",
    ]


def test_case_service_extracts_syndrome_from_truncated_report() -> None:
    state = {
        "answer": (
            "### 核心辨证结论（必须首先完整输出）\n\n"
            "> **证型：心脾两虚证**\n> 病机：心血不足兼脾气虚弱。\n\n"
            "### 一、四诊摘要\n响应在此处被截断"
        )
    }

    assert CaseService._extract_syndrome(state)["name"] == "心脾两虚证"


def test_case_service_keeps_structured_tongue_analysis() -> None:
    tongue = {"tongue_color": "淡白", "coating_quality": "滑", "confidence": 0.8}

    assert CaseService._extract_tongue_analysis({"tongue_analysis": tongue}) == tongue


def test_case_service_keeps_structured_report_analysis() -> None:
    report = {"report_type": "血常规", "summary": "白细胞轻度升高"}

    assert CaseService._extract_report_analysis({"report_analysis": report}) == report


def test_case_payload_keeps_chronic_disease_and_medication_evidence() -> None:
    payload = CaseService._extract_diagnosis_payload({
        "diagnosis_result": {"syndrome": "未明确"},
        "collected_info": {
            "medical_history": ["高血压", "糖尿病"],
            "current_medications": ["多种药物（具体名称待补充）"],
        },
    })

    assert payload is not None
    assert payload["input_evidence"]["medical_history"] == ["高血压", "糖尿病"]
    assert payload["input_evidence"]["current_medications"] == [
        "多种药物（具体名称待补充）"
    ]


@pytest.mark.asyncio
async def test_failed_or_uncertain_diagnosis_is_not_persisted() -> None:
    with patch(
        "app.src.agent.components.diagnose.nodes.save_case.CaseService.save_case_from_state",
        new=AsyncMock(),
    ) as save_mock:
        await save_case_node({
            "user_id": str(uuid4()),
            "conversation_id": str(uuid4()),
            "error": "结构化辨证生成失败",
            "diagnosis_result": {"syndrome": "未明确", "confidence": 0.0},
        })

    save_mock.assert_not_awaited()
