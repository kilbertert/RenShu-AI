"""病例 API 多模态字段序列化测试。"""

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

from app.src.controller.case_controller import _case_to_dict, _detail_to_dict


def test_case_api_exposes_tongue_analysis() -> None:
    tongue = {"tongue_color": "淡红", "source_attachment_id": str(uuid4())}
    report = {"report_type": "血常规", "source_attachment_id": str(uuid4())}
    case = SimpleNamespace(
        id=uuid4(),
        conversation_id=uuid4(),
        thread_id="thread",
        chief_complaint="乏力",
        complexity_level="moderate",
        syndrome_id=None,
        syndrome_name="脾气虚证",
        syndrome_confidence=0.8,
        diagnosis_text="辨证结果",
        diagnosis_payload={"syndrome": "脾气虚证"},
        tongue_analysis=tongue,
        report_analysis=report,
        created_at=datetime.now(),
    )

    assert _case_to_dict(case)["tongue_analysis"] == tongue
    detail = _detail_to_dict({
        "case": case,
        "symptoms": [],
        "syndromes": [],
        "prescriptions": [],
    })
    assert detail["tongue_analysis"] == tongue
    assert detail["case"]["tongue_analysis"] == tongue
    assert _case_to_dict(case)["report_analysis"] == report
    assert detail["report_analysis"] == report
    assert detail["case"]["report_analysis"] == report
