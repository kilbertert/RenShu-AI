"""医疗报告图片/PDF 安全解析与结构化兼容测试。"""

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pymupdf
import pytest
from langchain_core.messages import AIMessage
from pypdf import PdfWriter

from app.src.agent.tcm_report_analyzer import (
    ReportAnalyzer,
    looks_like_medical_report_text,
    parse_medical_report_text,
)
from app.src.agent.tcm_states import LLMConfig
from app.src.response.exception.exceptions import BusinessException
from app.src.schema.attachment_schema import ReportAnalysisPayload
from app.src.service.chat_servcie import ChatService


def _config() -> LLMConfig:
    return LLMConfig(
        provider_name="qwen",
        model_name="qwen3.6-flash",
        api_key_encrypted="plain-test-key",
        base_url="https://example.invalid/compatible-mode/v1",
    )


def _write_text_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Blood Test Report\nWBC 12.0 x10^9/L Reference 3.5-9.5 HIGH\nCRP 18 mg/L Reference 0-10 HIGH",
    )
    document.save(path)
    document.close()


def test_pasted_blood_count_is_parsed_without_llm() -> None:
    text = """血常规检验报告
血红蛋白：92 g/L，参考范围 115-150，偏低
白细胞：6.2×10^9/L，参考范围 3.5-9.5，正常
血小板：235×10^9/L，参考范围 125-350，正常

请解读这份报告，并结合我最近乏力、心悸、头晕的情况给出健康建议。"""

    assert looks_like_medical_report_text(text)
    result = parse_medical_report_text(text)

    assert result is not None
    assert result.report_type == "血常规"
    assert result.extraction_method == "text"
    assert len(result.metrics) == 3
    hemoglobin = next(item for item in result.metrics if item.name == "血红蛋白")
    assert hemoglobin.value == "92"
    assert hemoglobin.unit == "g/L"
    assert hemoglobin.reference_range == "115-150"
    assert hemoglobin.abnormal_flag == "low"
    assert "贫血" in result.summary
    assert any("MCV" in item and "铁蛋白" in item for item in result.medical_attention_advice)
    assert result.urgent_warning is False


def test_plain_symptom_text_is_not_misclassified_as_report() -> None:
    text = "我最近乏力、心悸、头晕，想做一次中医问诊。"

    assert not looks_like_medical_report_text(text)
    assert parse_medical_report_text(text) is None


@pytest.mark.asyncio
async def test_chat_service_routes_pasted_report_into_report_analysis() -> None:
    service = ChatService.__new__(ChatService)
    service._analyze_report_attachments = AsyncMock(return_value=None)
    text = """血常规检验报告
血红蛋白：92 g/L，参考范围 115-150，偏低
白细胞：6.2×10^9/L，参考范围 3.5-9.5，正常"""

    result = await service._analyze_report_input(
        attachments=[],
        query=text,
        model_config={},
    )

    assert result is not None
    assert result["report_type"] == "血常规"
    assert result["metrics"][0]["abnormal_flag"] == "low"


@pytest.mark.asyncio
async def test_text_pdf_uses_local_text_path_and_sets_provenance(tmp_path) -> None:
    path = tmp_path / "report.pdf"
    _write_text_pdf(path)
    attachment_id = uuid4()
    analyzer = ReportAnalyzer()
    analyzer._invoke = AsyncMock(return_value=ReportAnalysisPayload(
        report_type="血液检验",
        summary="白细胞和 CRP 高于参考范围",
    ))

    result = await analyzer.analyze_file(
        path,
        "application/pdf",
        llm_config=_config(),
        vision_enabled=False,
        attachment_id=attachment_id,
    )

    assert result.extraction_method == "text"
    assert result.page_count == 1
    assert result.analyzed_pages == [1]
    assert result.source_attachment_id == attachment_id
    invoke_content = analyzer._invoke.await_args.kwargs["content"]
    assert any("WBC 12.0" in item.get("text", "") for item in invoke_content)


@pytest.mark.asyncio
async def test_scanned_pdf_requires_vision_and_renders_limited_pages(tmp_path) -> None:
    path = tmp_path / "scan.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()

    with pytest.raises(BusinessException, match="支持图片输入"):
        await ReportAnalyzer().analyze_file(
            path,
            "application/pdf",
            llm_config=_config(),
            vision_enabled=False,
        )

    analyzer = ReportAnalyzer()
    analyzer._invoke = AsyncMock(return_value=ReportAnalysisPayload(report_type="扫描报告"))
    result = await analyzer.analyze_file(
        path,
        "application/pdf",
        llm_config=_config(),
        vision_enabled=True,
    )
    assert result.extraction_method == "vision"
    content = analyzer._invoke.await_args.kwargs["content"]
    assert any(item.get("type") == "image_url" for item in content)


@pytest.mark.asyncio
async def test_pdf_with_embedded_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "embedded.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_attachment("payload.txt", b"untrusted")
    with path.open("wb") as stream:
        writer.write(stream)

    with pytest.raises(BusinessException, match="嵌入文件"):
        await ReportAnalyzer().analyze_file(
            path,
            "application/pdf",
            llm_config=_config(),
            vision_enabled=True,
        )


@pytest.mark.asyncio
async def test_pdf_with_javascript_is_rejected(tmp_path) -> None:
    path = tmp_path / "javascript.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_js("app.alert('unsafe')")
    with path.open("wb") as stream:
        writer.write(stream)

    with pytest.raises(BusinessException, match="自动动作|脚本"):
        await ReportAnalyzer().analyze_file(
            path,
            "application/pdf",
            llm_config=_config(),
            vision_enabled=True,
        )


@pytest.mark.asyncio
async def test_encrypted_pdf_is_rejected(tmp_path) -> None:
    path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.encrypt("password")
    with path.open("wb") as stream:
        writer.write(stream)

    with pytest.raises(BusinessException, match="加密"):
        await ReportAnalyzer().analyze_file(
            path,
            "application/pdf",
            llm_config=_config(),
            vision_enabled=True,
        )


class _BoundQwen:
    def __init__(self):
        self.ainvoke = AsyncMock(return_value=AIMessage(content='''{
            "report_type": "血常规",
            "metrics": [{"name": "WBC", "value": 12.0, "abnormal_flag": "偏高"}],
            "key_findings": "白细胞升高；建议结合感染症状复核",
            "summary": "白细胞高于参考范围",
            "medical_attention_advice": null,
            "urgent_warning": false,
            "limitations": null
        }'''))


class _QwenLLM:
    def __init__(self):
        self.bound = _BoundQwen()
        self.bind_kwargs = None

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self.bound


@pytest.mark.asyncio
async def test_qwen_report_analysis_uses_json_mode_and_normalizes_fields() -> None:
    llm = _QwenLLM()
    with patch("app.src.agent.tcm_builder.get_llm", return_value=llm):
        result = await ReportAnalyzer()._invoke(
            content="report text",
            llm_config=_config(),
        )

    assert llm.bind_kwargs == {
        "extra_body": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    assert result.metrics[0].abnormal_flag == "high"
    assert result.metrics[0].value == "12.0"
    assert result.key_findings == ["白细胞升高", "建议结合感染症状复核"]
    assert result.medical_attention_advice == []
