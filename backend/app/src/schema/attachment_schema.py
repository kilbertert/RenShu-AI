"""聊天附件与舌像分析的数据契约。"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class AttachmentKind(str, Enum):
    GENERIC_IMAGE = "generic_image"
    TONGUE_IMAGE = "tongue_image"
    MEDICAL_REPORT = "medical_report"


class AttachmentStatus(str, Enum):
    UPLOADED = "uploaded"
    ATTACHED = "attached"
    ANALYZED = "analyzed"
    ANALYSIS_FAILED = "analysis_failed"


class ChatAttachmentRef(BaseModel):
    id: UUID


class TongueAnalysisPayload(BaseModel):
    """可进入 LangGraph、病例和舌诊历史的统一结构。"""

    is_tongue_image: bool = Field(
        default=False,
        description="图片中是否存在清晰、可辨认的人体舌部主体",
    )
    rejection_reason: str = ""
    tongue_color: str = ""
    tongue_shape: str = ""
    coating_color: str = ""
    coating_quality: str = ""
    analysis: str = ""
    syndrome_hints: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    image_quality: str = "unknown"
    source_attachment_id: UUID | None = None
    warning: str = "舌象只能作为四诊合参的一部分，不能单独作为诊断或用药依据。"

    @field_validator(
        "tongue_color",
        "tongue_shape",
        "coating_color",
        "coating_quality",
        "analysis",
        "image_quality",
        "rejection_reason",
        mode="before",
    )
    @classmethod
    def normalize_nullable_text(cls, value):
        return "" if value is None else value

    @field_validator("syndrome_hints", mode="before")
    @classmethod
    def normalize_nullable_hints(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [
                item.strip()
                for item in re.split(r"[、,，;；]", value)
                if item.strip()
            ]
        return value

    @field_validator("is_tongue_image", mode="before")
    @classmethod
    def normalize_tongue_flag(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"是", "有", "true", "yes", "有效", "舌像"}:
                return True
            if normalized in {"否", "无", "false", "no", "无效", "非舌像"}:
                return False
        return bool(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_nullable_confidence(cls, value):
        if value is None:
            return 0.0
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {
                "高": 0.85,
                "较高": 0.75,
                "中": 0.6,
                "中等": 0.6,
                "一般": 0.5,
                "低": 0.35,
                "较低": 0.4,
                "high": 0.85,
                "medium": 0.6,
                "low": 0.35,
            }
            if normalized in aliases:
                return aliases[normalized]
            if normalized.endswith("%"):
                return float(normalized[:-1]) / 100
        return value

    def is_clinically_usable(self) -> bool:
        """只有明确识别为舌像且存在可观察舌色/苔色时才进入问诊证据。"""
        invalid_quality = {"无效", "invalid", "not_tongue", "非舌像", "无法判断"}
        return bool(
            self.is_tongue_image
            and self.confidence >= 0.35
            and (self.tongue_color or self.coating_color)
            and self.image_quality.strip().lower() not in invalid_quality
        )


class ReportMetric(BaseModel):
    """报告中可核验的单项指标。"""

    name: str = ""
    value: str = ""
    unit: str = ""
    reference_range: str = ""
    abnormal_flag: Literal[
        "high", "low", "abnormal", "normal", "positive", "negative", "unknown"
    ] = "unknown"
    note: str = ""

    @field_validator(
        "name", "value", "unit", "reference_range", "abnormal_flag", "note",
        mode="before",
    )
    @classmethod
    def normalize_metric_text(cls, value):
        return "" if value is None else str(value)

    @field_validator("abnormal_flag", mode="before")
    @classmethod
    def normalize_abnormal_flag(cls, value):
        normalized = str(value or "unknown").strip().lower()
        aliases = {
            "h": "high",
            "↑": "high",
            "偏高": "high",
            "l": "low",
            "↓": "low",
            "偏低": "low",
            "异常": "abnormal",
            "正常": "normal",
            "+": "positive",
            "阳性": "positive",
            "-": "negative",
            "阴性": "negative",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {"high", "low", "abnormal", "normal", "positive", "negative", "unknown"}
        return normalized if normalized in allowed else "unknown"


class ReportAnalysisPayload(BaseModel):
    """医疗报告图片/PDF 的统一、可审计结构化结果。"""

    report_type: str = ""
    report_date: str = ""
    institution: str = ""
    metrics: list[ReportMetric] = Field(default_factory=list, max_length=100)
    key_findings: list[str] = Field(default_factory=list, max_length=30)
    summary: str = ""
    tcm_supporting_interpretation: str = ""
    medical_attention_advice: list[str] = Field(default_factory=list, max_length=20)
    urgent_warning: bool = False
    limitations: list[str] = Field(default_factory=list, max_length=20)
    extraction_method: Literal["vision", "text", "hybrid"] = "vision"
    page_count: int = Field(default=1, ge=1, le=100)
    analyzed_pages: list[int] = Field(default_factory=list, max_length=10)
    source_attachment_id: UUID | None = None
    warning: str = (
        "报告解读仅供辅助理解，不能替代医生结合病史、查体和完整检查作出的诊断。"
    )

    @field_validator(
        "report_type",
        "report_date",
        "institution",
        "summary",
        "tcm_supporting_interpretation",
        "extraction_method",
        mode="before",
    )
    @classmethod
    def normalize_report_text(cls, value):
        return "" if value is None else str(value)

    @field_validator("key_findings", "medical_attention_advice", "limitations", mode="before")
    @classmethod
    def normalize_report_lists(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[\n；;]", value) if item.strip()]
        return value

    @field_validator("analyzed_pages", mode="before")
    @classmethod
    def normalize_analyzed_pages(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[、,，;；\s]+", value) if item.strip()]
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def normalize_metrics(cls, value):
        if value is None:
            return []
        return value

class AttachmentResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    kind: AttachmentKind
    original_filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    status: AttachmentStatus
    download_url: str
    analysis_result: dict[str, Any] | None = None


class AttachmentContext(BaseModel):
    """传给智能体的安全附件描述，不含文件内容和服务端路径。"""

    id: UUID
    kind: AttachmentKind
    original_filename: str
    mime_type: str
    size_bytes: int
    status: AttachmentStatus
    download_url: str
    analysis_result: dict[str, Any] | None = None
    analysis_error: str | None = None


class AttachmentChatInputMixin(BaseModel):
    attachments: list[ChatAttachmentRef] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_unique_attachments(self):
        ids = [item.id for item in self.attachments]
        if len(ids) != len(set(ids)):
            raise ValueError("附件 ID 不得重复")
        return self
