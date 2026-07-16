"""
P2 阶段：病例库结构化模型 (case library structuring)

由 diagnose subgraph 在诊断完成后通过 CaseService 写入；
供 case_controller 跨会话查询。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, Index, SQLModel


class Case(SQLModel, table=True):
    """病例主表：一次问诊的元信息与辨证结论。"""
    __tablename__ = "cases"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True, description="病例ID")
    user_id: UUID = Field(foreign_key="accounts.id", index=True, description="患者ID")
    conversation_id: UUID = Field(description="所属会话ID")
    thread_id: Optional[str] = Field(default=None, max_length=100, description="LangGraph thread_id")
    chief_complaint: str = Field(description="主诉（患者自述）")
    complexity_level: Optional[str] = Field(default=None, max_length=20, description="simple/moderate/complex")
    syndrome_id: Optional[str] = Field(default=None, max_length=50, description="Neo4j 证型 ID (如 S001)")
    syndrome_name: Optional[str] = Field(default=None, max_length=100, description="证型中文名")
    syndrome_confidence: Optional[float] = Field(default=None, description="辨证置信度 0-1")
    diagnosis_text: Optional[str] = Field(default=None, description="完整辨证文本")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

    __table_args__ = (
        Index("idx_cases_created_at", "created_at"),
        Index("idx_cases_syndrome_name", "syndrome_name"),
        {"extend_existing": True},
    )


class CaseSymptom(SQLModel, table=True):
    """病例-症状一对多：单次问诊拆出的具体症状条目。"""
    __tablename__ = "case_symptoms"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True, description="记录ID")
    case_id: UUID = Field(foreign_key="cases.id", description="所属病例ID")
    symptom_name: str = Field(max_length=100, description="症状名")
    category: Optional[str] = Field(default=None, max_length=50, description="症状分类")
    severity: Optional[int] = Field(default=None, description="严重程度 1-5")

    __table_args__ = (
        Index("idx_case_symptoms_case_id", "case_id"),
        Index("idx_case_symptoms_name", "symptom_name"),
        {"extend_existing": True},
    )


class CaseSyndrome(SQLModel, table=True):
    """病例-证型多对多：支持联合辨证（主证 + 兼证）。"""
    __tablename__ = "case_syndromes"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True, description="记录ID")
    case_id: UUID = Field(foreign_key="cases.id", description="所属病例ID")
    syndrome_name: str = Field(max_length=100, description="证型名")
    confidence: Optional[float] = Field(default=None, description="置信度 0-1")
    is_primary: bool = Field(default=False, description="是否主证")

    __table_args__ = (
        Index("idx_case_syndromes_case_id", "case_id"),
        {"extend_existing": True},
    )


class CasePrescription(SQLModel, table=True):
    """病例-方剂一对多：辨证后推荐的方剂（含组成/用法/出处）。"""
    __tablename__ = "case_prescriptions"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True, description="记录ID")
    case_id: UUID = Field(foreign_key="cases.id", description="所属病例ID")
    prescription_name: str = Field(max_length=100, description="方剂名")
    composition: Optional[str] = Field(default=None, description="组成（药材 + 剂量）")
    usage: Optional[str] = Field(default=None, description="用法用量")
    source: Optional[str] = Field(default=None, max_length=200, description="方剂出处")
    recommendation_rank: int = Field(default=1, description="推荐顺序 1=首选")

    __table_args__ = (
        Index("idx_case_prescriptions_case_id", "case_id"),
        {"extend_existing": True},
    )


class UserHealthProfile(SQLModel, table=True):
    """用户健康档案聚合行：每用户 1 行，由 cases 触发器自动维护。"""
    __tablename__ = "user_health_profiles"

    user_id: UUID = Field(foreign_key="accounts.id", primary_key=True, description="用户ID")
    constitution: Optional[str] = Field(default=None, max_length=50, description="体质（如气虚/阳虚）")
    chronic_conditions: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="慢病标签",
    )
    allergies: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="过敏标签",
    )
    last_case_at: Optional[datetime] = Field(default=None, description="最近一次问诊时间")
    total_cases: int = Field(default=0, description="累计问诊次数")
    most_common_syndrome: Optional[str] = Field(default=None, max_length=100, description="最高频证型")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
