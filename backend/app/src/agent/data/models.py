"""
TCM Data Models
中医数据模型

定义古籍和医案的数据结构
"""

from typing import Optional
from pydantic import BaseModel, Field


class ClassicRecord(BaseModel):
    """古籍条文记录"""
    book_name: str = Field(description="书名，如《伤寒论》《金匮要略》")
    chapter: str = Field(default="", description="章节名称")
    title: str = Field(default="", description="条文标题或编号")
    content: str = Field(description="条文原文内容")
    interpretation: str = Field(default="", description="条文释义/白话解释")
    related_syndromes: list[str] = Field(
        default_factory=list,
        description="相关证型，如['太阳病', '风寒表证']"
    )
    related_prescriptions: list[str] = Field(
        default_factory=list,
        description="相关方剂，如['桂枝汤', '麻黄汤']"
    )
    related_herbs: list[str] = Field(
        default_factory=list,
        description="相关药材"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="关键词标签"
    )
    content_embedding: Optional[list[float]] = Field(
        default=None,
        description="内容向量嵌入"
    )


class CaseRecord(BaseModel):
    """医案记录"""
    case_id: str = Field(description="医案唯一标识")
    source: str = Field(default="", description="医案来源/出处")
    doctor_name: str = Field(default="", description="医家姓名")
    patient_info: str = Field(default="", description="患者信息（脱敏）")
    chief_complaint: str = Field(description="主诉")
    symptoms: list[str] = Field(
        default_factory=list,
        description="症状列表"
    )
    tongue: str = Field(default="", description="舌象描述")
    pulse: str = Field(default="", description="脉象描述")
    syndrome: str = Field(description="辨证/证型")
    treatment_principle: str = Field(default="", description="治则治法")
    prescription: str = Field(description="处方")
    prescription_herbs: list[dict] = Field(
        default_factory=list,
        description="处方药材及用量，如[{'herb': '桂枝', 'dosage': '9g'}]"
    )
    outcome: str = Field(default="", description="疗效/转归")
    notes: str = Field(default="", description="按语/备注")
    case_embedding: Optional[list[float]] = Field(
        default=None,
        description="医案向量嵌入"
    )


class IngestResult(BaseModel):
    """导入结果"""
    success: bool = Field(description="是否成功")
    total_records: int = Field(default=0, description="总记录数")
    imported_count: int = Field(default=0, description="成功导入数")
    failed_count: int = Field(default=0, description="失败数")
    errors: list[str] = Field(default_factory=list, description="错误信息列表")
    message: str = Field(default="", description="结果消息")


class SyndromeNode(BaseModel):
    """证型节点"""
    name: str = Field(description="证型名称")
    category: str = Field(default="", description="证型分类")
    description: str = Field(default="", description="证型描述")
    symptoms: list[str] = Field(default_factory=list, description="典型症状")
    treatment_principle: str = Field(default="", description="治则")


class PrescriptionNode(BaseModel):
    """方剂节点"""
    name: str = Field(description="方剂名称")
    source: str = Field(default="", description="出处")
    composition: list[dict] = Field(
        default_factory=list,
        description="组成"
    )
    effects: str = Field(default="", description="功效")
    indications: str = Field(default="", description="主治")


class HerbNode(BaseModel):
    """药材节点"""
    name: str = Field(description="药材名称")
    pinyin: str = Field(default="", description="拼音")
    category: str = Field(default="", description="分类")
    nature: str = Field(default="", description="药性")
    flavor: list[str] = Field(default_factory=list, description="药味")
    meridians: list[str] = Field(default_factory=list, description="归经")
    effects: list[str] = Field(default_factory=list, description="功效")
