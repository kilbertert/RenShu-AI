"""
诊断子图数据模型

定义诊断过程中使用的核心数据结构
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    """复杂度级别枚举"""
    SIMPLE = "simple"       # 简单：LLM 直接辨证
    MODERATE = "moderate"   # 中等：RAG + 预定义 Cypher
    COMPLEX = "complex"     # 复杂：DeepSearch Agent


class CollectedDiagnoseInfo(BaseModel):
    """
    已收集的诊断信息 - 基于中医十问

    中医十问歌：
    一问寒热二问汗，三问头身四问便，
    五问饮食六问胸，七聋八渴俱当辨，
    九问旧病十问因，再兼服药参机变。
    """

    # === 主诉 ===
    chief_complaint: Optional[str] = None       # 主诉
    onset_time: Optional[str] = None            # 发病时间
    duration: Optional[str] = None              # 病程

    # === 十问信息 ===
    cold_heat: Optional[str] = None             # 寒热：恶寒、发热、寒热往来
    sweat: Optional[str] = None                 # 汗出：有汗、无汗、盗汗、自汗
    head_body: Optional[str] = None             # 头身：头痛、头晕、身痛、乏力
    urine_stool: Optional[str] = None           # 二便：大便、小便情况
    diet: Optional[str] = None                  # 饮食：食欲、口渴、口苦、口淡
    chest_abdomen: Optional[str] = None         # 胸腹：胸闷、腹胀、心悸
    sleep: Optional[str] = None                 # 睡眠：失眠、多梦、嗜睡
    emotion: Optional[str] = None               # 情志：烦躁、抑郁、焦虑

    # === 望诊（多模态）===
    tongue: Optional[Dict[str, str]] = None     # 舌象：舌色、舌形、苔色、苔质
    complexion: Optional[str] = None            # 面色

    # === 既往史 ===
    medical_history: Optional[List[str]] = None  # 既往病史
    current_medications: Optional[List[str]] = None  # 当前用药
    allergies: Optional[List[str]] = None       # 过敏史

    # === 女性专属 ===
    menstruation: Optional[str] = None          # 月经情况（女性）

    # === 其他症状 ===
    other_symptoms: Optional[List[str]] = None  # 其他症状列表

    def get_missing_categories(self) -> List[str]:
        """获取缺失的必要信息类别"""
        required = {
            "cold_heat": "寒热",
            "sweat": "汗出",
            "head_body": "头身",
            "urine_stool": "二便",
            "diet": "饮食",
            "sleep": "睡眠",
        }
        missing = []
        for field, name in required.items():
            if getattr(self, field) is None:
                missing.append(name)
        return missing

    def is_sufficient(self, min_categories: int = 4) -> bool:
        """判断信息是否足够（至少收集到 N 类信息）"""
        missing = self.get_missing_categories()
        return len(missing) <= (6 - min_categories)

    def get_filled_count(self) -> int:
        """获取已填充的信息类别数量"""
        count = 0
        fields = [
            "chief_complaint", "cold_heat", "sweat", "head_body",
            "urine_stool", "diet", "chest_abdomen", "sleep", "emotion"
        ]
        for field in fields:
            if getattr(self, field) is not None:
                count += 1
        return count

    def get_all_symptoms(self) -> List[str]:
        """获取所有已收集的症状"""
        symptoms = []

        # 从各字段提取症状
        if self.chief_complaint:
            symptoms.append(self.chief_complaint)
        if self.cold_heat:
            symptoms.append(self.cold_heat)
        if self.sweat:
            symptoms.append(self.sweat)
        if self.head_body:
            symptoms.append(self.head_body)
        if self.urine_stool:
            symptoms.append(self.urine_stool)
        if self.diet:
            symptoms.append(self.diet)
        if self.chest_abdomen:
            symptoms.append(self.chest_abdomen)
        if self.sleep:
            symptoms.append(self.sleep)
        if self.emotion:
            symptoms.append(self.emotion)
        if self.other_symptoms:
            symptoms.extend(self.other_symptoms)

        return symptoms

    def to_summary(self) -> str:
        """生成信息摘要"""
        parts = []

        if self.chief_complaint:
            parts.append(f"主诉：{self.chief_complaint}")
        if self.duration:
            parts.append(f"病程：{self.duration}")
        if self.cold_heat:
            parts.append(f"寒热：{self.cold_heat}")
        if self.sweat:
            parts.append(f"汗出：{self.sweat}")
        if self.head_body:
            parts.append(f"头身：{self.head_body}")
        if self.urine_stool:
            parts.append(f"二便：{self.urine_stool}")
        if self.diet:
            parts.append(f"饮食：{self.diet}")
        if self.chest_abdomen:
            parts.append(f"胸腹：{self.chest_abdomen}")
        if self.sleep:
            parts.append(f"睡眠：{self.sleep}")
        if self.emotion:
            parts.append(f"情志：{self.emotion}")
        if self.tongue:
            tongue_str = "、".join([f"{k}:{v}" for k, v in self.tongue.items()])
            parts.append(f"舌象：{tongue_str}")
        if self.complexion:
            parts.append(f"面色：{self.complexion}")
        if self.medical_history:
            parts.append(f"既往史：{', '.join(self.medical_history)}")

        return "\n".join(parts) if parts else "暂无收集到的信息"


class ComplexityAssessment(BaseModel):
    """复杂度评估结果"""
    level: ComplexityLevel
    score: int = Field(ge=0, le=10)             # 0-10 分
    factors: Dict[str, int] = Field(default_factory=dict)  # 各因素得分
    reasoning: str = ""                          # 评估理由

    # === 评估因素说明 ===
    # symptom_count: 症状数量 (1-3: 0分, 4-5: 1分, >5: 2分)
    # organ_systems: 涉及脏腑 (1: 0分, 2: 1分, >2: 2分)
    # duration: 病程 (<2周: 0分, 2周-3月: 1分, >3月: 2分)
    # contradiction: 症状矛盾 (无: 0分, 有: 2分)
    # chronic_conditions: 既往慢性病 (0: 0分, 1-2: 1分, >2: 2分)


class DiagnosisResult(BaseModel):
    """辨证结果"""

    # === 八纲辨证 ===
    ba_gang: Dict[str, str] = Field(default_factory=dict)
    # 示例: {"阴阳": "阳证", "表里": "表证", "寒热": "热证", "虚实": "实证"}

    # === 证型 ===
    syndrome: str = ""                          # 主要证型
    syndrome_secondary: Optional[List[str]] = None  # 兼证

    # === 病因病机 ===
    etiology: Optional[str] = None              # 病因
    pathogenesis: Optional[str] = None          # 病机

    # === 治则治法 ===
    treatment_principle: Optional[str] = None   # 治则
    treatment_method: Optional[str] = None      # 治法

    # === 建议 ===
    recommendations: Optional[List[str]] = None  # 调理建议
    warnings: Optional[List[str]] = None         # 注意事项
    should_seek_doctor: bool = False             # 是否建议就医

    # === 置信度 ===
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # 辨证置信度 0-1
    reasoning_chain: List[str] = Field(default_factory=list)  # 推理链

    # === 参考来源 ===
    references: Optional[List[Dict[str, Any]]] = None  # 参考医案/文献

    def to_display(self) -> str:
        """生成用于显示的格式化文本"""
        parts = []

        # 证型
        parts.append(f"**证型**：{self.syndrome}")
        if self.syndrome_secondary:
            parts.append(f"**兼证**：{', '.join(self.syndrome_secondary)}")

        # 八纲
        if self.ba_gang:
            ba_gang_str = "、".join([f"{k}:{v}" for k, v in self.ba_gang.items()])
            parts.append(f"**八纲**：{ba_gang_str}")

        # 病因病机
        if self.etiology:
            parts.append(f"**病因**：{self.etiology}")
        if self.pathogenesis:
            parts.append(f"**病机**：{self.pathogenesis}")

        # 治则治法
        if self.treatment_principle:
            parts.append(f"**治则**：{self.treatment_principle}")
        if self.treatment_method:
            parts.append(f"**治法**：{self.treatment_method}")

        # 建议
        if self.recommendations:
            parts.append("**调理建议**：")
            for i, rec in enumerate(self.recommendations, 1):
                parts.append(f"  {i}. {rec}")

        # 注意事项
        if self.warnings:
            parts.append("**注意事项**：")
            for warning in self.warnings:
                parts.append(f"  - {warning}")

        return "\n".join(parts)


class CollectionRecord(BaseModel):
    """信息收集记录"""
    round_number: int                           # 轮次
    user_input: str                             # 用户输入
    extracted_info: Dict[str, Any]              # 提取的信息
    follow_up_question: Optional[str] = None    # 追问问题


class TongueAnalysisResult(BaseModel):
    """舌像分析结果"""
    tongue_color: Optional[str] = None          # 舌色：淡白、淡红、红、绛红、紫暗
    tongue_shape: Optional[str] = None          # 舌形：胖大、瘦薄、齿痕、裂纹
    coating_color: Optional[str] = None         # 苔色：白、黄、灰黑
    coating_quality: Optional[str] = None       # 苔质：薄、厚、腻、燥、剥
    analysis: Optional[str] = None              # 综合分析
    confidence: float = 0.0                     # 置信度


class ReportAnalysisResult(BaseModel):
    """检验报告解读结果"""
    report_type: Optional[str] = None           # 报告类型
    abnormal_items: Optional[List[Dict[str, Any]]] = None  # 异常指标
    tcm_interpretation: Optional[str] = None    # 中医解读
    suggestions: Optional[List[str]] = None     # 建议
