"""
诊断子图数据模型

定义诊断过程中使用的核心数据结构
"""

from typing import List, Optional, Dict, Any
from enum import Enum
import json
import re

from pydantic import BaseModel, Field, field_validator


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
    pulse: Optional[Dict[str, str]] = None      # 脉象：文字描述及来源
    complexion: Optional[str] = None            # 面色

    # === 既往史 ===
    medical_history: Optional[List[str]] = None  # 既往病史
    current_medications: Optional[List[str]] = None  # 当前用药
    allergies: Optional[List[str]] = None       # 过敏史

    # === 女性专属 ===
    menstruation: Optional[str] = None          # 月经情况（女性）

    # === 其他症状 ===
    other_symptoms: Optional[List[str]] = None  # 其他症状列表
    negated_symptoms: List[str] = Field(
        default_factory=list,
        description="患者明确否认的症状，不作为阳性症状参与检索或落库",
    )

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
        """获取阳性症状，排除否定项和“正常/如常”等非病理观察。"""
        symptoms: List[str] = []

        # 从各字段提取症状
        values = [
            self.chief_complaint,
            self.cold_heat,
            self.sweat,
            self.head_body,
            self.urine_stool,
            self.diet,
            self.chest_abdomen,
            self.sleep,
            self.emotion,
        ]
        if self.other_symptoms:
            values.extend(self.other_symptoms)

        negated = {self._normalize_observation(item) for item in self.negated_symptoms}
        for value in values:
            if not value:
                continue
            for statement in re.split(r"[，,、；;\n]+", str(value)):
                statement = statement.strip()
                if not statement or self._is_negated_statement(statement, negated):
                    continue
                self._append_distinct_symptom(symptoms, statement)
        return symptoms

    @classmethod
    def _append_distinct_symptom(cls, symptoms: List[str], statement: str) -> None:
        """合并“头晕/偶尔头晕”一类包含关系，保留信息更完整的表述。"""
        normalized = cls._normalize_observation(statement)
        for index, existing in enumerate(symptoms):
            existing_normalized = cls._normalize_observation(existing)
            if normalized == existing_normalized:
                return
            if (
                len(normalized) >= 2
                and len(existing_normalized) >= 2
                and (normalized in existing_normalized or existing_normalized in normalized)
            ):
                if len(normalized) > len(existing_normalized):
                    symptoms[index] = statement
                return
        symptoms.append(statement)

    @staticmethod
    def _normalize_observation(value: str) -> str:
        return re.sub(r"[\s，,。；;、]", "", str(value or ""))

    @classmethod
    def _is_negated_statement(cls, statement: str, negated: set[str]) -> bool:
        """区分阳性症状、普通否定和仅表示正常的非病理观察。"""
        normalized = cls._normalize_observation(statement)
        if not normalized:
            return True
        if normalized in {"无汗", "不欲饮", "口不渴"}:
            return False
        if cls._is_non_pathological_statement(normalized):
            return True
        if normalized in negated:
            return True
        if any(
            normalized.endswith(item) or item.endswith(normalized)
            for item in negated
            if item
        ):
            return True
        return bool(
            re.match(
                r"^(?:无|没有|未见|未出现|否认|不伴|并无|不|没)(?!汗)",
                normalized,
            )
        )

    @staticmethod
    def _is_non_pathological_statement(normalized: str) -> bool:
        """识别十问中用于表示“已回答但无异常”的内容。"""
        if normalized in {
            "正常", "无异常", "未见异常", "如常", "尚可", "良好", "平稳",
            "大便", "小便", "二便", "饮食", "饭量", "食欲", "睡眠", "寒热",
            "汗出", "胸腹", "情绪", "脉平", "脉平和",
            "大便正常", "小便正常", "二便正常", "饮食正常", "饭量正常",
            "食欲一般", "胃口一般", "胃口还行", "饭量还行", "睡眠尚可",
            "睡眠正常", "汗出正常", "无明显寒热",
        }:
            return True
        return bool(
            re.search(
                r"(?:均|都|也|基本|大致|较为|尚|还)?"
                r"(?:正常|无异常|未见异常|如常|尚可|良好|平稳|调|可)$",
                normalized,
            )
        )

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
            tongue_str = "、".join(
                f"{k}:{v}" for k, v in self.tongue.items() if k != "source"
            )
            parts.append(f"舌象：{tongue_str}")
        if self.pulse:
            pulse_desc = self.pulse.get("description") or self.pulse.get("pulse")
            if pulse_desc:
                parts.append(f"脉象：{pulse_desc}")
        if self.complexion:
            parts.append(f"面色：{self.complexion}")
        if self.medical_history:
            parts.append(f"既往史：{', '.join(self.medical_history)}")
        if self.negated_symptoms:
            parts.append(f"明确否认：{'、'.join(self.negated_symptoms)}")

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


class PrescriptionRelationEvidence(BaseModel):
    """最终证型与方剂之间可回溯的 Neo4j 关系证据。"""

    source_db: str
    syndrome_id: Optional[str] = None
    syndrome_name: str
    formula_id: Optional[str] = None
    formula_name: str
    relationship_type: str
    relationship_id: str
    relationship_path: List[str] = Field(default_factory=list)


class DiagnosisPrescription(BaseModel):
    """结构化方剂建议；剂量必须由专业医师结合个体情况确认。"""

    name: str
    composition: List[Dict[str, str]] = Field(default_factory=list)
    usage: Optional[str] = None
    source: Optional[str] = None
    rationale: Optional[str] = None
    cautions: List[str] = Field(default_factory=list)
    relation_evidence: Optional[PrescriptionRelationEvidence] = None

    @field_validator("composition", mode="before")
    @classmethod
    def normalize_composition(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            return [{"herb": value, "dosage": ""}]
        if isinstance(value, list):
            return [
                {"herb": item, "dosage": ""} if isinstance(item, str) else item
                for item in value
            ]
        return value

    @field_validator("cautions", mode="before")
    @classmethod
    def normalize_cautions(cls, value):
        if not value:
            return []
        return [value] if isinstance(value, str) else value


class DiagnosisCitation(BaseModel):
    """诊断使用的检索证据。"""

    source_type: str = Field(
        description="graph_path/formula_relation/treatment_pattern/formula/classic/profile"
    )
    title: str
    source: Optional[str] = None
    evidence: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    citation_id: Optional[str] = None
    node_ids: List[str] = Field(default_factory=list)
    relationship_ids: List[str] = Field(default_factory=list)
    relationship_path: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field(default_factory=list)
    symptom_role: Optional[str] = None
    evidence_weight: Optional[float] = Field(default=None, ge=0.0)


class DiagnosisResult(BaseModel):
    """可供患者展示、病例落库和后续审计的统一辨证结果。"""

    # === 八纲辨证 ===
    ba_gang: Dict[str, str] = Field(default_factory=dict)
    # 示例: {"阴阳": "阳证", "表里": "表证", "寒热": "热证", "虚实": "实证"}

    # === 证型 ===
    syndrome: str = "未明确"                    # 主要证型
    syndrome_id: Optional[str] = None
    syndrome_secondary: List[str] = Field(default_factory=list)  # 兼证
    syndrome_evidence: List[str] = Field(default_factory=list)

    # === 病因病机 ===
    etiology: Optional[str] = None              # 病因
    pathogenesis: Optional[str] = None          # 病机

    # === 治则治法 ===
    treatment_principle: Optional[str] = None   # 治则
    treatment_method: Optional[str] = None      # 治法

    # === 建议 ===
    recommendations: List[str] = Field(default_factory=list)  # 调理建议
    warnings: List[str] = Field(default_factory=list)         # 注意事项
    should_seek_doctor: bool = False             # 是否建议就医

    # === 方剂与证据 ===
    prescriptions: List[DiagnosisPrescription] = Field(default_factory=list)
    citations: List[DiagnosisCitation] = Field(default_factory=list)

    # === 面向患者的最终回答 ===
    patient_answer: str = ""

    # === 置信度 ===
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # 辨证置信度 0-1
    reasoning_summary: List[str] = Field(
        default_factory=list,
        description="面向审计的简要依据摘要，不包含隐藏思维链",
    )

    # === 参考来源 ===
    references: List[Dict[str, Any]] = Field(default_factory=list)  # 兼容旧字段

    @field_validator("citations", mode="before")
    @classmethod
    def normalize_citations(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
                return decoded if isinstance(decoded, list) else []
            except json.JSONDecodeError:
                return []
        if not isinstance(value, list):
            return []
        normalized = []
        for item in value:
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence") or item.get("description")
            source = item.get("source")
            normalized.append({
                **item,
                "source_type": item.get("source_type") or item.get("type") or "knowledge",
                "title": (
                    item.get("title")
                    or item.get("name")
                    or evidence
                    or source
                    or str(item.get("id") or "检索证据")
                ),
                "source": source,
                "evidence": evidence,
            })
        return normalized

    @field_validator(
        "syndrome_secondary",
        "syndrome_evidence",
        "recommendations",
        "warnings",
        "reasoning_summary",
        mode="before",
    )
    @classmethod
    def normalize_string_lists(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    decoded = json.loads(stripped)
                    if isinstance(decoded, list):
                        return [str(item).strip() for item in decoded if str(item).strip()]
                except json.JSONDecodeError:
                    # 兼容模型把 JSON 数组塞进字符串且末尾截断的情况，只保留完整引号项。
                    quoted_items = re.findall(r'"((?:[^"\\]|\\.)*)"', stripped)
                    recovered: list[str] = []
                    for item in quoted_items:
                        try:
                            normalized = json.loads(f'"{item}"').strip()
                        except (json.JSONDecodeError, AttributeError):
                            normalized = item.strip()
                        if normalized:
                            recovered.append(normalized)
                    if recovered:
                        return recovered
            parts = [part.strip() for part in re.split(r"\n+", value) if part.strip()]
            return parts or [value]
        return value

    @field_validator("should_seek_doctor", mode="before")
    @classmethod
    def normalize_should_seek_doctor(cls, value):
        """兼容模型把就医建议句子错误放进布尔字段的常见输出。"""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"false", "no", "否", "不需要", "无需", "0"}:
            return False
        if text in {"true", "yes", "是", "需要", "建议", "1"}:
            return True
        # 非空的就医建议说明模型实际表达了需要线下复核。
        return bool(text)

    def to_display(self) -> str:
        """生成用于显示的格式化文本"""
        if self.patient_answer.strip():
            return self.patient_answer.strip()
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

        if self.prescriptions:
            parts.append("**方剂参考**：")
            for prescription in self.prescriptions:
                parts.append(f"  - {prescription.name}")

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
