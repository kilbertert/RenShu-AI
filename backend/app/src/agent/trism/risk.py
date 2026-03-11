"""
TRiSM - Risk (风险) 模块

识别和缓解 AI 相关风险
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """风险类别"""
    HALLUCINATION = "hallucination"       # 幻觉风险
    BIAS = "bias"                         # 偏见风险
    PRIVACY = "privacy"                   # 隐私风险
    SAFETY = "safety"                     # 安全风险
    RELIABILITY = "reliability"           # 可靠性风险
    PERFORMANCE = "performance"           # 性能风险
    COMPLIANCE = "compliance"           # 合规风险


@dataclass
class RiskMitigation:
    """风险缓解措施"""
    description: str
    priority: int = 0                    # 优先级 (0-10)
    automated: bool = False              # 是否可自动执行
    effectiveness: float = 0.0            # 有效性 (0-1)

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "priority": self.priority,
            "automated": self.automated,
            "effectiveness": self.effectiveness
        }


@dataclass
class RiskAssessment:
    """风险评估"""
    category: RiskCategory
    level: RiskLevel
    description: str
    probability: float = 0.0             # 发生概率 (0-1)
    impact: float = 0.0                  # 影响程度 (0-1)
    mitigations: List[RiskMitigation] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)

    @property
    def risk_score(self) -> float:
        """风险分数 = 概率 × 影响"""
        return self.probability * self.impact

    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "level": self.level.value,
            "description": self.description,
            "probability": self.probability,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "mitigations": [m.to_dict() for m in self.mitigations],
            "detected_at": self.detected_at.isoformat()
        }


class RiskAnalyzer:
    """风险分析器"""

    def __init__(self):
        self.assessments: List[RiskAssessment] = []
        self.thresholds = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.9
        }

    def assess_hallucination_risk(
        self,
        confidence: float,
        sources_count: int,
        verification_status: str
    ) -> RiskAssessment:
        """评估幻觉风险"""
        # 低置信度 + 无来源 = 高风险
        probability = max(0, 1 - confidence) * (0.5 if sources_count > 0 else 1.0)
        impact = 0.8  # 幻觉影响较大

        level = self._calculate_level(probability * impact)

        mitigations = []
        if level != RiskLevel.LOW:
            mitigations.append(RiskMitigation(
                description="添加来源验证",
                priority=8,
                automated=True,
                effectiveness=0.7
            ))
            mitigations.append(RiskMitigation(
                description="降低置信度阈值",
                priority=6,
                automated=True,
                effectiveness=0.5
            ))

        return RiskAssessment(
            category=RiskCategory.HALLUCINATION,
            level=level,
            description=f"幻觉风险: 置信度={confidence:.2f}, 来源数={sources_count}",
            probability=probability,
            impact=impact,
            mitigations=mitigations
        )

    def assess_privacy_risk(
        self,
        sensitive_fields: List[str],
        encryption: bool,
        access_control: bool
    ) -> RiskAssessment:
        """评估隐私风险"""
        # 敏感字段 + 无加密 = 高风险
        has_sensitive = len(sensitive_fields) > 0
        probability = 0.7 if has_sensitive else 0.1
        if has_sensitive and not encryption:
            probability += 0.2
        if has_sensitive and not access_control:
            probability += 0.1

        probability = min(1.0, probability)
        impact = 0.9  # 隐私影响严重

        level = self._calculate_level(probability * impact)

        mitigations = []
        if not encryption:
            mitigations.append(RiskMitigation(
                description="启用数据加密",
                priority=10,
                automated=False,
                effectiveness=0.9
            ))
        if not access_control:
            mitigations.append(RiskMitigation(
                description="实施访问控制",
                priority=9,
                automated=False,
                effectiveness=0.8
            ))

        return RiskAssessment(
            category=RiskCategory.PRIVACY,
            level=level,
            description=f"隐私风险: 敏感字段={len(sensitive_fields)}, 加密={encryption}",
            probability=probability,
            impact=impact,
            mitigations=mitigations
        )

    def assess_safety_risk(
        self,
        medical_advice: bool,
        prescription_generated: bool,
        verification: bool
    ) -> RiskAssessment:
        """评估安全风险 (医疗场景)"""
        probability = 0.1  # 基础风险

        if medical_advice and not verification:
            probability += 0.5
        if prescription_generated and not verification:
            probability += 0.4

        probability = min(1.0, probability)
        impact = 1.0  # 医疗安全影响最严重

        level = self._calculate_level(probability * impact)

        mitigations = []
        if medical_advice or prescription_generated:
            mitigations.append(RiskMitigation(
                description="添加专家审核",
                priority=10,
                automated=False,
                effectiveness=0.9
            ))
            mitigations.append(RiskMitigation(
                description="添加免责声明",
                priority=8,
                automated=True,
                effectiveness=0.6
            ))

        return RiskAssessment(
            category=RiskCategory.SAFETY,
            level=level,
            description=f"安全风险: 医疗建议={medical_advice}, 处方={prescription_generated}",
            probability=probability,
            impact=impact,
            mitigations=mitigations
        )

    def assess_bias_risk(
        self,
        diversity_score: float,
        representation_check: bool
    ) -> RiskAssessment:
        """评估偏见风险"""
        probability = max(0, 1 - diversity_score) * 0.5
        if not representation_check:
            probability += 0.2

        probability = min(1.0, probability)
        impact = 0.7

        level = self._calculate_level(probability * impact)

        return RiskAssessment(
            category=RiskCategory.BIAS,
            level=level,
            description=f"偏见风险: 多样性分数={diversity_score:.2f}",
            probability=probability,
            impact=impact
        )

    def _calculate_level(self, risk_score: float) -> RiskLevel:
        """根据风险分数计算风险级别"""
        if risk_score >= self.thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def add_assessment(self, assessment: RiskAssessment) -> None:
        """添加风险评估"""
        self.assessments.append(assessment)
        logger.info(
            f"[Risk] {assessment.category.value} risk assessed: "
            f"{assessment.level.value} (score: {assessment.risk_score:.2f})"
        )

    def get_risk_summary(self) -> Dict:
        """获取风险摘要"""
        if not self.assessments:
            return {"total": 0, "by_level": {}, "by_category": {}}

        by_level = {level.value: 0 for level in RiskLevel}
        by_category = {cat.value: 0 for cat in RiskCategory}
        total_score = 0

        for assessment in self.assessments:
            by_level[assessment.level.value] += 1
            by_category[assessment.category.value] += 1
            total_score += assessment.risk_score

        return {
            "total": len(self.assessments),
            "by_level": by_level,
            "by_category": by_category,
            "average_risk_score": total_score / len(self.assessments),
            "critical_count": by_level[RiskLevel.CRITICAL.value],
            "high_count": by_level[RiskLevel.HIGH.value]
        }

    def get_top_risks(self, n: int = 5) -> List[RiskAssessment]:
        """获取前 N 个最高风险"""
        return sorted(
            self.assessments,
            key=lambda r: r.risk_score,
            reverse=True
        )[:n]

    def clear_old_assessments(self, hours: int = 24) -> None:
        """清除旧的风险评估"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        self.assessments = [
            a for a in self.assessments
            if a.detected_at > cutoff
        ]
        logger.info(f"[Risk] Cleared assessments older than {hours}h")
