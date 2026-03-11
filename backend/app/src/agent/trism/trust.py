"""
TRiSM - Trust (信任) 模块

建立用户对 AI 系统的信任
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


@dataclass
class TrustScore:
    """信任分数"""
    overall: float              # 总体信任分数 (0-1)
    accuracy: float             # 准确性信任
    reliability: float          # 可靠性信任
    transparency: float         # 透明度信任
    privacy: float              # 隐私保护信任
    safety: float               # 安全性信任

    def __post_init__(self):
        # 确保所有值在 [0, 1] 范围内
        for name in ["overall", "accuracy", "reliability", "transparency", "privacy", "safety"]:
            value = getattr(self, name)
            setattr(self, name, max(0.0, min(1.0, value)))

    @classmethod
    def from_metrics(cls, metrics: "TrustMetrics") -> "TrustScore":
        """从信任指标计算信任分数"""
        return cls(
            overall=metrics.calculate_overall(),
            accuracy=metrics.accuracy_score,
            reliability=metrics.reliability_score,
            transparency=metrics.transparency_score,
            privacy=metrics.privacy_score,
            safety=metrics.safety_score
        )


@dataclass
class TrustMetrics:
    """信任指标"""
    # 准确性相关
    correct_predictions: int = 0
    total_predictions: int = 0
    accuracy_score: float = 0.0

    # 可靠性相关
    successful_responses: int = 0
    total_requests: int = 0
    reliability_score: float = 0.0

    # 透明度相关
    explained_decisions: int = 0
    total_decisions: int = 0
    transparency_score: float = 0.0

    # 隐私保护相关
    privacy_violations: int = 0
    privacy_compliant: int = 0
    privacy_score: float = 1.0  # 默认满分

    # 安全性相关
    safety_violations: int = 0
    safe_interactions: int = 0
    safety_score: float = 1.0  # 默认满分

    def update_accuracy(self, is_correct: bool) -> None:
        """更新准确性指标"""
        self.total_predictions += 1
        if is_correct:
            self.correct_predictions += 1
        self.accuracy_score = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0

    def update_reliability(self, success: bool) -> None:
        """更新可靠性指标"""
        self.total_requests += 1
        if success:
            self.successful_responses += 1
        self.reliability_score = self.successful_responses / self.total_requests if self.total_requests > 0 else 0.0

    def update_transparency(self, has_explanation: bool) -> None:
        """更新透明度指标"""
        self.total_decisions += 1
        if has_explanation:
            self.explained_decisions += 1
        self.transparency_score = self.explained_decisions / self.total_decisions if self.total_decisions > 0 else 0.0

    def update_privacy(self, violation: bool) -> None:
        """更新隐私保护指标"""
        if violation:
            self.privacy_violations += 1
        else:
            self.privacy_compliant += 1

        total = self.privacy_violations + self.privacy_compliant
        if total > 0:
            self.privacy_score = self.privacy_compliant / total

    def update_safety(self, violation: bool) -> None:
        """更新安全性指标"""
        if violation:
            self.safety_violations += 1
        else:
            self.safe_interactions += 1

        total = self.safety_violations + self.safe_interactions
        if total > 0:
            self.safety_score = self.safe_interactions / total

    def calculate_overall(self) -> float:
        """计算总体信任分数"""
        weights = {
            "accuracy": 0.3,
            "reliability": 0.25,
            "transparency": 0.2,
            "privacy": 0.15,
            "safety": 0.1
        }
        return (
            self.accuracy_score * weights["accuracy"] +
            self.reliability_score * weights["reliability"] +
            self.transparency_score * weights["transparency"] +
            self.privacy_score * weights["privacy"] +
            self.safety_score * weights["safety"]
        )


@dataclass
class TransparencyReport:
    """透明度报告"""
    timestamp: datetime = field(default_factory=datetime.now)
    decision_explanation: str = ""
    data_sources: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    uncertainty_factors: List[str] = field(default_factory=list)
    alternative_considerations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "decision_explanation": self.decision_explanation,
            "data_sources": self.data_sources,
            "confidence_level": self.confidence_level,
            "uncertainty_factors": self.uncertainty_factors,
            "alternative_considerations": self.alternative_considerations
        }


class TrustEngine:
    """信任引擎"""

    def __init__(self):
        self.metrics = TrustMetrics()
        self.reports: List[TransparencyReport] = []

    def record_prediction(self, correct: bool) -> None:
        """记录预测结果"""
        self.metrics.update_accuracy(correct)
        logger.debug(f"[Trust] Prediction recorded: {'correct' if correct else 'incorrect'}")

    def record_request(self, success: bool) -> None:
        """记录请求结果"""
        self.metrics.update_reliability(success)
        logger.debug(f"[Trust] Request recorded: {'success' if success else 'failed'}")

    def record_decision(
        self,
        has_explanation: bool,
        explanation: Optional[str] = None
    ) -> None:
        """记录决策"""
        self.metrics.update_transparency(has_explanation)

        if explanation:
            report = TransparencyReport(decision_explanation=explanation)
            self.reports.append(report)

    def record_privacy_event(self, violation: bool, details: str = "") -> None:
        """记录隐私事件"""
        self.metrics.update_privacy(violation)
        if violation:
            logger.warning(f"[Trust] Privacy violation: {details}")

    def record_safety_event(self, violation: bool, details: str = "") -> None:
        """记录安全事件"""
        self.metrics.update_safety(violation)
        if violation:
            logger.warning(f"[Trust] Safety violation: {details}")

    def get_trust_score(self) -> TrustScore:
        """获取当前信任分数"""
        return TrustScore.from_metrics(self.metrics)

    def get_metrics(self) -> TrustMetrics:
        """获取信任指标"""
        return self.metrics

    def generate_report(self) -> Dict:
        """生成信任报告"""
        score = self.get_trust_score()

        return {
            "trust_score": {
                "overall": score.overall,
                "accuracy": score.accuracy,
                "reliability": score.reliability,
                "transparency": score.transparency,
                "privacy": score.privacy,
                "safety": score.safety
            },
            "metrics": {
                "correct_predictions": self.metrics.correct_predictions,
                "total_predictions": self.metrics.total_predictions,
                "successful_responses": self.metrics.successful_responses,
                "total_requests": self.metrics.total_requests,
                "explained_decisions": self.metrics.explained_decisions,
                "total_decisions": self.metrics.total_decisions,
                "privacy_violations": self.metrics.privacy_violations,
                "safety_violations": self.metrics.safety_violations
            },
            "recent_reports": [r.to_dict() for r in self.reports[-5:]]
        }

    def reset_metrics(self) -> None:
        """重置指标"""
        self.metrics = TrustMetrics()
        self.reports.clear()
        logger.info("[Trust] Metrics reset")
