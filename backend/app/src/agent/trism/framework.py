"""
TRiSM 框架主模块

整合 Trust、Risk、Security 三个维度
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .trust import TrustEngine, TrustScore, TransparencyReport
from .risk import RiskAnalyzer, RiskAssessment, RiskLevel
from .security import SecurityMonitor, SecurityPolicy, SecurityEvent, SecurityEventType, create_tcm_security_policies


logger = logging.getLogger(__name__)


@dataclass
class TRiSMConfig:
    """TRiSM 配置"""
    # Trust 配置
    track_accuracy: bool = True
    track_reliability: bool = True
    track_transparency: bool = True

    # Risk 配置
    assess_hallucination: bool = True
    assess_privacy: bool = True
    assess_safety: bool = True
    assess_bias: bool = True

    # Security 配置
    enable_policies: bool = True
    log_security_events: bool = True

    # 阈值配置
    high_risk_threshold: float = 0.7
    critical_risk_threshold: float = 0.9
    low_trust_threshold: float = 0.5


@dataclass
class TRiSMReport:
    """TRiSM 报告"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Trust 维度
    trust_score: Optional[TrustScore] = None
    trust_metrics: Optional[Dict] = None

    # Risk 维度
    risk_summary: Optional[Dict] = None
    top_risks: List[RiskAssessment] = field(default_factory=list)

    # Security 维度
    security_summary: Optional[Dict] = None
    recent_events: List[SecurityEvent] = field(default_factory=list)
    unresolved_violations: int = 0

    # 整体评估
    overall_status: str = "unknown"  # safe, warning, critical
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trust": {
                "score": self.trust_score.__dict__ if self.trust_score else None,
                "metrics": self.trust_metrics
            },
            "risk": {
                "summary": self.risk_summary,
                "top_risks": [r.to_dict() for r in self.top_risks]
            },
            "security": {
                "summary": self.security_summary,
                "recent_events": [e.to_dict() for e in self.recent_events],
                "unresolved_violations": self.unresolved_violations
            },
            "overall": {
                "status": self.overall_status,
                "recommendations": self.recommendations
            }
        }


class TRiSMFramework:
    """
    TRiSM 框架

    整合 Trust、Risk、Security 三个维度的安全管理
    """

    def __init__(self, config: Optional[TRiSMConfig] = None):
        self.config = config or TRiSMConfig()

        # 初始化三个组件
        self.trust = TrustEngine()
        self.risk = RiskAnalyzer()
        self.security = SecurityMonitor()

        # 添加 TCM 默认安全策略
        if self.config.enable_policies:
            for policy in create_tcm_security_policies():
                self.security.add_policy(policy)

        logger.info("[TRiSM] Framework initialized")

    def record_agent_interaction(
        self,
        success: bool,
        correct: Optional[bool] = None,
        has_explanation: bool = False,
        explanation: Optional[str] = None
    ) -> None:
        """
        记录 Agent 交互

        Args:
            success: 请求是否成功
            correct: 预测是否正确 (可选)
            has_explanation: 是否有解释
            explanation: 解释内容 (可选)
        """
        # Trust 记录
        self.trust.record_request(success)
        if correct is not None:
            self.trust.record_prediction(correct)
        if has_explanation:
            self.trust.record_decision(True, explanation)

    def record_medical_action(
        self,
        action_type: str,
        verified: bool = False,
        has_disclaimer: bool = False,
        **kwargs
    ) -> None:
        """
        记录医疗行为

        Args:
            action_type: 行为类型 (diagnosis, prescription, advice)
            verified: 是否经过验证
            has_disclaimer: 是否有免责声明
            **kwargs: 其他上下文
        """
        # 风险评估
        if action_type == "prescription":
            assessment = self.risk.assess_safety_risk(
                medical_advice=True,
                prescription_generated=True,
                verification=verified
            )
            self.risk.add_assessment(assessment)

        # 安全策略检查
        context = {
            "generating_prescription": action_type == "prescription",
            "verified": verified,
            "medical_advice": action_type in ["diagnosis", "advice"],
            "has_disclaimer": has_disclaimer,
            **kwargs
        }
        self.security.evaluate_policies(context)

    def assess_current_risks(
        self,
        confidence: float = 0.8,
        sources_count: int = 1,
        sensitive_fields: List[str] = None,
        **kwargs
    ) -> List[RiskAssessment]:
        """
        评估当前风险

        Args:
            confidence: 置信度
            sources_count: 来源数量
            sensitive_fields: 敏感字段列表
            **kwargs: 其他上下文

        Returns:
            风险评估列表
        """
        assessments = []

        if self.config.assess_hallucination:
            assessment = self.risk.assess_hallucination_risk(
                confidence, sources_count, "verified"
            )
            assessments.append(assessment)
            self.risk.add_assessment(assessment)

        if self.config.assess_privacy:
            assessment = self.risk.assess_privacy_risk(
                sensitive_fields or [],
                encryption=kwargs.get("encrypted", False),
                access_control=kwargs.get("access_control", False)
            )
            assessments.append(assessment)
            self.risk.add_assessment(assessment)

        if self.config.assess_safety:
            assessment = self.risk.assess_safety_risk(
                medical_advice=kwargs.get("medical_advice", False),
                prescription_generated=kwargs.get("prescription", False),
                verification=kwargs.get("verified", False)
            )
            assessments.append(assessment)
            self.risk.add_assessment(assessment)

        return assessments

    def generate_report(self) -> TRiSMReport:
        """生成 TRiSM 报告"""
        report = TRiSMReport()

        # Trust 维度
        report.trust_score = self.trust.get_trust_score()
        report.trust_metrics = self.trust.generate_report()

        # Risk 维度
        report.risk_summary = self.risk.get_risk_summary()
        report.top_risks = self.risk.get_top_risks(5)

        # Security 维度
        report.security_summary = self.security.get_security_summary()
        report.recent_events = self.security.get_recent_events(10)
        report.unresolved_violations = report.security_summary.get("unresolved_violations", 0)

        # 整体评估
        report = self._calculate_overall_status(report)

        return report

    def _calculate_overall_status(self, report: TRiSMReport) -> TRiSMReport:
        """计算整体状态"""
        critical_count = 0
        high_count = 0
        recommendations = []

        # 检查 Trust
        if report.trust_score and report.trust_score.overall < self.config.low_trust_threshold:
            recommendations.append("信任分数较低，建议提高系统准确性")
            high_count += 1

        # 检查 Risk
        if report.risk_summary:
            critical_count += report.risk_summary.get("critical_count", 0)
            high_count += report.risk_summary.get("high_count", 0)

        for risk in report.top_risks:
            if risk.level == RiskLevel.CRITICAL:
                recommendations.append(f"关键风险: {risk.description}")
            elif risk.level == RiskLevel.HIGH:
                recommendations.append(f"高风险: {risk.description}")

        # 检查 Security
        if report.unresolved_violations > 0:
            recommendations.append(f"存在 {report.unresolved_violations} 个未解决的安全违规")
            high_count += report.unresolved_violations

        # 确定整体状态
        if critical_count > 0:
            report.overall_status = "critical"
        elif high_count > 2:
            report.overall_status = "warning"
        else:
            report.overall_status = "safe"

        report.recommendations = recommendations

        return report

    def is_safe_to_proceed(self) -> tuple[bool, List[str]]:
        """
        检查是否可以安全继续

        Returns:
            (是否安全, 阻止原因列表)
        """
        blockers = []

        # 检查关键风险
        for assessment in self.risk.assessments:
            if assessment.level == RiskLevel.CRITICAL:
                blockers.append(f"关键风险: {assessment.description}")

        # 检查未解决的安全违规
        for violation in self.security.violations:
            if not violation.event.resolved and violation.event.severity in ["high", "critical"]:
                blockers.append(f"安全违规: {violation.event.description}")

        # 检查信任分数
        if self.config.track_accuracy:
            score = self.trust.get_trust_score()
            if score.overall < self.config.low_trust_threshold:
                blockers.append(f"信任分数过低: {score.overall:.2f}")

        return len(blockers) == 0, blockers

    def reset_metrics(self) -> None:
        """重置所有指标"""
        self.trust.reset_metrics()
        self.risk.assessments.clear()
        self.security.events.clear()
        logger.info("[TRiSM] All metrics reset")


def create_tcm_trism_framework() -> TRiSMFramework:
    """创建 TCM 专用的 TRiSM 框架"""
    config = TRiSMConfig(
        track_accuracy=True,
        assess_hallucination=True,
        assess_privacy=True,
        assess_safety=True,
        high_risk_threshold=0.7,
        critical_risk_threshold=0.9,
        low_trust_threshold=0.6
    )
    return TRiSMFramework(config)
