"""
TRiSM - Security (安全) 模块

确保系统安全性
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """安全事件类型"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_LEAK = "data_leak"
    INJECTION_ATTACK = "injection_attack"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ABNORMAL_BEHAVIOR = "abnormal_behavior"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class SecurityEvent:
    """安全事件"""
    event_type: SecurityEventType
    severity: str                           # low, medium, high, critical
    description: str
    source: str = ""                        # 事件来源
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity,
            "description": self.description,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "resolved": self.resolved
        }


@dataclass
class SecurityViolation:
    """安全违规"""
    event: SecurityEvent
    policy_name: str
    remediation: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "event": self.event.to_dict(),
            "policy_name": self.policy_name,
            "remediation": self.remediation
        }


@dataclass
class SecurityPolicy:
    """安全策略"""
    name: str
    description: str
    rules: List[Dict] = field(default_factory=list)
    enabled: bool = True

    def evaluate(self, context: Dict) -> Optional[SecurityViolation]:
        """评估策略，返回违规或None"""
        if not self.enabled:
            return None

        for rule in self.rules:
            if self._check_rule(rule, context):
                return SecurityViolation(
                    event=SecurityEvent(
                        event_type=SecurityEventType.POLICY_VIOLATION,
                        severity=rule.get("severity", "medium"),
                        description=rule.get("message", f"Policy {self.name} violated"),
                        details={"rule": rule, "context": context}
                    ),
                    policy_name=self.name,
                    remediation=rule.get("remediation")
                )
        return None

    def _check_rule(self, rule: Dict, context: Dict) -> bool:
        """检查单条规则"""
        condition = rule.get("condition")
        if callable(condition):
            return condition(context)
        elif isinstance(condition, dict):
            # 简化的键值匹配
            for key, value in condition.items():
                if context.get(key) != value:
                    return False
            return True
        return False


class SecurityMonitor:
    """安全监控器"""

    def __init__(self):
        self.policies: List[SecurityPolicy] = []
        self.events: List[SecurityEvent] = []
        self.violations: List[SecurityViolation] = []
        self.event_handlers: Dict[SecurityEventType, List[Callable]] = {}

    def add_policy(self, policy: SecurityPolicy) -> None:
        """添加安全策略"""
        self.policies.append(policy)
        logger.info(f"[Security] Policy added: {policy.name}")

    def remove_policy(self, name: str) -> None:
        """移除安全策略"""
        self.policies = [p for p in self.policies if p.name != name]
        logger.info(f"[Security] Policy removed: {name}")

    def evaluate_policies(self, context: Dict) -> List[SecurityViolation]:
        """评估所有策略"""
        violations = []
        for policy in self.policies:
            violation = policy.evaluate(context)
            if violation:
                violations.append(violation)
                self.violations.append(violation)
                logger.warning(
                    f"[Security] Policy violation: {policy.name} - "
                    f"{violation.event.description}"
                )
        return violations

    def record_event(
        self,
        event_type: SecurityEventType,
        severity: str,
        description: str,
        source: str = "",
        details: Optional[Dict] = None
    ) -> SecurityEvent:
        """记录安全事件"""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            source=source,
            details=details or {}
        )
        self.events.append(event)

        # 触发事件处理器
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"[Security] Event handler error: {e}")

        # 根据严重级别记录日志
        log_fn = logger.error if severity in ["high", "critical"] else logger.warning
        log_fn(f"[Security] Event: {event_type.value} - {description}")

        return event

    def register_handler(
        self,
        event_type: SecurityEventType,
        handler: Callable[[SecurityEvent], None]
    ) -> None:
        """注册事件处理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def get_security_summary(self) -> Dict:
        """获取安全摘要"""
        if not self.events:
            return {"total_events": 0, "by_type": {}, "by_severity": {}}

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        unresolved_violations = 0

        for event in self.events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_severity[event.severity] = by_severity.get(event.severity, 0) + 1

        for violation in self.violations:
            if not violation.event.resolved:
                unresolved_violations += 1

        return {
            "total_events": len(self.events),
            "by_type": by_type,
            "by_severity": by_severity,
            "total_violations": len(self.violations),
            "unresolved_violations": unresolved_violations,
            "active_policies": len([p for p in self.policies if p.enabled])
        }

    def get_recent_events(self, n: int = 10) -> List[SecurityEvent]:
        """获取最近的N个事件"""
        return sorted(self.events, key=lambda e: e.timestamp, reverse=True)[:n]

    def clear_old_events(self, hours: int = 24) -> None:
        """清除旧事件"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        self.events = [e for e in self.events if e.timestamp > cutoff]
        logger.info(f"[Security] Cleared events older than {hours}h")


# ==================== TCM 专用安全策略 ====================

def create_tcm_security_policies() -> List[SecurityPolicy]:
    """创建 TCM 系统的默认安全策略"""

    policies = []

    # 处方生成必须验证
    policies.append(SecurityPolicy(
        name="prescription_verification",
        description="处方生成必须经过验证",
        rules=[
            {
                "condition": lambda ctx: ctx.get("generating_prescription") and not ctx.get("verified", False),
                "severity": "high",
                "message": "处方生成未经验证",
                "remediation": "启用专家审核或验证流程"
            }
        ]
    ))

    # 敏感数据必须加密
    policies.append(SecurityPolicy(
        name="sensitive_data_encryption",
        description="敏感数据必须加密存储",
        rules=[
            {
                "condition": lambda ctx: ctx.get("has_sensitive_data") and not ctx.get("encrypted", False),
                "severity": "critical",
                "message": "敏感数据未加密",
                "remediation": "启用数据加密"
            }
        ]
    ))

    # 医疗建议需要免责声明
    policies.append(SecurityPolicy(
        name="medical_disclaimer",
        description="医疗建议需要免责声明",
        rules=[
            {
                "condition": lambda ctx: ctx.get("medical_advice") and not ctx.get("has_disclaimer", False),
                "severity": "medium",
                "message": "医疗建议缺少免责声明",
                "remediation": "添加免责声明"
            }
        ]
    ))

    return policies
