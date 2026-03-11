"""
LLMbda 信息流安全分析

实现基于格理论的信息流控制
确保敏感数据不会泄露到不安全的输出
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """安全级别 (基于格理论)"""
    PUBLIC = 0      # 公开数据
    INTERNAL = 1    # 内部数据
    CONFIDENTIAL = 2  # 机密数据
    SECRET = 3      # 秘密数据
    TOP_SECRET = 4  # 绝密数据

    def can_flow_to(self, other: "SecurityLevel") -> bool:
        """检查是否可以流向其他级别 (不升密原则)"""
        return self.value <= other.value

    def join(self, other: "SecurityLevel") -> "SecurityLevel":
        """格的上界操作 (max)"""
        return SecurityLevel(max(self.value, other.value))

    @staticmethod
    def from_str(s: str) -> "SecurityLevel":
        """从字符串解析"""
        mapping = {
            "public": SecurityLevel.PUBLIC,
            "internal": SecurityLevel.INTERNAL,
            "confidential": SecurityLevel.CONFIDENTIAL,
            "secret": SecurityLevel.SECRET,
            "top_secret": SecurityLevel.TOP_SECRET,
        }
        return mapping.get(s.lower(), SecurityLevel.PUBLIC)


@dataclass
class SecurityLabel:
    """安全标签"""
    level: SecurityLevel
    tags: Set[str] = field(default_factory=set)  # 额外的安全标记

    def can_flow_to(self, other: "SecurityLabel") -> bool:
        """检查是否可以流向另一个标签"""
        if not self.level.can_flow_to(other.level):
            return False

        # 检查标签冲突
        # 如果目标标签禁止某些标签，需要检查
        return True

    def join(self, other: "SecurityLabel") -> "SecurityLabel":
        """合并标签"""
        return SecurityLabel(
            level=self.level.join(other.level),
            tags=self.tags | other.tags
        )


@dataclass
class FlowPolicy:
    """信息流策略"""
    # 变量安全级别映射
    variable_labels: Dict[str, SecurityLabel] = field(default_factory=dict)

    # 工具安全级别
    tool_levels: Dict[str, SecurityLevel] = field(default_factory=dict)

    # 默认安全级别
    default_level: SecurityLevel = SecurityLevel.PUBLIC

    def get_variable_label(self, name: str) -> SecurityLabel:
        """获取变量安全标签"""
        if name in self.variable_labels:
            return self.variable_labels[name]
        return SecurityLabel(self.default_level)

    def set_variable_label(
        self,
        name: str,
        level: SecurityLevel,
        tags: Optional[Set[str]] = None
    ) -> None:
        """设置变量安全标签"""
        self.variable_labels[name] = SecurityLabel(level, tags or set())

    def get_tool_level(self, tool_name: str) -> SecurityLevel:
        """获取工具安全级别"""
        return self.tool_levels.get(tool_name, self.default_level)

    def set_tool_level(self, tool_name: str, level: SecurityLevel) -> None:
        """设置工具安全级别"""
        self.tool_levels[tool_name] = level


class InformationFlowAnalyzer:
    """
    信息流分析器

    检测程序中的信息流违规
    """

    def __init__(self, policy: Optional[FlowPolicy] = None):
        self.policy = policy or FlowPolicy()
        self.violations: List[str] = []

    def check_expr(
        self,
        expr: object,
        expected_level: Optional[SecurityLevel] = None
    ) -> SecurityLabel:
        """
        检查表达式的信息流

        Returns:
            表达式的安全标签
        """
        from .core import Variable, ToolCall, Message, Sequence, Application

        if isinstance(expr, Variable):
            return self._check_variable(expr)
        elif isinstance(expr, ToolCall):
            return self._check_tool_call(expr)
        elif isinstance(expr, Message):
            return self._check_message(expr)
        elif isinstance(expr, Sequence):
            return self._check_sequence(expr)
        elif isinstance(expr, Application):
            return self._check_application(expr)
        else:
            # 抽象等表达式不产生信息流
            return SecurityLabel(SecurityLevel.PUBLIC)

    def _check_variable(self, var: object) -> SecurityLabel:
        """检查变量访问"""
        label = self.policy.get_variable_label(var.name)
        logger.debug(f"[FlowAnalysis] Variable {var.name} has level {label.level}")
        return label

    def _check_tool_call(self, call: object) -> SecurityLabel:
        """检查工具调用"""
        # 检查参数的标签
        arg_labels: List[SecurityLabel] = []
        for arg_expr in call.arguments.values():
            arg_label = self.check_expr(arg_expr)
            arg_labels.append(arg_label)

        # 计算输入标签 (join)
        input_label = SecurityLabel(SecurityLevel.PUBLIC)
        for label in arg_labels:
            input_label = input_label.join(label)

        # 获取工具级别
        tool_level = self.policy.get_tool_level(call.tool_name)

        # 检查是否可以调用
        if not input_label.level.can_flow_to(tool_level):
            violation = (
                f"Security violation: Tool '{call.tool_name}' requires level {tool_level}, "
                f"but input has level {input_label.level}"
            )
            self.violations.append(violation)
            logger.warning(f"[FlowAnalysis] {violation}")

        # 输出标签取工具级别和输入标签的较大值
        output_level = input_label.level.join(tool_level)
        return SecurityLabel(output_level, input_label.tags)

    def _check_message(self, msg: object) -> SecurityLabel:
        """检查消息发送"""
        content_label = self.check_expr(msg.content)

        # 消息内容会公开，所以需要检查
        if msg.recipient:
            recipient_level = self._get_recipient_level(msg.recipient)
            if not content_label.level.can_flow_to(recipient_level):
                violation = (
                    f"Security violation: Cannot send {content_label.level} data "
                    f"to {msg.recipient} (level {recipient_level})"
                )
                self.violations.append(violation)
                logger.warning(f"[FlowAnalysis] {violation}")

        return content_label

    def _check_sequence(self, seq: object) -> SecurityLabel:
        """检查序列"""
        self.check_expr(seq.first)
        return self.check_expr(seq.second)

    def _check_application(self, app: object) -> SecurityLabel:
        """检查函数应用"""
        func_label = self.check_expr(app.function)
        arg_label = self.check_expr(app.argument)

        # 简化处理：取两者的较大值
        return func_label.join(arg_label)

    def _get_recipient_level(self, recipient: str) -> SecurityLevel:
        """获取接收者的安全级别"""
        # 可以根据 recipient 名称或配置确定
        # 这里简化处理
        if "external" in recipient.lower() or "public" in recipient.lower():
            return SecurityLevel.PUBLIC
        return SecurityLevel.INTERNAL

    def get_violations(self) -> List[str]:
        """获取所有违规"""
        return self.violations.copy()

    def clear_violations(self) -> None:
        """清除违规记录"""
        self.violations.clear()

    def is_safe(self) -> bool:
        """检查是否有违规"""
        return len(self.violations) == 0


# ==================== TCM 安全策略 ====================

def create_tcm_security_policy() -> FlowPolicy:
    """
    创建 TCM 系统的安全策略

    定义了不同数据和工具的安全级别
    """
    policy = FlowPolicy()

    # 患者敏感信息 - 高安全级别
    policy.set_variable_label("patient_name", SecurityLevel.CONFIDENTIAL)
    policy.set_variable_label("patient_id", SecurityLevel.SECRET)
    policy.set_variable_label("patient_phone", SecurityLevel.CONFIDENTIAL)
    policy.set_variable_label("patient_address", SecurityLevel.CONFIDENTIAL)
    policy.set_variable_label("medical_history", SecurityLevel.CONFIDENTIAL)

    # 诊断结果 - 机密
    policy.set_variable_label("diagnosis_result", SecurityLevel.CONFIDENTIAL)
    policy.set_variable_label("syndrome", SecurityLevel.CONFIDENTIAL)

    # 处方信息 - 机密
    policy.set_variable_label("prescription", SecurityLevel.CONFIDENTIAL)
    policy.set_variable_label("herb_dosage", SecurityLevel.CONFIDENTIAL)

    # 公开的医疗知识 - 公开
    policy.set_variable_label("herb_properties", SecurityLevel.PUBLIC)
    policy.set_variable_label("prescription_info", SecurityLevel.PUBLIC)
    policy.set_variable_label("tcm_knowledge", SecurityLevel.PUBLIC)

    # 工具安全级别
    policy.set_tool_level("database_write", SecurityLevel.CONFIDENTIAL)
    policy.set_tool_level("database_read", SecurityLevel.INTERNAL)
    policy.set_tool_level("external_api", SecurityLevel.PUBLIC)
    policy.set_tool_level("log_write", SecurityLevel.INTERNAL)

    return policy
