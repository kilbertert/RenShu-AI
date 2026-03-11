"""
消息优先级分配系统

基于上下文工程最佳实践，为消息分配优先级，
用于在 Token 超限时决定哪些消息应该被保留或删除。

优先级规则：
1. CRITICAL (4): 系统消息，绝不删除
2. HIGH (3): 工具消息、边缘位置消息（首尾）
3. NORMAL (2): 默认优先级
4. LOW (1): 短消息且无问号，优先删除
"""

from enum import Enum
from typing import List, Dict, Any, Union
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


class MessagePriority(Enum):
    """消息优先级枚举"""
    CRITICAL = 4    # 绝不删除：系统消息
    HIGH = 3        # 高优先保留：工具消息、边缘位置
    NORMAL = 2      # 默认优先级
    LOW = 1         # 优先删除：短消息且无问号


@dataclass
class PrioritizedMessage:
    """带优先级的消息"""
    message: Any  # BaseMessage or Dict
    priority: MessagePriority
    position: int
    estimated_tokens: int


class MessagePriorityAssigner:
    """
    消息优先级分配器

    根据消息类型、位置和内容特征分配优先级。
    """

    # 阈值配置
    LONG_MESSAGE_THRESHOLD = 800    # tokens - 长消息阈值
    SHORT_MESSAGE_THRESHOLD = 20    # tokens - 短消息阈值
    EDGE_POSITION_RATIO = 0.1       # 前后 10% 为边缘位置

    # 中医关键词（表示重要信息）
    TCM_KEYWORDS = [
        "主诉", "症状", "舌象", "脉象", "辨证", "证型",
        "方剂", "处方", "用药", "诊断", "治则", "治法",
    ]

    def __init__(
        self,
        long_threshold: int = 800,
        short_threshold: int = 20,
        edge_ratio: float = 0.1,
    ):
        """
        初始化优先级分配器

        Args:
            long_threshold: 长消息 token 阈值
            short_threshold: 短消息 token 阈值
            edge_ratio: 边缘位置比例
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.edge_ratio = edge_ratio

    def assign_priorities(
        self,
        messages: List[Any]
    ) -> List[PrioritizedMessage]:
        """
        为消息列表分配优先级

        Args:
            messages: 消息列表（BaseMessage 或 Dict）

        Returns:
            带优先级的消息列表
        """
        total = len(messages)
        result = []

        for i, msg in enumerate(messages):
            tokens = self._estimate_tokens(msg)
            priority = self._assign_single_priority(msg, i, total, tokens)

            result.append(PrioritizedMessage(
                message=msg,
                priority=priority,
                position=i,
                estimated_tokens=tokens,
            ))

        return result

    def _assign_single_priority(
        self,
        message: Any,
        position: int,
        total: int,
        tokens: int
    ) -> MessagePriority:
        """
        单条消息优先级分配

        Args:
            message: 消息
            position: 位置索引
            total: 消息总数
            tokens: 估算的 token 数

        Returns:
            消息优先级
        """
        role = self._get_role(message)
        content = self._get_content(message)

        # 规则 1: 系统消息 → CRITICAL
        if role == "system" or isinstance(message, SystemMessage):
            return MessagePriority.CRITICAL

        # 规则 2: 工具相关 → HIGH
        if role == "tool" or self._has_tool_calls(message):
            return MessagePriority.HIGH

        # 规则 3: 边缘位置（首尾消息）→ HIGH
        edge_count = max(1, int(total * self.edge_ratio))
        if position < edge_count or position >= total - edge_count:
            return MessagePriority.HIGH

        # 规则 4: 包含中医关键词 → HIGH
        if self._contains_tcm_keywords(content):
            return MessagePriority.HIGH

        # 规则 5: 长消息 → NORMAL (保持默认，信息量大)
        if tokens >= self.long_threshold:
            return MessagePriority.NORMAL

        # 规则 6: 短消息且无问号 → LOW
        if tokens < self.short_threshold:
            if "?" not in content and "？" not in content:
                return MessagePriority.LOW

        return MessagePriority.NORMAL

    def _get_role(self, message: Any) -> str:
        """获取消息角色"""
        if isinstance(message, dict):
            return message.get("role", "")
        elif isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif hasattr(message, "type"):
            return message.type
        return ""

    def _get_content(self, message: Any) -> str:
        """获取消息内容"""
        if isinstance(message, dict):
            return message.get("content", "")
        elif hasattr(message, "content"):
            return message.content or ""
        return ""

    def _has_tool_calls(self, message: Any) -> bool:
        """检查是否包含工具调用"""
        if isinstance(message, dict):
            return "tool_calls" in message or "tool_call_id" in message
        elif hasattr(message, "tool_calls"):
            return bool(message.tool_calls)
        return False

    def _contains_tcm_keywords(self, content: str) -> bool:
        """检查是否包含中医关键词"""
        return any(kw in content for kw in self.TCM_KEYWORDS)

    def _estimate_tokens(self, message: Any) -> int:
        """
        估算消息的 token 数量

        中文约 1.5 字符/token，英文约 4 字符/token
        """
        content = self._get_content(message)
        if not content:
            return 0

        chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(content) - chinese_chars

        return int(chinese_chars / 1.5 + other_chars / 4)

    def get_messages_by_priority(
        self,
        prioritized: List[PrioritizedMessage],
        min_priority: MessagePriority = MessagePriority.LOW
    ) -> List[PrioritizedMessage]:
        """
        获取指定优先级以上的消息

        Args:
            prioritized: 带优先级的消息列表
            min_priority: 最低优先级

        Returns:
            过滤后的消息列表
        """
        return [
            pm for pm in prioritized
            if pm.priority.value >= min_priority.value
        ]

    def calculate_total_tokens(
        self,
        prioritized: List[PrioritizedMessage]
    ) -> int:
        """计算消息列表的总 token 数"""
        return sum(pm.estimated_tokens for pm in prioritized)
