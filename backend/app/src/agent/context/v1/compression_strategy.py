# """
# 压缩策略选择器,被动压缩
#
# 根据 Token 使用情况动态选择压缩策略：
# - Level 1: 裁剪低优先级消息
# - Level 2: 摘要中间对话
# - Level 3: 激进压缩（仅保留关键信息）
# """
#
# from enum import Enum
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass
#
# from .message_priority import MessagePriority, PrioritizedMessage
#
#
# class CompressionLevel(Enum):
#     """压缩级别"""
#     NONE = 0        # 不压缩
#     LIGHT = 1       # 轻度：裁剪低优先级
#     MEDIUM = 2      # 中度：摘要中间对话
#     AGGRESSIVE = 3  # 激进：仅保留关键信息
#
#
# @dataclass
# class CompressionStrategy:
#     """压缩策略"""
#     level: CompressionLevel
#     target_tokens: int
#     keep_system: bool = True
#     keep_last_n: int = 3
#     summarize_middle: bool = False
#     drop_low_priority: bool = False
#     trim_tool_results: bool = False
#
#
# class CompressionStrategySelector:
#     """
#     压缩策略选择器
#
#     根据当前 Token 使用情况和目标限制，选择合适的压缩策略。
#     """
#
#     # Token 阈值配置
#     DEFAULT_MAX_TOKENS = 8000       # 默认最大 Token
#     WARNING_THRESHOLD = 0.7         # 70% 时开始预警
#     LIGHT_THRESHOLD = 0.8           # 80% 时轻度压缩
#     MEDIUM_THRESHOLD = 0.9          # 90% 时中度压缩
#     AGGRESSIVE_THRESHOLD = 0.95     # 95% 时激进压缩
#
#     def __init__(
#         self,
#         max_tokens: int = 8000,
#         warning_threshold: float = 0.7,
#         light_threshold: float = 0.8,
#         medium_threshold: float = 0.9,
#         aggressive_threshold: float = 0.95,
#     ):
#         """
#         初始化策略选择器
#
#         Args:
#             max_tokens: 最大 Token 限制
#             warning_threshold: 预警阈值
#             light_threshold: 轻度压缩阈值
#             medium_threshold: 中度压缩阈值
#             aggressive_threshold: 激进压缩阈值
#         """
#         self.max_tokens = max_tokens
#         self.warning_threshold = warning_threshold
#         self.light_threshold = light_threshold
#         self.medium_threshold = medium_threshold
#         self.aggressive_threshold = aggressive_threshold
#
#     def select_strategy(
#         self,
#         current_tokens: int,
#         message_count: int = 0,
#     ) -> CompressionStrategy:
#         """
#         选择压缩策略
#
#         Args:
#             current_tokens: 当前 Token 使用量
#             message_count: 消息数量（可选）
#
#         Returns:
#             压缩策略
#         """
#         usage_ratio = current_tokens / self.max_tokens
#
#         if usage_ratio >= self.aggressive_threshold:
#             return CompressionStrategy(
#                 level=CompressionLevel.AGGRESSIVE,
#                 target_tokens=int(self.max_tokens * 0.6),
#                 keep_system=True,
#                 keep_last_n=2,
#                 summarize_middle=True,
#                 drop_low_priority=True,
#                 trim_tool_results=True,
#             )
#         elif usage_ratio >= self.medium_threshold:
#             return CompressionStrategy(
#                 level=CompressionLevel.MEDIUM,
#                 target_tokens=int(self.max_tokens * 0.7),
#                 keep_system=True,
#                 keep_last_n=3,
#                 summarize_middle=True,
#                 drop_low_priority=True,
#                 trim_tool_results=True,
#             )
#         elif usage_ratio >= self.light_threshold:
#             return CompressionStrategy(
#                 level=CompressionLevel.LIGHT,
#                 target_tokens=int(self.max_tokens * 0.75),
#                 keep_system=True,
#                 keep_last_n=5,
#                 summarize_middle=False,
#                 drop_low_priority=True,
#                 trim_tool_results=False,
#             )
#         else:
#             return CompressionStrategy(
#                 level=CompressionLevel.NONE,
#                 target_tokens=self.max_tokens,
#                 keep_system=True,
#                 keep_last_n=10,
#                 summarize_middle=False,
#                 drop_low_priority=False,
#                 trim_tool_results=False,
#             )
#
#     def apply_strategy(
#         self,
#         prioritized_messages: List[PrioritizedMessage],
#         strategy: CompressionStrategy,
#         summarizer: Optional[Any] = None,
#     ) -> List[Any]:
#         """
#         应用压缩策略
#
#         Args:
#             prioritized_messages: 带优先级的消息列表
#             strategy: 压缩策略
#             summarizer: 摘要生成器（可选）
#
#         Returns:
#             压缩后的消息列表
#         """
#         if strategy.level == CompressionLevel.NONE:
#             return [pm.message for pm in prioritized_messages]
#
#         result = []
#         total = len(prioritized_messages)
#
#         # 分离不同部分
#         system_messages = []
#         head_messages = []
#         middle_messages = []
#         tail_messages = []
#
#         for i, pm in enumerate(prioritized_messages):
#             if pm.priority == MessagePriority.CRITICAL:
#                 system_messages.append(pm)
#             elif i < strategy.keep_last_n:
#                 head_messages.append(pm)
#             elif i >= total - strategy.keep_last_n:
#                 tail_messages.append(pm)
#             else:
#                 middle_messages.append(pm)
#
#         # 1. 保留系统消息
#         result.extend([pm.message for pm in system_messages])
#
#         # 2. 保留头部消息
#         result.extend([pm.message for pm in head_messages])
#
#         # 3. 处理中间消息
#         if strategy.summarize_middle and middle_messages:
#             # TODO: 使用 summarizer 生成摘要
#             # 暂时简化处理：过滤低优先级
#             if strategy.drop_low_priority:
#                 filtered = [
#                     pm for pm in middle_messages
#                     if pm.priority.value >= MessagePriority.NORMAL.value
#                 ]
#                 result.extend([pm.message for pm in filtered])
#             else:
#                 result.extend([pm.message for pm in middle_messages])
#         elif not strategy.drop_low_priority:
#             result.extend([pm.message for pm in middle_messages])
#         else:
#             # 仅保留非低优先级
#             filtered = [
#                 pm for pm in middle_messages
#                 if pm.priority.value >= MessagePriority.NORMAL.value
#             ]
#             result.extend([pm.message for pm in filtered])
#
#         # 4. 保留尾部消息
#         result.extend([pm.message for pm in tail_messages])
#
#         return result
#
#     def get_usage_status(self, current_tokens: int) -> Dict[str, Any]:
#         """
#         获取使用状态
#
#         Args:
#             current_tokens: 当前 Token 使用量
#
#         Returns:
#             使用状态信息
#         """
#         usage_ratio = current_tokens / self.max_tokens
#
#         if usage_ratio >= self.aggressive_threshold:
#             status = "critical"
#         elif usage_ratio >= self.medium_threshold:
#             status = "warning"
#         elif usage_ratio >= self.light_threshold:
#             status = "notice"
#         elif usage_ratio >= self.warning_threshold:
#             status = "approaching"
#         else:
#             status = "normal"
#
#         return {
#             "status": status,
#             "current_tokens": current_tokens,
#             "max_tokens": self.max_tokens,
#             "usage_ratio": usage_ratio,
#             "remaining_tokens": self.max_tokens - current_tokens,
#         }
