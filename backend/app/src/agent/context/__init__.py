"""
上下文工程模块

提供智能上下文管理功能：
- MessagePriority: 消息优先级分配
- CompressionStrategy: 压缩策略选择
- TCMSummarizer: TCM 专用摘要
- ToolTrimmer: 工具消息裁剪
- HierarchicalMemory: 分层记忆
"""

from .message_priority import (
    MessagePriority,
    MessagePriorityAssigner,
    PrioritizedMessage,
)
from .compression_strategy import (
    CompressionLevel,
    CompressionStrategy,
    CompressionStrategySelector,
)
from .summarization import (
    TCMSummarizer,
    TCMSummaryResult,
    TCM_SUMMARY_TEMPLATE,
)
from .tool_trimmer import (
    ToolType,
    ToolRound,
    TrimmedToolRound,
    SmartToolTrimmer,
)
from .hierarchical_memory import (
    MemoryLevel,
    MemoryEntry,
    PatientProfile,
    HierarchicalMemory,
)

__all__ = [
    # Message Priority
    "MessagePriority",
    "MessagePriorityAssigner",
    "PrioritizedMessage",

    # Compression Strategy
    "CompressionLevel",
    "CompressionStrategy",
    "CompressionStrategySelector",

    # Summarization
    "TCMSummarizer",
    "TCMSummaryResult",
    "TCM_SUMMARY_TEMPLATE",

    # Tool Trimmer
    "ToolType",
    "ToolRound",
    "TrimmedToolRound",
    "SmartToolTrimmer",

    # Hierarchical Memory
    "MemoryLevel",
    "MemoryEntry",
    "PatientProfile",
    "HierarchicalMemory",
]
