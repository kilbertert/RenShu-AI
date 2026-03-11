"""
统一上下文引擎

基于 Focus Sawtooth (arXiv:2601.07190) 的统一上下文管理系统。
吸收了原 5 套上下文/压缩系统的功能：

1. context/focus_manager.py → FocusContextManager (核心 Sawtooth 引擎)
2. context/compression_strategy.py → CompressionStrategySelector (压缩级别选择)
3. context/summarization.py → TCMSummarizer (TCM 专用摘要)
4. context/message_priority.py → MessagePriorityAssigner (消息优先级) [独立保留]
5. context/tool_trimmer.py → SmartToolTrimmer (工具裁剪) [独立保留]

FocusContextMiddleware 成为唯一的上下文管理中间件 (P5)。
"""

from .focus_context_middleware import FocusContextMiddleware, FocusContextMiddlewareConfig
from .focus_engine import FocusContextEngine, FocusConfig, KnowledgeBlock, FocusPhase

__all__ = [
    "FocusContextMiddleware",
    "FocusContextMiddlewareConfig",
    "FocusContextEngine",
    "FocusConfig",
    "KnowledgeBlock",
    "FocusPhase",
]
