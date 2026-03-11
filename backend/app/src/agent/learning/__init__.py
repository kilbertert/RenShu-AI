"""
自学习机制 (Self-Learning Mechanism)

基于 2026 年最新研究的 AI Agent 自适应学习

实现三层学习架构：
1. Immediate Feedback Loop - 即时反馈学习
2. Self-Reflection - 自我反思改进
3. Long-term Evolution - 长期进化优化

参考论文：
- Reflexion (NeurIPS 2023): Language agents with self-reflection
- Self-Evolving Agents (arXiv:2507.21046): Long-term evolution
"""

from .feedback import (
    FeedbackCollector,
    UserFeedback,
    FeedbackType,
    FeedbackAggregator,
)
from .reflection import (
    SelfReflection,
    ReflectionPrompt,
    ReflectionResult,
    ReflectionMemory,
)
from .evolution import (
    EvolutionEngine,
    EvolutionStrategy,
    EvolutionRecord,
    PerformanceMetrics,
)
from .learner import (
    SelfLearner,
    LearningConfig,
    LearningSession,
    LearningReport,
)

__all__ = [
    # Feedback
    "FeedbackCollector",
    "UserFeedback",
    "FeedbackType",
    "FeedbackAggregator",

    # Reflection
    "SelfReflection",
    "ReflectionPrompt",
    "ReflectionResult",
    "ReflectionMemory",

    # Evolution
    "EvolutionEngine",
    "EvolutionStrategy",
    "EvolutionRecord",
    "PerformanceMetrics",

    # Learner
    "SelfLearner",
    "LearningConfig",
    "LearningSession",
    "LearningReport",
]
