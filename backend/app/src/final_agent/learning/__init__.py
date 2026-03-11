"""
中医问诊自学习机制 (TCM Self-Learning Mechanism)

核心目标：持续提升诊断准确率

三层学习架构：
1. Feedback Layer (反馈层) - 收集准确率相关的用户反馈
2. Reflection Layer (反思层) - 分析错误原因，生成改进规则
3. Evolution Layer (进化层) - 从案例中提取知识，提升准确率

设计原则：
- 准确率优先：所有学习活动直接服务于提升准确率
- 主图与子图分离：主图学习意图识别，子图学习诊断全流程
- 单线程 + 跨线程：当前对话立即纠正，历史案例提取模式
- 可操作性：反思和进化必须产生具体的规则，不是模糊的建议
"""

from .feedback import (
    TCMFeedbackCollector,
    TCMUserFeedback,
    TCMFeedbackType,
    TCMFeedbackAggregator,
)
from .reflection import (
    TCMReflectionEngine,
    TCMReflectionResult,
    TCMReflectionType,
    TCM_REFLECTION_PROMPTS,
)
from .evolution import (
    AccuracyEvolutionEngine,
    AccuracyMetrics,
    EvolutionStrategy,
    EvolutionRecord,
    IntentRuleEvolution,
    ToolSelectionEvolution,
    DiagnosisRuleEvolution,
    InquiryOptimization,
    ErrorPreventionEvolution,
)
from .events import (
    LearningEvent,
    LearningEventType,
    IntentLearningContext,
    ThreadLearningContext,
    IntentRouteCorrection,
    ToolSelectionCorrection,
    UserOutputCorrection,
)
from .learner import (
    SelfLearner,
    LearningConfig,
    LearningSession,
    LearningReport,
)
from .storage import (
    ThreadLearningStorage,
    FeedbackStorage,
    ReflectionStorage,
    CrossThreadKnowledgeStorage,
    EvolutionStorage,
)

__all__ = [
    # Feedback (准确率反馈收集)
    "TCMFeedbackCollector",
    "TCMUserFeedback",
    "TCMFeedbackType",
    "TCMFeedbackAggregator",

    # Reflection (准确率反思分析)
    "TCMReflectionEngine",
    "TCMReflectionResult",
    "TCMReflectionType",
    "TCM_REFLECTION_PROMPTS",

    # Evolution (准确率进化引擎)
    "AccuracyEvolutionEngine",
    "AccuracyMetrics",
    "EvolutionStrategy",
    "EvolutionRecord",
    "IntentRuleEvolution",
    "ToolSelectionEvolution",
    "DiagnosisRuleEvolution",
    "InquiryOptimization",
    "ErrorPreventionEvolution",

    # Events (学习事件)
    "LearningEvent",
    "LearningEventType",
    "IntentLearningContext",
    "ThreadLearningContext",
    "IntentRouteCorrection",
    "ToolSelectionCorrection",
    "UserOutputCorrection",

    # Learner (自学习器)
    "SelfLearner",
    "LearningConfig",
    "LearningSession",
    "LearningReport",

    # Storage (存储层)
    "ThreadLearningStorage",
    "FeedbackStorage",
    "ReflectionStorage",
    "CrossThreadKnowledgeStorage",
    "EvolutionStorage",
]
