"""
TCM Intent Recognition Module
中医意图识别模块

架构升级（2026-02）：
本模块专注于意图分类，守卫层已移至中间件架构：
- L0 红线层（急救阻断）→ TCMGuardrailsMiddleware
- 上下文增强 → TCMContextManagerMiddleware

当前架构：
L1: 规则层 - 高频快速匹配（正则/关键词）
L3: LLM层 - 深度分类（大模型）

四大核心意图：
- wellness (养生类): L1简单 / L2复杂
- prescription (方剂类): 查询 / 组成 / 推荐 / 对比
- herb (药材类): 功效 / 禁忌 / 用法 / 鉴别
- diagnosis (问诊类): 症状 / 舌诊 / 综合 / 医案

注意：EmergencyInterceptor 和 TCMContextEnricher 保留用于向后兼容，
但新代码应使用中间件架构（TCMGuardrailsMiddleware + TCMContextManagerMiddleware）
"""

from .schemas import (
    IntentType,
    WellnessLevel,
    WellnessSubType,
    PrescriptionSubType,
    HerbSubType,
    DiagnosisSubType,
    EmergencyType,
    OOSReason,
    SentimentAnalysis,
    ExtractedEntities,
    IntentClassification,
    IntentRouteResult,
    EmergencyResult,
    OOSResult,
)
from .diagnosis_state import (
    DiagnosisStage,
    CollectedSymptoms,
    DiagnosisState,
    should_trigger_tongue_analysis,
    TONGUE_KEYWORDS,
)
# from .emergency_interceptor import EmergencyInterceptor, get_emergency_interceptor
from .router.rule_router import RuleBasedRouter, get_rule_router
from .context_enricher import TCMContextEnricher, create_context_enricher
from .intent_classifier import IntentClassifier, create_intent_classifier
from .router.wellness_router import WellnessRouter, get_wellness_router
from .router.intent_router import IntentRouter, create_intent_router

__all__ = [
    # 核心类
    # "EmergencyInterceptor",
    "RuleBasedRouter",
    "TCMContextEnricher",
    "IntentClassifier",
    "WellnessRouter",
    "IntentRouter",
    # 问诊状态管理
    "DiagnosisStage",
    "CollectedSymptoms",
    "DiagnosisState",
    "should_trigger_tongue_analysis",
    "TONGUE_KEYWORDS",
    # 工厂函数
    # "get_emergency_interceptor",
    "get_rule_router",
    "create_context_enricher",
    "create_intent_classifier",
    "get_wellness_router",
    "create_intent_router",
    # 枚举类型
    "IntentType",
    "WellnessLevel",
    "WellnessSubType",
    "PrescriptionSubType",
    "HerbSubType",
    "DiagnosisSubType",
    "EmergencyType",
    "OOSReason",
    # 数据模型
    "SentimentAnalysis",
    "ExtractedEntities",
    "IntentClassification",
    "IntentRouteResult",
    "EmergencyResult",
    "OOSResult",
]
