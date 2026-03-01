"""
Intent Router - 意图路由主入口

架构升级（2026-02）：
- L0 红线层（急救阻断）已移至 TCMGuardrailsMiddleware
- 上下文增强已移至 TCMContextManagerMiddleware
- 本模块专注于：L1规则层 + L3 LLM层 意图分类

简化后的架构：
L1: 规则层 - 高频快速匹配（正则/关键词）
L3: LLM层 - 深度分类（大模型）
"""

import time
from typing import Optional


from .rule_router import RuleBasedRouter, get_rule_router

from .wellness_router import WellnessRouter, get_wellness_router
from app.src.agent.intent_recognition.schemas import(
    IntentRouteResult,
    IntentClassification,
    IntentType,
    OOSResult,
    OOSReason,
    EnrichedContext,
    UserProfile,
    EnvironmentContext,
    ConversationContext,
)

from app.src.agent.intent_recognition.intent_classifier import IntentClassifier, create_intent_classifier




class IntentRouter:
    """
    意图路由器（简化版）

    专注于意图分类：
    1. L1规则层：高频快速匹配
    2. L3 LLM层：深度分类

    注意：紧急情况检测和上下文增强已移至中间件处理
    """

    # 拒识置信度阈值
    OOS_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        rule_router: Optional[RuleBasedRouter] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        wellness_router: Optional[WellnessRouter] = None,
        # 以下参数保留用于向后兼容，但不再使用
        emergency_interceptor=None,
        context_enricher=None,
        db_session=None,
        redis_client=None,
    ):
        """
        初始化路由器

        Args:
            rule_router: 规则路由器
            intent_classifier: LLM分类器
            wellness_router: 养生路由器
        """
        self.rule_router = rule_router or get_rule_router()
        self.intent_classifier = intent_classifier or create_intent_classifier()
        self.wellness_router = wellness_router or get_wellness_router()

    async def route(
        self,
        query: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        has_image: bool = False,
        # 以下参数保留用于向后兼容
        skip_emergency_check: bool = True,
        skip_context_enrichment: bool = True,
    ) -> IntentRouteResult:
        """
        执行意图路由

        Args:
            query: 用户输入
            user_id: 用户ID
            conversation_id: 会话ID
            has_image: 是否包含图片

        Returns:
            IntentRouteResult: 路由结果
        """
        start_time = time.time()
        route_path = []

        result = IntentRouteResult()

        # 创建空上下文（实际上下文由中间件注入到 state）
        result.context = EnrichedContext(
            user_profile=UserProfile(user_id=user_id),
            environment=EnvironmentContext(),
            conversation=ConversationContext()
        )

        # ========== L1: 规则层匹配路由 - 高频快速匹配 ==========
        route_path.append("L1:rule_matching")
        rule_result = self.rule_router.route(query)

        if rule_result and rule_result.confidence >= 0.85:
            # 规则层高置信度命中
            route_path.append(f"L1:matched:{rule_result.primary_intent.value}")

            # 如果是养生类，判定L1/L2
            if rule_result.primary_intent == IntentType.WELLNESS:
                wellness_level = self.wellness_router.determine_level(
                    rule_result, query, result.context
                )
                rule_result.wellness_level = wellness_level
                route_path.append(f"wellness:{wellness_level.value}")

            result.classification = rule_result
            result.route_path = route_path
            result.latency_ms = (time.time() - start_time) * 1000
            return result

        # ========== L3: LLM层 - 深度分类 ==========
        route_path.append("L3:llm_classification")

        llm_result = await self.intent_classifier.classify_with_fallback(
            query=query,
            rule_result=rule_result,
            context=result.context,
            has_image=has_image,
        )

        route_path.append(f"L3:classified:{llm_result.primary_intent.value}")

        # ========== 拒识检测 ==========
        oos_result = self._check_oos(llm_result, query)
        result.oos = oos_result

        if oos_result.is_oos:
            route_path.append(f"OOS:{oos_result.reason.value if oos_result.reason else 'unknown'}")
            result.classification = llm_result
            result.route_path = route_path
            result.latency_ms = (time.time() - start_time) * 1000
            return result

        # ========== 养生分流判定 ==========
        if llm_result.primary_intent == IntentType.WELLNESS:
            wellness_level = self.wellness_router.determine_level(
                llm_result, query, result.context
            )
            llm_result.wellness_level = wellness_level
            route_path.append(f"wellness:{wellness_level.value}")



            

        result.classification = llm_result
        result.route_path = route_path
        result.latency_ms = (time.time() - start_time) * 1000

        return result

    def _check_oos(
        self,
        classification: IntentClassification,
        query: str
    ) -> OOSResult:
        """
        拒识检测,仅仅处理llm可能的低置信度情况

        Args:
            classification: 分类结果
            query: 用户查询

        Returns:
            OOSResult: 拒识结果
        """
        # 置信度检测
        if classification.confidence < self.OOS_CONFIDENCE_THRESHOLD:
            return OOSResult(
                is_oos=True,
                reason=OOSReason.LOW_CONFIDENCE,
                action="clarify",
                response="您的问题我不太确定理解对了，能否再具体描述一下？比如：\n"
                         "- 如果是身体不适，请描述具体症状\n"
                         "- 如果是咨询药材，请告诉我药材名称\n"
                         "- 如果是养生问题，请说明您的具体需求"
            )


        return OOSResult(is_oos=False, action="allow")

    def get_routing_config(self, result: IntentRouteResult) -> dict:
        """
        获取路由配置（供下游使用）

        Args:
            result: 路由结果

        Returns:
            dict: 路由配置
        """
        if result.should_terminate():
            return {
                "action": "terminate",
                "response": result.get_final_response(),
            }

        if not result.classification:
            return {
                "action": "error",
                "response": "意图识别失败，请稍后重试",
            }

        classification = result.classification
        intent = classification.primary_intent

        # 基础配置
        config = {
            "action": "process",
            "intent": intent.value,
            "sub_type": classification.sub_type,
            "confidence": classification.confidence,
            "entities": classification.entities.model_dump(),
            "route_source": classification.route_source,
            "route_path": result.route_path,
            "latency_ms": result.latency_ms,
        }

        # 养生类特殊配置
        if intent == IntentType.WELLNESS and classification.wellness_level:
            wellness_config = self.wellness_router.get_routing_advice(
                classification.wellness_level
            )
            config.update(wellness_config)

        # 问诊类特殊配置
        if intent == IntentType.DIAGNOSIS:
            config["requires_follow_up"] = True
            config["suggested_follow_up"] = classification.suggested_follow_up
            # 舌诊分析需要图片
            if classification.sub_type == "tongue":
                config["requires_image"] = True
                if not classification.has_image:
                    config["action"] = "request_image"
                    config["response"] = "请上传舌象图片，我来帮您进行舌诊分析。"

        return config


# 工厂函数
def create_intent_router() -> IntentRouter:
    """创建意图路由器"""
    return IntentRouter()
