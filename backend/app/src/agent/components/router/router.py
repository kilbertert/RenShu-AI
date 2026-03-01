"""
TCM 路由器

负责意图分类和路由决策。

架构说明（2026-02 升级）：
- 紧急情况检测已移至 TCMGuardrailsMiddleware
- 用户画像注入已移至 TCMContextManagerMiddleware
- 路由器专注于意图分类和业务路由
"""

from langchain_core.messages import HumanMessage

from app.src.common.config.setting_config import settings
from app.src.utils import get_logger
from ...tcm_states import TCMAgentState, TCMRouter

logger = get_logger("tcm_router")


async def analyze_and_route_query(state: TCMAgentState) -> dict:
    """
    分析用户输入并路由到相应的处理节点

    注意：紧急情况和上下文增强已由中间件处理，此处只做意图分类。

    Args:
        state: 当前状态

    Returns:
        dict: 路由结果
    """
    messages = state.messages
    if not messages:
        return {"error": "No messages to process"}

    # 获取最后一条用户消息
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        last_user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_query = msg.content
                break
        if not last_user_query:
            return {"error": "Last message is not from user"}
    else:
        last_user_query = last_message.content

    # 创建意图分类器和路由器
    from ...intent_recognition.router.intent_router import IntentRouter

    intent_classifier = _create_intent_classifier(state.llm_config)

    router = IntentRouter(intent_classifier=intent_classifier)

    try:
        route_result = await router.route(
            query=last_user_query,
            user_id=state.user_id or "default_user",
            conversation_id=state.conversation_id,
            has_image=state.router.has_image if state.router else False,
        )

        # OOS/闲聊 -> tcm-chat
        if route_result.oos.is_oos:
            return {
                "router": TCMRouter(
                    query_type="tcm-chat",
                    reasoning=f"OOS识别: {route_result.oos.reason}",
                    confidence=0.9,
                    extracted_entities={},
                ),
                "answer": route_result.oos.response,
                "steps": ["路由分析: 命中一般性回答(General)流"],
            }

        # 业务意图映射
        query_type = "tcm-chat"
        extracted_entities = {}

        if route_result.classification:
            cls = route_result.classification
            intent_map = {
                "general": "tcm-chat",
                "wellness": "tcm-wellness",
                "prescription": "tcm-prescription",
                "herb": "tcm-herb",
                "diagnosis": "tcm-diagnose"
            }
            query_type = intent_map.get(cls.primary_intent.value, "tcm-chat")
            extracted_entities = cls.entities.model_dump() if cls.entities else {}

            # 舌诊需要图片
            if cls.primary_intent.value == "diagnosis" and cls.sub_type == "tongue":
                query_type = "tcm-image"

        return {
            "router": TCMRouter(
                classification=route_result.classification,
                query_type=query_type,
                reasoning=f"路径: {' -> '.join(route_result.route_path)}",
                confidence=route_result.classification.confidence if route_result.classification else 0.5,
                extracted_entities=extracted_entities,
            ),
            "steps": [f"路由分析完成: {query_type}"],
        }

    except Exception as e:
        logger.error(f"路由分析失败: {e}", exc_info=True)
        return {
            "error": f"Router failed: {str(e)}",
            "router": TCMRouter(
                query_type="tcm-chat",
                reasoning="降级到General",
                confidence=0.5
            ),
        }


def _create_intent_classifier(llm_config):
    """
    创建意图分类器

    优先级：state.llm_config > settings
    """
    from ...intent_recognition.intent_classifier import create_intent_classifier

    if llm_config and llm_config.provider_name and llm_config.model_name:
        logger.debug(
            f"使用前端传入的模型配置: provider={llm_config.provider_name}, "
            f"model={llm_config.model_name}"
        )
        return create_intent_classifier(
            provider_name=llm_config.provider_name,
            model_name=llm_config.model_name,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            top_p=llm_config.top_p,
            temperature=llm_config.temperature
        )

    if settings.DEEPSEEK_API_KEY:
        logger.debug("使用 settings 中的 DeepSeek 配置")
        return create_intent_classifier(
            provider_name="deepseek",
            model_name="deepseek-chat",
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL or None,
        )

    if settings.OPENAI_API_KEY:
        logger.debug("使用 settings 中的 OpenAI 配置")
        return create_intent_classifier(
            provider_name="openai",
            model_name="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
        )

    logger.warning("未找到有效的模型配置，使用默认配置")
    return create_intent_classifier()


def route_query(state: TCMAgentState) -> str:
    """
    根据路由结果决定下一个节点

    Args:
        state: 当前状态

    Returns:
        str: 下一个节点名称
    """
    # 如果有错误或已有答案，直接进入后置中间件（跳过总结 agent）
    if state.error or state.answer:
        return "middleware_after"

    router = state.router
    if not router:
        return "respond_to_general_query"

    query_type = router.query_type

    route_map = {
        "tcm-chat": "respond_to_general_query",
        "tcm-wellness": "wellness_subgraph_node",
        "tcm-diagnose": "handle_diagnose_query",
        "tcm-herb": "handle_herb_query",
        "tcm-prescription": "handle_prescription_query",
    }

    return route_map.get(query_type, "respond_to_general_query")
