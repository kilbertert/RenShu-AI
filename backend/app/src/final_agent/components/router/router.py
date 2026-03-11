"""
TCM 路由器 (final_agent)

负责意图分类和路由决策。
路由器专注于意图分类和业务路由。

上下文工程分工：
- enriched_context 由主图中间件（MemoryMiddleware/FocusContextMiddleware）构建
- 本模块从 state 中读取，传递给 LLM 分类器辅助分类
- 规则层只看 query 原文，不受 context 影响
"""

from langchain_core.messages import HumanMessage

from app.src.common.config.setting_config import settings
from app.src.utils import get_logger
from ...states import MainState, TCMRouter

logger = get_logger("final_agent_router")


_UNDERSPECIFIED_EXACT_QUERIES = {
    "还有吗",
    "还有么",
    "还有呢",
    "还有别的吗",
    "还有别的么",
    "那呢",
    "这个呢",
    "那个呢",
    "然后呢",
    "怎么办",
    "怎么调",
    "怎么调理",
    "怎么弄",
    "怎么说",
    "为什么",
    "严重吗",
    "可以吗",
    "能吃吗",
    "能用吗",
    "多久",
    "咋办",
}

_UNDERSPECIFIED_PREFIXES = (
    "这个",
    "那个",
    "这样",
    "那样",
    "这种",
    "那种",
)


async def analyze_and_route_query(state: MainState) -> dict:
    """
    分析用户输入并路由到相应的处理节点

    从 state 中读取 enriched_context（由中间件 P1/P2 构建），
    传递给 LLM 分类器以提高分类准确度。

    Args:
        state: 当前状态

    Returns:
        dict: 路由结果
    """
    messages = state["messages"]
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

    intent_query = _build_intent_query(messages, last_user_query)

    # 从 state 获取中间件已构建的增强上下文
    enriched_context = state.get("enriched_context", None)

    # 创建意图分类器和路由器
    from ...intent_recognition.router.intent_router import IntentRouter

    intent_classifier = _create_intent_classifier(state.get("llm_config"))

    router = IntentRouter(intent_classifier=intent_classifier)

    try:
        existing_router = state.get("router")
        route_result = await router.route(
            query=last_user_query,
            intent_query=intent_query,
            user_id=state.get("user_id", "default_user"),
            conversation_id=state.get("conversation_id"),
            has_image=existing_router.has_image if existing_router else False,
            enriched_context=enriched_context,
        )

        # OOS/闲聊 -> tcm-chat
        if route_result.oos.is_oos:
            return {
                "router": TCMRouter(
                    query_type="tcm-chat",
                    reasoning=f"OOS识别: {route_result.oos.reason}",
                    confidence=0.9,
                ),
                "answer": route_result.oos.response,
                "steps": ["路由分析: 命中一般性回答(General)流"],
            }

        # 业务意图映射
        query_type = "tcm-chat"

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

            # 舌诊需要图片
            if cls.primary_intent.value == "diagnosis" and cls.sub_type == "tongue":
                query_type = "tcm-image"

        return {
            "router": TCMRouter(
                query_type=query_type,
                reasoning=f"路径: {' -> '.join(route_result.route_path)}",
                confidence=route_result.classification.confidence if route_result.classification else 0.5,
                primary_intent=route_result.classification.primary_intent.value if route_result.classification else None,
                sub_type=route_result.classification.sub_type if route_result.classification else None,
            ),
            "enriched_context": route_result.context,
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


def _build_intent_query(messages, current_query: str) -> str:
    if not _is_underspecified_query(current_query):
        return current_query

    previous_queries = _get_recent_user_queries(messages, limit=2, exclude_current=True)
    if not previous_queries:
        return current_query

    parts = [
        "请以【当前用户问题】为主判断意图；如果当前问题是续问或省略句，可以参考前面的用户问题补全语义。",
        "",
    ]

    for index, query in enumerate(previous_queries, start=1):
        parts.extend([f"【前序用户问题{index}】", query, ""])

    parts.extend(["【当前用户问题】", current_query])
    return "\n".join(parts)


def _is_underspecified_query(query: str) -> bool:
    normalized_query = "".join(str(query or "").split())
    if not normalized_query:
        return False

    if normalized_query in _UNDERSPECIFIED_EXACT_QUERIES:
        return True

    if len(normalized_query) > 8:
        return False

    return any(normalized_query.startswith(prefix) for prefix in _UNDERSPECIFIED_PREFIXES)


def _get_recent_user_queries(messages, limit: int = 2, exclude_current: bool = True) -> list[str]:
    user_queries: list[str] = []

    for message in messages:
        content = _extract_user_message_content(message)
        if content:
            user_queries.append(content)

    if exclude_current and user_queries:
        user_queries = user_queries[:-1]

    if limit <= 0:
        return []

    return user_queries[-limit:]


def _extract_user_message_content(message) -> str:
    if isinstance(message, HumanMessage):
        content = message.content
    elif isinstance(message, dict) and message.get("role") in ("user", "human"):
        content = message.get("content", "")
    elif getattr(message, "type", None) in ("human", "user"):
        content = getattr(message, "content", "")
    else:
        return ""

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


def _create_intent_classifier(llm_config):
    """
    创建意图分类器

    优先级：state.llm_config > settings
    """
    from ...intent_recognition.intent_classifier import create_intent_classifier

    if llm_config and getattr(llm_config, "provider_name", None) and getattr(llm_config, "model_name", None):
        logger.debug(
            f"使用前端传入的模型配置: provider={llm_config.provider_name}, "
            f"model={llm_config.model_name}"
        )
        return create_intent_classifier(
            provider_name=llm_config.provider_name,
            model_name=llm_config.model_name,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            top_p=getattr(llm_config, "top_p", None),
            temperature=getattr(llm_config, "temperature", None),
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


def route_query(state: MainState) -> str:
    """
    根据路由结果决定下一个节点

    Args:
        state: 当前状态

    Returns:
        str: 下一个节点名称
    """
    if state.get("error") or state.get("answer"):
        return "middleware_after"

    router = state.get("router")
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
