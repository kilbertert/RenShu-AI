"""
Final Agent 主图构建
架构设计（精简版）:
- 主图只负责请求级别的通用处理
- 中间件精简为 4 个：Guardrails, Memory, Learning, PII
- 业务中间件移至 DiagnoseSubgraph

流程:
START → Guardrails → Memory → Router → [子图] → Learning → PII → END
"""

import uuid
import logging
from typing import Optional, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from .states import MainState, MainInput, MainOutput, TCMRouter
from .middleware import (
    MiddlewareChain,
    get_tcm_guardrails_middleware,
    MemoryMiddleware,
    FocusContextMiddleware,
    FocusContextMiddlewareConfig,
    create_main_graph_focus_config,
    LearningMiddleware,
    get_tcm_pii_middleware,
)

logger = logging.getLogger(__name__)

# 全局单例
_middleware_chain: Optional[MiddlewareChain] = None
_learner = None


def get_llm(
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
) -> BaseChatModel:
    """
    获取LLM实例

    复用 super_agent 的 LLM Provider
    """
    from app.src.core.language_model.llm_provider import get_langchain_llm
    from app.src.common.config.setting_config import settings
    import os

    service = os.getenv("AGENT_SERVICE", "DEEPSEEK")

    if service == "DEEPSEEK" or settings.DEEPSEEK_API_KEY:
        return get_langchain_llm(
            provider_name="deepseek",
            model_name=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=settings.DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=settings.DEEPSEEK_BASE_URL or os.getenv("DEEPSEEK_BASE_URL"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif service == "OLLAMA":
        return get_langchain_llm(
            provider_name="ollama",
            model_name=model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return get_langchain_llm(
            provider_name="openai",
            model_name=model or "gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_learner():
    """获取 SelfLearner 单例"""
    global _learner
    if _learner is None:
        from .learning import SelfLearner, LearningConfig

        _learner = SelfLearner(
            config=LearningConfig(
                enable_feedback=True,
                enable_reflection=True,
                enable_evolution=True,
            ),
            llm=get_llm(),
        )
    return _learner


def get_middleware_chain() -> MiddlewareChain:
    """
    获取主图中间件链 - 精简版
    包含 5 个中间件:
    1. GuardrailsMiddleware (P0) - 安全检查
    2. MemoryMiddleware (P1) - 用户记忆加载/保存 (Mem0)
    3. FocusContextMiddleware (P5) - Focus上下文工程 (知识块、压缩)
    4. LearningMiddleware (P30) - 学习经验加载/记录
    5. PIIMiddleware (P40) - 输出脱敏
    """
    global _middleware_chain
    if _middleware_chain is None:
        _middleware_chain = MiddlewareChain()
        learner = get_learner()

        # === P0: 安全检查 ===
        _middleware_chain.add(get_tcm_guardrails_middleware())

        # === P1: 记忆加载 (Mem0) ===
        _middleware_chain.add(MemoryMiddleware())

        # === P5: Focus上下文工程（主图专用：总结-裁剪，禁用工具裁剪） ===
        _middleware_chain.add(
            FocusContextMiddleware(
                config=create_main_graph_focus_config(),
                llm=get_llm(),
            )
        )

        # === P30: 学习经验加载/记录（主图：仅意图学习） ===
        _middleware_chain.add(LearningMiddleware(
            learner=learner,
            source="main_graph",
        ))

        # === P40: 输出脱敏 ===
        _middleware_chain.add(get_tcm_pii_middleware())

    return _middleware_chain


# ============== 节点函数 ==============
async def middleware_before_handler(state: MainState) -> dict:
    """中间件前置处理节点"""
    chain = get_middleware_chain()
    steps = ["[中间件] 开始执行前置检查"]

    result = chain.execute_before_model(state, runtime=None)

    if result:
        if result.get("jump_to") == "end":
            if result.get("steps"):
                steps.extend(result["steps"])
            result["steps"] = steps
            return result
        if result.get("steps"):
            steps.extend(result["steps"])
        steps.append("[中间件] 前置检查完成")
        result["steps"] = steps
        return result

    return {"steps": ["[中间件] 前置检查完成，全部通过"]}


async def middleware_after_handler(state: MainState) -> dict:
    """中间件后置处理节点"""
    chain = get_middleware_chain()
    steps = ["[中间件] 开始执行后置检查"]

    updates = chain.execute_after_model(state, runtime=None)

    if updates and updates.get("steps"):
        steps.extend(updates["steps"])

    steps.append("[中间件] 后置检查完成")

    if updates:
        updates["steps"] = steps
        return updates

    return {"steps": steps}


def route_after_middleware(state: MainState) -> str:
    """中间件后的路由决策"""
    if getattr(state, "jump_to", None) == "end":
        return "middleware_after"
    return "analyze_and_route_query"


# ============== 路由组件 ==============
def _import_router():
    """延迟导入 final_agent 自己的路由组件"""
    from .components.router.router import (
        analyze_and_route_query as _analyze,
        route_query as _route,
    )

    return _analyze, _route


def _import_handlers():
    """延迟导入处理器"""
    from app.src.super_agent.components.general.handlers import respond_to_general_query
    from app.src.super_agent.components.wellness.handlers import call_wellness_subgraph
    from app.src.super_agent.components.herb.handlers import handle_herb_query
    from app.src.super_agent.components.prescription.handlers import (
        handle_prescription_query,
    )
    from app.src.super_agent.components.diagnose.handlers import handle_diagnose_query

    return (
        respond_to_general_query,
        call_wellness_subgraph,
        handle_herb_query,
        handle_prescription_query,
        handle_diagnose_query,
    )


async def analyze_and_route_query(state: MainState) -> dict:
    """路由分析节点 - 使用 final_agent 自己的路由器"""
    _analyze, _ = _import_router()
    return await _analyze(state)


def route_query(state: MainState) -> str:
    """路由决策 - 使用 final_agent 自己的路由器"""
    _, _route = _import_router()
    return _route(state)


# ============== 图构建 ==============
def build_main_graph():
    """
    构建主图 - 精简版
    流程:
    START → middleware_before → [条件判断]
                                  ├── 被拦截 → middleware_after → END
                                  └── 通过 → analyze_and_route_query → [业务节点] → middleware_after → END
    """
    # 导入处理器
    (
        respond_to_general_query,
        call_wellness_subgraph,
        handle_herb_query,
        handle_prescription_query,
        handle_diagnose_query,
    ) = _import_handlers()

    # 创建状态图
    graph = StateGraph(MainState, input=MainInput, output=MainOutput)

    # ============== 添加节点 ==============
    # 中间件节点
    graph.add_node("middleware_before", middleware_before_handler)
    graph.add_node("middleware_after", middleware_after_handler)

    # 路由节点
    graph.add_node("analyze_and_route_query", analyze_and_route_query)

    # 业务节点 - 复用 super_agent 的处理器
    graph.add_node("respond_to_general_query", respond_to_general_query)
    graph.add_node("wellness_subgraph_node", call_wellness_subgraph)
    graph.add_node("handle_diagnose_query", handle_diagnose_query)
    graph.add_node("handle_herb_query", handle_herb_query)
    graph.add_node("handle_prescription_query", handle_prescription_query)

    # ============== 添加边 ==============
    # 1. 入口 → 中间件前置处理
    graph.add_edge(START, "middleware_before")

    # 2. 中间件前置 → 条件路由
    graph.add_conditional_edges(
        "middleware_before",
        route_after_middleware,
        {
            "analyze_and_route_query": "analyze_and_route_query",
            "middleware_after": "middleware_after",
        },
    )

    # 3. 路由分析 → 各个处理节点
    graph.add_conditional_edges(
        "analyze_and_route_query",
        route_query,
        {
            "wellness_subgraph_node": "wellness_subgraph_node",
            "respond_to_general_query": "respond_to_general_query",
            "handle_diagnose_query": "handle_diagnose_query",
            "handle_herb_query": "handle_herb_query",
            "handle_prescription_query": "handle_prescription_query",
            "middleware_after": "middleware_after",
        },
    )

    # 4. 各处理节点 → 中间件后置
    graph.add_edge("respond_to_general_query", "middleware_after")
    graph.add_edge("wellness_subgraph_node", "middleware_after")
    graph.add_edge("handle_diagnose_query", "middleware_after")
    graph.add_edge("handle_herb_query", "middleware_after")
    graph.add_edge("handle_prescription_query", "middleware_after")

    # 5. 中间件后置 → 结束
    graph.add_edge("middleware_after", END)

    # 编译图
    from app.src.final_agent.checkpointer import get_checkpointer

    checkpointer = get_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    return compiled_graph


def new_thread_id() -> str:
    """生成新的线程ID"""
    return str(uuid.uuid4())