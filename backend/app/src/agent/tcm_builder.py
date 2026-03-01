"""
TCM Multi-Agent Graph Builder
中医多智能体图构建器

使用LangGraph构建中医问诊系统的多智能体工作流

架构升级（2026-02）：
- 集成中间件链（Guardrails, PII, Logging, CostControl）
- 中间件在图执行前后自动运行
"""

import uuid
from typing import Literal, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from .checkpointer import get_checkpointer  # 2026-02-05: 支持 PostgresSaver
from .tcm_states import (
    TCMInputState,
    TCMAgentState,
    TCMOutputState,

    LLMConfig,
)



# 导入中间件
from .middleware import (
    MiddlewareChain,
    # TCM 自定义中间件
    get_tcm_guardrails_middleware,
    get_tcm_pii_middleware,
    get_tcm_logging_middleware,
    get_tcm_context_manager_middleware,
    get_tcm_filesystem_middleware,
    # LangChain 内置中间件（2026-02-05 集成）
    TCMLangChainMiddlewareConfig,
    get_model_call_limit_middleware,
    get_tool_call_limit_middlewares,
    # get_model_fallback_middleware,
    get_tool_retry_middleware,
    get_model_retry_middleware,
    get_summarization_middleware,
)

# 导入统一的 LLM Provider
from app.src.core.language_model.llm_provider import get_langchain_llm


def get_llm(
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,  # 增加默认值，避免长文本截断
    llm_config: Optional[LLMConfig] = None
) -> BaseChatModel:
    """
    获取LLM实例

    优先级：
    1. llm_config（从前端传入的配置）
    2. 环境变量配置

    Args:
        model: 模型名称（可选，会被 llm_config 覆盖）
        temperature: 温度参数（可选，会被 llm_config 覆盖）
        max_tokens: 最大token数（可选，会被 llm_config 覆盖），默认4096
        llm_config: LLM配置对象（从 state.llm_config 传入）

    Returns:
        BaseChatModel: LLM实例
    """
    import os
    from app.src.common.config.setting_config import settings

    # 1. 优先使用传入的 llm_config
    if llm_config and llm_config.provider_name and llm_config.model_name:
        # 检查模型是否支持 tool_call（需要从数据库查询模型特性）
        enable_web_search = _check_model_supports_tool_call(
            llm_config.provider_name, 
            llm_config.model_name
        )
        
        return get_langchain_llm(
            provider_name=llm_config.provider_name,
            model_name=llm_config.model_name,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            max_tokens=llm_config.max_tokens,
            enable_thinking=llm_config.enable_thinking,
            enable_web_search=enable_web_search,
        )

    # 2. 回退到环境变量配置
    service = os.getenv("AGENT_SERVICE", "DEEPSEEK")

    if service == "DEEPSEEK" or settings.DEEPSEEK_API_KEY:
        return get_langchain_llm(
            provider_name="deepseek",
            model_name=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=settings.DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=settings.DEEPSEEK_BASE_URL or os.getenv("DEEPSEEK_BASE_URL"),
            temperature=temperature,
            max_tokens=max_tokens,  # 传递 max_tokens
        )
    elif service == "OLLAMA":
        return get_langchain_llm(
            provider_name="ollama",
            model_name=model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            max_tokens=max_tokens,  # 传递 max_tokens
        )
    elif settings.OPENAI_API_KEY:
        return get_langchain_llm(
            provider_name="openai",
            model_name=model or "gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,  # 传递 max_tokens
        )
    else:
        # 默认使用OpenAI兼容接口
        return get_langchain_llm(
            provider_name="openai",
            model_name=model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,  # 传递 max_tokens
        )


def _check_model_supports_tool_call(provider_name: str, model_name: str) -> bool:
    """
    检查模型是否支持 tool_call 特性
    
    通过查询数据库中的模型配置来判断
    
    Args:
        provider_name: 提供商名称
        model_name: 模型名称
        
    Returns:
        bool: 是否支持 tool_call
    """
    try:
        from app.src.service.language_model_service import LanguageModelService
        from app.src.database.database import get_async_session
        
        # 这里需要同步查询，但 get_async_session 是异步的
        # 为了简化，我们使用一个简单的映射表
        # 实际生产环境中，应该从数据库查询
        
        # 已知支持 tool_call 的模型（可以扩展）
        tool_call_models = {
            "openai": ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "anthropic": ["claude-3", "claude-3.5"],
            "google": ["gemini"],
            "qwen": ["qwen-max", "qwen-plus"],
        }
        
        provider_key = provider_name.lower()
        if provider_key in tool_call_models:
            # 检查模型名称是否匹配
            for supported_model in tool_call_models[provider_key]:
                if supported_model in model_name.lower():
                    return True
        
        return False
    except Exception as e:
        from app.src.utils import get_logger
        logger = get_logger("tcm_builder")
        logger.warning(f"Failed to check tool_call support: {e}")
        return False


# ============== 节点导入 ==============

from .components.router.router import analyze_and_route_query, route_query
from .components.general.handlers import respond_to_general_query
from .components.wellness.handlers import call_wellness_subgraph
from .components.herb.handlers import handle_herb_query
from .components.prescription.handlers import handle_prescription_query
from .components.diagnose.handlers import handle_diagnose_query

# ============== 节点函数 ==============

# 全局中间件链（在图构建时初始化）
_middleware_chain: Optional[MiddlewareChain] = None


def get_middleware_chain() -> MiddlewareChain:
    """
    获取全局中间件链
    
    中间件执行顺序（按优先级从小到大）：
    
    1. 安全与合规（P0）
       - Guardrails: TCM 专用安全守卫
    
    2. 稳定性（P1）
       - ModelRetry: 模型调用重试
       - ToolRetry: 工具调用重试
       - ModelFallback: 模型降级
    
    3. 上下文管理（P2）
       - Filesystem: 大结果驱逐
       - ContextManager: 用户画像 + 工具裁剪
       - Summarization: 对话历史摘要
    
    4. 成本控制（P3）
       - ModelCallLimit: 模型调用次数限制
       - ToolCallLimit: 工具调用次数限制
    
    5. 日志与安全（P4）
       - Logging: 日志记录
       - PII: 个人信息脱敏
    """
    global _middleware_chain
    if _middleware_chain is None:
        _middleware_chain = MiddlewareChain()
        
        # LangChain 内置中间件配置
        lc_config = TCMLangChainMiddlewareConfig(
            # 成本控制
            model_call_thread_limit=50,
            model_call_run_limit=10,
            tool_call_limits={
                "web_search": 5,
                "kg_syndrome_search": 10,
            },
            # 降级配置
            enable_model_fallback=True,
            fallback_models=["deepseek:deepseek-chat", "openai:gpt-4o-mini"],
            # 重试配置
            enable_tool_retry=True,
            tool_retry_max_retries=3,
            enable_model_retry=True,
            model_retry_max_retries=2,
            # 摘要配置
            enable_summarization=True,
            summarization_trigger_tokens=6000,
            summarization_keep_messages=10,
        )
        
        # === P0: 安全与合规 ===
        _middleware_chain.add(get_tcm_guardrails_middleware())      # Priority: 0
        
        # === P1: 稳定性 ===
        model_retry = get_model_retry_middleware(lc_config)
        if model_retry:
            _middleware_chain.add(model_retry)                      # Priority: 2
        
        tool_retry = get_tool_retry_middleware(lc_config)
        if tool_retry:
            _middleware_chain.add(tool_retry)                       # Priority: 3
        #暂时注释
        # fallback = get_model_fallback_middleware(lc_config)
        # if fallback:
        #     _middleware_chain.add(fallback)                         # Priority: 4
        
        # === P2: 上下文管理 ===
        _middleware_chain.add(get_tcm_filesystem_middleware())      # Priority: 5
        _middleware_chain.add(get_tcm_context_manager_middleware()) # Priority: 6
        
        summarization = get_summarization_middleware(lc_config)
        if summarization:
            _middleware_chain.add(summarization)                    # Priority: 7
        
        # === P3: 成本控制（替代旧 cost_control.py） ===
        _middleware_chain.add(get_model_call_limit_middleware(lc_config))  # Priority: 10
        
        for tool_limit in get_tool_call_limit_middlewares(lc_config):
            _middleware_chain.add(tool_limit)                       # Priority: 11+
        
        # === P4: 日志与安全 ===
        # _middleware_chain.add(get_tcm_logging_middleware())         # Priority: 15
        _middleware_chain.add(get_tcm_pii_middleware())             # Priority: 20
        
    return _middleware_chain


async def middleware_before_handler(state: TCMAgentState) -> dict:
    """
    中间件前置处理节点

    在路由之前执行所有中间件的 before_model 钩子
    """
    middleware_chain = get_middleware_chain()

    # 记录中间件开始执行
    execution_steps = ["[中间件] 开始执行前置检查"]
    
    # 执行中间件链
    result = middleware_chain.execute_before_model(state, runtime=None)

    if result:
        # 如果中间件返回了拦截结果
        if result.get("jump_to") == "end":
            # 合并中间件返回的 steps
            if result.get("steps"):
                execution_steps.extend(result["steps"])
            result["steps"] = execution_steps
            # 直接跳转到结束
            return result
        # 否则返回状态更新
        if result.get("steps"):
            execution_steps.extend(result["steps"])
        execution_steps.append("[中间件] 前置检查完成")
        result["steps"] = execution_steps
        return result

    # 中间件全部通过
    return {"steps": ["[中间件] 前置检查完成，全部通过"]}


async def middleware_after_handler(state: TCMAgentState) -> dict:
    """
    中间件后置处理节点

    在生成最终响应后执行所有中间件的 after_model 钩子
    """
    middleware_chain = get_middleware_chain()

    # 记录中间件开始执行
    execution_steps = ["[中间件] 开始执行后置检查"]
    
    # 执行中间件链
    updates = middleware_chain.execute_after_model(state, runtime=None)

    if updates and updates.get("steps"):
        execution_steps.extend(updates["steps"])
    
    execution_steps.append("[中间件] 后置检查完成")
    
    if updates:
        updates["steps"] = execution_steps
        return updates
    
    return {"steps": execution_steps}


def route_after_middleware(state: TCMAgentState) -> str:
    """
    中间件后的路由决策

    如果中间件拦截了请求（jump_to: end），直接跳转到后置中间件
    否则继续正常路由
    """
    # 检查是否被中间件拦截
    if getattr(state, "jump_to", None) == "end":
        return "middleware_after"

    # 否则继续正常路由
    return "analyze_and_route_query"


# async def generate_final_response(state: TCMAgentState) -> dict:
#     """
#     生成最终响应（已禁用 - 2026-02-05）
#     
#     现在各路由直接流式输出，不需要总结 agent
#     """
#     if state.answer:
#         return {
#             "messages": [AIMessage(content=state.answer)],
#         }
#     return {
#         "answer": TCM_ERROR_RESPONSE,
#         "messages": [AIMessage(content=TCM_ERROR_RESPONSE)],
#     }


# ============== 图构建 ==============

def build_tcm_graph():
    """
    构建中医多智能体工作流图

    架构升级（2026-02）：
    - 集成中间件链（before/after hooks）
    - 移除旧的 guardrails_check 节点，改用中间件实现
    - 中间件在路由前后自动执行
    - 移除总结 agent，各路由直接流式输出

    流程：
    START → middleware_before → [条件判断]
                                  ├── 被拦截 → middleware_after → END
                                  └── 通过 → analyze_and_route_query → [业务节点] → middleware_after → END

    Returns:
        CompiledGraph: 编译后的LangGraph图
    """
    # 创建状态图
    graph = StateGraph(TCMAgentState, input=TCMInputState, output=TCMOutputState)

    # ============== 添加节点 ==============

    # 中间件节点
    graph.add_node("middleware_before", middleware_before_handler)
    graph.add_node("middleware_after", middleware_after_handler)

    # 路由节点
    graph.add_node("analyze_and_route_query", analyze_and_route_query)

    # 一般性回答
    graph.add_node("respond_to_general_query", respond_to_general_query)

    # 养生流子图节点
    graph.add_node("wellness_subgraph_node", call_wellness_subgraph)

    # 诊断节点
    graph.add_node("handle_diagnose_query", handle_diagnose_query)

    # 药材询问节点
    graph.add_node("handle_herb_query", handle_herb_query)

    # 方剂询问节点
    graph.add_node("handle_prescription_query", handle_prescription_query)

    # 注释掉总结 agent 节点（2026-02-05）
    # graph.add_node("generate_final_response", generate_final_response)

    # ============== 添加边 ==============

    # 1. 入口 → 中间件前置处理
    graph.add_edge(START, "middleware_before")

    # 2. 中间件前置 → 条件路由（检查是否被拦截）
    graph.add_conditional_edges(
        "middleware_before",
        route_after_middleware,
        {
            "analyze_and_route_query": "analyze_and_route_query",
            # 被拦截时直接进入后置中间件
            "middleware_after": "middleware_after",
        }
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
            # 直接进入后置中间件（跳过总结 agent）
            "middleware_after": "middleware_after",
        }
    )

    # 4. 各处理节点 → 直接进入后置中间件（跳过总结 agent）
    graph.add_edge("respond_to_general_query", "middleware_after")
    graph.add_edge("wellness_subgraph_node", "middleware_after")
    graph.add_edge("handle_diagnose_query", "middleware_after")
    graph.add_edge("handle_herb_query", "middleware_after")
    graph.add_edge("handle_prescription_query", "middleware_after")

    # 5. 中间件后置 → 结束
    graph.add_edge("middleware_after", END)

    # 编译图（使用可配置的 Checkpointer）
    checkpointer = get_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    return compiled_graph


def new_thread_id() -> str:
    """生成新的线程ID"""
    return str(uuid.uuid4())
