"""
LangChain 内置中间件适配层

封装 LangChain 官方提供的中间件，提供统一的配置接口。
参考文档：https://docs.langchain.com/oss/python/langchain/middleware/built-in

集成的中间件：
- ModelCallLimitMiddleware: 模型调用次数限制（替代 cost_control.py）
- ToolCallLimitMiddleware: 工具调用次数限制
- ModelFallbackMiddleware: 模型降级
- ToolRetryMiddleware: 工具调用重试
- ModelRetryMiddleware: 模型调用重试
- SummarizationMiddleware: 对话历史摘要

更新日期：2026-02-05
"""

from typing import Optional, List, Any
from dataclasses import dataclass, field
import logging

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
)

logger = logging.getLogger(__name__)


# ============================================================
# 包装器类（统一接口）
# ============================================================

class LangChainMiddlewareWrapper:
    """
    LangChain 中间件轻量包装器
    
    只为 LangChain 内置中间件添加必要的属性（priority、name、enabled），
    其他方法通过 __getattr__ 直接委托给原始中间件。
    
    优点：
    - 代码量减少 50%
    - 性能更好（减少一层继承开销）
    - 逻辑更清晰
    """
    
    def __init__(self, wrapped_middleware, priority: int, name: str, enabled: bool = True):
        """
        初始化包装器
        
        Args:
            wrapped_middleware: LangChain 原生中间件实例
            priority: 优先级（数字越小越先执行）
            name: 中间件名称
            enabled: 是否启用
        """
        self.wrapped = wrapped_middleware
        self.priority = priority
        self.name = name
        self.enabled = enabled
    
    def __getattr__(self, attr):
        """
        属性访问代理
        
        除了 priority、name、enabled、wrapped 外，
        其他属性和方法都委托给包装的中间件
        """
        return getattr(self.wrapped, attr)


# ============================================================
# 配置类
# ============================================================

@dataclass
class TCMLangChainMiddlewareConfig:
    """LangChain 中间件统一配置"""

    # === 成本控制 ===
    # 模型调用限制
    model_call_thread_limit: int = 50   # 每个会话最大模型调用次数
    model_call_run_limit: int = 10      # 单次运行最大模型调用次数
    model_call_exit_behavior: str = "end"  # "end" 或 "error"

    # 工具调用限制（按工具名）
    tool_call_limits: dict = field(default_factory=lambda: {
        "web_search": 5,           # 网络搜索限制 5 次
        "kg_syndrome_search": 10,  # 知识图谱查询限制 10 次
    })
    tool_call_exit_behavior: str = "continue"

    # === 健壮性 ===
    # 模型降级
    enable_model_fallback: bool = True
    fallback_models: List[str] = field(default_factory=lambda: [
        "deepseek:deepseek-chat",    # 主模型（DeepSeek）
        "openai:gpt-4o-mini",        # 备选模型
    ])

    # 重试配置
    enable_tool_retry: bool = True
    tool_retry_max_retries: int = 3
    tool_retry_backoff_factor: float = 2.0

    enable_model_retry: bool = True
    model_retry_max_retries: int = 2
    model_retry_backoff_factor: float = 1.5

    # === 摘要 ===
    enable_summarization: bool = True
    summarization_model: str = "deepseek:deepseek-chat"
    summarization_trigger_tokens: int = 6000
    summarization_trigger_messages: int = 20
    summarization_keep_messages: int = 10

    # TCM 专用摘要模板
    summarization_prompt: str = """请对以下中医对话进行摘要，保留关键医学信息。

## 摘要要求
1. 保留所有症状描述
2. 保留舌象、脉象等诊断信息
3. 保留辨证结论和证型
4. 保留方剂和用药建议
5. 忽略寒暄和重复内容

## 对话内容
{messages}

## 输出格式
请按以下格式输出摘要：

【主诉】患者主要症状和就诊原因
【症状】详细症状列表
【四诊信息】舌象、脉象等
【辨证】证型判断
【建议】已给出的方剂或建议

如果某项信息未提及，标注"未提及"。
"""


# ============================================================
# 工厂函数
# ============================================================

def get_model_call_limit_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> LangChainMiddlewareWrapper:
    """
    获取模型调用限制中间件

    替代原有的 TCMCostControlMiddleware，提供更完善的限制功能：
    - thread_limit: 跨运行累计的总调用限制
    - run_limit: 单次运行的调用限制
    - exit_behavior: 超限时的行为（end/error）

    Args:
        config: 配置对象

    Returns:
        包装后的 ModelCallLimitMiddleware
    """
    config = config or TCMLangChainMiddlewareConfig()

    wrapped = ModelCallLimitMiddleware(
        thread_limit=config.model_call_thread_limit,
        run_limit=config.model_call_run_limit,
        exit_behavior=config.model_call_exit_behavior,
    )

    logger.info(
        f"ModelCallLimitMiddleware 已创建: "
        f"thread_limit={config.model_call_thread_limit}, "
        f"run_limit={config.model_call_run_limit}"
    )
    
    return LangChainMiddlewareWrapper(
        wrapped,
        priority=10,
        name="ModelCallLimitMiddleware"
    )


def get_tool_call_limit_middlewares(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> List[LangChainMiddlewareWrapper]:
    """
    获取工具调用限制中间件列表

    为每个需要限制的工具创建独立的中间件。

    Args:
        config: 配置对象

    Returns:
        包装后的 ToolCallLimitMiddleware 实例列表
    """
    config = config or TCMLangChainMiddlewareConfig()

    middlewares = []
    for idx, (tool_name, limit) in enumerate(config.tool_call_limits.items()):
        wrapped = ToolCallLimitMiddleware(
            tool_name=tool_name,
            thread_limit=limit,
            exit_behavior=config.tool_call_exit_behavior,
        )
        
        middleware = LangChainMiddlewareWrapper(
            wrapped,
            priority=11 + idx,
            name=f"ToolCallLimitMiddleware_{tool_name}"
        )
        middlewares.append(middleware)
        logger.debug(f"ToolCallLimitMiddleware 已创建: {tool_name}={limit}")

    logger.info(f"创建了 {len(middlewares)} 个工具调用限制中间件")
    return middlewares


# def get_model_fallback_middleware(
#     config: Optional[TCMLangChainMiddlewareConfig] = None
# ) -> Optional[LangChainMiddlewareWrapper]:
#     """
#     获取模型降级中间件
#
#     当主模型不可用时自动切换到备选模型。
#
#     Args:
#         config: 配置对象
#
#     Returns:
#         包装后的 ModelFallbackMiddleware，如果未启用则返回 None
#     """
#     config = config or TCMLangChainMiddlewareConfig()
#
#     if not config.enable_model_fallback:
#         return None
#
#     wrapped = ModelFallbackMiddleware(
#         additional_models=config.fallback_models,
#     )
#
#     logger.info(f"ModelFallbackMiddleware 已创建: fallback_models={config.fallback_models}")
#
#     return LangChainMiddlewareWrapper(
#         wrapped,
#         priority=4,
#         name="ModelFallbackMiddleware"
#     )


def get_tool_retry_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> Optional[LangChainMiddlewareWrapper]:
    """
    获取工具重试中间件

    工具调用失败时自动重试，支持指数退避。

    Args:
        config: 配置对象

    Returns:
        包装后的 ToolRetryMiddleware，如果未启用则返回 None
    """
    config = config or TCMLangChainMiddlewareConfig()

    if not config.enable_tool_retry:
        return None

    wrapped = ToolRetryMiddleware(
        max_retries=config.tool_retry_max_retries,
        backoff_factor=config.tool_retry_backoff_factor,
    )

    logger.info(
        f"ToolRetryMiddleware 已创建: "
        f"max_retries={config.tool_retry_max_retries}, "
        f"backoff_factor={config.tool_retry_backoff_factor}"
    )
    
    return LangChainMiddlewareWrapper(
        wrapped,
        priority=3,
        name="ToolRetryMiddleware"
    )


def get_model_retry_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> Optional[LangChainMiddlewareWrapper]:
    """
    获取模型重试中间件

    模型调用失败时自动重试，支持指数退避。

    Args:
        config: 配置对象

    Returns:
        包装后的 ModelRetryMiddleware，如果未启用则返回 None
    """
    config = config or TCMLangChainMiddlewareConfig()

    if not config.enable_model_retry:
        return None

    wrapped = ModelRetryMiddleware(
        max_retries=config.model_retry_max_retries,
        backoff_factor=config.model_retry_backoff_factor,
    )

    logger.info(
        f"ModelRetryMiddleware 已创建: "
        f"max_retries={config.model_retry_max_retries}, "
        f"backoff_factor={config.model_retry_backoff_factor}"
    )
    
    return LangChainMiddlewareWrapper(
        wrapped,
        priority=2,
        name="ModelRetryMiddleware"
    )


def get_summarization_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> Optional[LangChainMiddlewareWrapper]:
    """
    获取对话摘要中间件

    当对话历史接近 Token 限制时自动生成摘要。
    使用 TCM 专用摘要模板，保留关键医学信息。

    Args:
        config: 配置对象

    Returns:
        包装后的 SummarizationMiddleware，如果未启用则返回 None
    """
    config = config or TCMLangChainMiddlewareConfig()

    if not config.enable_summarization:
        return None

    # 延迟导入，避免循环依赖
    from ..tcm_builder import get_llm
    
    middleware = SummarizationMiddleware(
        model=get_llm(),
        trigger=[
            ("tokens", config.summarization_trigger_tokens),
            ("messages", config.summarization_trigger_messages),
        ],
        keep=("messages", config.summarization_keep_messages),
        summary_prompt=config.summarization_prompt,
    )

    logger.info(
        f"SummarizationMiddleware 已创建: "
        f"trigger_tokens={config.summarization_trigger_tokens}, "
        f"trigger_messages={config.summarization_trigger_messages}, "
        f"keep_messages={config.summarization_keep_messages}"
    )
    
    return LangChainMiddlewareWrapper(
        middleware,
        priority=7,
        name="SummarizationMiddleware"
    )


# ============================================================
# 统一获取所有 LangChain 中间件
# ============================================================

def get_all_langchain_middlewares(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> List[Any]:
    """
    获取所有 LangChain 内置中间件

    返回按优先级排序的中间件列表，可直接添加到 MiddlewareChain。

    Args:
        config: 配置对象

    Returns:
        中间件列表
    """
    config = config or TCMLangChainMiddlewareConfig()

    middlewares = []

    # 1. 重试中间件（最先执行，优先级最高）
    model_retry = get_model_retry_middleware(config)
    if model_retry:
        middlewares.append(model_retry)

    tool_retry = get_tool_retry_middleware(config)
    if tool_retry:
        middlewares.append(tool_retry)

    # 2. 降级中间件
    # fallback = get_model_fallback_middleware(config)
    # if fallback:
    #     middlewares.append(fallback)

    # 3. 摘要中间件
    summarization = get_summarization_middleware(config)
    if summarization:
        middlewares.append(summarization)

    # 4. 成本控制中间件
    model_limit = get_model_call_limit_middleware(config)
    middlewares.append(model_limit)

    tool_limits = get_tool_call_limit_middlewares(config)
    middlewares.extend(tool_limits)

    logger.info(f"共创建 {len(middlewares)} 个 LangChain 内置中间件")
    return middlewares


# ============================================================
# 导出
# ============================================================

__all__ = [
    "TCMLangChainMiddlewareConfig",
    "get_model_call_limit_middleware",
    "get_tool_call_limit_middlewares",
    # "get_model_fallback_middleware",
    "get_tool_retry_middleware",
    "get_model_retry_middleware",
    "get_summarization_middleware",
    "get_all_langchain_middlewares",
]
