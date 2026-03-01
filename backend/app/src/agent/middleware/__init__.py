"""
TCM Agent 中间件模块

提供各类中间件实现：

== 自定义中间件（TCM 专用）==
- Guardrails: 安全守卫（紧急情况/超范围拦截）
- PII: 个人信息检测与脱敏（中国场景优化）
- ContextManager: 上下文工程管理（用户画像注入）
- Filesystem: 大结果自动驱逐

== LangChain 内置中间件（2026-02-05 集成）==
- ModelCallLimitMiddleware: 模型调用次数限制（替代旧 CostControl）
- ToolCallLimitMiddleware: 工具调用次数限制
- ModelFallbackMiddleware: 模型降级
- ToolRetryMiddleware: 工具调用重试
- ModelRetryMiddleware: 模型调用重试
- SummarizationMiddleware: 对话历史摘要
"""

from .base import BaseMiddleware, MiddlewareChain, MiddlewareConfig
from .guardrails import TCMGuardrailsMiddleware, get_tcm_guardrails_middleware
from .pii import TCMPIIMiddleware, get_tcm_pii_middleware
from .logging import TCMLoggingMiddleware, get_tcm_logging_middleware
from .context_manager import (
    TCMContextManagerMiddleware,
    ContextManagerConfig,
    get_tcm_context_manager_middleware,
)
from .filesystem import (
    TCMFilesystemMiddleware,
    FilesystemConfig,
    get_tcm_filesystem_middleware,
    create_read_file_tool,
)

# LangChain 内置中间件适配层（2026-02-05 新增）
from .langchain_middleware import (
    TCMLangChainMiddlewareConfig,
    get_model_call_limit_middleware,
    get_tool_call_limit_middlewares,
    # get_model_fallback_middleware,
    get_tool_retry_middleware,
    get_model_retry_middleware,
    get_summarization_middleware,
    get_all_langchain_middlewares,
)

__all__ = [
    # Base
    "BaseMiddleware",
    "MiddlewareChain",
    "MiddlewareConfig",

    # Guardrails
    "TCMGuardrailsMiddleware",
    "get_tcm_guardrails_middleware",

    # PII
    "TCMPIIMiddleware",
    "get_tcm_pii_middleware",

    # Logging
    "TCMLoggingMiddleware",
    "get_tcm_logging_middleware",

    # Context Manager
    "TCMContextManagerMiddleware",
    "ContextManagerConfig",
    "get_tcm_context_manager_middleware",

    # Filesystem
    "TCMFilesystemMiddleware",
    "FilesystemConfig",
    "get_tcm_filesystem_middleware",
    "create_read_file_tool",

    # LangChain 内置中间件（2026-02-05 新增）
    "TCMLangChainMiddlewareConfig",
    "get_model_call_limit_middleware",
    "get_tool_call_limit_middlewares",
    # "get_model_fallback_middleware",
    "get_tool_retry_middleware",
    "get_model_retry_middleware",
    "get_summarization_middleware",
    "get_all_langchain_middlewares",
]
