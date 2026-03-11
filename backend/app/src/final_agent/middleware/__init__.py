"""
Final Agent 中间件模块

主图中间件（精简版）:
1. GuardrailsMiddleware - 安全检查
2. MemoryMiddleware - 用户记忆加载/保存
3. FocusContextMiddleware - 上下文工程
4. LearningMiddleware - 自我学习
5. PIIMiddleware - 输出脱敏
"""

# 复用 super_agent 的基础中间件
from app.src.super_agent.middleware.base import (
    BaseMiddleware,
    MiddlewareChain,
    MiddlewareConfig,
)
from app.src.super_agent.middleware.guardrails import (
    TCMGuardrailsMiddleware,
    get_tcm_guardrails_middleware,
)
from app.src.super_agent.middleware.pii import (
    TCMPIIMiddleware,
    get_tcm_pii_middleware,
)
from .context import (
    FocusContextMiddleware,
    FocusContextMiddlewareConfig,
    create_main_graph_focus_config,
    create_diagnose_focus_config,
    create_wellness_focus_config,
)

# 新创建的中间件
from .memory import MemoryMiddleware, MemoryMiddlewareConfig
from .learning import LearningMiddleware, LearningMiddlewareConfig

__all__ = [
    # Base
    "BaseMiddleware",
    "MiddlewareChain",
    "MiddlewareConfig",
    # Guardrails
    "TCMGuardrailsMiddleware",
    "get_tcm_guardrails_middleware",
    # Memory
    "MemoryMiddleware",
    "MemoryMiddlewareConfig",
    # FocusContext
    "FocusContextMiddleware",
    "FocusContextMiddlewareConfig",
    "create_main_graph_focus_config",
    "create_diagnose_focus_config",
    "create_wellness_focus_config",
    # Learning
    "LearningMiddleware",
    "LearningMiddlewareConfig",
    # PII
    "TCMPIIMiddleware",
    "get_tcm_pii_middleware",
]
