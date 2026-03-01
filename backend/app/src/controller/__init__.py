"""
控制器模块

包含所有的API控制器。
"""

from .account_controller import router as account_router
from .model_config_controller import router as model_config_router
from .chat_controller import router as chat_router
from .conversation_controller import router as conversation_router
from .tongue_analysis_controller import router as tongue_analysis_router

# TCM Agent 路由

__all__ = [
    "account_router",
    "model_config_router",
    "chat_router",
    "conversation_router",
    "tongue_analysis_router",
]
