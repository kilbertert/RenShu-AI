"""
统一工具系统

提供 ToolRegistry 注册中心和合并后的 @tool 函数。
"""

from .registry import ToolRegistry, get_tool_registry
from .tcm_tools import (
    kg_syndrome_search,
    kg_organ_query,
    case_vector_search,
    classics_search,
    web_search,
    medical_research_search,
)
from .monitoring import ToolMonitor

__all__ = [
    "ToolRegistry",
    "get_tool_registry",
    "ToolMonitor",
    # Tools
    "kg_syndrome_search",
    "kg_organ_query",
    "case_vector_search",
    "classics_search",
    "web_search",
    "medical_research_search",
]
