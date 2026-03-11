"""
工具注册中心

统一管理所有 TCM Agent 工具的注册、发现和获取。
支持按类别获取、按 agent 类型获取、MCP 工具批量注册。
"""

import logging
from typing import Dict, List, Optional

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Agent 类型 → 工具类别映射
AGENT_TOOL_MAPPING = {
    "deep_search": ["kg", "vector", "classics", "web"],
    "moderate": ["kg", "vector"],
    "simple": [],
    "wellness": [],
    "herb": ["kg"],
    "prescription": ["kg", "vector"],
}


class ToolRegistry:
    """
    工具注册中心（单例）

    功能：
    1. 注册单个工具（按类别）
    2. 批量注册 MCP 工具
    3. 按类别获取工具列表
    4. 按 agent 类型获取工具列表
    """

    def __init__(self):
        # category -> [tool]
        self._tools: Dict[str, List[StructuredTool]] = {}
        # tool_name -> tool（去重索引）
        self._name_index: Dict[str, StructuredTool] = {}

    def register(self, tool: StructuredTool, category: str) -> None:
        """
        注册单个工具

        Args:
            tool: LangChain StructuredTool
            category: 工具类别 (kg, vector, classics, web, mcp, etc.)
        """
        if tool.name in self._name_index:
            logger.debug(f"工具已注册，跳过: {tool.name}")
            return

        if category not in self._tools:
            self._tools[category] = []

        self._tools[category].append(tool)
        self._name_index[tool.name] = tool
        logger.debug(f"注册工具: {tool.name} -> {category}")

    def register_mcp_tools(self, toolkit) -> int:
        """
        批量注册 MCP 工具

        Args:
            toolkit: MCPToolkit 实例（来自 agent/mcp/adapter.py）

        Returns:
            注册的工具数量
        """
        count = 0
        try:
            tools = toolkit.get_tools()
            for tool in tools:
                self.register(tool, "mcp")
                count += 1
            logger.info(f"批量注册 {count} 个 MCP 工具")
        except Exception as e:
            logger.warning(f"MCP 工具注册失败: {e}")
        return count

    def get_tools(self, categories: List[str]) -> List[StructuredTool]:
        """
        按类别获取工具列表

        Args:
            categories: 类别列表

        Returns:
            匹配的工具列表（去重）
        """
        seen = set()
        result = []
        for cat in categories:
            for tool in self._tools.get(cat, []):
                if tool.name not in seen:
                    seen.add(tool.name)
                    result.append(tool)
        return result

    def get_tools_for_agent(self, agent_type: str) -> List[StructuredTool]:
        """
        按 agent 类型获取工具列表

        Args:
            agent_type: Agent 类型 (deep_search, moderate, simple, etc.)

        Returns:
            该 agent 需要的工具列表
        """
        categories = AGENT_TOOL_MAPPING.get(agent_type, [])
        return self.get_tools(categories)

    def get_tool_by_name(self, name: str) -> Optional[StructuredTool]:
        """按名称获取单个工具"""
        return self._name_index.get(name)

    def list_categories(self) -> Dict[str, int]:
        """列出所有类别及其工具数量"""
        return {cat: len(tools) for cat, tools in self._tools.items()}

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名"""
        return list(self._name_index.keys())


# ============== 单例 ==============

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """
    获取 ToolRegistry 单例

    首次调用时自动注册内置 TCM 工具。
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _register_builtin_tools(_registry)
    return _registry


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """注册内置 TCM 工具"""
    from .tcm_tools import (
        kg_syndrome_search,
        kg_organ_query,
        case_vector_search,
        classics_search,
        web_search,
        medical_research_search,
    )

    # 知识图谱工具
    registry.register(kg_syndrome_search, "kg")
    registry.register(kg_organ_query, "kg")

    # 向量检索工具
    registry.register(case_vector_search, "vector")

    # 古籍检索工具
    registry.register(classics_search, "classics")

    # 网络搜索工具
    registry.register(web_search, "web")
    registry.register(medical_research_search, "web")

    logger.info(f"注册了 {len(registry.list_tools())} 个内置 TCM 工具")
