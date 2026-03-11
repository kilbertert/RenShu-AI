"""
MCP Adapter for LangChain/LangGraph
将MCP工具适配为LangChain可用的工具

这个模块提供：
1. MCPToolWrapper: 将MCP工具包装为LangChain StructuredTool
2. MCPToolkit: 管理多个MCP工具的工具包
3. get_mcp_tools: 快捷函数，获取所有可用的MCP工具
"""

import json
from typing import Any, Optional, Sequence, Callable
from functools import wraps

from app.src.utils import get_logger

logger = get_logger("mcp_adapter")

# 检查 LangChain 是否可用
try:
    from langchain_core.tools import StructuredTool
    from langchain_core.pydantic_v1 import BaseModel, Field
    from pydantic import create_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    StructuredTool = None
    BaseModel = None
    Field = None
    create_model = None
    logger.warning("LangChain not available")

# 导入MCP客户端
from .client import (
    MCPClient,
    MCPClientPool,
    MCPTool,
    MCPServerConfig,
    MCPToolResult,
)


# ============== 动态模型创建 ==============

def create_input_model(tool: MCPTool) -> type[BaseModel]:
    """
    根据MCP工具的input_schema创建Pydantic输入模型

    Args:
        tool: MCP工具定义

    Returns:
        Pydantic模型类
    """
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("LangChain not available")

    schema = tool.input_schema
    fields = {}

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "string")
        description = field_def.get("description", "")
        is_required = field_name in required

        # 类型映射
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        python_type = type_mapping.get(field_type, str)

        # 设置默认值
        if not is_required:
            if "default" in field_def:
                default_value = field_def["default"]
            else:
                default_value = None
        else:
            default_value = ...

        # 创建Field
        field_info = Field(description=description, default=default_value)

        # 处理数组类型
        if field_type == "array" and "items" in field_def:
            items_type = field_def["items"].get("type", "string")
            if items_type == "string":
                python_type = list[str]
            elif items_type == "integer":
                python_type = list[int]
            elif items_type == "number":
                python_type = list[float]

        fields[field_name] = (python_type, field_info)

    # 创建模型
    model_name = f"{tool.name.capitalize()}Input"

    # 使用pydantic create_model
    try:
        from pydantic import create_model as pydantic_create_model
        return pydantic_create_model(
            model_name,
            **fields
        )
    except Exception as e:
        logger.warning(f"Failed to create model with pydantic.create_model: {e}")
        # 回退到简单模型
        return type(
            model_name,
            (BaseModel,),
            {"__annotations__": fields}
        )


# ============== MCP工具包装器 ==============

class MCPToolWrapper:
    """
    MCP工具包装器

    将MCP工具包装为LangChain StructuredTool
    """

    def __init__(
        self,
        tool: MCPTool,
        client: MCPClient,
        pool: Optional[MCPClientPool] = None
    ):
        """
        初始化工具包装器

        Args:
            tool: MCP工具定义
            client: MCP客户端
            pool: MCP连接池（可选）
        """
        self.tool = tool
        self.client = client
        self.pool = pool
        self._input_model = None

    @property
    def input_model(self) -> type[BaseModel]:
        """获取输入模型"""
        if self._input_model is None:
            self._input_model = create_input_model(self.tool)
        return self._input_model

    async def _invoke(self, **kwargs) -> str:
        """
        调用MCP工具

        Args:
            **kwargs: 工具参数

        Returns:
            JSON格式的结果字符串
        """
        try:
            result = await self.client.call_tool(self.tool.name, kwargs)

            if result.is_error:
                return json.dumps({
                    "error": result.error_message,
                    "tool": self.tool.name
                }, ensure_ascii=False)

            # 提取文本内容
            texts = []
            for content in result.content:
                if content.get("type") == "text":
                    texts.append(content.get("text", ""))

            return "\n\n".join(texts) if texts else "{}"

        except Exception as e:
            logger.error(f"Failed to invoke tool {self.tool.name}: {e}")
            return json.dumps({
                "error": str(e),
                "tool": self.tool.name
            }, ensure_ascii=False)

    def to_langchain_tool(self) -> "StructuredTool":
        """
        转换为LangChain StructuredTool

        Returns:
            StructuredTool实例
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available")

        return StructuredTool.from_function(
            coroutine=self._invoke,
            name=self.tool.name,
            description=self.tool.description,
            args_schema=self.input_model,
        )

    def to_langchain_tool_sync(self) -> "StructuredTool":
        """
        转换为同步版本的LangChain工具

        Returns:
            StructuredTool实例（同步版本）
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available")

        def sync_invoke(**kwargs) -> str:
            """同步调用（在事件循环中运行异步函数）"""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果已经在事件循环中，使用create_task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            self._invoke(**kwargs)
                        )
                        return future.result()
                else:
                    return asyncio.run(self._invoke(**kwargs))
            except Exception as e:
                logger.error(f"Sync invoke failed: {e}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        return StructuredTool.from_function(
            func=sync_invoke,
            name=self.tool.name,
            description=self.tool.description,
            args_schema=self.input_model,
        )


# ============== MCP工具包 ==============

class MCPToolkit:
    """
    MCP工具包

    管理多个MCP服务器的工具，提供统一的访问接口。
    """

    def __init__(self, pool: Optional[MCPClientPool] = None):
        """
        初始化工具包

        Args:
            pool: MCP连接池
        """
        self.pool = pool or MCPClientPool()
        self._tools: dict[str, MCPToolWrapper] = {}
        self._loaded = False

    async def load_tools(self) -> None:
        """
        从所有连接的服务器加载工具
        """
        self._tools.clear()

        for server_name in self.pool.connected_servers:
            client = self.pool.get_client(server_name)
            if not client:
                continue

            try:
                mcp_tools = await client.list_tools()
                for tool in mcp_tools:
                    wrapper = MCPToolWrapper(tool, client, self.pool)
                    self._tools[tool.name] = wrapper
            except Exception as e:
                logger.error(f"Failed to load tools from {server_name}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._tools)} tools from MCP servers")

    def get_tool(self, name: str) -> Optional[MCPToolWrapper]:
        """
        获取指定工具

        Args:
            name: 工具名称

        Returns:
            工具包装器，如果不存在则返回None
        """
        return self._tools.get(name)

    def get_all_tools(self) -> list[MCPToolWrapper]:
        """获取所有工具"""
        return list(self._tools.values())

    def to_langchain_tools(self) -> list["StructuredTool"]:
        """
        转换所有工具为LangChain格式

        Returns:
            LangChain工具列表
        """
        return [wrapper.to_langchain_tool() for wrapper in self._tools.values()]

    async def call_tool(
        self,
        name: str,
        **kwargs
    ) -> str:
        """
        直接调用工具

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果（JSON字符串）
        """
        wrapper = self.get_tool(name)
        if not wrapper:
            return json.dumps({
                "error": f"Tool not found: {name}"
            }, ensure_ascii=False)

        return await wrapper._invoke(**kwargs)

    @property
    def tool_names(self) -> list[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())

    @property
    def is_loaded(self) -> bool:
        """是否已加载工具"""
        return self._loaded


# ============== LangGraph工具绑定 ==============

async def bind_mcp_tools_to_agent(
    agent_builder,
    toolkit: MCPToolkit
) -> None:
    """
    将MCP工具绑定到LangGraph Agent

    Args:
        agent_builder: LangGraph Agent构建器
        toolkit: MCP工具包
    """
    if not toolkit.is_loaded:
        await toolkit.load_tools()

    langchain_tools = toolkit.to_langchain_tools()

    # 绑定工具到agent
    if hasattr(agent_builder, "bind_tools"):
        agent_builder = agent_builder.bind_tools(langchain_tools)
    elif hasattr(agent_builder, "with_tools"):
        agent_builder = agent_builder.with_tools(langchain_tools)
    else:
        logger.warning("Agent builder does not support tool binding")

    return agent_builder


# ============== 中间件集成 ==============

class MCPPromptInjectionMiddleware:
    """
    MCP工具注入中间件

    在模型调用前，将可用的MCP工具信息注入到系统提示中。
    """

    def __init__(self, toolkit: MCPToolkit):
        self.toolkit = toolkit

    def get_tool_prompt(self) -> str:
        """
        生成工具提示

        Returns:
            工具描述的提示文本
        """
        tools_info = []

        for name in self.toolkit.tool_names:
            wrapper = self.toolkit.get_tool(name)
            if wrapper:
                tools_info.append(
                    f"- {wrapper.tool.name}: {wrapper.tool.description}"
                )

        if not tools_info:
            return ""

        return "## 可用工具\n\n" + "\n".join(tools_info)

    def get_tool_usage_example(self, tool_name: str) -> str:
        """
        获取工具使用示例

        Args:
            tool_name: 工具名称

        Returns:
            使用示例文本
        """
        wrapper = self.toolkit.get_tool(tool_name)
        if not wrapper:
            return ""

        schema = wrapper.tool.input_schema
        properties = schema.get("properties", {})

        example = f"### {wrapper.tool.name}\n\n"
        example += f"**描述**: {wrapper.tool.description}\n\n"
        example += "**参数**:\n"

        for param_name, param_def in properties.items():
            param_type = param_def.get("type", "string")
            desc = param_def.get("description", "")
            required = param_name in schema.get("required", [])

            example += f"- `{param_name}` ({param_type}"
            if required:
                example += ", 必需"
            example += f"): {desc}\n"

        return example


# ============== 快捷函数 ==============

async def create_local_tcm_toolkit() -> Optional[MCPToolkit]:
    """
    创建连接到本地TCM MCP服务器的工具包

    Returns:
        MCPToolkit实例，如果创建失败则返回None
    """
    from .client import create_local_tcm_client

    client = await create_local_tcm_client()
    if not client:
        return None

    pool = MCPClientPool()
    pool._clients["local-tcm"] = client

    toolkit = MCPToolkit(pool)
    await toolkit.load_tools()

    return toolkit


async def get_mcp_tools(
    server_configs: Optional[list[MCPServerConfig]] = None
) -> list["StructuredTool"]:
    """
    获取所有MCP工具（LangChain格式）

    Args:
        server_configs: 服务器配置列表，如果为None则尝试连接本地服务器

    Returns:
        LangChain工具列表
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available")
        return []

    if server_configs is None:
        toolkit = await create_local_tcm_toolkit()
    else:
        pool = MCPClientPool()
        for config in server_configs:
            pool.add_server(config)
        await pool.connect_all()

        toolkit = MCPToolkit(pool)
        await toolkit.load_tools()

    if not toolkit:
        return []

    return toolkit.to_langchain_tools()


def create_mock_mcp_tools() -> list["StructuredTool"]:
    """
    创建模拟的MCP工具（用于测试）

    不需要连接到实际的MCP服务器，直接返回模拟工具。

    Returns:
        LangChain工具列表
    """
    if not LANGCHAIN_AVAILABLE:
        return []

    from .server import TCMToolExecutor

    executor = TCMToolExecutor()

    async def search_herbs(query: str, limit: int = 10) -> str:
        result = await executor.search_herbs(query, limit)
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def search_prescriptions(
        syndrome: Optional[str] = None,
        symptoms: Optional[list[str]] = None,
        prescription_name: Optional[str] = None,
        limit: int = 10
    ) -> str:
        result = await executor.search_prescriptions(syndrome, symptoms, prescription_name, limit)
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def diagnose_syndrome(
        symptoms: dict,
        pulse: Optional[str] = None,
        tongue: Optional[dict] = None
    ) -> str:
        result = await executor.diagnose_syndrome(symptoms, pulse, tongue)
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def get_contraindications(
        herbs: list[str],
        constitution: Optional[str] = None,
        pregnancy: bool = False
    ) -> str:
        result = await executor.get_contraindications(herbs, constitution, pregnancy)
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def search_classics(
        keywords: list[str],
        books: Optional[list[str]] = None,
        max_results: int = 5
    ) -> str:
        result = await executor.search_classics(keywords, books, max_results)
        return json.dumps(result, ensure_ascii=False, indent=2)

    # 创建Pydantic模型
    SearchHerbsInput = create_model(
        "SearchHerbsInput",
        query=(str, Field(..., description="搜索关键词")),
        limit=(int, Field(10, description="返回结果数量限制")),
    )

    SearchPrescriptionsInput = create_model(
        "SearchPrescriptionsInput",
        syndrome=(Optional[str], Field(None, description="证型名称")),
        symptoms=(Optional[list[str]], Field(None, description="症状列表")),
        prescription_name=(Optional[str], Field(None, description="方剂名称")),
        limit=(int, Field(10, description="返回结果数量限制")),
    )

    DiagnoseSyndromeInput = create_model(
        "DiagnoseSyndromeInput",
        symptoms=(dict, Field(..., description="症状对象")),
        pulse=(Optional[str], Field(None, description="脉象描述")),
        tongue=(Optional[dict], Field(None, description="舌象描述")),
    )

    GetContraindicationsInput = create_model(
        "GetContraindicationsInput",
        herbs=(list[str], Field(..., description="药材列表")),
        constitution=(Optional[str], Field(None, description="患者体质")),
        pregnancy=(bool, Field(False, description="是否孕期")),
    )

    SearchClassicsInput = create_model(
        "SearchClassicsInput",
        keywords=(list[str], Field(..., description="关键词列表")),
        books=(Optional[list[str]], Field(None, description="限定检索的书籍列表")),
        max_results=(int, Field(5, description="最多返回结果数")),
    )

    return [
        StructuredTool.from_function(
            coroutine=search_herbs,
            name="mcp_search_herbs",
            description="搜索中药材信息。根据药材名称或关键词查询药材的性味归经、功效主治等。",
            args_schema=SearchHerbsInput,
        ),
        StructuredTool.from_function(
            coroutine=search_prescriptions,
            name="mcp_search_prescriptions",
            description="搜索方剂信息。根据证型、症状或方剂名称查询方剂的组成、功效等。",
            args_schema=SearchPrescriptionsInput,
        ),
        StructuredTool.from_function(
            coroutine=diagnose_syndrome,
            name="mcp_diagnose_syndrome",
            description="根据症状进行中医辨证分析。综合症状、脉象、舌象等信息分析可能的证型。",
            args_schema=DiagnoseSyndromeInput,
        ),
        StructuredTool.from_function(
            coroutine=get_contraindications,
            name="mcp_get_contraindications",
            description="获取药物禁忌信息。检查药材配伍是否安全，包括十八反、十九畏等。",
            args_schema=GetContraindicationsInput,
        ),
        StructuredTool.from_function(
            coroutine=search_classics,
            name="mcp_search_classics",
            description="从中医古籍中检索相关论述。根据关键词在古籍库中搜索相关条文。",
            args_schema=SearchClassicsInput,
        ),
    ]


# ============== 便捷装饰器 ==============

def with_mcp_tools(toolkit: MCPToolkit):
    """
    装饰器：为节点函数添加MCP工具支持

    用法：
        @with_mcp_tools(my_toolkit)
        async def my_node(state: TCMAgentState) -> dict:
            # 可以通过 state.mcp_tools 访问工具
            result = await state.mcp_tools["search_herbs"](query="人参")
            return {"herbs": result}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(state, *args, **kwargs):
            # 注入工具到状态
            if hasattr(state, "__setitem__"):
                state["mcp_toolkit"] = toolkit
                state["mcp_tools"] = {
                    name: lambda **kwargs, n=name: toolkit.call_tool(n, **kwargs)
                    for name in toolkit.tool_names
                }

            result = await func(state, *args, **kwargs)
            return result

        return wrapper

    return decorator


__all__ = [
    "MCPToolWrapper",
    "MCPToolkit",
    "MCPPromptInjectionMiddleware",
    "create_input_model",
    "bind_mcp_tools_to_agent",
    "create_local_tcm_toolkit",
    "get_mcp_tools",
    "create_mock_mcp_tools",
    "with_mcp_tools",
    "LANGCHAIN_AVAILABLE",
]
