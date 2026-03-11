"""
MCP (Model Context Protocol) Module for SmartTCM
基于 Model Context Protocol 标准的中医知识服务模块

这个模块提供：
1. Server: MCP服务器实现（backend/app/src/agent/mcp/server.py）
2. Client: MCP客户端实现（backend/app/src/agent/mcp/client.py）
3. Adapter: LangChain/LangGraph适配器（backend/app/src/agent/mcp/adapter.py）

使用示例：

# 1. 作为独立服务器运行
python -m app.src.agent.mcp.server

# 2. 在代码中使用MCP工具
from app.src.agent.mcp import create_mock_mcp_tools

tools = await create_mock_mcp_tools()
# 将工具绑定到LangGraph Agent

# 3. 使用MCP客户端
from app.src.agent.mcp import MCPClient, MCPServerConfig

config = MCPServerConfig(name="tcm", command="python", args=["-m", "tcm_server"])
client = MCPClient(config)
await client.connect()
result = await client.call_tool("search_herbs", {"query": "人参"})

参考：https://modelcontextprotocol.io/
符合 MCP Industry Standard (Linux Foundation AAIF)
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "SmartTCM Team"

# 从各个子模块导出公共接口

# ============== Server 导出 ==============
from .server import (
    # 工具定义
    TCM_TOOLS,
    # 执行器
    TCMToolExecutor,
    TCMToolContext,
    # 服务器创建
    create_mcp_server,
    create_mcp_server_with_executor,
    run_mcp_server,
    get_tool_executor,
    # 可用性标志
    MCP_AVAILABLE as SERVER_MCP_AVAILABLE,
)

# ============== Client 导出 ==============
from .client import (
    # 客户端
    MCPClient,
    MCPClientPool,
    # 数据结构
    MCPTool,
    MCPServerConfig,
    MCPToolResult,
    # 上下文管理器
    mcp_client_context,
    mcp_pool_context,
    # 便捷函数
    create_local_tcm_client,
    create_pool,
    # 可用性标志
    MCP_CLIENT_AVAILABLE,
)

# ============== Adapter 导出 ==============
from .adapter import (
    # 工具包装
    MCPToolWrapper,
    MCPToolkit,
    # 中间件
    MCPPromptInjectionMiddleware,
    # 工具函数
    create_input_model,
    bind_mcp_tools_to_agent,
    create_local_tcm_toolkit,
    get_mcp_tools,
    create_mock_mcp_tools,
    with_mcp_tools,
    # 可用性标志
    LANGCHAIN_AVAILABLE,
)


# ============== 统一的可用性标志 ==============

def is_mcp_available() -> bool:
    """
    检查MCP库是否可用

    Returns:
        bool: 是否可用
    """
    return SERVER_MCP_AVAILABLE or MCP_CLIENT_AVAILABLE


def is_full_stack_available() -> bool:
    """
    检查完整的MCP技术栈是否可用

    需要：
    1. MCP服务器库
    2. MCP客户端库
    3. LangChain

    Returns:
        bool: 是否完整可用
    """
    return SERVER_MCP_AVAILABLE and MCP_CLIENT_AVAILABLE and LANGCHAIN_AVAILABLE


# ============== 便捷入口函数 ==============

async def get_available_tcm_tools(
    use_mock: bool = True,
    server_configs: list = None
) -> list:
    """
    获取可用的TCM工具

    Args:
        use_mock: 如果无法连接真实MCP服务器，是否使用模拟工具
        server_configs: MCP服务器配置列表

    Returns:
        可用的LangChain工具列表
    """
    if is_full_stack_available() and server_configs:
        # 尝试连接真实MCP服务器
        try:
            return await get_mcp_tools(server_configs)
        except Exception as e:
            from app.src.utils import get_logger
            logger = get_logger("mcp")
            logger.warning(f"Failed to connect to MCP servers: {e}, falling back to mock tools")

    if use_mock:
        return create_mock_mcp_tools()

    return []


def get_tool_executor_instance() -> TCMToolExecutor:
    """
    获取工具执行器实例

    Returns:
        TCMToolExecutor实例
    """
    return get_tool_executor()


# ============== 模块信息 ==============

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    # 可用性检查
    "is_mcp_available",
    "is_full_stack_available",
    # Server
    "TCM_TOOLS",
    "TCMToolExecutor",
    "TCMToolContext",
    "create_mcp_server",
    "create_mcp_server_with_executor",
    "run_mcp_server",
    "get_tool_executor",
    "SERVER_MCP_AVAILABLE",
    # Client
    "MCPClient",
    "MCPClientPool",
    "MCPTool",
    "MCPServerConfig",
    "MCPToolResult",
    "mcp_client_context",
    "mcp_pool_context",
    "create_local_tcm_client",
    "create_pool",
    "MCP_CLIENT_AVAILABLE",
    # Adapter
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
    # 便捷入口
    "get_available_tcm_tools",
    "get_tool_executor_instance",
]


# ============== 模块初始化日志 ==============

from app.src.utils import get_logger
logger = get_logger("mcp")

logger.info(f"SmartTCM MCP Module v{__version__}")
logger.info(f"MCP Server: {'Available' if SERVER_MCP_AVAILABLE else 'Not Available'}")
logger.info(f"MCP Client: {'Available' if MCP_CLIENT_AVAILABLE else 'Not Available'}")
logger.info(f"LangChain Integration: {'Available' if LANGCHAIN_AVAILABLE else 'Not Available'}")

if not is_mcp_available():
    logger.warning("MCP library not installed. Install with: pip install mcp")
    logger.warning("Falling back to mock tools for development")
