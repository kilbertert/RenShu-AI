"""
MCP Client Implementation for SmartTCM
MCP客户端，用于连接和调用MCP服务器

参考：https://modelcontextprotocol.io/
支持两种传输模式：
1. stdio: 子进程通信（本地MCP服务器）
2. SSE: Server-Sent Events（远程MCP服务器）
"""

import asyncio
import json
from typing import Any, Optional, Sequence, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from app.src.utils import get_logger

logger = get_logger("mcp_client")

# 检查 MCP 客户端库是否可用
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters
    from mcp.client.sse import sse_client
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    ClientSession = None
    logger.warning("MCP client library not installed. Install with: pip install mcp")


# ============== 数据结构 ==============

@dataclass
class MCPTool:
    """MCP工具定义"""
    name: str
    description: str
    input_schema: dict

    @classmethod
    def from_dict(cls, data: dict) -> "MCPTool":
        """从字典创建工具"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {})
        )


@dataclass
class MCPServerConfig:
    """MCP服务器配置"""
    name: str
    transport: str = "stdio"  # stdio 或 sse
    command: Optional[str] = None  # stdio模式：启动命令
    args: list[str] = field(default_factory=list)  # stdio模式：命令参数
    url: Optional[str] = None  # SSE模式：服务器URL
    timeout: int = 30  # 请求超时时间（秒）


@dataclass
class MCPToolResult:
    """工具调用结果"""
    content: list[dict]
    is_error: bool = False
    error_message: Optional[str] = None


# ============== MCP 客户端 ==============

class MCPClient:
    """
    MCP客户端

    用于连接MCP服务器并调用工具。支持stdio和SSE两种传输模式。
    """

    def __init__(self, config: MCPServerConfig):
        """
        初始化MCP客户端

        Args:
            config: 服务器配置
        """
        if not MCP_CLIENT_AVAILABLE:
            raise RuntimeError("MCP client library not available")

        self.config = config
        self._session: Optional[ClientSession] = None
        self._server_params: Optional[StdioServerParameters] = None
        self._close_callback: Optional[Callable] = None

    async def connect(self) -> bool:
        """
        连接到MCP服务器

        Returns:
            是否连接成功
        """
        try:
            if self.config.transport == "stdio":
                return await self._connect_stdio()
            elif self.config.transport == "sse":
                return await self._connect_sse()
            else:
                logger.error(f"Unknown transport: {self.config.transport}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False

    async def _connect_stdio(self) -> bool:
        """使用stdio模式连接"""
        if not self.config.command:
            logger.error("stdio mode requires command")
            return False

        self._server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args
        )

        # 创建stdio服务器连接
        from mcp.client.stdio import stdio_client

        stdio_context = stdio_client(self._server_params)
        read_stream, write_stream = await stdio_context.__aenter__()

        # 创建会话
        self._session = ClientSession(read_stream, write_stream)
        await self._session.initialize()

        # 保存关闭回调
        self._close_callback = lambda: asyncio.create_task(stdio_context.__aexit__(None, None, None))

        logger.info(f"Connected to MCP server via stdio: {self.config.name}")
        return True

    async def _connect_sse(self) -> bool:
        """使用SSE模式连接"""
        if not self.config.url:
            logger.error("sse mode requires url")
            return False

        # 创建SSE连接
        context = sse_client(self.config.url)
        read_stream, write_stream = await context.__aenter__()

        self._session = ClientSession(read_stream, write_stream)
        await self._session.initialize()

        # 保存关闭回调
        self._close_callback = lambda: asyncio.create_task(context.__aexit__(None, None, None))

        logger.info(f"Connected to MCP server via SSE: {self.config.url}")
        return True

    async def close(self):
        """关闭连接"""
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            self._session = None

        if self._close_callback:
            try:
                self._close_callback()
            except Exception as e:
                logger.warning(f"Error closing server connection: {e}")
            self._close_callback = None

    async def list_tools(self) -> list[MCPTool]:
        """
        列出服务器提供的所有工具

        Returns:
            工具列表
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self._session.list_tools()
            return [MCPTool.from_dict(t.model_dump()) for t in response.tools]
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    async def call_tool(
        self,
        name: str,
        arguments: dict
    ) -> MCPToolResult:
        """
        调用工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self._session.call_tool(name, arguments)

            contents = []
            is_error = False
            error_message = None

            for item in response.content:
                content_dict = {
                    "type": item.type,
                }
                if hasattr(item, "text"):
                    content_dict["text"] = item.text
                if hasattr(item, "data"):
                    content_dict["data"] = item.data
                if hasattr(item, "mimeType"):
                    content_dict["mimeType"] = item.mimeType
                contents.append(content_dict)

                # 检查是否是错误响应
                if item.type == "text":
                    try:
                        data = json.loads(item.text)
                        if "error" in data:
                            is_error = True
                            error_message = data.get("error")
                    except (json.JSONDecodeError, TypeError):
                        pass

            return MCPToolResult(
                content=contents,
                is_error=is_error,
                error_message=error_message
            )

        except Exception as e:
            logger.error(f"Failed to call tool {name}: {e}")
            return MCPToolResult(
                content=[],
                is_error=True,
                error_message=str(e)
            )

    async def list_resources(self) -> list[dict]:
        """列出服务器提供的资源"""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self._session.list_resources()
            return [r.model_dump() for r in response.resources]
        except Exception as e:
            logger.error(f"Failed to list resources: {e}")
            return []

    async def read_resource(self, uri: str) -> str:
        """读取资源内容"""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self._session.read_resource(uri)
            # 返回第一个内容块
            for item in response.contents:
                if hasattr(item, "text"):
                    return item.text
            return ""
        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {e}")
            return ""

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._session is not None

    @property
    def server_name(self) -> str:
        """服务器名称"""
        return self.config.name


# ============== 连接池 ==============

class MCPClientPool:
    """
    MCP客户端连接池

    管理多个MCP服务器连接，支持工具路由和负载均衡。
    """

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._tool_index: dict[str, list[str]] = {}  # 工具名 -> 服务器名列表

    def add_server(self, config: MCPServerConfig) -> None:
        """
        添加服务器配置

        Args:
            config: 服务器配置
        """
        self._clients[config.name] = MCPClient(config)
        logger.info(f"Added MCP server config: {config.name}")

    async def connect_all(self) -> dict[str, bool]:
        """
        连接所有配置的服务器

        Returns:
            服务器名 -> 连接状态的映射
        """
        results = {}
        for name, client in self._clients.items():
            try:
                results[name] = await client.connect()
                if results[name]:
                    # 构建工具索引
                    tools = await client.list_tools()
                    for tool in tools:
                        if tool.name not in self._tool_index:
                            self._tool_index[tool.name] = []
                        self._tool_index[tool.name].append(name)
            except Exception as e:
                logger.error(f"Failed to connect {name}: {e}")
                results[name] = False

        return results

    async def close_all(self) -> None:
        """关闭所有连接"""
        for client in self._clients.values():
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        self._clients.clear()
        self._tool_index.clear()

    def get_client(self, server_name: str) -> Optional[MCPClient]:
        """
        获取指定服务器的客户端

        Args:
            server_name: 服务器名称

        Returns:
            MCP客户端，如果不存在则返回None
        """
        return self._clients.get(server_name)

    def get_client_for_tool(self, tool_name: str) -> Optional[MCPClient]:
        """
        获取提供指定工具的客户端

        Args:
            tool_name: 工具名称

        Returns:
            MCP客户端，如果找不到则返回None
        """
        server_names = self._tool_index.get(tool_name, [])
        if server_names:
            return self._clients.get(server_names[0])
        return None

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
        server_name: Optional[str] = None
    ) -> MCPToolResult:
        """
        调用工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            server_name: 指定服务器名（可选）

        Returns:
            工具执行结果
        """
        client = None

        if server_name:
            client = self.get_client(server_name)
        else:
            client = self.get_client_for_tool(tool_name)

        if not client:
            return MCPToolResult(
                content=[],
                is_error=True,
                error_message=f"No client found for tool: {tool_name}"
            )

        return await client.call_tool(tool_name, arguments)

    async def list_all_tools(self) -> dict[str, list[MCPTool]]:
        """
        列出所有服务器的工具

        Returns:
            服务器名 -> 工具列表的映射
        """
        result = {}
        for name, client in self._clients.items():
            if client.is_connected:
                try:
                    result[name] = await client.list_tools()
                except Exception as e:
                    logger.error(f"Failed to list tools for {name}: {e}")
                    result[name] = []
        return result

    @property
    def servers(self) -> list[str]:
        """获取所有服务器名称"""
        return list(self._clients.keys())

    @property
    def connected_servers(self) -> list[str]:
        """获取已连接的服务器名称"""
        return [name for name, client in self._clients.items() if client.is_connected]


# ============== 上下文管理器 ==============

@asynccontextmanager
async def mcp_client_context(config: MCPServerConfig):
    """
    MCP客户端上下文管理器

    用法：
        async with mcp_client_context(config) as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_herbs", {"query": "人参"})

    Args:
        config: 服务器配置

    Yields:
        MCPClient实例
    """
    client = MCPClient(config)
    try:
        connected = await client.connect()
        if not connected:
            raise RuntimeError(f"Failed to connect to MCP server: {config.name}")
        yield client
    finally:
        await client.close()


@asynccontextmanager
async def mcp_pool_context(configs: list[MCPServerConfig]):
    """
    MCP连接池上下文管理器

    用法：
        configs = [config1, config2]
        async with mcp_pool_context(configs) as pool:
            result = await pool.call_tool("search_herbs", {"query": "人参"})

    Args:
        configs: 服务器配置列表

    Yields:
        MCPClientPool实例
    """
    pool = MCPClientPool()
    for config in configs:
        pool.add_server(config)

    try:
        await pool.connect_all()
        yield pool
    finally:
        await pool.close_all()


# ============== 内置本地服务器客户端 ==============

async def create_local_tcm_client() -> Optional[MCPClient]:
    """
    创建连接到本地TCM MCP服务器的客户端

    本地服务器通过子进程运行，使用stdio通信。

    Returns:
        MCPClient实例，如果创建失败则返回None
    """
    if not MCP_CLIENT_AVAILABLE:
        return None

    # 配置本地TCM服务器
    # 假设服务器启动命令为: python -m app.src.agent.mcp.server
    config = MCPServerConfig(
        name="local-tcm",
        transport="stdio",
        command="python",
        args=["-m", "app.src.agent.mcp.server"],
        timeout=30
    )

    client = MCPClient(config)
    success = await client.connect()

    if success:
        return client
    else:
        return None


# ============== 快捷函数 ==============

def create_pool() -> MCPClientPool:
    """创建新的连接池"""
    return MCPClientPool()


__all__ = [
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
]
