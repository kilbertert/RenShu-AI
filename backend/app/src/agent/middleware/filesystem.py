"""
文件系统中间件

实现大结果自动驱逐功能：
- 监控工具调用结果大小
- 超过阈值时自动写入文件
- 上下文中只保留摘要和文件引用

参考 DeepAgents FilesystemMiddleware 设计
"""

import json
import os
import hashlib
from typing import Any, Optional, Dict, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .base import BaseMiddleware, MiddlewareConfig
from app.src.utils import get_logger

logger = get_logger("filesystem_middleware")


@dataclass
class FilesystemConfig(MiddlewareConfig):
    """文件系统中间件配置"""

    # Token 阈值（超过此值触发驱逐）
    tool_token_limit_before_evict: int = 15000

    # 摘要保留的 token 数
    summary_token_limit: int = 500

    # 文件存储根目录
    storage_root: str = "./data/agent_files"

    # 路由配置（不同类型的结果存储到不同目录）
    routes: Dict[str, str] = field(default_factory=lambda: {
        "kg_results": "kg_results",      # 知识图谱查询结果
        "case_library": "case_library",  # 医案库
        "classics": "classics",          # 古籍
        "patient_history": "patient_history",  # 患者档案
    })

    # 是否持久化（True: 写入磁盘, False: 仅内存）
    persist_to_disk: bool = True

    # 文件保留时间（秒），0 表示永久保留
    file_ttl: int = 3600  # 1小时

    # 启用的工具名称（只对这些工具的结果进行驱逐）
    enabled_tools: List[str] = field(default_factory=lambda: [
        "kg_syndrome_search",
        "kg_organ_query",
        "case_vector_search",
        "classics_search",
        "web_search",
    ])


class TCMFilesystemMiddleware(BaseMiddleware):
    """
    TCM 文件系统中间件

    功能：
    1. 监控工具调用结果大小
    2. 超过阈值时自动写入文件
    3. 上下文中只保留摘要和文件引用
    4. 提供 read_file 工具供 Agent 按需读取

    工作流程：
    1. wrap_tool_call() 包装工具调用
    2. 工具执行后检查结果大小
    3. 超过阈值时：
       - 将完整结果写入文件
       - 生成摘要
       - 返回摘要 + 文件引用
    """

    def __init__(self, config: Optional[FilesystemConfig] = None):
        """
        初始化文件系统中间件

        Args:
            config: 文件系统配置
        """
        super().__init__(config or FilesystemConfig(
            name="TCMFilesystemMiddleware",
            priority=3,  # 在 ContextManager 之后执行
        ))
        self.fs_config: FilesystemConfig = self.config

        # 内存文件存储（用于非持久化模式）
        self._memory_store: Dict[str, Any] = {}

        # 确保存储目录存在
        if self.fs_config.persist_to_disk:
            self._ensure_storage_dirs()

    def _ensure_storage_dirs(self):
        """确保存储目录存在"""
        root = Path(self.fs_config.storage_root)
        root.mkdir(parents=True, exist_ok=True)

        for route_name, route_path in self.fs_config.routes.items():
            (root / route_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"文件存储目录已创建: {root}")

    def _estimate_tokens(self, content: Any) -> int:
        """
        估算内容的 token 数

        简单估算：中文约 1.5 字符/token，英文约 4 字符/token
        """
        if content is None:
            return 0

        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False, default=str)

        # 简单估算：平均 2 字符/token
        return len(text) // 2

    def _generate_file_id(self, tool_name: str, content: Any) -> str:
        """生成文件 ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(
            json.dumps(content, ensure_ascii=False, default=str).encode()
        ).hexdigest()[:8]
        return f"{tool_name}_{timestamp}_{content_hash}"

    def _get_route_for_tool(self, tool_name: str) -> str:
        """根据工具名称获取存储路由"""
        # 根据工具名称映射到路由
        tool_route_map = {
            "kg_syndrome_search": "kg_results",
            "kg_organ_query": "kg_results",
            "case_vector_search": "case_library",
            "classics_search": "classics",
            "web_search": "kg_results",  # 网络搜索结果也放到 kg_results
        }
        return tool_route_map.get(tool_name, "kg_results")

    def _generate_summary(self, content: Any, tool_name: str) -> str:
        """
        生成内容摘要

        Args:
            content: 原始内容
            tool_name: 工具名称

        Returns:
            摘要文本
        """
        if isinstance(content, dict):
            # 提取关键信息
            summary_parts = []

            # 证型查询结果
            if "syndromes" in content:
                syndromes = content["syndromes"]
                summary_parts.append(f"找到 {len(syndromes)} 个相关证型")
                if syndromes:
                    top_names = [s.get("name", "未知") for s in syndromes[:3]]
                    summary_parts.append(f"主要证型: {', '.join(top_names)}")

            # 医案查询结果
            if "similar_cases" in content:
                cases = content["similar_cases"]
                summary_parts.append(f"找到 {len(cases)} 个相似医案")
                if cases:
                    top_syndromes = [c.get("syndrome", "未知") for c in cases[:3]]
                    summary_parts.append(f"主要证型: {', '.join(top_syndromes)}")

            # 古籍查询结果
            if "citations" in content:
                citations = content["citations"]
                summary_parts.append(f"找到 {len(citations)} 条古籍引用")
                if citations:
                    books = list(set(c.get("book", "未知") for c in citations[:5]))
                    summary_parts.append(f"来源: {', '.join(books)}")

            # 方剂查询结果
            if "prescriptions" in content:
                prescriptions = content["prescriptions"]
                summary_parts.append(f"找到 {len(prescriptions)} 个相关方剂")
                if prescriptions:
                    top_names = [p.get("name", "未知") for p in prescriptions[:3]]
                    summary_parts.append(f"主要方剂: {', '.join(top_names)}")

            # 通用处理
            if not summary_parts:
                keys = list(content.keys())[:5]
                summary_parts.append(f"包含字段: {', '.join(keys)}")

            return "\n".join(summary_parts)

        elif isinstance(content, list):
            return f"列表结果，共 {len(content)} 项"

        elif isinstance(content, str):
            # 截取前 200 字符
            return content[:200] + ("..." if len(content) > 200 else "")

        else:
            return f"结果类型: {type(content).__name__}"

    def _write_file(self, file_id: str, route: str, content: Any) -> str:
        """
        写入文件

        Args:
            file_id: 文件 ID
            route: 存储路由
            content: 文件内容

        Returns:
            文件路径
        """
        file_path = f"/{route}/{file_id}.json"

        if self.fs_config.persist_to_disk:
            # 写入磁盘
            full_path = Path(self.fs_config.storage_root) / route / f"{file_id}.json"
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"文件已写入磁盘: {full_path}")
        else:
            # 写入内存
            self._memory_store[file_path] = content
            logger.debug(f"文件已写入内存: {file_path}")

        return file_path

    def read_file(self, file_path: str) -> Any:
        """
        读取文件

        Args:
            file_path: 文件路径（如 /kg_results/xxx.json）

        Returns:
            文件内容
        """
        # 安全校验：防止路径遍历攻击
        clean_path = file_path.lstrip("/")
        
        # 检查是否包含路径遍历字符
        if ".." in clean_path or clean_path.startswith("/"):
            logger.warning(f"路径安全校验失败: {file_path}")
            return None
        
        # 检查路径是否在允许的路由中
        path_parts = clean_path.split("/")
        if path_parts and path_parts[0] not in self.fs_config.routes.values():
            logger.warning(f"路径不在允许的路由中: {file_path}")
            return None
        
        if self.fs_config.persist_to_disk:
            # 从磁盘读取
            full_path = Path(self.fs_config.storage_root) / clean_path
            
            # 确保解析后的路径仍在存储根目录下
            try:
                resolved_path = full_path.resolve()
                storage_root_resolved = Path(self.fs_config.storage_root).resolve()
                if not str(resolved_path).startswith(str(storage_root_resolved)):
                    logger.warning(f"路径遍历攻击检测: {file_path}")
                    return None
            except Exception as e:
                logger.warning(f"路径解析失败: {e}")
                return None
            
            if full_path.exists():
                with open(full_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"文件不存在: {full_path}")
                return None
        else:
            # 从内存读取
            return self._memory_store.get(file_path)

    def wrap_tool_call(
        self,
        tool_call: Callable,
        tool_name: str,
        state: Dict[str, Any]
    ) -> Callable:
        """
        包装工具调用，实现大结果自动驱逐

        Args:
            tool_call: 原始工具调用函数
            tool_name: 工具名称
            state: 当前状态

        Returns:
            包装后的工具调用函数
        """
        # 检查是否需要对此工具进行驱逐
        if tool_name not in self.fs_config.enabled_tools:
            return tool_call

        async def wrapped_tool_call(*args, **kwargs):
            """包装后的工具调用"""
            # 执行原始工具调用
            result = await tool_call(*args, **kwargs)

            # 估算结果大小
            token_count = self._estimate_tokens(result)

            logger.debug(f"工具 {tool_name} 结果大小: {token_count} tokens")

            # 检查是否需要驱逐
            if token_count > self.fs_config.tool_token_limit_before_evict:
                logger.info(
                    f"工具 {tool_name} 结果超过阈值 "
                    f"({token_count} > {self.fs_config.tool_token_limit_before_evict})，"
                    f"触发驱逐"
                )

                # 生成文件 ID 和路由
                file_id = self._generate_file_id(tool_name, result)
                route = self._get_route_for_tool(tool_name)

                # 写入文件
                file_path = self._write_file(file_id, route, result)

                # 生成摘要
                summary = self._generate_summary(result, tool_name)

                # 返回摘要 + 文件引用
                evicted_result = {
                    "_evicted": True,
                    "_file_path": file_path,
                    "_token_count": token_count,
                    "_summary": summary,
                    "message": f"结果已保存到文件 {file_path}，以下是摘要：\n{summary}\n\n如需完整内容，请使用 read_file 工具读取。"
                }

                return evicted_result

            return result

        return wrapped_tool_call

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前：注入文件系统工具说明
        """
        # 可以在这里注入 read_file 工具的说明
        return None

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后：清理过期文件
        """
        if self.fs_config.file_ttl > 0 and self.fs_config.persist_to_disk:
            self._cleanup_expired_files()
        return None

    def _cleanup_expired_files(self):
        """清理过期文件"""
        import time

        root = Path(self.fs_config.storage_root)
        current_time = time.time()

        for route_path in self.fs_config.routes.values():
            route_dir = root / route_path
            if not route_dir.exists():
                continue

            for file_path in route_dir.glob("*.json"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > self.fs_config.file_ttl:
                    file_path.unlink()
                    logger.debug(f"已清理过期文件: {file_path}")


# 工厂函数
def get_tcm_filesystem_middleware(
    token_limit: int = 15000,
    storage_root: str = "./data/agent_files",
    persist_to_disk: bool = True,
) -> TCMFilesystemMiddleware:
    """
    获取 TCM 文件系统中间件实例

    Args:
        token_limit: 触发驱逐的 token 阈值
        storage_root: 文件存储根目录
        persist_to_disk: 是否持久化到磁盘

    Returns:
        TCMFilesystemMiddleware 实例
    """
    config = FilesystemConfig(
        name="TCMFilesystemMiddleware",
        priority=3,
        tool_token_limit_before_evict=token_limit,
        storage_root=storage_root,
        persist_to_disk=persist_to_disk,
    )
    return TCMFilesystemMiddleware(config)


# 创建 read_file 工具供 Agent 使用
def create_read_file_tool(middleware: TCMFilesystemMiddleware):
    """
    创建 read_file 工具

    Args:
        middleware: 文件系统中间件实例

    Returns:
        read_file 工具函数
    """
    from langchain.tools import tool

    @tool
    async def read_file(file_path: str) -> Dict[str, Any]:
        """
        读取之前保存的文件内容

        当工具返回结果过大被驱逐到文件时，使用此工具读取完整内容。

        Args:
            file_path: 文件路径，如 /kg_results/xxx.json

        Returns:
            文件内容
        """
        content = middleware.read_file(file_path)
        if content is None:
            return {"error": f"文件不存在: {file_path}"}
        return content

    return read_file
