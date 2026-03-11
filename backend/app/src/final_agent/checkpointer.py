"""
LangGraph Checkpointer 配置

支持两种持久化方式：
1. PostgresSaver - 生产环境，会话持久化到数据库
2. MemorySaver - 开发环境，会话存储在内存中

2026-02-05: 新增 PostgresSaver 支持
"""

import os
from typing import Optional
from urllib.parse import quote_plus

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.src.utils import get_logger

logger = get_logger("checkpointer")

# 全局 checkpointer 实例
_checkpointer: Optional[BaseCheckpointSaver] = None

# 是否使用 PostgreSQL（通过环境变量控制）
USE_POSTGRES_CHECKPOINTER = os.getenv("USE_POSTGRES_CHECKPOINTER", "false").lower() == "true"


def _get_postgres_connection_string() -> str:
    """
    获取 PostgreSQL 连接字符串（psycopg3 格式）

    LangGraph PostgresSaver 需要 psycopg3 格式的连接字符串
    """
    from app.src.common.config.setting_config import settings

    encoded_password = quote_plus(settings.POSTGRESQL_PASSWORD)

    # psycopg3 格式（不带驱动前缀）
    return (
        f"postgresql://"
        f"{settings.POSTGRESQL_USER_NAME}:{encoded_password}@"
        f"{settings.POSTGRESQL_HOST}:{settings.POSTGRESQL_PORT}/"
        f"{settings.POSTGRESQL_DATABASE_NAME}"
    )


def _create_postgres_checkpointer() -> BaseCheckpointSaver:
    """
    创建 PostgreSQL Checkpointer

    使用 psycopg_pool 连接池
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        connection_string = _get_postgres_connection_string()

        # 创建连接池
        pool = ConnectionPool(
            conninfo=connection_string,
            min_size=5,
            max_size=20,
            open=True,  # 立即打开连接
        )

        # 创建 PostgresSaver
        checkpointer = PostgresSaver(pool)

        # 初始化表结构（如果不存在）
        checkpointer.setup()

        logger.info("PostgresSaver 初始化成功")
        return checkpointer

    except ImportError as e:
        logger.warning(f"无法导入 PostgresSaver 依赖: {e}，降级到 MemorySaver")
        return MemorySaver()
    except Exception as e:
        logger.error(f"PostgresSaver 初始化失败: {e}，降级到 MemorySaver")
        return MemorySaver()


def _create_async_postgres_checkpointer() -> BaseCheckpointSaver:
    """
    创建异步 PostgreSQL Checkpointer

    使用 psycopg_pool 异步连接池
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        connection_string = _get_postgres_connection_string()

        # 创建异步连接池
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            min_size=5,
            max_size=20,
            open=False,  # 延迟打开
        )

        # 创建 AsyncPostgresSaver
        checkpointer = AsyncPostgresSaver(pool)

        logger.info("AsyncPostgresSaver 初始化成功")
        return checkpointer

    except ImportError as e:
        logger.warning(f"无法导入 AsyncPostgresSaver 依赖: {e}，降级到 MemorySaver")
        return MemorySaver()
    except Exception as e:
        logger.error(f"AsyncPostgresSaver 初始化失败: {e}，降级到 MemorySaver")
        return MemorySaver()


def get_checkpointer(force_memory: bool = False) -> BaseCheckpointSaver:
    """
    获取 Checkpointer 实例（单例）

    Args:
        force_memory: 强制使用 MemorySaver（用于测试）

    Returns:
        BaseCheckpointSaver 实例
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    if force_memory:
        logger.info("使用 MemorySaver（强制内存模式）")
        _checkpointer = MemorySaver()
    elif USE_POSTGRES_CHECKPOINTER:
        logger.info("使用 PostgresSaver（生产模式）")
        _checkpointer = _create_postgres_checkpointer()
    else:
        logger.info("使用 MemorySaver（开发模式）")
        _checkpointer = MemorySaver()

    return _checkpointer


def reset_checkpointer():
    """
    重置 Checkpointer（用于测试）
    """
    global _checkpointer
    _checkpointer = None


async def setup_postgres_tables():
    """
    初始化 PostgreSQL 表结构

    在应用启动时调用
    """
    if not USE_POSTGRES_CHECKPOINTER:
        return

    try:
        checkpointer = get_checkpointer()
        if hasattr(checkpointer, 'setup'):
            checkpointer.setup()
            logger.info("PostgresSaver 表结构初始化完成")
    except Exception as e:
        logger.error(f"PostgresSaver 表结构初始化失败: {e}")
