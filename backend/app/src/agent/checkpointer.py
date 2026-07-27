"""LangGraph Checkpointer 生命周期管理。"""

from typing import Optional
from urllib.parse import quote_plus

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.src.common.config.setting_config import settings
from app.src.utils import get_logger

logger = get_logger("checkpointer")

_checkpointer: Optional[BaseCheckpointSaver] = None
_postgres_pool = None


def _get_postgres_connection_string() -> str:
    encoded_password = quote_plus(settings.POSTGRESQL_PASSWORD)
    return (
        "postgresql://"
        f"{settings.POSTGRESQL_USER_NAME}:{encoded_password}@"
        f"{settings.POSTGRESQL_HOST}:{settings.POSTGRESQL_PORT}/"
        f"{settings.POSTGRESQL_DATABASE_NAME}"
    )


async def initialize_checkpointer() -> BaseCheckpointSaver:
    """在应用启动阶段打开连接池并初始化 checkpoint 表。"""
    global _checkpointer, _postgres_pool

    if _checkpointer is not None:
        return _checkpointer

    if not settings.USE_POSTGRES_CHECKPOINTER:
        _checkpointer = MemorySaver()
        logger.warning("LangGraph 使用内存 Checkpointer；进程重启后状态会丢失")
        return _checkpointer

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=_get_postgres_connection_string(),
        min_size=settings.CHECKPOINTER_POOL_MIN_SIZE,
        max_size=settings.CHECKPOINTER_POOL_MAX_SIZE,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    await pool.open()
    await pool.wait()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    _postgres_pool = pool
    _checkpointer = checkpointer
    logger.info("AsyncPostgresSaver 初始化完成，多轮状态可跨进程恢复")
    return checkpointer


def get_checkpointer(force_memory: bool = False) -> BaseCheckpointSaver:
    """返回已初始化的 Checkpointer；持久模式禁止静默降级。"""
    global _checkpointer

    if force_memory:
        if _checkpointer is None:
            _checkpointer = MemorySaver()
        return _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    if not settings.USE_POSTGRES_CHECKPOINTER:
        _checkpointer = MemorySaver()
        return _checkpointer

    raise RuntimeError("PostgreSQL Checkpointer 尚未初始化，请先执行 initialize_checkpointer()")


async def close_checkpointer() -> None:
    """关闭持久化连接池并释放单例。"""
    global _checkpointer, _postgres_pool
    if _postgres_pool is not None:
        await _postgres_pool.close()
    _postgres_pool = None
    _checkpointer = None


def reset_checkpointer() -> None:
    """测试辅助：仅重置未持有连接池的实例。"""
    global _checkpointer
    if _postgres_pool is not None:
        raise RuntimeError("请先 await close_checkpointer() 再重置")
    _checkpointer = None


async def setup_postgres_tables() -> None:
    """兼容旧调用名。"""
    await initialize_checkpointer()
