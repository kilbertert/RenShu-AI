"""
P2 阶段：diagnose subgraph 末尾的病例落库节点

把 LangGraph state 持久化到 PostgreSQL。
异常被吞，不影响诊断结果返回。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.src.common.config.prosgresql_config import async_db_manager
from app.src.service.case_service import CaseService
from app.src.utils import get_logger

logger = get_logger("save_case_node")


async def save_case_node(state: dict[str, Any]) -> dict[str, Any]:
    """在 diagnose 流程结束后把病例写库。

    state 期望字段：user_id (str|UUID), conversation_id (str|UUID), thread_id (str)
    任何字段缺失或异常均不影响上层状态返回。
    """
    try:
        user_id_raw = state.get("user_id")
        conversation_id_raw = state.get("conversation_id")
        if not user_id_raw or not conversation_id_raw:
            logger.debug("save_case_node 跳过：缺少 user_id / conversation_id")
            return {}

        user_id = UUID(str(user_id_raw))
        conversation_id = UUID(str(conversation_id_raw))
        thread_id = state.get("thread_id")

        if async_db_manager.async_engine is None:
            logger.warning("save_case_node 跳过：DB 引擎未初始化")
            return {}

        async with async_db_manager.get_session() as session:
            service = CaseService(session)
            await service.save_case_from_state(
                user_id=user_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
                state=dict(state),
            )
    except Exception as exc:
        logger.error("save_case_node 异常: %s", exc, exc_info=True)

    return {}
