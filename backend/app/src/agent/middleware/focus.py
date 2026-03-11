"""
Focus 上下文压缩中间件

基于 arXiv:2601.07190 - 主动上下文压缩
实现 Sawtooth 模式，每 10-15 次工具调用压缩一次

关键发现：
- 激进的压缩提示是关键
- 22.7% token 节省，准确率不变
- 探索-重度任务收益最大
"""

import logging
from typing import Any, Optional, Dict
from dataclasses import dataclass

from .base import BaseMiddleware, MiddlewareConfig
from ..context.focus_manager import FocusContextManager, FocusConfig


logger = logging.getLogger(__name__)


@dataclass
class FocusMiddlewareConfig(MiddlewareConfig):
    """Focus 中间件配置"""
    # 压缩间隔（工具调用次数）
    compress_interval: int = 12
    # 激进模式（更频繁的压缩）
    aggressive: bool = True
    # 自动压缩（达到阈值时自动触发）
    auto_compress: bool = True
    # 提醒阈值（超过此次数发送提醒）
    reminder_threshold: int = 15
    # 最大阶段数（超过此数量强制压缩）
    max_phases: int = 10


class FocusMiddleware(BaseMiddleware):
    """
    Focus 上下文压缩中间件

    功能：
    1. 跟踪工具调用次数
    2. 在达到阈值时提醒或自动压缩
    3. 管理知识块注入
    4. 提供压缩统计

    使用场景：
    - 长时间运行的诊断会话
    - 多轮工具调用的复杂任务
    - 需要节省 token 的场景
    """

    def __init__(self, config: Optional[FocusMiddlewareConfig] = None, llm=None):
        """
        初始化 Focus 中间件

        Args:
            config: 中间件配置
            llm: LLM 实例（用于生成总结）
        """
        config = config or FocusMiddlewareConfig(
            enabled=True,
            priority=8,  # P2: 上下文管理
            name="FocusMiddleware",
            compress_interval=12,
            aggressive=True,
            auto_compress=False,  # 默认不自动压缩，由用户或系统触发
            reminder_threshold=15,
            max_phases=10
        )

        super().__init__(config)

        # 创建 Focus 配置
        focus_config = FocusConfig(
            compress_interval=config.compress_interval,
            aggressive=config.aggressive,
            auto_compress=config.auto_compress,
            max_phases=config.max_phases,
            reminder_threshold=config.reminder_threshold
        )

        # 创建 Focus 管理器
        self.focus_manager = FocusContextManager(config=focus_config, llm=llm)

        # 每个会话的状态
        self._session_states: Dict[str, Dict] = {}

    def _get_session_state(self, session_id: str) -> Dict:
        """获取或创建会话状态"""
        if session_id not in self._session_states:
            self._session_states[session_id] = {
                "tool_call_count": 0,
                "current_phase": None,
                "context_messages": [],
                "phase_count": 0
            }
        return self._session_states[session_id]

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前执行

        功能：
        1. 检查是否需要压缩提醒
        2. 注入知识上下文
        """
        session_id = self._get_state_value(state, "session_id", "default")
        session_state = self._get_session_state(session_id)

        # 检查是否需要压缩提醒
        reminder = self.focus_manager.get_compression_reminder(
            session_state["tool_call_count"]
        )

        updates = {}

        if reminder:
            logger.info(
                f"[Focus] 会话 {session_id}: 工具调用 {session_state['tool_call_count']} 次，"
                f"发送压缩提醒"
            )
            updates["focus_reminder"] = reminder

        # 注入知识上下文
        context_messages = self._get_state_value(state, "messages", [])
        if context_messages:
            enhanced_messages = self.focus_manager.inject_knowledge_to_context(
                context_messages,
                position="after_system"
            )
            if len(enhanced_messages) != len(context_messages):
                updates["messages"] = enhanced_messages
                logger.debug(f"[Focus] 注入知识上下文到会话 {session_id}")

        return updates if updates else None

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后执行

        功能：
        1. 更新工具调用计数
        2. 检查是否需要自动压缩
        3. 返回压缩统计
        """
        session_id = self._get_state_value(state, "session_id", "default")
        session_state = self._get_session_state(session_id)

        updates = {}

        # 更新工具调用计数
        tool_calls = self._get_state_value(state, "tool_calls", [])
        if tool_calls:
            session_state["tool_call_count"] += len(tool_calls)
            logger.debug(
                f"[Focus] 会话 {session_id}: 工具调用累计 "
                f"{session_state['tool_call_count']} 次"
            )

        # 检查是否应该自动压缩
        if self.config.auto_compress:
            should_compress = self.focus_manager.should_compress(
                session_state["tool_call_count"]
            )

            if should_compress:
                logger.info(f"[Focus] 会话 {session_id}: 触发自动压缩")
                updates["focus_auto_compress_triggered"] = True
                # 这里可以触发自动压缩逻辑
                # 实际压缩需要调用 complete_phase

        # 返回统计信息
        stats = self.focus_manager.get_stats()
        updates["focus_stats"] = stats

        return updates if updates else None

    def start_phase(
        self,
        session_id: str,
        name: str,
        goal: str,
        messages: list
    ) -> tuple:
        """
        开始新的 Focus 阶段

        Args:
            session_id: 会话 ID
            name: 阶段名称
            goal: 阶段目标
            messages: 上下文消息

        Returns:
            (系统指令消息, 阶段开始索引)
        """
        session_state = self._get_session_state(session_id)
        session_state["phase_count"] += 1

        instruction, start_idx = self.focus_manager.start_phase(
            name=name,
            goal=goal,
            context_messages=messages
        )

        session_state["current_phase"] = name
        logger.info(f"[Focus] 会话 {session_id}: 开始阶段 '{name}'")

        return instruction, start_idx

    async def complete_phase(
        self,
        session_id: str,
        result: str,
        messages: list,
        llm=None
    ) -> Dict[str, Any]:
        """
        完成当前 Focus 阶段并执行压缩

        Args:
            session_id: 会话 ID
            result: 阶段结果
            messages: 上下文消息
            llm: LLM 实例

        Returns:
            压缩操作结果
        """
        session_state = self._get_session_state(session_id)

        compression_result = await self.focus_manager.complete_phase(
            result=result,
            context_messages=messages,
            llm=llm or self.focus_manager.llm
        )

        if compression_result.get("action") == "compress":
            session_state["current_phase"] = None
            logger.info(
                f"[Focus] 会话 {session_id}: 压缩完成，"
                f"删除 {compression_result['messages_dropped']} 条消息"
            )

        return compression_result

    def get_stats(self, session_id: str = None) -> Dict[str, Any]:
        """
        获取统计信息

        Args:
            session_id: 会话 ID（可选）

        Returns:
            统计信息字典
        """
        stats = self.focus_manager.get_stats()

        if session_id and session_id in self._session_states:
            session_state = self._session_states[session_id]
            stats["session"] = {
                "tool_call_count": session_state["tool_call_count"],
                "current_phase": session_state["current_phase"],
                "phase_count": session_state["phase_count"]
            }

        return stats

    def reset_session(self, session_id: str):
        """重置会话状态"""
        if session_id in self._session_states:
            del self._session_states[session_id]
            logger.info(f"[Focus] 重置会话 {session_id}")


def get_focus_middleware(
    compress_interval: int = 12,
    aggressive: bool = True,
    auto_compress: bool = False,
    llm=None
) -> Optional[FocusMiddleware]:
    """
    创建 Focus 中间件的便捷函数

    Args:
        compress_interval: 压缩间隔（工具调用次数）
        aggressive: 激进模式
        auto_compress: 自动压缩
        llm: LLM 实例

    Returns:
        FocusMiddleware 实例
    """
    try:
        config = FocusMiddlewareConfig(
            enabled=True,
            priority=8,
            name="FocusMiddleware",
            compress_interval=compress_interval,
            aggressive=aggressive,
            auto_compress=auto_compress,
            reminder_threshold=compress_interval + 3
        )
        return FocusMiddleware(config=config, llm=llm)
    except Exception as e:
        logger.warning(f"无法创建 Focus 中间件: {e}")
        return None
