"""
Focus 上下文管理中间件 (P5)

统一替代原先 3 个上下文中间件：
- TCMContextManagerMiddleware (P6) → 用户画像注入 + token 监控
- SummarizationMiddleware (P7) → 对话摘要
- FocusMiddleware (P8) → Sawtooth 压缩

现在 FocusContextMiddleware 是唯一的上下文管理中间件。

触发条件（双重）：
1. 工具调用次数达阈值 (Sawtooth 模式)
2. Token 使用超过阈值
"""

import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from .base import BaseMiddleware, MiddlewareConfig
from ..context.focus_engine import FocusContextEngine, FocusConfig
from ..utils.tcm_utils import estimate_messages_tokens, get_message_content, get_message_role

logger = logging.getLogger(__name__)


@dataclass
class FocusContextMiddlewareConfig(MiddlewareConfig):
    """Focus 上下文中间件配置"""
    # Sawtooth 参数
    compress_interval: int = 12
    aggressive: bool = True
    auto_compress: bool = True
    reminder_threshold: int = 15

    # Token 限制（从 ContextManagerConfig 吸收）
    max_tokens: int = 8000
    warning_threshold: float = 0.7

    # 用户画像注入（从 context_manager.py 吸收）
    enable_profile_injection: bool = True
    enable_environment_context: bool = True

    # 工具裁剪
    max_tokens_per_tool: int = 300
    max_total_tool_tokens: int = 2000

    # 总结-裁剪策略（医疗场景）
    enable_summarization: bool = False
    summarize_before_trim: bool = False
    keep_recent_messages: int = 10


class FocusContextMiddleware(BaseMiddleware):
    """
    统一上下文管理中间件 (P5)

    功能（整合 3 个旧中间件 + 学习洞察注入）：
    1. 注入用户画像和环境上下文（from ContextManager）
    2. 注入学习洞察（反思经验、反馈统计、进化建议 from SelfLearner）
    3. 注入知识块 (Focus Sawtooth)
    4. 检查 Sawtooth 压缩触发（工具调用次数 OR token 阈值）
    5. 如触发 → 执行压缩 (tool_trim + priority_filter + summarize)
    6. 跟踪工具调用计数和阶段进度

    上下文工程 vs 记忆工程的分工：
    - LearningMiddleware (P30) = 记忆工程: "记什么、怎么记"
    - 本中间件 (P5) = 上下文工程: "给模型看什么、怎么拼进 prompt"
    """

    def __init__(
        self,
        config: Optional[FocusContextMiddlewareConfig] = None,
        llm=None,
        learner=None,
    ):
        config = config or FocusContextMiddlewareConfig(
            enabled=True,
            priority=5,
            name="FocusContextMiddleware",
        )
        super().__init__(config)

        # 创建 Focus 引擎
        focus_config = FocusConfig(
            compress_interval=config.compress_interval,
            aggressive=config.aggressive,
            auto_compress=config.auto_compress,
            max_tokens=config.max_tokens,
            warning_threshold=config.warning_threshold,
            reminder_threshold=config.reminder_threshold,
            max_tokens_per_tool=config.max_tokens_per_tool,
            max_total_tool_tokens=config.max_total_tool_tokens,
        )
        self.engine = FocusContextEngine(config=focus_config, llm=llm)

        # 学习器引用（记忆工程 → 上下文工程的桥梁）
        self.learner = learner

        # 会话状态
        self._session_states: Dict[str, Dict] = {}

        # 统计
        self._stats = {
            "compressions_applied": 0,
            "tokens_saved": 0,
            "profiles_injected": 0,
            "learning_injected": 0,
        }

    def _get_session_state(self, session_id: str) -> Dict:
        if session_id not in self._session_states:
            self._session_states[session_id] = {
                "tool_call_count": 0,
                "current_phase": None,
                "phase_count": 0,
            }
        return self._session_states[session_id]

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前：

        1. 总结-裁剪策略（启用时：先总结旧消息，再裁剪）
        2. 检查 Sawtooth 压缩触发
        3. 如触发 → 执行压缩 (tool_trim + priority_filter + summarize)

        注意：不在这里注入用户画像或学习经验到 messages
        这些数据已经在 state 中，由子图自己决定如何使用
        """
        if not self.config.enabled:
            return None

        messages = self._get_state_value(state, "messages", [])
        if not messages:
            return None

        conversation_id = self._get_state_value(state, "conversation_id", "default")
        session_state = self._get_session_state(conversation_id)
        config: FocusContextMiddlewareConfig = self.config
        updates = {}

        # 1. 总结-裁剪策略（医疗场景：先总结再裁剪）
        current_tokens = estimate_messages_tokens(messages)
        if (
            config.enable_summarization
            and config.summarize_before_trim
            and current_tokens > config.max_tokens * config.warning_threshold
        ):
            trimmed = self._summarize_then_trim(messages)
            if trimmed is not None and len(trimmed) < len(messages):
                tokens_before = current_tokens
                current_tokens = estimate_messages_tokens(trimmed)
                updates["messages"] = trimmed
                messages = trimmed
                self._stats["compressions_applied"] += 1
                self._stats["tokens_saved"] += tokens_before - current_tokens
                logger.info(
                    f"[Focus] 总结-裁剪: {tokens_before} -> {current_tokens} tokens, "
                    f"消息数 {len(state.get('messages', []))} -> {len(trimmed)}"
                )

        # 2. 检查 Sawtooth 压缩触发（双重条件：工具调用次数 OR token 阈值）
        should_compress = self.engine.should_compress(session_state["tool_call_count"], current_tokens)
        compression_result = None

        if should_compress and config.auto_compress:
            compression_result = self.engine.apply_token_compression(messages, current_tokens)
            if compression_result:
                updates.update(compression_result)
                self._stats["compressions_applied"] += 1
                self._stats["tokens_saved"] += compression_result.get(
                    "compression_applied", {}
                ).get("tokens_saved", 0)

        # 3. 压缩提醒
        reminder = self.engine.get_compression_reminder(session_state["tool_call_count"])
        if reminder:
            updates["focus_reminder"] = reminder

        # 4. Token 使用状态
        usage_status = self.engine.strategy_selector.get_usage_status(current_tokens)
        updates["context_status"] = usage_status

        # 5. 设置 enriched_context (仅包含 Focus 相关内容)
        if updates:
            compression_details = compression_result.get("compression_applied") if should_compress and compression_result else None
            updates["enriched_context"] = {
                "focus_stats": self.engine.get_stats(),
                "context_status": usage_status,
                "compression_applied": bool(compression_details),
            }
            if compression_details:
                updates["enriched_context"]["compression_details"] = compression_details

        return updates if updates else None

    def _summarize_then_trim(self, messages: List) -> Optional[List]:
        """
        总结-裁剪策略：先总结旧消息，再裁剪

        医疗场景中直接裁剪会丢失关键信息（症状、证型、用药反馈）。
        此方法先将超出窗口的旧消息摘要为一条 SystemMessage，再保留最近 N 条消息。

        Returns:
            裁剪后的消息列表，或 None（不需要裁剪时）
        """
        from langchain_core.messages import SystemMessage

        config: FocusContextMiddlewareConfig = self.config
        keep_recent = config.keep_recent_messages

        if len(messages) <= keep_recent:
            return None

        # 分离旧消息和最近消息
        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]

        # 使用 TCMSummarizer 生成摘要
        summary_result = self.engine.summarizer.summarize_messages(
            old_messages, use_llm=bool(self.engine.llm)
        )
        summary_text = summary_result.get("summary", "")
        if not summary_text:
            return None

        # 构建摘要 SystemMessage
        summary_msg = SystemMessage(
            content=f"[对话历史摘要（保留关键医疗信息）]\n{summary_text}"
        )

        # 拼接：摘要 + 最近消息
        return [summary_msg] + list(recent_messages)

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后：

        1. 更新工具调用计数
        2. 跟踪阶段进度
        3. 返回统计信息
        """
        if not self.config.enabled:
            return None

        conversation_id = self._get_state_value(state, "conversation_id", "default")
        session_state = self._get_session_state(conversation_id)

        updates = {}

        # 更新工具调用计数
        tool_calls = self._get_state_value(state, "tool_calls", [])
        if tool_calls:
            session_state["tool_call_count"] += len(tool_calls)

        # 统计信息
        updates["focus_stats"] = self.engine.get_stats()

        return updates if updates else None

    def start_phase(self, session_id: str, name: str, goal: str, messages: list) -> tuple:
        """开始新的 Focus 阶段"""
        session_state = self._get_session_state(session_id)
        session_state["phase_count"] += 1
        session_state["current_phase"] = name
        return self.engine.start_phase(name, goal, messages)

    async def complete_phase(
        self, session_id: str, result: str, messages: list, llm=None
    ) -> Dict[str, Any]:
        """完成当前 Focus 阶段并执行压缩"""
        session_state = self._get_session_state(session_id)
        compression_result = await self.engine.complete_phase(result, messages, llm)
        if compression_result.get("action") == "compress":
            session_state["current_phase"] = None
        return compression_result

    def get_stats(self, session_id: str = None) -> Dict[str, Any]:
        stats = {**self._stats, **self.engine.get_stats()}
        if session_id and session_id in self._session_states:
            stats["session"] = self._session_states[session_id]
        return stats

    def reset_session(self, session_id: str):
        if session_id in self._session_states:
            del self._session_states[session_id]


def get_focus_context_middleware(
    max_tokens: int = 8000,
    compress_interval: int = 12,
    auto_compress: bool = True,
    llm=None,
) -> FocusContextMiddleware:
    """
    创建 FocusContextMiddleware 的便捷函数

    替代原先的：
    - get_tcm_context_manager_middleware()
    - get_summarization_middleware()
    - get_focus_middleware()
    """
    config = FocusContextMiddlewareConfig(
        enabled=True,
        priority=5,
        name="FocusContextMiddleware",
        max_tokens=max_tokens,
        compress_interval=compress_interval,
        auto_compress=auto_compress,
    )
    return FocusContextMiddleware(config=config, llm=llm)


# ============== 配置工厂方法 ==============

def create_main_graph_focus_config() -> FocusContextMiddlewareConfig:
    """
    主图专用 Focus 配置

    特点：
    - 总结-裁剪策略（医疗场景，先总结再裁剪）
    - 禁用工具裁剪（主图无工具调用）
    - 轻量级压缩（主图路由快速）
    """
    return FocusContextMiddlewareConfig(
        enabled=True,
        priority=5,
        name="FocusContextMiddleware_MainGraph",
        max_tokens=8000,
        auto_compress=True,
        compress_interval=12,
        max_tokens_per_tool=0,          # 禁用工具裁剪
        max_total_tool_tokens=0,
        aggressive=False,
        warning_threshold=0.8,
        # 总结-裁剪策略
        enable_summarization=True,
        summarize_before_trim=True,
        keep_recent_messages=10,
    )


def create_diagnose_focus_config() -> FocusContextMiddlewareConfig:
    """
    诊断子图专用 Focus 配置

    特点：
    - 总结-裁剪策略（医疗场景，保留关键诊断信息）
    - 激进压缩（多轮信息收集）
    - 启用工具裁剪（KG/向量检索输出大）
    - 阶段性压缩（collection → assessment → diagnosis）
    """
    return FocusContextMiddlewareConfig(
        enabled=True,
        priority=5,
        name="FocusContextMiddleware_Diagnose",
        max_tokens=8000,
        auto_compress=True,
        compress_interval=6,            # 每6次工具调用触发
        max_tokens_per_tool=300,        # 单个工具输出最多300 tokens
        max_total_tool_tokens=2000,     # 所有工具输出总计最多2000 tokens
        aggressive=True,
        warning_threshold=0.65,         # 65% 就开始压缩
        reminder_threshold=15,
        # 总结-裁剪策略（诊断场景同样需要）
        enable_summarization=True,
        summarize_before_trim=True,
        keep_recent_messages=8,         # 诊断子图保留最近8条
    )


def create_wellness_focus_config() -> FocusContextMiddlewareConfig:
    """
    养生子图专用 Focus 配置

    特点：
    - 轻量级（养生回复不需要太大窗口）
    - 禁用工具裁剪（养生工具轻量）
    - 低阈值触发（养生轮次少）
    """
    return FocusContextMiddlewareConfig(
        enabled=True,
        priority=5,
        name="FocusContextMiddleware_Wellness",
        max_tokens=6000,
        auto_compress=True,
        compress_interval=8,
        max_tokens_per_tool=0,          # 禁用工具裁剪
        max_total_tool_tokens=0,
        aggressive=False,
        warning_threshold=0.75,
        # 养生子图不需要总结-裁剪（对话简短）
        enable_summarization=False,
    )

