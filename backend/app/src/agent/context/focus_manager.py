"""
Focus上下文管理器 #主动压缩策略

基于arXiv:2601.07190 - 主动上下文压缩
实现Sawtooth模式，每10-15次调用压缩一次

关键发现：
- 激进的压缩提示是关键（每10-15次调用）
- 22.7% token节省，准确率不变
- 保留学习内容到Knowledge块，删除原始交互
"""

import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage

logger = logging.getLogger(__name__)


class CompressionTrigger(str, Enum):
    """压缩触发条件"""
    TOOL_CALL_COUNT = "tool_call_count"      # 工具调用次数
    TOKEN_LIMIT = "token_limit"              # Token限制
    MANUAL = "manual"                        # 手动触发


@dataclass
class FocusPhase:
    """Focus阶段"""
    name: str
    goal: str
    start_index: int
    start_time: datetime = field(default_factory=datetime.now)
    end_index: int = 0
    end_time: Optional[datetime] = None
    result: str = ""
    tool_calls_count: int = 0

    @property
    def duration(self) -> float:
        """阶段持续时间（秒）"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


@dataclass
class KnowledgeBlock:
    """知识块 - 压缩后的学习内容"""
    phase_name: str
    timestamp: datetime
    summary: str
    key_findings: List[str]
    next_step: str
    dropped_messages: int

    def format(self) -> str:
        """格式化为上下文字符串"""
        return f"""
## {self.phase_name} ({self.timestamp.strftime('%H:%M')})

**总结**: {self.summary}

**关键发现**:
{chr(10).join(f'- {f}' for f in self.key_findings)}

**下一步**: {self.next_step}
""".strip()


@dataclass
class FocusConfig:
    """Focus配置"""
    compress_interval: int = 12          # 每12次工具调用压缩
    aggressive: bool = True              # 激进模式
    auto_compress: bool = True           # 自动压缩
    max_phases: int = 10                 # 最大阶段数
    reminder_threshold: int = 15          # 提醒阈值


class FocusContextManager:
    """
    Focus上下文管理器

    实现Sawtooth上下文模式：
    1. 探索阶段（start_focus）
    2. 积累学习内容
    3. 压缩阶段（complete_focus）
    4. 重复

    Token趋势：
      ↑
    │   ╱╲    ╱╲    ╱╲
    │  ╱  ╲  ╱  ╲  ╱  ╲    ← Sawtooth
    │ ╱    ╲╱    ╲╱    ╲
    └─────────────────────→ 时间
       探索 压缩 探索 压缩
    """

    def __init__(
        self,
        config: Optional[FocusConfig] = None,
        llm=None
    ):
        self.config = config or FocusConfig()
        self.llm = llm

        self.phases: List[FocusPhase] = []
        self.knowledge_blocks: List[KnowledgeBlock] = []
        self.current_phase: Optional[FocusPhase] = None

        # 状态追踪
        self.total_compressions = 0
        self.total_dropped_messages = 0
        self.total_tokens_saved = 0

    def start_phase(
        self,
        name: str,
        goal: str,
        context_messages: List[BaseMessage]
    ) -> tuple[SystemMessage, int]:
        """
        开始新的Focus阶段

        Args:
            name: 阶段名称（如 "exploration", "implementation", "verification"）
            goal: 阶段目标
            context_messages: 当前上下文消息列表

        Returns:
            (系统指令消息, 阶段开始索引)
        """
        # 关闭上一个阶段
        if self.current_phase is not None:
            self._close_current_phase()

        # 创建新阶段
        self.current_phase = FocusPhase(
            name=name,
            goal=goal,
            start_index=len(context_messages)
        )
        self.phases.append(self.current_phase)

        instruction = SystemMessage(content=f"""
## Focus阶段: {name}

**目标**: {goal}

请专注于完成此目标。完成或遇到障碍后，请告知可以开始下一阶段。

当前已有 {len(self.knowledge_blocks)} 个知识块可供参考。
""")

        logger.info(f"[Focus] 开始阶段: {name}, 目标: {goal}")

        return instruction, self.current_phase.start_index

    async def complete_phase(
        self,
        result: str,
        context_messages: List[BaseMessage],
        llm=None
    ) -> Dict[str, Any]:
        """
        完成当前Focus阶段并执行压缩

        Args:
            result: 阶段结果
            context_messages: 当前上下文消息
            llm: LLM实例（用于生成总结）

        Returns:
            压缩操作结果
        """
        if self.current_phase is None:
            logger.warning("[Focus] 没有活动阶段，跳过压缩")
            return {"action": "skipped", "reason": "no_active_phase"}

        # 关闭阶段
        self.current_phase.end_index = len(context_messages)
        self.current_phase.end_time = datetime.now()
        self.current_phase.result = result

        logger.info(
            f"[Focus] 完成阶段: {self.current_phase.name}, "
            f"调用次数: {self.current_phase.tool_calls_count}, "
            f"持续: {self.current_phase.duration:.1f}s"
        )

        # 生成学习总结
        summary = await self._generate_summary(
            self.current_phase,
            context_messages,
            llm or self.llm
        )

        # 创建知识块
        dropped_count = self.current_phase.end_index - self.current_phase.start_index

        knowledge_block = KnowledgeBlock(
            phase_name=self.current_phase.name,
            timestamp=datetime.now(),
            summary=summary["summary"],
            key_findings=summary["key_findings"],
            next_step=summary["next_step"],
            dropped_messages=dropped_count
        )
        self.knowledge_blocks.append(knowledge_block)

        # 更新统计
        self.total_compressions += 1
        self.total_dropped_messages += dropped_count
        self.total_tokens_saved += dropped_count * 500  # 估算

        logger.info(
            f"[Focus] 压缩完成: 删除{dropped_count}条消息, "
            f"累计节省~{self.total_tokens_saved} tokens"
        )

        # 清除当前阶段
        self.current_phase = None

        return {
            "action": "compress",
            "knowledge_added": knowledge_block.format(),
            "drop_range": (knowledge_block.timestamp,
                          self.current_phase.start_index if self.current_phase else 0,
                          self.current_phase.end_index if self.current_phase else 0),
            "messages_dropped": dropped_count,
            "knowledge_block": knowledge_block
        }

    async def _generate_summary(
        self,
        phase: FocusPhase,
        context_messages: List[BaseMessage],
        llm
    ) -> Dict[str, Any]:
        """生成阶段总结"""

        # 获取阶段内的消息（限制长度）
        phase_messages = context_messages[phase.start_index:phase.end_index]
        phase_content = self._truncate_messages(phase_messages, max_chars=3000)

        summary_prompt = f"""
总结以下Focus阶段的学习内容：

【阶段】{phase.name}
【目标】{phase.goal}
【持续时间】{phase.duration:.1f}秒
【工具调用】{phase.tool_calls_count}次
【结果】{phase.result}

【交互记录】
{phase_content}

请提取关键信息（每项不超过30字）：

1. summary: 一句话总结这个阶段完成了什么
2. key_findings: 3-5个关键发现（列表格式）
3. next_step: 下一步应该做什么

返回JSON格式：
```json
{{
  "summary": "完成功能X的实现",
  "key_findings": ["发现1", "发现2", "发现3"],
  "next_step": "开始测试"
}}
```
"""

        try:
            response = await llm.ainvoke(summary_prompt)

            # 尝试解析JSON
            import json
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                # 解析失败，返回默认值
                pass
        except Exception as e:
            logger.warning(f"[Focus] LLM总结失败: {e}")

        # 默认总结
        return {
            "summary": phase.result[:100] if phase.result else "已完成",
            "key_findings": [
                f"调用{phase.tool_calls_count}次工具",
                f"持续时间{phase.duration:.1f}秒"
            ],
            "next_step": "继续"
        }

    def _truncate_messages(self, messages: List[BaseMessage], max_chars: int) -> str:
        """截断消息到指定长度"""
        content_parts = []
        total_chars = 0

        for msg in messages:
            msg_content = msg.content[:200]  # 每条消息最多200字符
            if total_chars + len(msg_content) > max_chars:
                break
            content_parts.append(f"[{msg.type}]: {msg_content}")
            total_chars += len(msg_content) + 20  # 加上标签长度

        return "\n".join(content_parts)

    def should_compress(
        self,
        tool_call_count: int,
        current_tokens: int = 0,
        max_tokens: int = 100000
    ) -> bool:
        """判断是否应该压缩"""

        # 检查工具调用次数
        if self.config.aggressive and tool_call_count >= self.config.compress_interval:
            return True

        # 检查token限制
        if current_tokens > 0.8 * max_tokens:
            return True

        return False

    def get_compression_reminder(self, tool_call_count: int) -> Optional[str]:
        """获取压缩提醒消息"""

        if tool_call_count >= self.config.reminder_threshold:
            return f"""
⚠️ 提示：您已进行{tool_call_count}次工具调用。
建议调用complete_focus进行上下文压缩，保留关键学习内容。
"""

        return None

    def get_knowledge_context(self) -> str:
        """获取累积的知识上下文"""

        if not self.knowledge_blocks:
            return ""

        knowledge_parts = ["## 累积的学习内容\n"]

        for block in self.knowledge_blocks[-3:]:  # 只保留最近3个
            knowledge_parts.append(block.format())

        return "\n\n".join(knowledge_parts)

    def inject_knowledge_to_context(
        self,
        messages: List[BaseMessage],
        position: str = "after_system"  # after_system, before_user, append
    ) -> List[BaseMessage]:
        """
        将知识上下文注入到消息列表

        Args:
            messages: 原始消息列表
            position: 注入位置

        Returns:
            注入知识后的消息列表
        """
        knowledge_context = self.get_knowledge_context()

        if not knowledge_context:
            return messages

        knowledge_msg = SystemMessage(content=knowledge_context)

        if position == "after_system":
            # 在系统消息之后注入
            return [messages[0], knowledge_msg] + messages[1:] if messages else [knowledge_msg]

        elif position == "before_user":
            # 在用户消息之前注入
            return [knowledge_msg] + messages

        elif position == "append":
            # 追加到末尾
            return messages + [knowledge_msg]

        return messages

    def _close_current_phase(self):
        """关闭当前阶段（内部使用）"""
        if self.current_phase:
            self.current_phase.end_time = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_phases": len(self.phases),
            "total_compressions": self.total_compressions,
            "total_dropped_messages": self.total_dropped_messages,
            "total_tokens_saved": self.total_tokens_saved,
            "current_phase": self.current_phase.name if self.current_phase else None,
            "knowledge_blocks_count": len(self.knowledge_blocks),
            "compression_rate": (
                self.total_dropped_messages / max(1, self.total_dropped_messages + len(self.phases) * 20)
            )
        }


class FocusMiddleware:
    """
    Focus中间件

    集成到中间件系统，自动管理上下文压缩
    """

    def __init__(
        self,
        config: Optional[FocusConfig] = None,
        llm=None
    ):
        self.config = config or FocusConfig()
        self.llm = llm
        self.focus_manager = FocusContextManager(config, llm)

        # 每个会话的独立状态
        self._session_states: Dict[str, Dict] = {}

    def _get_session_state(self, session_id: str) -> Dict:
        """获取会话状态"""
        if session_id not in self._session_states:
            self._session_states[session_id] = {
                "tool_call_count": 0,
                "current_phase": None,
                "context_messages": []
            }
        return self._session_states[session_id]

    async def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        """模型调用前：检查压缩提醒"""

        session_id = state.get("session_id", "default")
        session_state = self._get_session_state(session_id)

        # 检查是否需要压缩提醒
        reminder = self.focus_manager.get_compression_reminder(
            session_state["tool_call_count"]
        )

        if reminder:
            logger.info(f"[FocusMiddleware] 发送压缩提醒: {session_id}")
            return {"focus_reminder": reminder}

        return None

    async def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        """模型调用后：更新工具调用计数"""

        session_id = state.get("session_id", "default")
        session_state = self._get_session_state(session_id)

        # 更新工具调用计数
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            session_state["tool_call_count"] += len(tool_calls)

        # 检查是否应该自动压缩
        if self.config.auto_compress:
            should_compress = self.focus_manager.should_compress(
                session_state["tool_call_count"]
            )

            if should_compress and session_state.get("current_phase"):
                logger.info(f"[FocusMiddleware] 触发自动压缩: {session_id}")
                # 这里可以触发自动压缩逻辑
                return {"auto_compress_triggered": True}

        return None

    def create_phase_instruction(
        self,
        phase_name: str,
        phase_goal: str,
        messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """创建阶段指令并添加到消息列表"""

        instruction, _ = self.focus_manager.start_phase(
            phase_name,
            phase_goal,
            messages
        )

        return [instruction] + messages


def create_focus_middleware(
    compress_interval: int = 12,
    aggressive: bool = True,
    auto_compress: bool = True
) -> FocusMiddleware:
    """
    创建Focus中间件的便捷函数
    """
    config = FocusConfig(
        compress_interval=compress_interval,
        aggressive=aggressive,
        auto_compress=auto_compress
    )
    return FocusMiddleware(config=config)
