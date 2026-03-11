"""
Focus 上下文引擎 (Sawtooth 模式)

基于 arXiv:2601.07190 - 主动上下文压缩

整合了原先分散在 4 个文件中的功能：
- FocusContextManager (focus_manager.py)
- CompressionStrategySelector (compression_strategy.py)
- TCMSummarizer (summarization.py)
- 并引用 MessagePriorityAssigner 和 SmartToolTrimmer

Sawtooth 模式：
  ↑ tokens
  │   ╱╲    ╱╲    ╱╲
  │  ╱  ╲  ╱  ╲  ╱  ╲
  │ ╱    ╲╱    ╲╱    ╲
  └─────────────────────→ time
     explore compress explore
"""
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from ..utils.tcm_utils import (
    estimate_tokens,
    estimate_messages_tokens,
    get_message_content,
    get_message_role,
)
from ..utils.tcm_constants import TCM_KEYWORDS_FLAT

logger = logging.getLogger(__name__)


# ============== 枚举和数据类 ==============

class CompressionLevel(Enum):
    """压缩级别"""
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    AGGRESSIVE = 3


@dataclass
class CompressionStrategy:
    """压缩策略"""
    level: CompressionLevel
    target_tokens: int
    keep_system: bool = True
    keep_last_n: int = 3
    summarize_middle: bool = False
    drop_low_priority: bool = False
    trim_tool_results: bool = False


@dataclass
class FocusPhase:
    """Focus 阶段"""
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
        return f"""## {self.phase_name} ({self.timestamp.strftime('%H:%M')})

**总结**: {self.summary}

**关键发现**:
{chr(10).join(f'- {f}' for f in self.key_findings)}

**下一步**: {self.next_step}"""


@dataclass
class FocusConfig:
    """Focus 引擎配置"""
    # Sawtooth 压缩参数
    compress_interval: int = 12
    aggressive: bool = True
    auto_compress: bool = True
    max_phases: int = 10
    reminder_threshold: int = 15

    # Token 限制（从 ContextManagerConfig 吸收）
    max_tokens: int = 8000
    warning_threshold: float = 0.7
    light_threshold: float = 0.8
    medium_threshold: float = 0.9
    aggressive_threshold: float = 0.95

    # 工具裁剪配置
    max_tokens_per_tool: int = 300
    max_total_tool_tokens: int = 2000

    # 摘要配置
    max_summary_tokens: int = 500

    # 用户画像注入
    enable_profile_injection: bool = True
    enable_environment_context: bool = True


# ============== TCM 摘要器（吸收自 summarization.py） ==============

TCM_SUMMARY_TEMPLATE = """请对以下中医对话进行摘要，保留关键医学信息。

## 摘要要求
1. 保留所有症状描述
2. 保留舌象、脉象等诊断信息
3. 保留辨证结论和证型
4. 保留方剂和用药建议
5. 忽略寒暄和重复内容

## 对话内容
{conversation}

## 输出格式
【主诉】患者主要症状和就诊原因
【症状】详细症状列表
【四诊信息】舌象、脉象等
【辨证】证型判断
【建议】已给出的方剂或建议

如果某项信息未提及，标注"未提及"。
"""


class TCMSummarizer:
    """TCM 专用摘要生成器（吸收自 context/summarization.py）"""

    SECTION_MARKERS = {
        "chief_complaint": ["主诉", "主要症状"],
        "symptoms": ["症状", "表现"],
        "diagnosis_info": ["四诊", "舌象", "脉象"],
        "syndrome": ["辨证", "证型"],
        "suggestions": ["建议", "方剂", "处方"],
    }

    def __init__(self, llm: Optional[BaseChatModel] = None, max_summary_tokens: int = 500):
        self.llm = llm
        self.max_summary_tokens = max_summary_tokens

    def summarize_messages(self, messages: List[Any], use_llm: bool = True) -> Dict[str, Any]:
        """
        对消息列表生成摘要

        Returns:
            {"summary": str, "original_tokens": int, "summary_tokens": int}
        """
        conversation_text = self._extract_conversation(messages)
        original_tokens = estimate_tokens(conversation_text)

        if use_llm and self.llm:
            summary = self._llm_summarize(conversation_text)
        else:
            summary = self._rule_based_summarize(messages)

        summary_tokens = estimate_tokens(summary)

        return {
            "summary": summary,
            "original_tokens": original_tokens,
            "summary_tokens": summary_tokens,
        }

    def _extract_conversation(self, messages: List[Any]) -> str:
        lines = []
        for msg in messages:
            role = get_message_role(msg)
            content = get_message_content(msg)
            if role == "system":
                continue
            role_name = "患者" if role in ["human", "user"] else "医生"
            if content:
                lines.append(f"{role_name}: {content}")
        return "\n".join(lines)

    def _llm_summarize(self, conversation: str) -> str:
        prompt = TCM_SUMMARY_TEMPLATE.format(conversation=conversation)
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception:
            return self._rule_based_summarize_text(conversation)

    def _rule_based_summarize(self, messages: List[Any]) -> str:
        sections = {k: [] for k in self.SECTION_MARKERS}

        for msg in messages:
            content = get_message_content(msg)
            if not content:
                continue
            for section, markers in self.SECTION_MARKERS.items():
                for marker in markers:
                    if marker in content:
                        sections[section].append(content)
                        break

        lines = []
        if sections["chief_complaint"]:
            lines.append(f"【主诉】{sections['chief_complaint'][0][:100]}")
        if sections["symptoms"]:
            lines.append(f"【症状】{'; '.join(s[:50] for s in sections['symptoms'][:5])}")
        if sections["diagnosis_info"]:
            lines.append(f"【四诊信息】{sections['diagnosis_info'][0][:100]}")
        if sections["syndrome"]:
            lines.append(f"【辨证】{sections['syndrome'][0][:100]}")
        if sections["suggestions"]:
            lines.append(f"【建议】{sections['suggestions'][-1][:200]}")

        return "\n".join(lines) if lines else "对话内容较少，无需摘要"

    def _rule_based_summarize_text(self, text: str) -> str:
        lines = text.split("\n")
        keywords = ["症状", "主诉", "舌", "脉", "方", "药", "证", "诊断", "建议"]
        important = [l for l in lines if any(kw in l for kw in keywords)]
        if important:
            return "\n".join(important[:10])
        return "\n".join(lines[:3] + ["..."] + lines[-3:])

    def create_summary_message(self, summary_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": "system",
            "content": f"[对话摘要]\n{summary_result['summary']}",
            "metadata": {
                "type": "summary",
                "original_tokens": summary_result.get("original_tokens", 0),
                "summary_tokens": summary_result.get("summary_tokens", 0),
            },
        }


# ============== 压缩策略选择器（吸收自 compression_strategy.py） ==============

class CompressionStrategySelector:
    """根据 token 使用情况选择压缩策略"""

    def __init__(self, config: FocusConfig):
        self.config = config

    def select_strategy(self, current_tokens: int, message_count: int = 0) -> CompressionStrategy:
        usage_ratio = current_tokens / self.config.max_tokens

        if usage_ratio >= self.config.aggressive_threshold:
            return CompressionStrategy(
                level=CompressionLevel.AGGRESSIVE,
                target_tokens=int(self.config.max_tokens * 0.6),
                keep_last_n=2, summarize_middle=True,
                drop_low_priority=True, trim_tool_results=True,
            )
        elif usage_ratio >= self.config.medium_threshold:
            return CompressionStrategy(
                level=CompressionLevel.MEDIUM,
                target_tokens=int(self.config.max_tokens * 0.7),
                keep_last_n=3, summarize_middle=True,
                drop_low_priority=True, trim_tool_results=True,
            )
        elif usage_ratio >= self.config.light_threshold:
            return CompressionStrategy(
                level=CompressionLevel.LIGHT,
                target_tokens=int(self.config.max_tokens * 0.75),
                keep_last_n=5, drop_low_priority=True,
            )
        else:
            return CompressionStrategy(
                level=CompressionLevel.NONE,
                target_tokens=self.config.max_tokens,
                keep_last_n=10,
            )

    def get_usage_status(self, current_tokens: int) -> Dict[str, Any]:
        usage_ratio = current_tokens / self.config.max_tokens
        if usage_ratio >= self.config.aggressive_threshold:
            status = "critical"
        elif usage_ratio >= self.config.medium_threshold:
            status = "warning"
        elif usage_ratio >= self.config.light_threshold:
            status = "notice"
        elif usage_ratio >= self.config.warning_threshold:
            status = "approaching"
        else:
            status = "normal"

        return {
            "status": status,
            "current_tokens": current_tokens,
            "max_tokens": self.config.max_tokens,
            "usage_ratio": usage_ratio,
            "remaining_tokens": self.config.max_tokens - current_tokens,
        }


# ============== Focus 上下文引擎（核心） ==============

class FocusContextEngine:
    """
    Focus 上下文引擎

    统一的 Sawtooth 上下文管理，整合：
    - FocusContextManager (原 focus_manager.py)
    - CompressionStrategySelector (原 compression_strategy.py)
    - TCMSummarizer (原 summarization.py)
    - 引用 MessagePriorityAssigner (保留独立)
    - 引用 SmartToolTrimmer (保留独立)

    complete_phase() 增强为完整压缩流程：
    1. strategy_selector.select_strategy()
    2. tool_trimmer.apply_trimming()
    3. priority_assigner.assign_priorities()
    4. summarizer.summarize_messages()
    5. 创建 KnowledgeBlock，删除原始消息
    """

    def __init__(self, config: Optional[FocusConfig] = None, llm=None):
        self.config = config or FocusConfig()
        self.llm = llm

        # 子组件
        self.strategy_selector = CompressionStrategySelector(self.config)
        self.summarizer = TCMSummarizer(llm=llm, max_summary_tokens=self.config.max_summary_tokens)

        # 延迟导入独立组件（保持向后兼容）
        self._priority_assigner = None
        self._tool_trimmer = None

        # 状态
        self.phases: List[FocusPhase] = []
        self.knowledge_blocks: List[KnowledgeBlock] = []
        self.current_phase: Optional[FocusPhase] = None

        # 统计
        self.total_compressions = 0
        self.total_dropped_messages = 0
        self.total_tokens_saved = 0

    @property
    def priority_assigner(self):
        if self._priority_assigner is None:
            from .message_priority import MessagePriorityAssigner
            self._priority_assigner = MessagePriorityAssigner()
        return self._priority_assigner

    @property
    def tool_trimmer(self):
        if self._tool_trimmer is None:
            from .tool_trimmer import SmartToolTrimmer
            self._tool_trimmer = SmartToolTrimmer(
                max_tokens_per_tool=self.config.max_tokens_per_tool,
                max_total_tool_tokens=self.config.max_total_tool_tokens,
            )
        return self._tool_trimmer

    def start_phase(self, name: str, goal: str, context_messages: List[BaseMessage]) -> tuple:
        """开始新的 Focus 阶段"""
        if self.current_phase is not None:
            self._close_current_phase()

        self.current_phase = FocusPhase(
            name=name, goal=goal, start_index=len(context_messages)
        )
        self.phases.append(self.current_phase)

        instruction = SystemMessage(content=f"""## Focus阶段: {name}

**目标**: {goal}

请专注于完成此目标。完成或遇到障碍后，请告知可以开始下一阶段。
当前已有 {len(self.knowledge_blocks)} 个知识块可供参考。""")

        logger.info(f"[Focus] 开始阶段: {name}, 目标: {goal}")
        return instruction, self.current_phase.start_index

    async def complete_phase(
        self,
        result: str,
        context_messages: List[BaseMessage],
        llm=None,
    ) -> Dict[str, Any]:
        """
        完成当前 Focus 阶段并执行完整压缩流程

        增强流程：
        1. strategy_selector 选择压缩级别
        2. tool_trimmer 裁剪工具结果
        3. priority_assigner 标记消息优先级
        4. summarizer 生成 TCM 专用摘要
        5. 创建 KnowledgeBlock，删除原始消息
        """
        if self.current_phase is None:
            return {"action": "skipped", "reason": "no_active_phase"}

        # 关闭阶段
        self.current_phase.end_index = len(context_messages)
        self.current_phase.end_time = datetime.now()
        self.current_phase.result = result

        # 获取阶段内的消息
        phase_messages = context_messages[self.current_phase.start_index:self.current_phase.end_index]
        current_tokens = estimate_messages_tokens(phase_messages)

        # 1. 选择压缩策略
        strategy = self.strategy_selector.select_strategy(current_tokens, len(phase_messages))

        compressed_messages = list(phase_messages)
        tokens_before = current_tokens

        # 2. 工具裁剪
        if strategy.trim_tool_results:
            compressed_messages, tool_stats = self.tool_trimmer.apply_trimming(
                compressed_messages, target_tokens=strategy.target_tokens // 2
            )

        # 3. 消息优先级过滤
        if strategy.drop_low_priority:
            from .message_priority import MessagePriority
            prioritized = self.priority_assigner.assign_priorities(compressed_messages)
            compressed_messages = [
                pm.message for pm in prioritized
                if pm.priority.value >= MessagePriority.NORMAL.value
            ]

        # 4. 生成摘要
        use_llm_for_summary = llm or self.llm
        summary_result = self.summarizer.summarize_messages(
            compressed_messages, use_llm=bool(use_llm_for_summary)
        )

        # 5. 创建知识块
        dropped_count = self.current_phase.end_index - self.current_phase.start_index
        key_findings = [
            f"调用{self.current_phase.tool_calls_count}次工具",
            f"持续{self.current_phase.duration:.1f}秒",
        ]
        if summary_result.get("summary"):
            # 从摘要提取要点
            summary_lines = summary_result["summary"].split("\n")
            key_findings.extend([l.strip() for l in summary_lines[:3] if l.strip()])

        knowledge_block = KnowledgeBlock(
            phase_name=self.current_phase.name,
            timestamp=datetime.now(),
            summary=summary_result.get("summary", result[:100]),
            key_findings=key_findings[:5],
            next_step="继续",
            dropped_messages=dropped_count,
        )
        self.knowledge_blocks.append(knowledge_block)

        # 更新统计
        tokens_after = estimate_messages_tokens(compressed_messages)
        self.total_compressions += 1
        self.total_dropped_messages += dropped_count
        self.total_tokens_saved += tokens_before - tokens_after

        logger.info(
            f"[Focus] 压缩完成: 删除{dropped_count}条消息, "
            f"tokens {tokens_before}->{tokens_after}, "
            f"累计节省~{self.total_tokens_saved} tokens"
        )

        # 清除当前阶段
        start_idx = self.current_phase.start_index
        end_idx = self.current_phase.end_index
        self.current_phase = None

        return {
            "action": "compress",
            "knowledge_added": knowledge_block.format(),
            "drop_range": (start_idx, end_idx),
            "messages_dropped": dropped_count,
            "tokens_saved": tokens_before - tokens_after,
            "knowledge_block": knowledge_block,
        }

    def should_compress(self, tool_call_count: int, current_tokens: int = 0) -> bool:
        """判断是否应该压缩（双重触发：工具调用次数 OR token 阈值）"""
        if self.config.aggressive and tool_call_count >= self.config.compress_interval:
            return True
        if current_tokens > 0 and current_tokens > self.config.max_tokens * self.config.warning_threshold:
            return True
        return False

    def get_compression_reminder(self, tool_call_count: int) -> Optional[str]:
        if tool_call_count >= self.config.reminder_threshold:
            return f"⚠️ 提示：您已进行{tool_call_count}次工具调用。建议进行上下文压缩。"
        return None

    def get_knowledge_context(self) -> str:
        """获取累积的知识上下文"""
        if not self.knowledge_blocks:
            return ""
        parts = ["## 累积的学习内容\n"]
        for block in self.knowledge_blocks[-3:]:
            parts.append(block.format())
        return "\n\n".join(parts)

    def inject_knowledge_to_context(
        self, messages: List[BaseMessage], position: str = "after_system"
    ) -> List[BaseMessage]:
        """将知识上下文注入到消息列表"""
        knowledge = self.get_knowledge_context()
        if not knowledge:
            return messages

        knowledge_msg = SystemMessage(content=knowledge)

        if position == "after_system":
            return [messages[0], knowledge_msg] + messages[1:] if messages else [knowledge_msg]
        elif position == "before_user":
            return [knowledge_msg] + messages
        return messages + [knowledge_msg]

    def apply_token_compression(self, messages: List[Any], current_tokens: int) -> Optional[Dict[str, Any]]:
        """
        应用 token 阈值触发的压缩（吸收自 context_manager.py:_apply_compression）

        当 token 使用超过阈值时调用。
        """
        strategy = self.strategy_selector.select_strategy(current_tokens, len(messages))
        if strategy.level == CompressionLevel.NONE:
            return None

        logger.info(f"应用压缩级别: {strategy.level.name}")
        compressed = list(messages)
        tokens_before = current_tokens

        # 工具裁剪
        if strategy.trim_tool_results:
            compressed, _ = self.tool_trimmer.apply_trimming(
                compressed, target_tokens=strategy.target_tokens // 2
            )

        # 优先级过滤
        if strategy.drop_low_priority:
            from .message_priority import MessagePriority
            prioritized = self.priority_assigner.assign_priorities(compressed)
            compressed = [
                pm.message for pm in prioritized
                if pm.priority.value >= MessagePriority.NORMAL.value
            ]

        # 中间消息摘要
        if strategy.summarize_middle and len(compressed) > strategy.keep_last_n * 2 + 5:
            middle = compressed[strategy.keep_last_n:-strategy.keep_last_n]
            summary_result = self.summarizer.summarize_messages(middle, use_llm=False)
            if summary_result.get("summary"):
                summary_msg = self.summarizer.create_summary_message(summary_result)
                compressed = (
                    compressed[:strategy.keep_last_n]
                    + [summary_msg]
                    + compressed[-strategy.keep_last_n:]
                )

        tokens_after = estimate_messages_tokens(compressed)
        tokens_saved = tokens_before - tokens_after
        self.total_tokens_saved += tokens_saved

        return {
            "messages": compressed,
            "compression_applied": {
                "level": strategy.level.name,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_saved,
            },
        }

    def _close_current_phase(self):
        if self.current_phase:
            self.current_phase.end_time = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_phases": len(self.phases),
            "total_compressions": self.total_compressions,
            "total_dropped_messages": self.total_dropped_messages,
            "total_tokens_saved": self.total_tokens_saved,
            "current_phase": self.current_phase.name if self.current_phase else None,
            "knowledge_blocks_count": len(self.knowledge_blocks),
        }
