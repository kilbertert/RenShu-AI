"""
TCM 上下文管理中间件

集成上下文工程组件：
- 消息优先级分配
- 压缩策略选择
- 工具消息裁剪
- 分层记忆管理
- 摘要生成
- 用户画像注入（集成 TCMContextEnricher）

在 Token 使用接近限制时自动触发压缩
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import logging

from langchain_core.messages import SystemMessage

from .base import BaseMiddleware, MiddlewareConfig

# 导入上下文工程组件
from ..context import (
    MessagePriority,
    MessagePriorityAssigner,
    PrioritizedMessage,
    CompressionLevel,
    CompressionStrategy,
    CompressionStrategySelector,
    TCMSummarizer,
    SmartToolTrimmer,
    HierarchicalMemory,
    MemoryLevel,
)

# 导入上下文增强器
from ..intent_recognition.context_enricher import TCMContextEnricher
from ..intent_recognition.schemas import EnrichedContext, UserProfile

logger = logging.getLogger(__name__)


@dataclass
class ContextManagerConfig(MiddlewareConfig):
    """上下文管理配置"""
    # Token 限制
    max_tokens: int = 8000
    warning_threshold: float = 0.7
    light_threshold: float = 0.8
    medium_threshold: float = 0.9
    aggressive_threshold: float = 0.95

    # 压缩选项
    enable_auto_compression: bool = True
    enable_tool_trimming: bool = True
    enable_summarization: bool = True
    enable_memory: bool = True

    # 用户画像注入
    enable_profile_injection: bool = True
    enable_environment_context: bool = True

    # 工具裁剪配置
    max_tokens_per_tool: int = 300
    max_total_tool_tokens: int = 2000

    # 记忆配置
    working_memory_size: int = 10
    episodic_memory_size: int = 50


class TCMContextManagerMiddleware(BaseMiddleware):
    """
    TCM 上下文管理中间件

    功能：
    1. 监控 Token 使用情况
    2. 自动触发上下文压缩
    3. 管理分层记忆
    4. 优化工具调用结果
    5. 用户画像注入（集成 TCMContextEnricher）
    6. 环境上下文注入（节气、季节等）
    """

    def __init__(self, config: Optional[ContextManagerConfig] = None):
        """
        初始化上下文管理中间件

        Args:
            config: 配置对象
        """
        # 先调用父类初始化
        super().__init__(config or ContextManagerConfig())
        # 保存配置引用（此时 self.config 已由父类设置）

        # 初始化组件
        self.priority_assigner = MessagePriorityAssigner()
        self.strategy_selector = CompressionStrategySelector(
            max_tokens=self.config.max_tokens,
            warning_threshold=self.config.warning_threshold,
            light_threshold=self.config.light_threshold,
            medium_threshold=self.config.medium_threshold,
            aggressive_threshold=self.config.aggressive_threshold,
        )
        self.tool_trimmer = SmartToolTrimmer(
            max_tokens_per_tool=self.config.max_tokens_per_tool,
            max_total_tool_tokens=self.config.max_total_tool_tokens,
        )
        self.summarizer = TCMSummarizer()
        self.memory = HierarchicalMemory(
            working_capacity=self.config.working_memory_size,
            episodic_capacity=self.config.episodic_memory_size,
        )

        # 上下文增强器（用户画像、环境上下文）
        self.context_enricher = TCMContextEnricher()

        # 统计信息
        self._stats = {
            "compressions_applied": 0,
            "tokens_saved": 0,
            "summaries_generated": 0,
            "profiles_injected": 0,
        }

    @property
    def name(self) -> str:
        return "TCMContextManager"

    def before_model(
        self,
        state: Any,
        runtime: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前处理

        1. 注入用户画像和环境上下文
        2. 检查 Token 使用情况
        3. 必要时应用压缩策略
        4. 更新工作记忆
        """
        if not self.config.enabled:
            return None

        messages = self._get_state_value(state, "messages", [])
        if not messages:
            return None

        updates = {}

        # 1. 用户画像和环境上下文注入
        if self.config.enable_profile_injection:
            profile_updates = self._inject_user_profile(state, messages)
            if profile_updates:
                updates.update(profile_updates)
                messages = updates.get("messages", messages)

        # 2. 计算当前 Token 使用量
        current_tokens = self._estimate_total_tokens(messages)

        # 3. 获取使用状态
        usage_status = self.strategy_selector.get_usage_status(current_tokens)
        logger.info(
            f"Context status: {usage_status['status']}, "
            f"tokens: {current_tokens}/{self.config.max_tokens} "
            f"({usage_status['usage_ratio']:.1%})"
        )

        updates["context_status"] = usage_status

        # 4. 根据使用情况决定是否压缩
        if self.config.enable_auto_compression and usage_status["usage_ratio"] >= self.config.warning_threshold:
            compression_result = self._apply_compression(messages, current_tokens)
            if compression_result:
                updates.update(compression_result)
                self._stats["compressions_applied"] += 1

        # 5. 更新工作记忆（最后一条用户消息）
        if self.config.enable_memory and messages:
            last_msg = messages[-1]
            content = self._get_message_content(last_msg)
            if content:
                importance = self._calculate_importance(content)
                self.memory.add_to_working(content, importance=importance)

        return updates if updates else None

    def _inject_user_profile(
        self,
        state: Any,
        messages: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        注入用户画像和环境上下文

        使用 TCMContextEnricher 从数据库获取用户画像，
        并将其注入到系统消息中。

        Args:
            state: 当前状态
            messages: 消息列表

        Returns:
            包含更新后消息的字典，或 None
        """
        # 检查是否已有用户画像消息
        if self._has_profile_message(messages):
            return None

        # 获取用户信息
        user_id = self._get_state_value(state, "user_id", "")
        conversation_id = self._get_state_value(state, "conversation_id", "")

        if not user_id:
            return None

        try:
            # 尝试获取已有的增强上下文（可能由路由阶段生成）
            enriched_context = self._get_state_value(state, "enriched_context", None)

            if enriched_context:
                # 使用已有的上下文
                profile = enriched_context.user_profile if hasattr(enriched_context, 'user_profile') else None
                environment = enriched_context.environment if hasattr(enriched_context, 'environment') else None
            else:
                # 如果没有，这里只创建基础画像（异步获取需要在外部处理）
                profile = None
                environment = None

            # 构建画像消息
            profile_content = self._create_profile_message_content(profile, environment)

            if profile_content:
                profile_msg = SystemMessage(content=profile_content)

                # 找到插入位置（在系统消息之后，对话消息之前）
                insert_pos = 0
                for i, msg in enumerate(messages):
                    if self._get_message_role(msg) == "system":
                        insert_pos = i + 1
                    else:
                        break

                new_messages = list(messages)
                new_messages.insert(insert_pos, profile_msg)

                self._stats["profiles_injected"] += 1
                logger.debug(f"Injected user profile for user_id={user_id}")

                return {"messages": new_messages}

        except Exception as e:
            logger.warning(f"Failed to inject user profile: {e}")

        return None

    def _has_profile_message(self, messages: List[Any]) -> bool:
        """检查是否已有用户画像消息"""
        for msg in messages:
            content = self._get_message_content(msg)
            if "【用户健康档案】" in content or "【环境上下文】" in content:
                return True
        return False

    def _create_profile_message_content(
        self,
        profile: Optional[UserProfile],
        environment: Optional[Any]
    ) -> Optional[str]:
        """
        创建用户画像消息内容

        Args:
            profile: 用户画像
            environment: 环境上下文

        Returns:
            格式化的消息内容
        """
        parts = []

        # 用户画像部分
        if profile:
            profile_parts = ["【用户健康档案】"]

            if profile.gender:
                profile_parts.append(f"- 性别：{profile.gender}")
            if profile.age_group:
                profile_parts.append(f"- 年龄段：{profile.age_group}")
            if profile.constitution:
                profile_parts.append(f"- 体质类型：{profile.constitution}")
            if profile.chronic_conditions:
                profile_parts.append(f"- 既往病史：{', '.join(profile.chronic_conditions)}")
            if profile.allergies:
                profile_parts.append(f"- 过敏史：{', '.join(profile.allergies)}")

            if len(profile_parts) > 1:  # 有实际内容
                parts.append("\n".join(profile_parts))

        # 环境上下文部分
        if self.config.enable_environment_context and environment:
            env_parts = ["【环境上下文】"]

            if hasattr(environment, 'season') and environment.season:
                env_parts.append(f"- 当前季节：{environment.season}")
            if hasattr(environment, 'solar_term') and environment.solar_term:
                env_parts.append(f"- 当前节气：{environment.solar_term}")
            if hasattr(environment, 'region') and environment.region:
                env_parts.append(f"- 地域：{environment.region}")

            if len(env_parts) > 1:
                parts.append("\n".join(env_parts))

        if parts:
            return "\n\n".join(parts) + "\n\n请根据以上信息提供个性化的中医建议。"

        return None

    def _get_message_role(self, message: Any) -> str:
        """获取消息角色"""
        if isinstance(message, dict):
            return message.get("role", "")
        elif hasattr(message, "type"):
            return message.type
        return ""

    def after_model(
        self,
        state: Any,
        runtime: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后处理

        1. 记录响应到记忆
        2. 更新统计
        """
        if not self.config.enabled:
            return None

        answer = self._get_state_value(state, "answer", "")
        if answer and self.config.enable_memory:
            importance = self._calculate_importance(answer)
            self.memory.add_to_working(
                answer,
                importance=importance,
                metadata={"type": "assistant_response"}
            )

        return None

    def _apply_compression(
        self,
        messages: List[Any],
        current_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """
        应用压缩策略

        Args:
            messages: 消息列表
            current_tokens: 当前 token 数

        Returns:
            状态更新字典
        """
        # 1. 选择压缩策略
        strategy = self.strategy_selector.select_strategy(
            current_tokens=current_tokens,
            message_count=len(messages)
        )

        if strategy.level == CompressionLevel.NONE:
            return None

        logger.info(f"Applying compression level: {strategy.level.name}")

        compressed_messages = list(messages)
        tokens_before = current_tokens

        # 2. 工具消息裁剪
        if self.config.enable_tool_trimming and strategy.trim_tool_results:
            compressed_messages, tool_stats = self.tool_trimmer.apply_trimming(
                compressed_messages,
                target_tokens=strategy.target_tokens // 2  # 分配一半给工具
            )
            if tool_stats.get("trimmed"):
                logger.info(f"Tool trimming saved {tool_stats.get('saved_tokens', 0)} tokens")

        # 3. 消息优先级过滤
        if strategy.drop_low_priority:
            prioritized = self.priority_assigner.assign_priorities(compressed_messages)
            compressed_messages = self.strategy_selector.apply_strategy(
                prioritized, strategy
            )

        # 4. 中间消息摘要
        if self.config.enable_summarization and strategy.summarize_middle:
            # 找出需要摘要的中间消息
            middle_count = len(compressed_messages) - strategy.keep_last_n * 2
            if middle_count > 5:  # 只有足够多的中间消息才摘要
                middle_messages = compressed_messages[strategy.keep_last_n:-strategy.keep_last_n]
                summary_result = self.summarizer.summarize_messages(
                    middle_messages,
                    use_llm=False  # 先用规则，快速
                )

                if summary_result.summary:
                    # 用摘要替换中间消息
                    summary_msg = self.summarizer.create_summary_message(summary_result)
                    compressed_messages = (
                        compressed_messages[:strategy.keep_last_n] +
                        [summary_msg] +
                        compressed_messages[-strategy.keep_last_n:]
                    )
                    self._stats["summaries_generated"] += 1

        # 5. 计算节省的 token
        tokens_after = self._estimate_total_tokens(compressed_messages)
        tokens_saved = tokens_before - tokens_after
        self._stats["tokens_saved"] += tokens_saved

        logger.info(
            f"Compression complete: {tokens_before} -> {tokens_after} tokens "
            f"(saved {tokens_saved}, {tokens_saved/tokens_before*100:.1f}%)"
        )

        return {
            "messages": compressed_messages,
            "compression_applied": {
                "level": strategy.level.name,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_saved,
            }
        }

    def _estimate_total_tokens(self, messages: List[Any]) -> int:
        """估算消息总 token 数"""
        total = 0
        for msg in messages:
            content = self._get_message_content(msg)
            total += self._estimate_tokens(content)
        return total

    def _estimate_tokens(self, text: str) -> int:
        """估算单段文本的 token 数"""
        if not text:
            return 0
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _get_message_content(self, message: Any) -> str:
        """获取消息内容"""
        if isinstance(message, dict):
            return message.get("content", "")
        elif hasattr(message, "content"):
            return message.content or ""
        return ""

    def _calculate_importance(self, content: str) -> float:
        """计算内容重要性"""
        importance = 0.5  # 基础分

        # TCM 关键词加分
        tcm_keywords = ["症状", "诊断", "方剂", "舌象", "脉象", "证型", "治法"]
        for kw in tcm_keywords:
            if kw in content:
                importance += 0.1

        # 问题加分
        if "?" in content or "？" in content:
            importance += 0.1

        # 长度加分（信息量）
        if len(content) > 200:
            importance += 0.1

        return min(1.0, importance)

    def get_memory_context(self, query: str, max_tokens: int = 300) -> str:
        """
        获取记忆上下文（用于增强提示词）

        Args:
            query: 当前查询
            max_tokens: 最大 token 数

        Returns:
            记忆上下文字符串
        """
        return self.memory.get_context_for_prompt(query, max_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "memory_stats": self.memory.get_stats(),
        }

    def reset_stats(self):
        """重置统计"""
        self._stats = {
            "compressions_applied": 0,
            "tokens_saved": 0,
            "summaries_generated": 0,
        }


def get_tcm_context_manager_middleware(
    max_tokens: int = 8000,
    enable_auto_compression: bool = True,
) -> TCMContextManagerMiddleware:
    """
    获取 TCM 上下文管理中间件实例

    Args:
        max_tokens: 最大 token 限制
        enable_auto_compression: 是否启用自动压缩

    Returns:
        配置好的中间件实例
    """
    config = ContextManagerConfig(
        enabled=True,
        priority=5,  # 较高优先级，在其他中间件之前处理
        max_tokens=max_tokens,
        enable_auto_compression=enable_auto_compression,
    )
    return TCMContextManagerMiddleware(config)
