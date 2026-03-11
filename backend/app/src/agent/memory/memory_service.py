"""
统一记忆协调服务

协调 3 层记忆系统：
- 层级 1: LangGraph Checkpointer (PostgreSQL) → 多轮对话状态快照、中断恢复
- 层级 2: Conversation/Message 表 (PostgreSQL) → 聊天历史持久化、token 统计
- 层级 3: Mem0 TCMMemory (Qdrant + Neo4j) → 跨会话语义记忆

MemoryService 为中间件层提供统一接口，替代直接调用各层 API 的零散代码。
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """会话所需的全部记忆上下文"""

    # 层级 2: 近期消息摘要
    recent_messages_summary: str = ""
    turn_count: int = 0

    # 层级 3: Mem0 语义记忆
    user_profile: Optional[Dict[str, Any]] = None
    diagnosis_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    contraindications: List[Dict[str, Any]] = field(default_factory=list)
    wellness_advices: List[Dict[str, Any]] = field(default_factory=list)

    loaded_at: str = ""

    def has_profile(self) -> bool:
        return self.user_profile is not None and bool(self.user_profile)

    def has_history(self) -> bool:
        return len(self.diagnosis_history) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_profile": self.user_profile,
            "diagnosis_history": self.diagnosis_history,
            "preferences": self.preferences,
            "contraindications": self.contraindications,
            "wellness_advices": self.wellness_advices,
            "loaded_at": self.loaded_at,
        }


class MemoryService:
    """
    统一记忆协调服务

    职责：
    1. load_context_for_session: 加载会话所需的全部记忆
    2. save_session_results: 会话结束后持久化重要发现到 Mem0
    3. 协调 3 层记忆，避免直接在中间件中散落各层 API 调用

    使用方式：
        service = MemoryService()
        context = await service.load_context_for_session(user_id, conversation_id)
        # ... 对话执行 ...
        await service.save_session_results(user_id, state)
    """

    def __init__(self, conversation_service=None, tcm_memory=None):
        """
        Args:
            conversation_service: ConversationService 实例（层级 2）
            tcm_memory: TCMMemory 实例（层级 3），None 则懒加载
        """
        self._conversation_service = conversation_service
        self._tcm_memory = tcm_memory

    async def _get_tcm_memory(self):
        """懒加载 TCMMemory"""
        if self._tcm_memory is None:
            try:
                from .tcm_memory import get_tcm_memory
                self._tcm_memory = await get_tcm_memory()
            except Exception as e:
                logger.warning(f"TCMMemory 初始化失败: {e}")
        return self._tcm_memory

    async def load_context_for_session(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        max_memories: int = 10,
    ) -> MemoryContext:
        """
        加载会话所需的全部记忆

        Args:
            user_id: 用户 ID
            conversation_id: 会话 ID（可选）
            max_memories: 最大加载记忆数

        Returns:
            MemoryContext 包含所有层级的记忆
        """
        context = MemoryContext(loaded_at=datetime.now().isoformat())

        # 层级 3: 从 Mem0 获取语义记忆
        memory = await self._get_tcm_memory()
        if memory:
            try:
                await self._load_mem0_context(memory, user_id, context, max_memories)
            except Exception as e:
                logger.error(f"Mem0 记忆加载失败: {e}")

        # 层级 2: 从 ConversationDB 获取近期消息
        if self._conversation_service and conversation_id:
            try:
                await self._load_conversation_context(conversation_id, context)
            except Exception as e:
                logger.error(f"会话历史加载失败: {e}")

        return context

    async def _load_mem0_context(
        self,
        memory,
        user_id: str,
        context: MemoryContext,
        max_memories: int,
    ) -> None:
        """从 Mem0 加载语义记忆"""
        import asyncio

        # 并行加载各类记忆
        results = await asyncio.gather(
            memory.get_user_profile(user_id),
            memory.get_user_diagnosis_history(user_id=user_id, limit=5),
            memory.search_relevant_context(
                query="用户偏好 饮食 运动 习惯",
                user_id=user_id,
                memory_types=["preference"],
                limit=5,
            ),
            memory.search_relevant_context(
                query="禁忌 过敏 妊娠",
                user_id=user_id,
                memory_types=["contraindication"],
                limit=max_memories,
            ),
            memory.search_relevant_context(
                query="养生建议",
                user_id=user_id,
                memory_types=["wellness"],
                limit=3,
            ),
            return_exceptions=True,
        )

        profile, history, preferences, contraindications, wellness = results

        if not isinstance(profile, Exception) and profile:
            context.user_profile = profile.get("metadata", {})

        if not isinstance(history, Exception) and history:
            context.diagnosis_history = history

        if not isinstance(preferences, Exception) and preferences:
            for pref in preferences:
                context.preferences.update(pref.get("metadata", {}))

        if not isinstance(contraindications, Exception) and contraindications:
            context.contraindications = contraindications

        if not isinstance(wellness, Exception) and wellness:
            context.wellness_advices = wellness

        logger.info(
            f"Mem0 记忆加载完成: profile={'有' if context.has_profile() else '无'}, "
            f"history={len(context.diagnosis_history)}, "
            f"contraindications={len(context.contraindications)}"
        )

    async def _load_conversation_context(
        self,
        conversation_id: str,
        context: MemoryContext,
    ) -> None:
        """从 ConversationDB 加载近期消息"""
        # 如果有 conversation_service，可以获取 turn_count 和摘要
        # 这里预留接口，具体实现取决于 ConversationService 的 API
        pass

    async def save_session_results(
        self,
        user_id: str,
        state: Dict[str, Any],
    ) -> None:
        """
        会话结束后持久化重要发现到 Mem0

        从 state 中提取：
        - 诊断结果（证型、症状、方剂）
        - 用户反馈
        - 新的禁忌信息

        Args:
            user_id: 用户 ID
            state: LangGraph 状态
        """
        memory = await self._get_tcm_memory()
        if not memory:
            return

        try:
            # 保存诊断结果
            diagnosis_result = state.get("diagnosis_result")
            if diagnosis_result:
                await self._save_diagnosis(memory, user_id, diagnosis_result)

            # 保存症状信息
            symptoms = state.get("symptoms") or state.get("collected_symptoms")
            if symptoms:
                await memory.add_symptom_memory(
                    user_id=user_id,
                    symptoms=symptoms,
                    confidence=0.8,
                )

            # 保存用户反馈
            feedback = state.get("user_feedback")
            if feedback:
                await self._save_feedback(memory, user_id, feedback)

            logger.info(f"会话结果已持久化到 Mem0, user_id={user_id}")

        except Exception as e:
            logger.error(f"保存会话结果失败: {e}")

    async def _save_diagnosis(
        self,
        memory,
        user_id: str,
        diagnosis_result: Dict[str, Any],
    ) -> None:
        """保存诊断记忆"""
        syndrome = diagnosis_result.get("syndrome") or diagnosis_result.get("证型")
        symptoms = diagnosis_result.get("symptoms") or diagnosis_result.get("症状", {})
        prescription = diagnosis_result.get("prescription") or diagnosis_result.get("方剂")

        if syndrome:
            await memory.add_diagnosis_memory(
                user_id=user_id,
                syndrome=syndrome,
                symptoms=symptoms,
                prescription=prescription,
            )

            if prescription:
                herbs = diagnosis_result.get("herbs") or diagnosis_result.get("药材", [])
                await memory.add_prescription_memory(
                    user_id=user_id,
                    prescription_name=prescription,
                    herbs=herbs if isinstance(herbs, list) else [],
                    syndrome=syndrome,
                    effectiveness=diagnosis_result.get("effectiveness"),
                )

    async def _save_feedback(
        self,
        memory,
        user_id: str,
        feedback: Dict[str, Any],
    ) -> None:
        """保存用户反馈记忆"""
        feedback_type = feedback.get("type")

        if feedback_type == "preference":
            await memory.add_wellness_memory(
                user_id=user_id,
                category=feedback.get("category", "偏好"),
                advice=feedback.get("content", ""),
                confidence=0.9,
            )
        elif feedback_type == "contraindication":
            await memory.add_contraindication_memory(
                user_id=user_id,
                contraindication_type=feedback.get("contraindication_type", "其他"),
                description=feedback.get("description", ""),
                related_herbs=feedback.get("related_herbs"),
                confidence=0.95,
            )

    def build_context_string(self, context: MemoryContext) -> str:
        """
        将 MemoryContext 构建为模型可用的上下文字符串

        替代 MemoryContextBuilder.build_context_string
        """
        import json

        parts = []

        if context.user_profile:
            profile = context.user_profile
            parts.append("【用户画像】")
            if profile.get("constitution"):
                parts.append(f"- 体质: {profile['constitution']}")
            if profile.get("preferences"):
                parts.append(f"- 偏好: {json.dumps(profile['preferences'], ensure_ascii=False)}")

        if context.diagnosis_history:
            parts.append("\n【历史诊断】")
            for i, diag in enumerate(context.diagnosis_history[:3], 1):
                syndrome = diag.get("metadata", {}).get("syndrome", "未知")
                content = diag.get("content", "")
                parts.append(f"{i}. {syndrome}: {content[:50]}...")

        if context.contraindications:
            parts.append("\n【禁忌信息】")
            for contra in context.contraindications:
                parts.append(f"- {contra.get('content', '')[:50]}")

        if context.wellness_advices:
            parts.append("\n【养生建议】")
            for advice in context.wellness_advices:
                parts.append(f"- {advice.get('content', '')[:50]}")

        return "\n".join(parts) if parts else ""
