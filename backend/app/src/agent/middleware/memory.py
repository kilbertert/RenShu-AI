"""
Mem0 记忆中间件

在对话过程中自动管理用户记忆：

1. before_model: 从Mem0加载相关记忆并注入到上下文
2. after_model: 提取对话中的新信息并保存到Mem0

优先级: P2 (在guardrails之后，logging之前)
"""

import json
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime

from .base import BaseMiddleware, MiddlewareConfig
from ..memory.mem0_config import Mem0Config
from ..memory.tcm_memory import TCMMemory, MemoryType, get_tcm_memory


logger = logging.getLogger(__name__)


class MemoryMiddlewareConfig(MiddlewareConfig):
    """记忆中间件配置"""

    def __init__(
        self,
        enabled: bool = True,
        priority: int = 20,  # P2 优先级
        max_memories_to_load: int = 10,
        memory_confidence_threshold: float = 0.3,
        enable_auto_save: bool = True,
        enable_decay_check: bool = False,
    ):
        super().__init__(enabled=enabled, priority=priority, name="MemoryMiddleware")
        self.max_memories_to_load = max_memories_to_load
        self.memory_confidence_threshold = memory_confidence_threshold
        self.enable_auto_save = enable_auto_save
        self.enable_decay_check = enable_decay_check


class MemoryMiddleware(BaseMiddleware):
    """
    Mem0 记忆中间件

    功能：
    1. 对话开始时加载用户画像、历史诊断、偏好等
    2. 对话结束时提取新信息（症状、诊断结果、用户反馈）
    3. 定期检查记忆衰减
    """

    def __init__(self, config: Optional[MemoryMiddlewareConfig] = None):
        super().__init__(config or MemoryMiddlewareConfig())
        self._tcm_memory: Optional[TCMMemory] = None

    async def _get_memory(self) -> TCMMemory:
        """获取记忆服务实例（懒加载）"""
        if self._tcm_memory is None:
            self._tcm_memory = await get_tcm_memory()
        return self._tcm_memory

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前：加载用户记忆

        加载顺序：
        1. 用户画像（体质、基本信息）
        2. 历史诊断（最近的几次）
        3. 用户偏好（饮食、运动等）
        4. 禁忌信息（过敏、妊娠等）
        """
        import asyncio

        # 获取用户ID
        user_id = self._get_state_value(state, "user_id")
        if not user_id:
            logger.debug("[MemoryMiddleware] 无用户ID，跳过记忆加载")
            return None

        # 检查是否已有记忆上下文（避免重复加载）
        existing_context = self._get_state_value(state, "memory_context")
        if existing_context:
            logger.debug("[MemoryMiddleware] 记忆已加载，跳过")
            return None

        # 异步加载记忆
        return asyncio.run(self._load_memories(state, user_id))

    async def _load_memories(
        self,
        state: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """异步加载用户记忆"""
        try:
            memory = await self._get_memory()
            config: MemoryMiddlewareConfig = self.config

            memory_context = {
                "user_profile": None,
                "diagnosis_history": [],
                "preferences": {},
                "contraindications": [],
                "wellness_advices": [],
                "loaded_at": datetime.now().isoformat()
            }

            # 1. 加载用户画像
            logger.info(f"[MemoryMiddleware] 为用户 {user_id} 加载记忆")
            profile = await memory.get_user_profile(user_id)
            if profile:
                memory_context["user_profile"] = profile.get("metadata", {})
                logger.debug(f"[MemoryMiddleware] 加载用户画像: {memory_context['user_profile']}")

            # 2. 加载诊断历史
            history = await memory.get_user_diagnosis_history(
                user_id=user_id,
                limit=5
            )
            if history:
                memory_context["diagnosis_history"] = history
                logger.debug(f"[MemoryMiddleware] 加载了 {len(history)} 条诊断历史")

            # 3. 加载用户偏好
            preferences = await memory.search_relevant_context(
                query="用户偏好 饮食 运动 习惯",
                user_id=user_id,
                memory_types=["preference"],
                limit=5
            )
            if preferences:
                for pref in preferences:
                    metadata = pref.get("metadata", {})
                    memory_context["preferences"].update(metadata)
                logger.debug(f"[MemoryMiddleware] 加载了 {len(preferences)} 条偏好记忆")

            # 4. 加载禁忌信息
            contraindications = await memory.search_relevant_context(
                query="禁忌 过敏 妊娠",
                user_id=user_id,
                memory_types=["contraindication"],
                limit=10
            )
            if contraindications:
                memory_context["contraindications"] = contraindications
                logger.debug(f"[MemoryMiddleware] 加载了 {len(contraindications)} 条禁忌")

            # 5. 加载养生建议
            wellness = await memory.search_relevant_context(
                query="养生建议",
                user_id=user_id,
                memory_types=["wellness"],
                limit=3
            )
            if wellness:
                memory_context["wellness_advices"] = wellness
                logger.debug(f"[MemoryMiddleware] 加载了 {len(wellness)} 条养生建议")

            # 可选：检查记忆衰减
            if config.enable_decay_check:
                await memory.decay_memory_confidence(user_id)

            return {"memory_context": memory_context}

        except Exception as e:
            logger.error(f"[MemoryMiddleware] 加载记忆失败: {e}")
            return None

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后：提取并保存新记忆

        提取信息：
        1. 对话中提到的症状
        2. 诊断结果
        3. 用户偏好/反馈
        4. 方剂信息
        """
        import asyncio

        config: MemoryMiddlewareConfig = self.config
        if not config.enable_auto_save:
            return None

        # 获取用户ID
        user_id = self._get_state_value(state, "user_id")
        if not user_id:
            return None

        # 异步保存记忆
        return asyncio.run(self._save_memories(state, user_id))

    async def _save_memories(
        self,
        state: Dict[str, Any],
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """异步保存新记忆"""
        try:
            memory = await self._get_memory()

            # 获取对话输出
            output = self._get_state_value(state, "output") or {}
            response_content = self._get_state_value(state, "response") or ""

            # 获取诊断结果（如果有）
            diagnosis_result = self._get_state_value(state, "diagnosis_result")
            if diagnosis_result:
                await self._save_diagnosis_memory(memory, user_id, diagnosis_result)

            # 获取症状信息
            symptoms = self._get_state_value(state, "symptoms") or self._get_state_value(state, "collected_symptoms")
            if symptoms:
                await memory.add_symptom_memory(
                    user_id=user_id,
                    symptoms=symptoms,
                    confidence=0.8
                )
                logger.debug(f"[MemoryMiddleware] 保存症状记忆: {symptoms}")

            # 获取用户反馈
            user_feedback = self._get_state_value(state, "user_feedback")
            if user_feedback:
                await self._save_feedback_memory(memory, user_id, user_feedback)

            return {"memory_saved": True}

        except Exception as e:
            logger.error(f"[MemoryMiddleware] 保存记忆失败: {e}")
            return None

    async def _save_diagnosis_memory(
        self,
        memory: TCMMemory,
        user_id: str,
        diagnosis_result: Dict[str, Any]
    ):
        """保存诊断记忆"""
        syndrome = diagnosis_result.get("syndrome") or diagnosis_result.get("证型")
        symptoms = diagnosis_result.get("symptoms") or diagnosis_result.get("症状", {})
        prescription = diagnosis_result.get("prescription") or diagnosis_result.get("方剂")

        if syndrome:
            memory_id = await memory.add_diagnosis_memory(
                user_id=user_id,
                syndrome=syndrome,
                symptoms=symptoms,
                prescription=prescription
            )
            logger.info(f"[MemoryMiddleware] 保存诊断记忆: {syndrome}, ID: {memory_id}")

            # 如果有方剂，也保存方剂记忆
            if prescription:
                herbs = diagnosis_result.get("herbs") or diagnosis_result.get("药材", [])
                effectiveness = diagnosis_result.get("effectiveness")

                await memory.add_prescription_memory(
                    user_id=user_id,
                    prescription_name=prescription,
                    herbs=herbs if isinstance(herbs, list) else [],
                    syndrome=syndrome,
                    effectiveness=effectiveness
                )

    async def _save_feedback_memory(
        self,
        memory: TCMMemory,
        user_id: str,
        feedback: Dict[str, Any]
    ):
        """保存用户反馈记忆"""
        feedback_type = feedback.get("type")

        if feedback_type == "preference":
            # 保存偏好记忆
            await memory.add_wellness_memory(
                user_id=user_id,
                category=feedback.get("category", "偏好"),
                advice=feedback.get("content", ""),
                confidence=0.9
            )

        elif feedback_type == "rating":
            # 更新记忆有效性评分
            memory_id = feedback.get("memory_id")
            if memory_id:
                score = feedback.get("score", 0.5)
                await memory.update_memory_effectiveness(memory_id, score)

        elif feedback_type == "contraindication":
            # 保存禁忌记忆
            await memory.add_contraindication_memory(
                user_id=user_id,
                contraindication_type=feedback.get("contraindication_type", "其他"),
                description=feedback.get("description", ""),
                related_herbs=feedback.get("related_herbs"),
                confidence=0.95
            )

    def wrap_tool_call(
        self,
        tool_call: Callable,
        tool_name: str,
        state: Dict[str, Any]
    ) -> Callable:
        """包装工具调用（如果需要监控特定工具）"""
        # 可以在这里监控特定工具的调用并记录相关记忆
        return tool_call


class MemoryContextBuilder:
    """
    记忆上下文构建器

    将加载的记忆转换为模型可用的上下文格式
    """

    @staticmethod
    def build_context_string(memory_context: Dict[str, Any]) -> str:
        """
        构建记忆上下文字符串

        Args:
            memory_context: 记忆上下文字典

        Returns:
            格式化的上下文字符串
        """
        parts = []

        # 用户画像
        if memory_context.get("user_profile"):
            profile = memory_context["user_profile"]
            parts.append(f"【用户画像】")
            if profile.get("constitution"):
                parts.append(f"- 体质: {profile['constitution']}")
            if profile.get("preferences"):
                parts.append(f"- 偏好: {json.dumps(profile['preferences'], ensure_ascii=False)}")

        # 诊断历史
        if memory_context.get("diagnosis_history"):
            parts.append(f"\n【历史诊断】")
            for i, diag in enumerate(memory_context["diagnosis_history"][:3], 1):
                syndrome = diag.get("metadata", {}).get("syndrome", "未知")
                content = diag.get("content", "")
                parts.append(f"{i}. {syndrome}: {content[:50]}...")

        # 禁忌
        if memory_context.get("contraindications"):
            parts.append(f"\n【禁忌信息】")
            for contra in memory_context["contraindications"]:
                parts.append(f"- {contra.get('content', '')[:50]}")

        # 养生建议
        if memory_context.get("wellness_advices"):
            parts.append(f"\n【养生建议】")
            for advice in memory_context["wellness_advices"]:
                parts.append(f"- {advice.get('content', '')[:50]}")

        return "\n".join(parts) if parts else ""

    @staticmethod
    def build_context_dict(memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建结构化记忆上下文

        Returns:
            结构化的上下文字典
        """
        return {
            "user_profile": memory_context.get("user_profile", {}),
            "recent_diagnoses": [
                {
                    "syndrome": d.get("metadata", {}).get("syndrome"),
                    "prescription": d.get("metadata", {}).get("prescription"),
                    "date": d.get("metadata", {}).get("created_at")
                }
                for d in memory_context.get("diagnosis_history", [])
            ],
            "preferences": memory_context.get("preferences", {}),
            "contraindications": [
                c.get("content", "") for c in memory_context.get("contraindications", [])
            ],
            "wellness": [
                w.get("content", "") for w in memory_context.get("wellness_advices", [])
            ]
        }


def create_memory_middleware(
    enabled: bool = True,
    max_memories: int = 10,
    auto_save: bool = True,
    enable_decay: bool = False
) -> MemoryMiddleware:
    """
    创建记忆中间件的便捷函数

    Args:
        enabled: 是否启用
        max_memories: 最大加载记忆数
        auto_save: 是否自动保存
        enable_decay: 是否启用衰减检查

    Returns:
        配置好的记忆中间件
    """
    config = MemoryMiddlewareConfig(
        enabled=enabled,
        max_memories_to_load=max_memories,
        enable_auto_save=auto_save,
        enable_decay_check=enable_decay
    )
    return MemoryMiddleware(config)
