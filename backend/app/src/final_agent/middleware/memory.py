"""
记忆中间件。

边界：
- 短期记忆由 LangGraph `checkpointer + thread_id` 负责。
- 长期记忆由 Mem0 负责，按 `user_id` 跨会话持久化。

职责：
1. `before_model`：加载跨线程长期上下文。
   - factual：数据库中的用户画像/base_profile
   - episodic：Mem0 中的情景摘要
   - semantic：Mem0 Graph 中的关系与稳定实体
2. `after_model`：只生成 episodic / semantic 长期记忆，不负责 factual 的生产。
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from sqlmodel import select

from app.src.common.config.prosgresql_config import async_db_manager
from app.src.model.account_model import Patient
from ..memory.tcm_memory import TCMMemory, get_tcm_memory
from .base import BaseMiddleware, MiddlewareConfig


logger = logging.getLogger(__name__)


class MemoryMiddlewareConfig(MiddlewareConfig):
    """记忆中间件配置。"""

    def __init__(
        self,
        enabled: bool = True,
        priority: int = 10,
        enable_auto_save: bool = True,
        retrieval_limit: int = 12,
        episodic_limit: int = 3,
    ):
        super().__init__(enabled=enabled, priority=priority, name="MemoryMiddleware")
        self.enable_auto_save = enable_auto_save
        self.retrieval_limit = retrieval_limit
        self.episodic_limit = episodic_limit


class MemoryMiddleware(BaseMiddleware):
    """负责长期记忆加载与持久化。"""

    def __init__(self, config: Optional[MemoryMiddlewareConfig] = None):
        super().__init__(config or MemoryMiddlewareConfig())
        self._tcm_memory: Optional[TCMMemory] = None

    async def _get_memory(self) -> TCMMemory:
        """获取记忆服务实例（懒加载）。"""
        if self._tcm_memory is None:
            self._tcm_memory = await get_tcm_memory()
        return self._tcm_memory

    def before_model(
        self, state: Dict[str, Any], runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """模型调用前：按当前 query 加载长期记忆。"""
        user_id = self._get_state_value(state, "user_id")
        if not user_id:
            return None

        if self._get_state_value(state, "memory_context"):
            return None

        query = self._extract_last_user_query(state)
        try:
            memory_context = asyncio.run(
                self._load_long_term_memory(
                    user_id=user_id,
                    query=query,
                )
            )
            if memory_context:
                logger.info(
                    "[MemoryMiddleware] 加载长期记忆: user_id=%s, query=%s, 记录数=%s",
                    user_id,
                    query or "<empty>",
                    len(memory_context.get("memories", [])),
                )
                return {"memory_context": memory_context}
        except Exception as error:
            logger.error("[MemoryMiddleware] 加载长期记忆失败: %s", error)

        return None

    async def _load_long_term_memory(
        self,
        user_id: str,
        query: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """从 Mem0 读取长期记忆，并整理成中间件上下文。"""
        try:
            memory = await self._get_memory()
            config: MemoryMiddlewareConfig = self.config

            if query:
                search_result = await memory.search_long_term_context(
                    query=query,
                    user_id=user_id,
                    limit=config.retrieval_limit,
                    episodic_limit=config.episodic_limit,
                )
            else:
                episodic_summaries = await memory.get_user_episode_summaries(
                    user_id=user_id,
                    limit=config.episodic_limit,
                )
                search_result = {
                    "user_id": user_id,
                    "query": query,
                    "memories": episodic_summaries,
                    "episodic_summaries": episodic_summaries,
                    "episodic_memories": episodic_summaries,
                    "semantic_relations": [],
                    "semantic_entities": self._collect_semantic_entities(
                        episodic_summaries
                    ),
                }

            user_profile_context = await self._load_db_user_profile(
                user_id=user_id,
            )
            memory_context = self._build_memory_context(
                search_result,
                user_profile_context=user_profile_context,
            )
            if (
                not memory_context.get("memories")
                and not memory_context.get("episodic_summaries")
                and not memory_context.get("factual", {}).get("user_profile")
            ):
                return None
            return memory_context
        except Exception as error:
            logger.error("[MemoryMiddleware] 加载长期记忆异常: %s", error)
            return None

    async def _load_db_user_profile(
        self,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            user_uuid = UUID(str(user_id))
        except (TypeError, ValueError):
            logger.warning("[MemoryMiddleware] user_id 不是合法 UUID: %s", user_id)
            return None

        try:
            async with async_db_manager.get_session() as session:
                patient_result = await session.exec(
                    select(Patient).where(Patient.account_id == user_uuid)
                )
                patient = patient_result.first()

                patient_profile = self._build_patient_profile(patient)

                locations: List[str] = []
                if patient:
                    locations.extend(
                        [
                            "patients.base_profile",
                            "patients.gender",
                            "patients.birth_date",
                        ]
                    )
                if not patient_profile and not locations:
                    return None

                return {
                    "source": "database",
                    "locations": locations,
                    "patient_profile": patient_profile,
                    "data": patient_profile,
                }
        except Exception as error:
            logger.warning("[MemoryMiddleware] 加载数据库画像失败: %s", error)
            return None

    def _build_patient_profile(self, patient: Optional[Patient]) -> Dict[str, Any]:
        if not patient:
            return {}

        base_profile = patient.base_profile or {}
        profile: Dict[str, Any] = {}

        if patient.gender:
            profile["gender"] = patient.gender
        if patient.birth_date:
            profile["birth_date"] = patient.birth_date.isoformat()

        field_mapping = {
            "constitution_type": "constitution",
            "taboo_items": "taboo_items",
            "medical_history": "medical_history",
            "family_history": "family_history",
            "allergy_info": "allergy_info",
        }
        for source_key, target_key in field_mapping.items():
            value = base_profile.get(source_key)
            if value not in (None, "", [], {}):
                profile[target_key] = value

        return profile

    def _build_memory_context(
        self,
        search_result: Dict[str, Any],
        user_profile_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """将记忆检索结果整理为主图可消费的上下文结构。"""
        memories = list(search_result.get("memories") or [])
        episodic_summaries = list(
            search_result.get("episodic_summaries")
            or search_result.get("episodic_memories")
            or []
        )
        semantic_relations = list(search_result.get("semantic_relations") or [])
        semantic_entities = list(search_result.get("semantic_entities") or [])
        if not semantic_entities:
            semantic_entities = self._collect_semantic_entities(
                episodic_summaries
            )

        factual = {
            "sources": {
                "user_profile": {
                    "source": (user_profile_context or {}).get("source", "database"),
                    "locations": (user_profile_context or {}).get("locations", []),
                },
            },
            "user_profile": (user_profile_context or {}).get("data", {}),
            "user_profile_details": user_profile_context or {},
        }

        episodic = {
            "sources": {
                "summaries": {
                    "source": "mem0",
                    "memory_type": "episodic_summary",
                    "memory_class": "episodic",
                }
            },
            "summaries": episodic_summaries,
        }

        semantic = {
            "sources": {
                "relations": {
                    "source": "mem0_graph",
                    "memory_class": "semantic",
                },
                "entities": {
                    "source": "derived",
                    "memory_class": "semantic",
                },
            },
            "relations": semantic_relations,
            "entities": semantic_entities,
        }

        return {
            "user_id": search_result.get("user_id"),
            "query": search_result.get("query"),
            "memories": memories,
            "factual": factual,
            "episodic": episodic,
            "semantic": semantic,
            "episodic_summaries": episodic_summaries,
            "episodic_memories": episodic_summaries,
            "semantic_relations": semantic_relations,
            "semantic_entities": semantic_entities,
            "user_profile": factual["user_profile"],
        }

    def after_model(
        self, state: Dict[str, Any], runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """模型调用后：提取并保存新的长期记忆。"""
        config: MemoryMiddlewareConfig = self.config
        if not config.enable_auto_save:
            return None

        user_id = self._get_state_value(state, "user_id")
        if not user_id:
            return None

        try:
            return asyncio.run(self._save_memories(state, user_id))
        except Exception as error:
            logger.error("[MemoryMiddleware] 保存记忆失败: %s", error)
            return None

    async def _save_memories(
        self, state: Dict[str, Any], user_id: str
    ) -> Optional[Dict[str, Any]]:
        """异步保存 episodic / semantic 长期记忆。"""
        try:
            session_id = self._get_state_value(state, "conversation_id", "default")
            memory = await self._get_memory()
            diagnosis_result = self._extract_structured_result(state)
            symptoms = self._extract_symptoms(state, diagnosis_result)

            episodic_memory_id = await self._save_episodic_memory(
                memory=memory,
                state=state,
                user_id=user_id,
                session_id=session_id,
                diagnosis_result=diagnosis_result,
                symptoms=symptoms,
            )

            return {
                "memory_saved": bool(episodic_memory_id),
                "episodic_memory_id": episodic_memory_id or None,
            }
        except Exception as error:
            logger.error("[MemoryMiddleware] 保存记忆失败: %s", error)
            return None

    async def _save_episodic_memory(
        self,
        memory: TCMMemory,
        state: Dict[str, Any],
        user_id: str,
        session_id: Optional[str],
        diagnosis_result: Optional[Dict[str, Any]],
        symptoms: Dict[str, Any],
    ) -> str:
        """自动生成并保存本轮情景记忆。"""
        query = self._extract_last_user_query(state)
        answer = self._extract_answer(state)
        task_type = self._extract_task_type(state)

        syndrome = None
        prescription = None
        if diagnosis_result:
            syndrome = diagnosis_result.get("syndrome") or diagnosis_result.get("证型")
            prescription = (
                diagnosis_result.get("prescription")
                or diagnosis_result.get("方剂")
                or diagnosis_result.get("prescription_name")
            )

        summary = self._build_episode_summary(
            query=query,
            answer=answer,
            task_type=task_type,
            syndrome=syndrome,
            prescription=prescription,
            symptoms=symptoms,
        )
        if not summary:
            return ""

        key_points = self._build_episode_key_points(
            query=query,
            answer=answer,
            syndrome=syndrome,
            prescription=prescription,
            symptoms=symptoms,
        )
        return await memory.add_episodic_memory(
            user_id=user_id,
            summary=summary,
            session_id=session_id,
            task_type=task_type,
            syndrome=syndrome,
            symptoms=symptoms or None,
            prescription=prescription,
            key_points=key_points,
            enable_graph=True,
        )

    def _extract_structured_result(
        self, state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """提取结构化业务结果。"""
        diagnosis_result = self._get_state_value(state, "diagnosis_result")
        if isinstance(diagnosis_result, dict):
            return diagnosis_result

        structured_data = self._get_state_value(state, "structured_data")
        if isinstance(structured_data, dict):
            return structured_data

        return None

    def _extract_last_user_query(self, state: Dict[str, Any]) -> Optional[str]:
        """提取最后一条用户消息。"""
        messages = self._get_state_value(state, "messages", [])
        for message in reversed(messages):
            role = self._get_message_role(message)
            if role in {"human", "user"}:
                content = self._get_message_content(message)
                if content:
                    return content
        return None

    def _extract_answer(self, state: Dict[str, Any]) -> Optional[str]:
        """提取最终回答文本。"""
        answer = self._get_state_value(state, "answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        messages = self._get_state_value(state, "messages", [])
        for message in reversed(messages):
            role = self._get_message_role(message)
            if role in {"ai", "assistant"}:
                content = self._get_message_content(message)
                if content:
                    return content
        return None

    def _extract_task_type(self, state: Dict[str, Any]) -> str:
        """提取当前任务类型。"""
        router = self._get_state_value(state, "router")
        if isinstance(router, dict):
            return router.get("query_type") or router.get("sub_type") or "consultation"

        query_type = getattr(router, "query_type", None)
        if query_type:
            return query_type

        sub_type = getattr(router, "sub_type", None)
        if sub_type:
            return sub_type

        return "consultation"

    def _extract_symptoms(
        self,
        state: Dict[str, Any],
        diagnosis_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """统一提取症状字段。"""
        candidates = [
            self._get_state_value(state, "symptoms"),
            self._get_state_value(state, "collected_symptoms"),
        ]
        if diagnosis_result:
            candidates.extend(
                [
                    diagnosis_result.get("symptoms"),
                    diagnosis_result.get("症状"),
                ]
            )

        for candidate in candidates:
            normalized = self._normalize_symptoms(candidate)
            if normalized:
                return normalized
        return {}

    @staticmethod
    def _normalize_symptoms(symptoms: Any) -> Dict[str, Any]:
        """将症状统一归一化为字典。"""
        if isinstance(symptoms, dict):
            return symptoms
        if isinstance(symptoms, list):
            return {str(item): True for item in symptoms if item}
        if isinstance(symptoms, str) and symptoms.strip():
            return {"主诉": symptoms.strip()}
        return {}

    @staticmethod
    def _normalize_string_list(values: Any) -> List[str]:
        """将输入统一归一化为字符串列表。"""
        if isinstance(values, list):
            return [str(item).strip() for item in values if str(item).strip()]
        if isinstance(values, str) and values.strip():
            return [values.strip()]
        return []

    @staticmethod
    def _truncate_text(text: Optional[str], limit: int = 80) -> str:
        """截断过长文本，便于写入记忆摘要。"""
        if not text:
            return ""
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1]}…"

    def _build_episode_summary(
        self,
        *,
        query: Optional[str],
        answer: Optional[str],
        task_type: str,
        syndrome: Optional[str],
        prescription: Optional[str],
        symptoms: Dict[str, Any],
    ) -> str:
        """构建规则版情景摘要。"""
        parts: List[str] = []
        if query:
            parts.append(f"用户发起了“{self._truncate_text(query, 60)}”相关{task_type}咨询")
        if symptoms:
            parts.append(f"提取到症状：{self._truncate_text(self._format_symptoms(symptoms), 80)}")
        if syndrome:
            parts.append(f"诊断倾向为{syndrome}")
        if prescription:
            parts.append(f"建议方剂为{prescription}")
        if answer:
            parts.append(f"最终回复聚焦：{self._truncate_text(answer, 100)}")
        return "；".join(parts)

    def _build_episode_key_points(
        self,
        *,
        query: Optional[str],
        answer: Optional[str],
        syndrome: Optional[str],
        prescription: Optional[str],
        symptoms: Dict[str, Any],
    ) -> List[str]:
        """构建情景记忆关键点。"""
        key_points: List[str] = []
        if query:
            key_points.append(f"主诉：{self._truncate_text(query, 40)}")
        if symptoms:
            key_points.append(
                f"症状：{self._truncate_text(self._format_symptoms(symptoms), 50)}"
            )
        if syndrome:
            key_points.append(f"诊断：{syndrome}")
        if prescription:
            key_points.append(f"方剂：{prescription}")
        if answer:
            key_points.append(f"答复：{self._truncate_text(answer, 50)}")
        return key_points[:5]

    @staticmethod
    def _get_message_role(message: Any) -> Optional[str]:
        """获取消息角色。"""
        if hasattr(message, "type"):
            return getattr(message, "type")
        if isinstance(message, dict):
            return message.get("role") or message.get("type")
        return None

    def _get_message_content(self, message: Any) -> str:
        """将消息 content 规范化为纯文本。"""
        if hasattr(message, "content"):
            return self._content_to_text(getattr(message, "content"))
        if isinstance(message, dict):
            return self._content_to_text(message.get("content"))
        return ""

    def _content_to_text(self, content: Any) -> str:
        """兼容字符串、分段内容列表与字典结构。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
            return "\n".join(part.strip() for part in parts if part.strip())
        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            return str(text).strip() if text else ""
        return str(content).strip()

    def _collect_semantic_entities(self, memories: List[Dict[str, Any]]) -> List[str]:
        """从记忆元数据中抽取一批高价值语义实体。"""
        entities: List[str] = []
        for memory in memories:
            metadata = memory.get("metadata") or {}
            candidates: List[str] = []
            for field in [
                metadata.get("syndrome"),
                metadata.get("prescription"),
                metadata.get("constitution"),
            ]:
                if field:
                    candidates.append(str(field))
            candidates.extend(self._normalize_string_list(metadata.get("herbs")))
            candidates.extend(
                self._normalize_string_list(metadata.get("related_entities"))
            )
            for item in candidates:
                if item and item not in entities:
                    entities.append(item)
        return entities

    def _format_symptoms(self, symptoms: Dict[str, Any]) -> str:
        """将症状结构格式化为紧凑文本。"""
        parts: List[str] = []
        for key, value in symptoms.items():
            if isinstance(value, bool) and value:
                parts.append(str(key))
            elif isinstance(value, str) and value:
                parts.append(f"{key}:{value}")
            elif isinstance(value, list) and value:
                parts.append(f"{key}:{','.join(str(item) for item in value)}")
            elif value not in (None, "", False):
                parts.append(f"{key}:{value}")
        return " ".join(parts)

    def wrap_tool_call(
        self, tool_call: Callable, tool_name: str, state: Dict[str, Any]
    ) -> Callable:
        """当前不包装工具调用。"""
        return tool_call


def create_memory_middleware(
    enabled: bool = True, auto_save: bool = True
) -> MemoryMiddleware:
    """创建记忆中间件的便捷函数。"""
    config = MemoryMiddlewareConfig(enabled=enabled, enable_auto_save=auto_save)
    return MemoryMiddleware(config)
