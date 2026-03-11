"""
TCM 长期记忆服务。

当前 final_agent 只保留两类 Mem0 能力：
1. `episodic_summary`：跨线程的情景摘要记忆
2. `semantic`：基于图检索得到的关系与稳定实体

说明：
- `factual` 用户画像由数据库提供，不在这里生产。
- 本模块只负责 Mem0 适配、情景记忆写入、情景/语义检索。
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .mem0_config import Mem0Config

logger = logging.getLogger(__name__)

try:
    from mem0 import AsyncMemory

    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    AsyncMemory = None


class TCMMemoryMetadata(BaseModel):
    """情景记忆元数据。"""

    memory_type: str = Field(
        default="episodic_summary",
        description="记忆类型",
        pattern="^episodic_summary$",
    )
    memory_class: str = Field(
        default="episodic",
        description="长期记忆分类",
        pattern="^(episodic|semantic)$",
    )

    user_id: str
    session_id: Optional[str] = None

    syndrome: Optional[str] = None
    symptoms: Optional[Dict[str, Any]] = None
    prescription: Optional[str] = None
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    task_type: Optional[str] = None
    related_entities: Optional[List[str]] = None

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = Field(default=0.75, ge=0, le=1)
    source: str = Field(default="agent", description="来源：agent/user/expert")


class MemoryType(str, Enum):
    """当前 final_agent 保留的记忆类型。"""

    EPISODIC_SUMMARY = "episodic_summary"


class LongTermMemoryClass(str, Enum):
    """当前 final_agent 保留的长期记忆分类。"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class TCMMemory:
    """基于 Mem0 的精简长期记忆服务。"""

    def __init__(self, config: Mem0Config):
        if not MEM0_AVAILABLE:
            raise ImportError(
                "请先安装 mem0ai: pip install mem0ai\n"
                "或使用: uv add mem0ai"
            )

        self.config = config
        self._memory: Optional[AsyncMemory] = None

    @staticmethod
    def _model_dump(model: BaseModel) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump(exclude_none=True)
        return model.dict(exclude_none=True)

    @staticmethod
    def _extract_results(result: Any) -> List[Dict[str, Any]]:
        if result is None:
            return []
        if isinstance(result, dict):
            records = result.get("results") or result.get("data") or []
            return records if isinstance(records, list) else []
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def _extract_relations(result: Any) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        relations = result.get("relations") or result.get("graph_relations") or []
        if isinstance(relations, list):
            return relations
        if isinstance(relations, dict):
            return [relations]
        return []

    def _extract_memory_id(self, result: Any) -> str:
        records = self._extract_results(result)
        if records:
            return str(records[0].get("id", ""))
        if isinstance(result, dict):
            return str(result.get("id", ""))
        return ""

    @staticmethod
    def _as_messages(content: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": content}]

    async def _add_text_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: TCMMemoryMetadata,
        memory_type: Optional[str] = None,
        enable_graph: bool = False,
        infer: bool = False,
    ) -> str:
        if self._memory is None:
            await self.initialize()

        payload = {
            "messages": self._as_messages(content),
            "user_id": user_id,
            "metadata": self._model_dump(metadata),
            "infer": infer,
        }
        if memory_type:
            payload["memory_type"] = memory_type
        if enable_graph:
            payload["enable_graph"] = True

        try:
            result = await self._memory.add(**payload)
        except TypeError:
            payload.pop("messages", None)
            payload["content"] = content
            if not enable_graph:
                payload.pop("enable_graph", None)
            result = await self._memory.add(**payload)

        return self._extract_memory_id(result)

    async def _search_memories(
        self,
        *,
        query: str,
        user_id: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        enable_graph: bool = False,
    ) -> Dict[str, Any]:
        if self._memory is None:
            await self.initialize()

        payload = {
            "query": query,
            "user_id": user_id,
            "limit": limit,
            "filters": filters or {"user_id": user_id},
        }
        if enable_graph:
            payload["enable_graph"] = True

        try:
            result = await self._memory.search(**payload)
        except TypeError:
            payload.pop("enable_graph", None)
            result = await self._memory.search(**payload)

        return result if isinstance(result, dict) else {"results": self._extract_results(result)}

    async def _get_all_memories(
        self,
        *,
        user_id: str,
        limit: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._memory is None:
            await self.initialize()

        result = await self._memory.get_all(user_id=user_id, limit=limit, filters=filters)
        return result if isinstance(result, dict) else {"results": self._extract_results(result)}

    @staticmethod
    def _normalize_memory_record(record: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(record or {})
        metadata = dict(normalized.get("metadata") or {})
        metadata["memory_type"] = metadata.get("memory_type") or MemoryType.EPISODIC_SUMMARY.value
        metadata["memory_class"] = metadata.get("memory_class") or LongTermMemoryClass.EPISODIC.value
        normalized["metadata"] = metadata
        normalized["memory_type"] = metadata["memory_type"]
        normalized["memory_class"] = metadata["memory_class"]
        return normalized

    @staticmethod
    def _format_semantic_relation(relation: Any) -> Dict[str, Any]:
        if isinstance(relation, dict):
            return relation
        return {"relation": relation}

    async def initialize(self) -> None:
        """初始化 Mem0 客户端。"""
        try:
            config_dict = self.config.to_mem0_config()
            self._memory = AsyncMemory.from_config(config_dict)
            logger.info(
                "[TCMMemory] Mem0 初始化成功, vector_store=%s",
                self.config.vector_store_provider,
            )
        except Exception as error:
            logger.error("[TCMMemory] Mem0 初始化失败: %s", error)
            raise

    async def add_episodic_memory(
        self,
        user_id: str,
        summary: str,
        session_id: Optional[str] = None,
        task_type: str = "consultation",
        syndrome: Optional[str] = None,
        symptoms: Optional[Dict[str, Any]] = None,
        prescription: Optional[str] = None,
        key_points: Optional[List[str]] = None,
        confidence: float = 0.75,
        enable_graph: bool = False,
    ) -> str:
        """添加情景记忆：过去互动/已完成任务的摘要。"""
        key_points = [item for item in (key_points or []) if item]
        metadata = TCMMemoryMetadata(
            memory_type=MemoryType.EPISODIC_SUMMARY.value,
            memory_class=LongTermMemoryClass.EPISODIC.value,
            user_id=user_id,
            session_id=session_id,
            syndrome=syndrome,
            symptoms=symptoms,
            prescription=prescription,
            summary=summary,
            task_type=task_type,
            key_points=key_points,
            confidence=confidence,
            related_entities=[item for item in [syndrome, prescription] if item],
        )

        content = f"情景记忆[{task_type}]: {summary}"
        if symptoms:
            content += f" | 症状: {self._format_symptoms(symptoms)}"
        if syndrome:
            content += f" | 诊断: {syndrome}"
        if prescription:
            content += f" | 方剂: {prescription}"
        if key_points:
            content += f" | 关键点: {'；'.join(key_points)}"

        return await self._add_text_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            memory_type=MemoryType.EPISODIC_SUMMARY.value,
            enable_graph=enable_graph,
        )

    async def search_semantic_relations(
        self,
        query: str,
        user_id: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """检索与当前查询相关的语义关系。"""
        results = await self._search_memories(
            query=query,
            user_id=user_id,
            limit=limit,
            filters={"user_id": user_id},
            enable_graph=True,
        )
        return [self._format_semantic_relation(item) for item in self._extract_relations(results)]

    async def get_user_episode_summaries(
        self,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """获取最近的情景记忆摘要。"""
        results = await self._get_all_memories(
            user_id=user_id,
            filters={"memory_type": MemoryType.EPISODIC_SUMMARY.value},
            limit=limit,
        )
        return [self._normalize_memory_record(item) for item in self._extract_results(results)]

    async def search_long_term_context(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        episodic_limit: int = 3,
    ) -> Dict[str, Any]:
        """统一检索跨线程的 episodic / semantic 长期记忆。"""
        search_result = await self._search_memories(
            query=query,
            user_id=user_id,
            limit=limit,
            filters={"user_id": user_id},
            enable_graph=True,
        )

        memories = [self._normalize_memory_record(item) for item in self._extract_results(search_result)]
        episodic_memories = [
            item
            for item in memories
            if item.get("memory_class") == LongTermMemoryClass.EPISODIC.value
        ]

        if len(episodic_memories) < episodic_limit:
            seen_ids = {item.get("id") for item in episodic_memories}
            recent_episodes = await self.get_user_episode_summaries(user_id, limit=episodic_limit)
            for episode in recent_episodes:
                if episode.get("id") not in seen_ids:
                    episodic_memories.append(episode)
                if len(episodic_memories) >= episodic_limit:
                    break

        relations = [self._format_semantic_relation(item) for item in self._extract_relations(search_result)]
        semantic_entities = self._collect_semantic_entities(episodic_memories)

        return {
            "user_id": user_id,
            "query": query,
            "memories": episodic_memories,
            "episodic_summaries": episodic_memories,
            "episodic_memories": episodic_memories,
            "semantic_relations": relations,
            "semantic_entities": semantic_entities,
        }

    def _collect_semantic_entities(self, memories: List[Dict[str, Any]]) -> List[str]:
        """从情景记忆中抽取一批高价值语义实体。"""
        entities: List[str] = []
        for memory in memories:
            metadata = memory.get("metadata") or {}
            candidates: List[str] = []
            for field in [metadata.get("syndrome"), metadata.get("prescription")]:
                if field:
                    candidates.append(str(field))
            candidates.extend(self._normalize_string_list(metadata.get("related_entities")))
            for item in candidates:
                if item and item not in entities:
                    entities.append(item)
        return entities

    @staticmethod
    def _normalize_string_list(values: Any) -> List[str]:
        if isinstance(values, list):
            return [str(item).strip() for item in values if str(item).strip()]
        if isinstance(values, str) and values.strip():
            return [values.strip()]
        return []

    def _format_symptoms(self, symptoms: Dict[str, Any]) -> str:
        """格式化症状描述。"""
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


_tcm_memory: Optional[TCMMemory] = None


async def get_tcm_memory() -> TCMMemory:
    """获取 TCM 记忆服务单例。"""
    global _tcm_memory
    if _tcm_memory is None:
        config = Mem0Config.from_env()
        _tcm_memory = TCMMemory(config)
        await _tcm_memory.initialize()
    return _tcm_memory
