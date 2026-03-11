"""
TCM 记忆层 - 基于 Mem0 的中医定制记忆服务

扩展 Mem0 能力：
1. TCM 元数据模式（症状、证型、方剂）
2. 跨会话用户画像
3. 诊断历史追踪
4. 记忆有效性评分
5. 时间衰减机制
6. 冲突检测与解决
7. 养生建议记忆
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from mem0 import AsyncMemory
    from mem0.configs import MemoryConfig
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    AsyncMemory = None
    MemoryConfig = None

from .mem0_config import Mem0Config


class TCMMemoryMetadata(BaseModel):
    """TCM 记忆元数据模式"""

    # 记忆类型
    memory_type: str = Field(
        ...,
        description="记忆类型",
        pattern="^(symptom|diagnosis|prescription|wellness|preference|profile)$"
    )

    # 用户相关
    user_id: str
    session_id: Optional[str] = None

    # TCM 特定字段
    syndrome: Optional[str] = None              # 证型
    symptoms: Optional[Dict[str, Any]] = None    # 症状
    prescription: Optional[str] = None           # 方剂
    herbs: Optional[List[str]] = None            # 药材列表

    # 时间相关
    season: Optional[str] = None                # 就诊季节
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    # 质量标记
    confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="记忆置信度"
    )
    source: str = Field(default="agent", description="来源：agent/user/expert")

    # 时间衰减相关
    expires_at: Optional[str] = None           # 过期时间
    access_count: int = Field(default=0, description="访问次数")
    last_accessed: Optional[str] = None        # 最后访问时间

    # 版本控制（用于冲突解决）
    version: int = Field(default=1, description="版本号")
    superseded_by: Optional[str] = None        # 被哪个记忆替代
    supersedes: Optional[str] = None           # 替代了哪个记忆


class MemoryType(str, Enum):
    """记忆类型枚举"""
    SYMPTOM = "symptom"           # 症状记忆
    DIAGNOSIS = "diagnosis"       # 诊断记忆
    PRESCRIPTION = "prescription" # 方剂记忆
    WELLNESS = "wellness"         # 养生记忆
    PREFERENCE = "preference"     # 偏好记忆
    PROFILE = "profile"           # 用户画像
    CONTRAINDICATION = "contraindication"  # 禁忌记忆


class ConflictResolution(str, Enum):
    """冲突解决策略"""
    KEEP_HIGHEST_CONFIDENCE = "keep_highest"  # 保留置信度最高的
    KEEP_MOST_RECENT = "keep_recent"          # 保留最新的
    MERGE = "merge"                            # 合并
    MANUAL = "manual"                          # 手动处理


class TCMMemory:
    """
    TCM 记忆服务

    基于 Mem0，提供中医领域定制的记忆操作
    """

    def __init__(self, config: Mem0Config):
        if not MEM0_AVAILABLE:
            raise ImportError(
                "请先安装 mem0ai: pip install mem0ai\n"
                "或使用: uv add mem0ai"
            )

        self.config = config
        self._memory: Optional[AsyncMemory] = None

    async def initialize(self) -> None:
        """初始化 Mem0 客户端"""
        try:
            mem0_config = MemoryConfig()

            # 转换为 Mem0 格式
            config_dict = self.config.to_mem0_config()

            # 配置向量存储
            if self.config.vector_store_provider == "qdrant":
                from mem0.configs.vector_stores import QdrantConfig

                qdrant_config = QdrantConfig(
                    host=self.config.vector_store_config["host"],
                    port=self.config.vector_store_config.get("port", 6333),
                    collection_name=self.config.vector_store_config.get("collection_name", "tcm_memories"),
                    embedding_model_dims=self.config.vector_store_config.get("embedding_model_dims", 1536),
                    on_disk=True
                )
                mem0_config.vector_store = qdrant_config

            # 配置图存储
            if self.config.graph_store_provider == "neo4j":
                from mem0.configs.graph_stores import Neo4jConfig

                neo4j_config = Neo4jConfig(
                    url=self.config.graph_store_config.get("url", "bolt://localhost:7687"),
                    username=self.config.graph_store_config.get("username", "neo4j"),
                    password=self.config.graph_store_config.get("password", "tcm_graph_2026")
                )
                mem0_config.graph_store = neo4j_config

            self._memory = AsyncMemory.from_config(mem0_config)
            print("[TCMMemory] Mem0 初始化成功")

        except Exception as e:
            print(f"[TCMMemory] Mem0 初始化失败: {e}")
            raise

    async def add_symptom_memory(
        self,
        user_id: str,
        symptoms: Dict[str, Any],
        confidence: float = 0.8
    ) -> str:
        """
        添加症状记忆

        Args:
            user_id: 用户ID
            symptoms: 症状数据
            confidence: 置信度

        Returns:
            记忆ID
        """
        if self._memory is None:
            await self.initialize()

        metadata = TCMMemoryMetadata(
            memory_type="symptom",
            user_id=user_id,
            symptoms=symptoms,
            confidence=confidence
        )

        result = await self._memory.add(
            content=self._format_symptoms(symptoms),
            user_id=user_id,
            metadata=metadata.dict()
        )

        if result.get("results"):
            return result["results"][0].get("id", "")

        return ""

    async def add_diagnosis_memory(
        self,
        user_id: str,
        syndrome: str,
        symptoms: Dict[str, Any],
        prescription: Optional[str] = None
    ) -> str:
        """添加诊断记忆"""
        if self._memory is None:
            await self.initialize()

        metadata = TCMMemoryMetadata(
            memory_type="diagnosis",
            user_id=user_id,
            syndrome=syndrome,
            symptoms=symptoms,
            prescription=prescription,
            confidence=0.9
        )

        content = f"诊断: {syndrome}"
        if symptoms:
            content += f" | 症状: {self._format_symptoms(symptoms)}"
        if prescription:
            content += f" | 方剂: {prescription}"

        result = await self._memory.add(
            content=content,
            user_id=user_id,
            metadata=metadata.dict()
        )

        if result.get("results"):
            return result["results"][0].get("id", "")

        return ""

    async def add_prescription_memory(
        self,
        user_id: str,
        prescription_name: str,
        herbs: List[str],
        syndrome: str,
        effectiveness: Optional[str] = None
    ) -> str:
        """添加方剂记忆"""
        if self._memory is None:
            await self.initialize()

        metadata = TCMMemoryMetadata(
            memory_type="prescription",
            user_id=user_id,
            syndrome=syndrome,
            prescription=prescription_name,
            herbs=herbs,
            confidence=0.95
        )

        content = f"方剂: {prescription_name} | 证型: {syndrome} | 药材: {', '.join(herbs)}"
        if effectiveness:
            content += f" | 效果: {effectiveness}"

        result = await self._memory.add(
            content=content,
            user_id=user_id,
            metadata=metadata.dict()
        )

        if result.get("results"):
            return result["results"][0].get("id", "")

        return ""

    async def add_user_profile(
        self,
        user_id: str,
        constitution: str,
        preferences: Dict[str, Any]
    ) -> str:
        """添加用户画像记忆"""
        if self._memory is None:
            await self.initialize()

        metadata = TCMMemoryMetadata(
            memory_type="profile",
            user_id=user_id,
            confidence=1.0
        )

        content = f"用户画像 | 体质: {constitution}"
        if preferences:
            content += f" | 偏好: {json.dumps(preferences, ensure_ascii=False)}"

        result = await self._memory.add(
            content=content,
            user_id=user_id,
            metadata=metadata.dict(),
            memory_type="user"  # Mem0 的用户级记忆
        )

        if result.get("results"):
            return result["results"][0].get("id", "")

        return ""

    async def search_relevant_context(
        self,
        query: str,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        检索相关上下文

        Args:
            query: 查询内容
            user_id: 用户ID
            memory_types: 记忆类型过滤
            limit: 返回数量

        Returns:
            相关记忆列表
        """
        if self._memory is None:
            await self.initialize()

        # 构建过滤器
        filters = {"user_id": user_id}
        if memory_types:
            filters["memory_type"] = memory_types[0] if len(memory_types) == 1 else memory_types

        results = await self._memory.search(
            query=query,
            user_id=user_id,
            limit=limit,
            filters=filters
        )

        return results.get("results", [])

    async def get_user_diagnosis_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取用户诊断历史"""
        return await self.search_relevant_context(
            query="诊断 治疗",
            user_id=user_id,
            memory_types=["diagnosis"],
            limit=limit
        )

    async def get_user_profile(
        self,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """获取用户画像"""
        if self._memory is None:
            await self.initialize()

        results = await self._memory.get_all(
            user_id=user_id,
            filters={"memory_type": "profile"},
            limit=1
        )

        if results.get("results"):
            return results["results"][0]
        return None

    async def update_memory_effectiveness(
        self,
        memory_id: str,
        effectiveness_score: float
    ) -> bool:
        """
        更新记忆的有效性评分

        用于学习闭环：根据用户反馈更新记忆置信度
        """
        if self._memory is None:
            await self.initialize()

        result = await self._memory.update(
            memory_id=memory_id,
            metadata={"confidence": effectiveness_score}
        )
        return result.get("success", False)

    def _format_symptoms(self, symptoms: Dict[str, Any]) -> str:
        """格式化症状描述"""
        parts = []
        for key, value in symptoms.items():
            if isinstance(value, bool) and value:
                parts.append(key)
            elif isinstance(value, str) and value:
                parts.append(f"{key}:{value}")
            elif isinstance(value, list):
                parts.append(f"{key}:{','.join(value)}")
        return " ".join(parts)

    # ==================== 养生记忆 ====================

    async def add_wellness_memory(
        self,
        user_id: str,
        category: str,
        advice: str,
        season: Optional[str] = None,
        constitution: Optional[str] = None,
        confidence: float = 0.8
    ) -> str:
        """
        添加养生记忆

        Args:
            user_id: 用户ID
            category: 养生类别（饮食、起居、运动、情志等）
            advice: 养生建议
            season: 适用季节
            constitution: 适用体质
            confidence: 置信度
        """
        if self._memory is None:
            await self.initialize()

        metadata = TCMMemoryMetadata(
            memory_type="wellness",
            user_id=user_id,
            season=season,
            confidence=confidence,
            source="agent"
        )

        content = f"养生建议[{category}]: {advice}"
        if constitution:
            content += f" | 适用体质: {constitution}"
        if season:
            content += f" | 适用季节: {season}"

        result = await self._memory.add(
            content=content,
            user_id=user_id,
            metadata=metadata.dict()
        )

        if result.get("results"):
            return result["results"][0].get("id", "")
        return ""

    async def add_contraindication_memory(
        self,
        user_id: str,
        contraindication_type: str,
        description: str,
        related_herbs: Optional[List[str]] = None,
        related_syndromes: Optional[List[str]] = None,
        confidence: float = 0.95
    ) -> str:
        """
        添加禁忌记忆

        Args:
            user_id: 用户ID
            contraindication_type: 禁忌类型（过敏、妊娠、饮食等）
            description: 禁忌描述
            related_herbs: 相关药材
            related_syndromes: 相关证型
            confidence: 置信度
        """
        if self._memory is None:
            await self.initialize()

        metadata = TCMMemoryMetadata(
            memory_type="contraindication",
            user_id=user_id,
            herbs=related_herbs,
            confidence=confidence,
            source="user"
        )

        content = f"禁忌[{contraindication_type}]: {description}"
        if related_herbs:
            content += f" | 相关药材: {', '.join(related_herbs)}"
        if related_syndromes:
            content += f" | 相关证型: {', '.join(related_syndromes)}"

        result = await self._memory.add(
            content=content,
            user_id=user_id,
            metadata=metadata.dict()
        )

        if result.get("results"):
            return result["results"][0].get("id", "")
        return ""

    # ==================== 时间衰减机制 ====================

    async def get_expired_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取已过期的记忆"""
        if self._memory is None:
            await self.initialize()

        now = datetime.now()
        filters = {"user_id": user_id}
        if memory_type:
            filters["memory_type"] = memory_type

        # 获取所有记忆并过滤过期的
        results = await self._memory.get_all(user_id=user_id, limit=1000)

        expired = []
        for memory in results.get("results", []):
            metadata = memory.get("metadata", {})
            expires_at_str = metadata.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if now > expires_at:
                    expired.append(memory)

        return expired

    async def decay_memory_confidence(
        self,
        user_id: str,
        decay_rate: float = 0.1,
        decay_period_days: int = 30
    ) -> int:
        """
        根据时间衰减记忆置信度

        Args:
            user_id: 用户ID
            decay_rate: 衰减率（每次减少的比例）
            decay_period_days: 衰减周期（天）

        Returns:
            更新的记忆数量
        """
        if self._memory is None:
            await self.initialize()

        results = await self._memory.get_all(user_id=user_id, limit=1000)
        updated_count = 0

        now = datetime.now()
        for memory in results.get("results", []):
            metadata = memory.get("metadata", {})
            created_at_str = metadata.get("created_at")
            if not created_at_str:
                continue

            created_at = datetime.fromisoformat(created_at_str)
            days_since_creation = (now - created_at).days

            # 计算衰减次数
            decay_cycles = days_since_creation // decay_period_days
            if decay_cycles > 0:
                current_confidence = metadata.get("confidence", 1.0)
                new_confidence = max(0.1, current_confidence * (1 - decay_rate * decay_cycles))

                await self._memory.update(
                    memory_id=memory["id"],
                    metadata={"confidence": new_confidence}
                )
                updated_count += 1

        logger.info(f"[TCMMemory] 衰减了 {updated_count} 条记忆的置信度")
        return updated_count

    # ==================== 冲突检测与解决 ====================

    async def detect_conflicts(
        self,
        user_id: str,
        new_memory_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        检测与新记忆冲突的已有记忆

        Args:
            user_id: 用户ID
            new_memory_data: 新记忆数据

        Returns:
            冲突记忆列表
        """
        if self._memory is None:
            await self.initialize()

        conflicts = []
        memory_type = new_memory_data.get("memory_type")

        # 检测用户画像冲突
        if memory_type == "profile":
            existing_profile = await self.get_user_profile(user_id)
            if existing_profile:
                # 检查体质是否改变
                new_constitution = new_memory_data.get("constitution")
                existing_metadata = existing_profile.get("metadata", {})
                old_constitution = existing_metadata.get("constitution")
                if old_constitution and new_constitution and old_constitution != new_constitution:
                    conflicts.append({
                        "type": "constitution_change",
                        "old_value": old_constitution,
                        "new_value": new_constitution,
                        "existing_memory": existing_profile
                    })

        # 检测症状冲突（互斥症状）
        elif memory_type == "symptom":
            symptoms = new_memory_data.get("symptoms", {})

            # 检测互斥症状对（如怕冷/怕热）
            exclusive_pairs = [
                ("怕冷", "怕热"),
                ("怕热", "怕冷"),
                ("口渴", "口不渴"),
                ("口不渴", "口渴"),
                ("便秘", "腹泻"),
                ("腹泻", "便秘"),
            ]

            for old_symptom, new_symptom in exclusive_pairs:
                if symptoms.get(new_symptom):
                    # 搜索是否已有相反症状
                    conflicting = await self.search_relevant_context(
                        query=old_symptom,
                        user_id=user_id,
                        memory_types=["symptom"],
                        limit=5
                    )
                    for mem in conflicting:
                        mem_symptoms = mem.get("metadata", {}).get("symptoms", {})
                        if mem_symptoms.get(old_symptom):
                            conflicts.append({
                                "type": "exclusive_symptoms",
                                "symptoms": (old_symptom, new_symptom),
                                "existing_memory": mem
                            })

        return conflicts

    async def resolve_conflict(
        self,
        user_id: str,
        conflict: Dict[str, Any],
        strategy: ConflictResolution = ConflictResolution.KEEP_HIGHEST_CONFIDENCE
    ) -> Dict[str, Any]:
        """
        解决记忆冲突

        Args:
            user_id: 用户ID
            conflict: 冲突信息
            strategy: 解决策略

        Returns:
            解决结果
        """
        conflict_type = conflict.get("type")
        existing_memory = conflict.get("existing_memory")

        if strategy == ConflictResolution.KEEP_MOST_RECENT:
            # 标记旧记忆为被替代
            await self._memory.update(
                memory_id=existing_memory["id"],
                metadata={"superseded": True}
            )
            return {"action": "superseded_old", "old_id": existing_memory["id"]}

        elif strategy == ConflictResolution.KEEP_HIGHEST_CONFIDENCE:
            # 比较置信度，保留更高的
            old_confidence = existing_memory.get("metadata", {}).get("confidence", 0.5)
            new_confidence = conflict.get("new_confidence", 0.5)

            if new_confidence > old_confidence:
                await self._memory.update(
                    memory_id=existing_memory["id"],
                    metadata={"superseded": True}
                )
                return {"action": "kept_new", "reason": "higher_confidence"}
            else:
                return {"action": "kept_old", "reason": "higher_confidence"}

        elif strategy == ConflictResolution.MERGE:
            # 合并记忆（适用于症状等可合并的情况）
            if conflict_type == "exclusive_symptoms":
                # 对于互斥症状，标记旧症状过期
                await self._memory.update(
                    memory_id=existing_memory["id"],
                    metadata={
                        "expires_at": datetime.now().isoformat(),
                        "superseded": True
                    }
                )
                return {"action": "merged", "result": "old_expired"}

        return {"action": "manual_review", "reason": "cannot_auto_resolve"}

    # ==================== 跨会话一致性 ====================

    async def ensure_cross_session_consistency(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        确保跨会话记忆一致性

        检查并修复：
        1. 重复记忆
        2. 过期记忆
        3. 置信度过低的记忆
        """
        if self._memory is None:
            await self.initialize()

        results = {
            "duplicates_removed": 0,
            "expired_archived": 0,
            "low_confidence_filtered": 0
        }

        # 获取所有记忆
        all_memories = await self._memory.get_all(user_id=user_id, limit=1000)
        memories = all_memories.get("results", [])

        # 检测重复（基于内容相似度）
        seen_contents = {}
        for memory in memories:
            content = memory.get("content", "")
            memory_id = memory.get("id")

            # 简单的去重（实际应使用更复杂的相似度算法）
            if content in seen_contents:
                # 标记为重复
                await self._memory.update(
                    memory_id=memory_id,
                    metadata={"duplicate": True, "duplicate_of": seen_contents[content]}
                )
                results["duplicates_removed"] += 1
            else:
                seen_contents[content] = memory_id

        # 过滤低置信度记忆
        for memory in memories:
            confidence = memory.get("metadata", {}).get("confidence", 1.0)
            if confidence < 0.3:
                await self._memory.update(
                    memory_id=memory["id"],
                    metadata={"low_confidence": True}
                )
                results["low_confidence_filtered"] += 1

        logger.info(f"[TCMMemory] 一致性检查完成: {results}")
        return results

    # ==================== 批量操作 ====================

    async def bulk_add_memories(
        self,
        user_id: str,
        memories: List[Dict[str, Any]]
    ) -> List[str]:
        """批量添加记忆"""
        memory_ids = []
        for memory_data in memories:
            memory_type = memory_data.get("memory_type")

            if memory_type == "symptom":
                memory_id = await self.add_symptom_memory(
                    user_id=user_id,
                    symptoms=memory_data.get("symptoms", {}),
                    confidence=memory_data.get("confidence", 0.8)
                )
            elif memory_type == "diagnosis":
                memory_id = await self.add_diagnosis_memory(
                    user_id=user_id,
                    syndrome=memory_data.get("syndrome"),
                    symptoms=memory_data.get("symptoms", {}),
                    prescription=memory_data.get("prescription")
                )
            elif memory_type == "wellness":
                memory_id = await self.add_wellness_memory(
                    user_id=user_id,
                    category=memory_data.get("category"),
                    advice=memory_data.get("advice"),
                    season=memory_data.get("season"),
                    constitution=memory_data.get("constitution")
                )
            else:
                continue

            if memory_id:
                memory_ids.append(memory_id)

        return memory_ids

    # ==================== 记忆统计 ====================

    async def get_memory_stats(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """获取用户记忆统计"""
        if self._memory is None:
            await self.initialize()

        all_memories = await self._memory.get_all(user_id=user_id, limit=1000)
        memories = all_memories.get("results", [])

        stats = {
            "total_count": len(memories),
            "by_type": {},
            "average_confidence": 0.0,
            "expired_count": 0,
            "low_confidence_count": 0
        }

        total_confidence = 0.0
        now = datetime.now()

        for memory in memories:
            metadata = memory.get("metadata", {})

            # 按类型统计
            memory_type = metadata.get("memory_type", "unknown")
            stats["by_type"][memory_type] = stats["by_type"].get(memory_type, 0) + 1

            # 置信度统计
            confidence = metadata.get("confidence", 0.5)
            total_confidence += confidence

            # 低置信度统计
            if confidence < 0.5:
                stats["low_confidence_count"] += 1

            # 过期统计
            expires_at_str = metadata.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if now > expires_at:
                    stats["expired_count"] += 1

        if memories:
            stats["average_confidence"] = total_confidence / len(memories)

        return stats


# 全局单例
_tcm_memory: Optional[TCMMemory] = None


async def get_tcm_memory() -> TCMMemory:
    """获取 TCM 记忆服务单例"""
    global _tcm_memory
    if _tcm_memory is None:
        config = Mem0Config.from_env()
        _tcm_memory = TCMMemory(config)
        await _tcm_memory.initialize()
    return _tcm_memory
