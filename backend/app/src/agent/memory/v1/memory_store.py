# """
# 记忆存储抽象层
#
# 定义记忆系统的持久化接口，支持多种存储后端。
# 参考 memory-systems skill 中的 Mem0/Zep 设计模式。
# """
#
# from abc import ABC, abstractmethod
# from typing import List, Optional, Dict, Any, Tuple
# from datetime import datetime
# from uuid import UUID
# import asyncio
# import logging
#
# from sqlmodel import SQLModel, Session, select, col
# from sqlalchemy.ext.asyncio import AsyncSession
#
# from app.src.model.memory_models import (
#     MemoryEntry, MemoryLevel, PatientProfile, MemoryConsolidationLog
# )
#
#
# logger = logging.getLogger(__name__)
#
#
# class MemoryStore(ABC):
#     """
#     记忆存储抽象基类
#
#     定义记忆持久化的标准接口，支持：
#     - 增删改查操作
#     - Temporal validity 查询
#     - 批量操作
#     """
#
#     @abstractmethod
#     async def save_entry(self, entry: MemoryEntry) -> bool:
#         """保存记忆条目"""
#         pass
#
#     @abstractmethod
#     async def load_entries(
#         self,
#         patient_id: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 100,
#         valid_at: Optional[datetime] = None,
#     ) -> List[MemoryEntry]:
#         """加载记忆条目"""
#         pass
#
#     @abstractmethod
#     async def search_entries(
#         self,
#         patient_id: str,
#         query: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 10,
#         valid_at: Optional[datetime] = None,
#     ) -> List[Tuple[MemoryEntry, float]]:
#         """搜索记忆条目，返回 (entry, score) 元组列表"""
#         pass
#
#     @abstractmethod
#     async def update_entry(self, entry_id: UUID, **updates) -> Optional[MemoryEntry]:
#         """更新记忆条目"""
#         pass
#
#     @abstractmethod
#     async def invalidate_entry(self, entry_id: UUID) -> bool:
#         """使记忆条目失效"""
#         pass
#
#     @abstractmethod
#     async def save_profile(self, profile: PatientProfile) -> bool:
#         """保存患者档案"""
#         pass
#
#     @abstractmethod
#     async def load_profile(
#         self,
#         patient_id: str,
#         valid_at: Optional[datetime] = None,
#     ) -> Optional[PatientProfile]:
#         """加载患者档案"""
#         pass
#
#     @abstractmethod
#     async def log_consolidation(self, log: MemoryConsolidationLog) -> bool:
#         """记录整理操作"""
#         pass
#
#
# class PostgresMemoryStore(MemoryStore):
#     """
#     PostgreSQL 记忆存储实现
#
#     使用 SQLAlchemy/SQLModel 实现持久化存储。
#     支持异步操作，适合 Web 应用场景。
#     """
#
#     def __init__(self, session_factory):
#         """
#         初始化 PostgreSQL 存储
#
#         Args:
#             session_factory: SQLAlchemy session factory
#         """
#         self.session_factory = session_factory
#
#     async def save_entry(self, entry: MemoryEntry) -> bool:
#         """保存记忆条目"""
#         try:
#             async with self.session_factory() as session:
#                 session.add(entry)
#                 await session.commit()
#                 await session.refresh(entry)
#                 logger.debug(f"Saved memory entry: {entry.id}")
#                 return True
#         except Exception as e:
#             logger.error(f"Failed to save memory entry: {e}")
#             return False
#
#     async def load_entries(
#         self,
#         patient_id: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 100,
#         valid_at: Optional[datetime] = None,
#     ) -> List[MemoryEntry]:
#         """加载记忆条目"""
#         try:
#             async with self.session_factory() as session:
#                 query = select(MemoryEntry).where(
#                     MemoryEntry.patient_id == patient_id
#                 )
#
#                 if level:
#                     query = query.where(MemoryEntry.level == level)
#
#                 # Temporal validity filter
#                 valid_at = valid_at or datetime.now()
#                 query = query.where(
#                     (MemoryEntry.valid_from <= valid_at) &
#                     ((MemoryEntry.valid_until.is_(None)) | (MemoryEntry.valid_until > valid_at))
#                 )
#
#                 query = query.order_by(
#                     col(MemoryEntry.importance).desc(),
#                     col(MemoryEntry.created_at).desc()
#                 ).limit(limit)
#
#                 result = await session.execute(query)
#                 entries = result.scalars().all()
#                 return list(entries)
#         except Exception as e:
#             logger.error(f"Failed to load memory entries: {e}")
#             return []
#
#     async def search_entries(
#         self,
#         patient_id: str,
#         query: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 10,
#         valid_at: Optional[datetime] = None,
#     ) -> List[Tuple[MemoryEntry, float]]:
#         """
#         搜索记忆条目
#
#         简单实现：基于关键词匹配
#         TODO: 接入向量检索（pgvector）实现语义搜索
#         """
#         try:
#             async with self.session_factory() as session:
#                 memory_query = select(MemoryEntry).where(
#                     MemoryEntry.patient_id == patient_id
#                 )
#
#                 if level:
#                     memory_query = memory_query.where(MemoryEntry.level == level)
#
#                 # Temporal validity filter
#                 valid_at = valid_at or datetime.now()
#                 memory_query = memory_query.where(
#                     (MemoryEntry.valid_from <= valid_at) &
#                     ((MemoryEntry.valid_until.is_(None)) | (MemoryEntry.valid_until > valid_at))
#                 )
#
#                 result = await session.execute(memory_query)
#                 all_entries = result.scalars().all()
#
#                 # 简单的关键词匹配评分
#                 query_lower = query.lower()
#                 scored_entries = []
#
#                 for entry in all_entries:
#                     content_lower = entry.content.lower()
#                     score = 0.0
#
#                     # 精确匹配
#                     if query_lower in content_lower:
#                         score += 0.5
#
#                     # 关键词匹配
#                     query_keywords = set(query_lower.split())
#                     content_keywords = set(content_lower.split())
#                     overlap = len(query_keywords & content_keywords)
#                     score += overlap * 0.1
#
#                     # 重要性加权
#                     score += entry.importance * 0.2
#
#                     if score > 0:
#                         scored_entries.append((entry, score))
#
#                 # 按分数排序
#                 scored_entries.sort(key=lambda x: x[1], reverse=True)
#                 return scored_entries[:limit]
#         except Exception as e:
#             logger.error(f"Failed to search memory entries: {e}")
#             return []
#
#     async def update_entry(self, entry_id: UUID, **updates) -> Optional[MemoryEntry]:
#         """更新记忆条目"""
#         try:
#             async with self.session_factory() as session:
#                 query = select(MemoryEntry).where(MemoryEntry.id == entry_id)
#                 result = await session.execute(query)
#                 entry = result.scalar_one_or_none()
#
#                 if entry:
#                     for key, value in updates.items():
#                         if hasattr(entry, key):
#                             setattr(entry, key, value)
#
#                     entry.updated_at = datetime.now()
#                     await session.commit()
#                     await session.refresh(entry)
#                     return entry
#                 return None
#         except Exception as e:
#             logger.error(f"Failed to update memory entry: {e}")
#             return None
#
#     async def invalidate_entry(self, entry_id: UUID) -> bool:
#         """使记忆条目失效"""
#         try:
#             entry = await self.update_entry(entry_id, valid_until=datetime.now())
#             return entry is not None
#         except Exception as e:
#             logger.error(f"Failed to invalidate memory entry: {e}")
#             return False
#
#     async def save_profile(self, profile: PatientProfile) -> bool:
#         """保存患者档案"""
#         try:
#             async with self.session_factory() as session:
#                 # 检查是否已存在
#                 query = select(PatientProfile).where(
#                     PatientProfile.patient_id == profile.patient_id
#                 )
#                 result = await session.execute(query)
#                 existing = result.scalar_one_or_none()
#
#                 if existing:
#                     # 更新现有档案
#                     for key, value in profile.model_dump(exclude={'id'}).items():
#                         setattr(existing, key, value)
#                     existing.updated_at = datetime.now()
#                     session.add(existing)
#                 else:
#                     session.add(profile)
#
#                 await session.commit()
#                 logger.debug(f"Saved patient profile: {profile.patient_id}")
#                 return True
#         except Exception as e:
#             logger.error(f"Failed to save patient profile: {e}")
#             return False
#
#     async def load_profile(
#         self,
#         patient_id: str,
#         valid_at: Optional[datetime] = None,
#     ) -> Optional[PatientProfile]:
#         """加载患者档案"""
#         try:
#             async with self.session_factory() as session:
#                 query = select(PatientProfile).where(
#                     PatientProfile.patient_id == patient_id
#                 )
#
#                 # Temporal validity filter
#                 valid_at = valid_at or datetime.now()
#                 query = query.where(
#                     (PatientProfile.valid_from <= valid_at) &
#                     ((PatientProfile.valid_until.is_(None)) | (PatientProfile.valid_until > valid_at))
#                 )
#
#                 query = query.order_by(PatientProfile.updated_at.desc()).limit(1)
#
#                 result = await session.execute(query)
#                 return result.scalar_one_or_none()
#         except Exception as e:
#             logger.error(f"Failed to load patient profile: {e}")
#             return None
#
#     async def log_consolidation(self, log: MemoryConsolidationLog) -> bool:
#         """记录整理操作"""
#         try:
#             async with self.session_factory() as session:
#                 session.add(log)
#                 await session.commit()
#                 return True
#         except Exception as e:
#             logger.error(f"Failed to log consolidation: {e}")
#             return False
#
#     async def batch_save_entries(self, entries: List[MemoryEntry]) -> int:
#         """批量保存记忆条目"""
#         try:
#             async with self.session_factory() as session:
#                 session.add_all(entries)
#                 await session.commit()
#                 logger.debug(f"Batch saved {len(entries)} memory entries")
#                 return len(entries)
#         except Exception as e:
#             logger.error(f"Failed to batch save memory entries: {e}")
#             return 0
#
#
# class InMemoryMemoryStore(MemoryStore):
#     """
#     内存记忆存储实现
#
#     用于测试和开发，不提供持久化。
#     """
#
#     def __init__(self):
#         self._entries: Dict[str, List[MemoryEntry]] = {}
#         self._profiles: Dict[str, PatientProfile] = {}
#         self._logs: List[MemoryConsolidationLog] = []
#
#     async def save_entry(self, entry: MemoryEntry) -> bool:
#         """保存记忆条目"""
#         patient_id = entry.patient_id or "default"
#         if patient_id not in self._entries:
#             self._entries[patient_id] = []
#         self._entries[patient_id].append(entry)
#         return True
#
#     async def load_entries(
#         self,
#         patient_id: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 100,
#         valid_at: Optional[datetime] = None,
#     ) -> List[MemoryEntry]:
#         """加载记忆条目"""
#         entries = self._entries.get(patient_id, [])
#
#         if level:
#             entries = [e for e in entries if e.level == level]
#
#         valid_at = valid_at or datetime.now()
#         entries = [e for e in entries if e.is_valid_at(valid_at)]
#
#         return sorted(
#             entries,
#             key=lambda e: (e.importance, e.created_at),
#             reverse=True
#         )[:limit]
#
#     async def search_entries(
#         self,
#         patient_id: str,
#         query: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 10,
#         valid_at: Optional[datetime] = None,
#     ) -> List[Tuple[MemoryEntry, float]]:
#         """搜索记忆条目"""
#         entries = await self.load_entries(patient_id, level, limit=limit, valid_at=valid_at)
#
#         query_lower = query.lower()
#         scored = []
#
#         for entry in entries:
#             content_lower = entry.content.lower()
#             score = 0.0
#
#             if query_lower in content_lower:
#                 score += 0.5
#
#             query_keywords = set(query_lower.split())
#             content_keywords = set(content_lower.split())
#             overlap = len(query_keywords & content_keywords)
#             score += overlap * 0.1
#             score += entry.importance * 0.2
#
#             if score > 0:
#                 scored.append((entry, score))
#
#         scored.sort(key=lambda x: x[1], reverse=True)
#         return scored[:limit]
#
#     async def update_entry(self, entry_id: UUID, **updates) -> Optional[MemoryEntry]:
#         """更新记忆条目"""
#         for entries in self._entries.values():
#             for entry in entries:
#                 if entry.id == entry_id:
#                     for key, value in updates.items():
#                         if hasattr(entry, key):
#                             setattr(entry, key, value)
#                     entry.updated_at = datetime.now()
#                     return entry
#         return None
#
#     async def invalidate_entry(self, entry_id: UUID) -> bool:
#         """使记忆条目失效"""
#         entry = await self.update_entry(entry_id, valid_until=datetime.now())
#         return entry is not None
#
#     async def save_profile(self, profile: PatientProfile) -> bool:
#         """保存患者档案"""
#         self._profiles[profile.patient_id] = profile
#         return True
#
#     async def load_profile(
#         self,
#         patient_id: str,
#         valid_at: Optional[datetime] = None,
#     ) -> Optional[PatientProfile]:
#         """加载患者档案"""
#         profile = self._profiles.get(patient_id)
#         if profile:
#             valid_at = valid_at or datetime.now()
#             if profile.is_valid_at(valid_at):
#                 return profile
#         return None
#
#     async def log_consolidation(self, log: MemoryConsolidationLog) -> bool:
#         """记录整理操作"""
#         self._logs.append(log)
#         return True
#
#
# class HybridMemoryStore(MemoryStore):
#     """
#     混合记忆存储实现
#
#     结合 Redis（热数据）和 PostgreSQL（持久化）。
#     参考内存系统 skill 中的混合检索策略。
#     """
#
#     def __init__(self, postgres_store, redis_client=None):
#         """
#         初始化混合存储
#
#         Args:
#             postgres_store: PostgreSQL 存储
#             redis_client: Redis 客户端（可选）
#         """
#         self.postgres = postgres_store
#         self.redis = redis_client
#         self._cache_ttl = 3600  # 缓存 1 小时
#
#     def _cache_key(self, patient_id: str, level: Optional[MemoryLevel]) -> str:
#         """生成缓存键"""
#         level_str = level.value if level else "all"
#         return f"memory:{patient_id}:{level_str}"
#
#     async def save_entry(self, entry: MemoryEntry) -> bool:
#         """保存记忆条目"""
#         # 先保存到 PostgreSQL
#         result = await self.postgres.save_entry(entry)
#
#         # 清除相关缓存
#         if result and self.redis:
#             cache_key = self._cache_key(entry.patient_id or "default", entry.level)
#             await self.redis.delete(cache_key)
#
#         return result
#
#     async def load_entries(
#         self,
#         patient_id: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 100,
#         valid_at: Optional[datetime] = None,
#     ) -> List[MemoryEntry]:
#         """加载记忆条目"""
#         # 尝试从 Redis 缓存加载
#         if self.redis:
#             cache_key = self._cache_key(patient_id, level)
#             cached = await self.redis.get(cache_key)
#             if cached:
#                 # TODO: 反序列化缓存数据
#                 pass
#
#         # 从 PostgreSQL 加载
#         entries = await self.postgres.load_entries(
#             patient_id, level, limit, valid_at
#         )
#
#         # 写入缓存
#         if self.redis and entries:
#             cache_key = self._cache_key(patient_id, level)
#             # TODO: 序列化并写入缓存
#
#         return entries
#
#     async def search_entries(
#         self,
#         patient_id: str,
#         query: str,
#         level: Optional[MemoryLevel] = None,
#         limit: int = 10,
#         valid_at: Optional[datetime] = None,
#     ) -> List[Tuple[MemoryEntry, float]]:
#         """搜索记忆条目"""
#         return await self.postgres.search_entries(
#             patient_id, query, level, limit, valid_at
#         )
#
#     async def update_entry(self, entry_id: UUID, **updates) -> Optional[MemoryEntry]:
#         """更新记忆条目"""
#         return await self.postgres.update_entry(entry_id, **updates)
#
#     async def invalidate_entry(self, entry_id: UUID) -> bool:
#         """使记忆条目失效"""
#         return await self.postgres.invalidate_entry(entry_id)
#
#     async def save_profile(self, profile: PatientProfile) -> bool:
#         """保存患者档案"""
#         return await self.postgres.save_profile(profile)
#
#     async def load_profile(
#         self,
#         patient_id: str,
#         valid_at: Optional[datetime] = None,
#     ) -> Optional[PatientProfile]:
#         """加载患者档案"""
#         return await self.postgres.load_profile(patient_id, valid_at)
#
#     async def log_consolidation(self, log: MemoryConsolidationLog) -> bool:
#         """记录整理操作"""
#         return await self.postgres.log_consolidation(log)
