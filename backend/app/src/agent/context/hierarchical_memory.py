"""HierarchicalMemory 占位实现（最小可用 stub）。

完整版位于 `app/src/agent/memory/v1/hierarchical_memory.py`
（暂未启用，全文件被 `#` 注释）。本 stub 提供与
`agent/middleware/context_manager.py` 兼容的构造与运行接口，
让中间件在未启用持久化记忆时仍能正常工作。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryLevel(Enum):
    """记忆层级"""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryEntry:
    """单条记忆"""
    content: Any
    level: MemoryLevel = MemoryLevel.WORKING
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PatientProfile:
    """患者画像（占位）"""
    patient_id: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[datetime] = None


class HierarchicalMemory:
    """分层记忆管理器 stub（仅 in-memory，不持久化）"""

    def __init__(self, working_capacity: int = 10, episodic_capacity: int = 50):
        self.working_capacity = working_capacity
        self.episodic_capacity = episodic_capacity
        self.semantic_capacity = 200
        self._working: List[MemoryEntry] = []
        self._episodic: List[MemoryEntry] = []
        self._semantic: List[MemoryEntry] = []
        self._patient_profiles: Dict[str, PatientProfile] = {}

    # ---------- working memory ----------
    def add_to_working(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            level=MemoryLevel.WORKING,
            metadata=metadata or {},
            importance=importance,
        )
        self._working.append(entry)
        if len(self._working) > self.working_capacity:
            # 简化：直接弹出最旧的；真实实现会做 consolidation
            self._working.pop(0)
        return entry

    # ---------- episodic memory ----------
    def add_to_episodic(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            level=MemoryLevel.EPISODIC,
            metadata=metadata or {},
            importance=importance,
        )
        self._episodic.append(entry)
        if len(self._episodic) > self.episodic_capacity:
            self._episodic.pop(0)
        return entry

    # ---------- semantic memory ----------
    def add_to_semantic(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            level=MemoryLevel.SEMANTIC,
            metadata=metadata or {},
            importance=importance,
        )
        self._semantic.append(entry)
        if len(self._semantic) > self.semantic_capacity:
            self._semantic.pop(0)
        return entry

    # ---------- generic add (used by add()) ----------
    def add(
        self,
        content: Any,
        level: MemoryLevel = MemoryLevel.WORKING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        if level == MemoryLevel.EPISODIC:
            return self.add_to_episodic(content, metadata)
        if level == MemoryLevel.SEMANTIC:
            return self.add_to_semantic(content, metadata)
        return self.add_to_working(content, metadata)

    # ---------- retrieve / context ----------
    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """简化：返回工作记忆最近 top_k 条"""
        return self._working[-top_k:]

    def get_context_for_prompt(self, query: str, max_tokens: int = 1500) -> str:
        """返回注入到 prompt 的记忆摘要"""
        recent = self._working[-3:]
        if not recent:
            return ""
        lines = ["[近期记忆]"]
        for e in recent:
            content_str = str(e.content)[:200]
            lines.append(f"- {content_str}")
        return "\n".join(lines)

    # ---------- patient profile ----------
    def update_patient_profile(
        self,
        patient_id: str,
        **updates: Any,
    ) -> PatientProfile:
        if patient_id not in self._patient_profiles:
            self._patient_profiles[patient_id] = PatientProfile(patient_id=patient_id)
        profile = self._patient_profiles[patient_id]
        for key, value in updates.items():
            if hasattr(profile, key):
                current = getattr(profile, key)
                if isinstance(current, list):
                    if isinstance(value, str):
                        current.append(value)
                    elif isinstance(value, list):
                        current.extend(value)
                else:
                    setattr(profile, key, value)
        profile.updated_at = datetime.now()
        return profile

    def get_patient_profile(self, patient_id: str) -> Optional[PatientProfile]:
        return self._patient_profiles.get(patient_id)

    # ---------- stats / maintenance ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "working_count": len(self._working),
            "working_capacity": self.working_capacity,
            "episodic_count": len(self._episodic),
            "episodic_capacity": self.episodic_capacity,
            "semantic_count": len(self._semantic),
            "semantic_capacity": self.semantic_capacity,
            "patient_profiles": len(self._patient_profiles),
        }

    def clear_working(self) -> None:
        self._working = []

    def clear(self) -> None:
        self._working.clear()
        self._episodic.clear()
        self._semantic.clear()

    def get_recent(self, limit: int = 5) -> List[MemoryEntry]:
        return self._working[-limit:]

    def get_episodic(self) -> List[MemoryEntry]:
        return list(self._episodic)
