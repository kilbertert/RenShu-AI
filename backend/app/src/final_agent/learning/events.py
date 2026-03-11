"""
线程学习事件与上下文结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union


def _normalize_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize_list(values: Optional[Any]) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    normalized = []
    for item in values:
        text = _normalize_text(item)
        if text:
            normalized.append(text)
    return normalized


class LearningEventType(str, Enum):
    """当前阶段允许的学习事件类型。"""

    INTENT_ROUTE_CORRECTION = "intent_route_correction"
    TOOL_SELECTION_CORRECTION = "tool_selection_correction"
    USER_OUTPUT_CORRECTION = "user_output_correction"


@dataclass
class LearningEvent:
    """统一的学习事件结构。"""

    event_type: LearningEventType
    conversation_id: str
    source: str
    trigger: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, LearningEventType):
            self.event_type = LearningEventType(str(self.event_type))
        self.conversation_id = _normalize_text(self.conversation_id)
        self.source = _normalize_text(self.source) or "main_graph"
        self.trigger = _normalize_text(self.trigger) or "unknown"
        self.payload = dict(self.payload or {})
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningEvent":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        return cls(
            event_type=LearningEventType(str(data.get("event_type"))),
            conversation_id=str(data.get("conversation_id") or ""),
            source=str(data.get("source") or "main_graph"),
            trigger=str(data.get("trigger") or "unknown"),
            payload=dict(data.get("payload") or {}),
            created_at=created_at,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "conversation_id": self.conversation_id,
            "source": self.source,
            "trigger": self.trigger,
            "payload": dict(self.payload),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass
class IntentRouteCorrection:
    """主图意图纠偏。"""

    user_input: str = ""
    wrong_route: Optional[str] = None
    correct_route: Optional[str] = None
    reason: str = ""
    hint: str = ""
    thread_intent_summary: str = ""
    source: str = "main_graph"
    trigger: str = "user_correction"
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_event(cls, event: LearningEvent) -> "IntentRouteCorrection":
        payload = event.payload
        return cls(
            user_input=_normalize_text(payload.get("user_input")),
            wrong_route=_normalize_text(payload.get("wrong_route")) or None,
            correct_route=_normalize_text(payload.get("correct_route")) or None,
            reason=_normalize_text(payload.get("reason")),
            hint=_normalize_text(payload.get("hint")),
            thread_intent_summary=_normalize_text(payload.get("thread_intent_summary")),
            source=event.source,
            trigger=event.trigger,
            created_at=event.created_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_input": self.user_input,
            "wrong_route": self.wrong_route,
            "correct_route": self.correct_route,
            "reason": self.reason,
            "hint": self.hint,
            "thread_intent_summary": self.thread_intent_summary,
            "source": self.source,
            "trigger": self.trigger,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ToolSelectionCorrection:
    """子图工具选择纠偏。"""

    wrong_tool: Optional[str] = None
    correct_tool: Optional[str] = None
    reason: str = ""
    missing_info: List[str] = field(default_factory=list)
    hint: str = ""
    source: str = ""
    subgraph: str = ""
    trigger: str = "tool_error"
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_event(cls, event: LearningEvent) -> "ToolSelectionCorrection":
        payload = event.payload
        source = _normalize_text(payload.get("subgraph") or event.source)
        return cls(
            wrong_tool=_normalize_text(payload.get("wrong_tool")) or None,
            correct_tool=_normalize_text(payload.get("correct_tool")) or None,
            reason=_normalize_text(payload.get("reason")),
            missing_info=_normalize_list(payload.get("missing_info")),
            hint=_normalize_text(payload.get("hint")),
            source=event.source,
            subgraph=source,
            trigger=event.trigger,
            created_at=event.created_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wrong_tool": self.wrong_tool,
            "correct_tool": self.correct_tool,
            "reason": self.reason,
            "missing_info": list(self.missing_info),
            "hint": self.hint,
            "source": self.source,
            "subgraph": self.subgraph,
            "trigger": self.trigger,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class UserOutputCorrection:
    """子图输出被用户纠正。"""

    wrong_understanding: str = ""
    user_correction: str = ""
    invalidated_assumption: str = ""
    hint: str = ""
    source: str = ""
    subgraph: str = ""
    trigger: str = "user_correction"
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_event(cls, event: LearningEvent) -> "UserOutputCorrection":
        payload = event.payload
        source = _normalize_text(payload.get("subgraph") or event.source)
        return cls(
            wrong_understanding=_normalize_text(payload.get("wrong_understanding")),
            user_correction=_normalize_text(payload.get("user_correction")),
            invalidated_assumption=_normalize_text(payload.get("invalidated_assumption")),
            hint=_normalize_text(payload.get("hint")),
            source=event.source,
            subgraph=source,
            trigger=event.trigger,
            created_at=event.created_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wrong_understanding": self.wrong_understanding,
            "user_correction": self.user_correction,
            "invalidated_assumption": self.invalidated_assumption,
            "hint": self.hint,
            "source": self.source,
            "subgraph": self.subgraph,
            "trigger": self.trigger,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class IntentLearningContext:
    """线程内的意图学习上下文。"""

    thread_intent_summary: str = ""
    intent_route_corrections: List[Dict[str, Any]] = field(default_factory=list)
    intent_disambiguation_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_intent_summary": self.thread_intent_summary,
            "intent_route_corrections": [dict(item) for item in self.intent_route_corrections],
            "intent_disambiguation_hints": list(self.intent_disambiguation_hints),
        }


@dataclass
class ThreadLearningContext:
    """按 conversation_id 保存的线程学习结果。"""

    conversation_id: str
    intent_learning: IntentLearningContext = field(default_factory=IntentLearningContext)
    tool_learning: List[Dict[str, Any]] = field(default_factory=list)
    correction_learning: List[Dict[str, Any]] = field(default_factory=list)
    last_updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "intent_learning": self.intent_learning.to_dict(),
            "tool_learning": [dict(item) for item in self.tool_learning],
            "correction_learning": [dict(item) for item in self.correction_learning],
            "last_updated_at": self.last_updated_at.isoformat(),
        }


LearningEventInput = Union[LearningEvent, Mapping[str, Any]]

