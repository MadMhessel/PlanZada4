"""Lightweight schemas and enums for AI interactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TypedDict, Any


class ActionKind(str, Enum):
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    OTHER = "OTHER"


class Topic(str, Enum):
    PERSONAL_NOTE = "PERSONAL_NOTE"
    PERSONAL_TASK = "PERSONAL_TASK"
    TEAM_TASK = "TEAM_TASK"
    CALENDAR = "CALENDAR"
    SYSTEM = "SYSTEM"
    OTHER = "OTHER"


@dataclass
class Classification:
    kind: ActionKind
    topic: Topic
    confidence: float


class ActionParams(TypedDict, total=False):
    scope: str
    title: str
    description: str
    due_date: str
    priority: str
    status: str
    tags: List[str]
    assignees: str
    add_to_calendar: bool
    calendar_event_id: str
    search_query: str
    clarify_question: str


@dataclass
class PlannedAction:
    method: str
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)

    def safe_params(self) -> Dict[str, Any]:
        return self.params or {}


DEBUG_META_KEYS = ["kind", "topic", "confidence", "method", "params"]
