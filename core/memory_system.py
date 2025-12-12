# core/memory_system.py
"""
Memory system definitions including MemoryType enum that includes EVENT.
Provides a simple Memory dataclass used across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional
import time


def _now() -> int:
    return int(time.time())


class MemoryType(Enum):
    # Ensure EVENT exists (fixes AttributeError: MemoryType.EVENT)
    EVENT = auto()
    REWARD = auto()
    SHARED = auto()
    PREDICTION = auto()
    PATTERN = auto()
    SHORT = auto()
    RESOURCE = auto()
    CELEBRATION = auto()


class Classification(Enum):
    PUBLIC = auto()
    PROTECTED = auto()
    PRIVATE = auto()
    CONFIDENTIAL = auto()


@dataclass
class Memory:
    content: Any
    importance: float = 0.5
    timestamp: int = field(default_factory=_now)
    memory_type: MemoryType = MemoryType.EVENT
    source_node: Optional[int] = None
    emotional_valence: Optional[float] = None
    classification: Classification = Classification.PUBLIC
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_seconds(self, now: Optional[int] = None) -> int:
        now = now or _now()
        return max(0, now - self.timestamp)

    def decayed_importance(self, decay_rate: float = 0.0001, now: Optional[int] = None) -> float:
        """Return importance after exponential decay (simple model)."""
        age = self.age_seconds(now)
        return max(0.0, self.importance * (1.0 - decay_rate) ** age)
