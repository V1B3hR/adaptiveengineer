"""Memory system for agent nodes with importance weighting and privacy controls.

This module provides the core memory infrastructure including:
- Memory types and classification
- Memory storage with decay and expiry
- Short-term memory store with LRU eviction
- Thread-safe operations
"""

import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """Types of memory for agent nodes."""
    REWARD = "reward"
    SHARED = "shared"
    PREDICTION = "prediction"
    PATTERN = "pattern"
    SHORT = "short"  # dedicated short-term memory type


class Classification(str, Enum):
    """Privacy classification levels for memory content."""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


@dataclass
class Memory:
    """Structured memory with importance weighting and privacy controls.
    
    Fields:
    - timestamp: integer epoch seconds when the memory was created (int(time.time()))
    - retention_limit: seconds to keep this memory before auto-expiry (None = keep)
    - size_mb: optional approximate size in megabytes (used by short-term store)
    """
    content: Any
    importance: float
    timestamp: int
    memory_type: MemoryType
    emotional_valence: float = 0.0  # -1.0 .. +1.0
    decay_rate: float = 0.95  # per age() call
    access_count: int = 0
    source_node: Optional[int] = None
    validation_count: int = 0

    # Privacy controls
    private: bool = False
    classification: Classification = Classification.PUBLIC
    retention_limit: Optional[int] = None  # seconds
    audit_log: List[str] = field(default_factory=list)

    # approx size in MB; if zero, stores/short-store will estimate
    size_mb: float = 0.0

    # internal lock for thread-safety when updating mutable fields
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        # normalize enum/string input
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)
        if isinstance(self.classification, str):
            self.classification = Classification(self.classification)

        # clamp vals
        try:
            self.emotional_valence = float(self.emotional_valence)
        except Exception:
            self.emotional_valence = 0.0
        self.emotional_valence = max(-1.0, min(1.0, self.emotional_valence))

        try:
            self.importance = max(0.0, float(self.importance))
        except Exception:
            self.importance = 0.0

        # compute timestamp default if invalid
        if not isinstance(self.timestamp, int) or self.timestamp <= 0:
            self.timestamp = int(time.time())

    def is_short(self) -> bool:
        return self.memory_type == MemoryType.SHORT

    def is_expired(self, current_time: Optional[int] = None) -> bool:
        """Return True if retention_limit is set and memory age exceeds it."""
        now = int(current_time or time.time())
        if self.retention_limit is None:
            return False
        return (now - self.timestamp) > self.retention_limit

    def age(self, current_time: Optional[int] = None) -> None:
        """Apply decay to importance and check expiry.

        Call periodically. Uses a dynamic decay factor influenced by emotional valence
        without permanently overwriting the configured decay_rate.
        """
        now = int(current_time or time.time())

        # compute dynamic decay; strong emotions reduce decay (i.e., memory persists longer)
        dynamic_decay = self.decay_rate
        ev = abs(self.emotional_valence)
        if ev > 0.7:
            # small bonus to reduce decay; clipped conservatively
            bonus = min(0.997 - self.decay_rate, (ev - 0.7) * 0.1)
            dynamic_decay = min(0.997, self.decay_rate + bonus)

        with self._lock:
            self.importance *= dynamic_decay
            # floor tiny values to zero
            if self.importance < 1e-9:
                self.importance = 0.0

            # expiration check based on current time
            if self.is_expired(now):
                self.importance = 0.0

    def access(self, accessor_id: int, current_time: Optional[int] = None, summary: bool = False) -> Any:
        """Access memory content with audit logging and privacy rules.

        - Logs actual access time.
        - Returns redacted/summary content depending on classification and accessor.
        """
        now = int(current_time or time.time())
        with self._lock:
            self.access_count += 1
            self.audit_log.append(f"accessed_by_{accessor_id}_at_{now}")

        # Access control
        if self.classification == Classification.PRIVATE and accessor_id != self.source_node:
            return "[REDACTED]"

        if self.classification == Classification.CONFIDENTIAL and accessor_id != self.source_node:
            return "[REDACTED - CONFIDENTIAL]"

        if self.classification == Classification.PROTECTED and accessor_id != self.source_node:
            text = str(self.content)
            if summary or len(text) > 100:
                return f"[SUMMARY: {text[:100]}...]"
            return f"[LIMITED: {text[:200]}]"

        # PUBLIC or owner/source_node: return full content
        return self.content

    def to_dict(self, redact_for: Optional[int] = None) -> Dict[str, Any]:
        """Serialize memory. If redact_for provided, content is produced via access(redact_for)."""
        data = {
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type.value,
            "emotional_valence": self.emotional_valence,
            "decay_rate": self.decay_rate,
            "access_count": self.access_count,
            "source_node": self.source_node,
            "validation_count": self.validation_count,
            "private": self.private,
            "classification": self.classification.value,
            "retention_limit": self.retention_limit,
            "audit_log": list(self.audit_log),
            "size_mb": self.size_mb,
        }
        if redact_for is not None:
            data["content"] = self.access(redact_for)
        return data

    def update_content(self, new_content: Any, importance_delta: Optional[float] = None) -> None:
        """Safely update content and optionally adjust importance."""
        with self._lock:
            self.content = new_content
            if importance_delta is not None:
                try:
                    self.importance = max(0.0, float(self.importance + importance_delta))
                except Exception:
                    pass


class ShortMemoryStore:
    """Thread-safe LRU short-term memory store with MB capacity accounting.

    - Default capacity is 4.0 MB (fast to hold and fetch).
    - Keys should be integers (e.g., memory id or timestamp).
    - Only stores Memory objects with memory_type == MemoryType.SHORT.
    - If a single Memory's size >= capacity, it will not be stored here.
    """

    def __init__(self, capacity_mb: float = 4.0):
        self.capacity_mb = float(capacity_mb)
        self._used_mb: float = 0.0
        self._store: "OrderedDict[int, Memory]" = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _estimate_size_mb(content: Any) -> float:
        """Conservative size estimation in MB for common types."""
        if content is None:
            return 0.0
        # bytes-like
        if isinstance(content, (bytes, bytearray)):
            return len(content) / (1024 * 1024)
        if isinstance(content, str):
            return len(content.encode("utf-8")) / (1024 * 1024)
        try:
            size = sys.getsizeof(content)
        except Exception:
            size = 0
        # shallow estimate to MB
        return max(0.0, size / (1024 * 1024))

    def put(self, key: int, mem: Memory) -> None:
        """Insert memory into short store; evict LRU entries when capacity exceeded."""
        if not mem.is_short():
            raise ValueError("ShortMemoryStore only accepts Memory objects with memory_type=SHORT")

        with self._lock:
            # ensure size_mb is set
            if not mem.size_mb:
                mem.size_mb = self._estimate_size_mb(mem.content)

            # skip storing objects larger than capacity to avoid thrashing
            if mem.size_mb >= self.capacity_mb:
                return

            # if key exists, remove old to update LRU position
            if key in self._store:
                old = self._store.pop(key)
                self._used_mb -= old.size_mb

            # evict until room
            while self._used_mb + mem.size_mb > self.capacity_mb and self._store:
                _, evicted = self._store.popitem(last=False)
                self._used_mb -= evicted.size_mb

            # insert as most-recent
            self._store[key] = mem
            self._used_mb += mem.size_mb

    def get(self, key: int) -> Optional[Memory]:
        """Fast fetch; returns Memory or None. Updates LRU position on hit."""
        with self._lock:
            mem = self._store.get(key)
            if mem is None:
                return None
            # mark as recently used
            self._store.move_to_end(key, last=True)
            return mem

    def remove(self, key: int) -> None:
        with self._lock:
            mem = self._store.pop(key, None)
            if mem:
                self._used_mb -= mem.size_mb

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._used_mb = 0.0

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {"capacity_mb": self.capacity_mb, "used_mb": self._used_mb, "count": len(self._store)}
