# core/social_signals.py
"""
Robust SocialSignal dataclass compatible with multiple call-sites.

This implementation accepts both:
 - source_id and source_node_id (alias)
 - target_id and target_node_id (alias)
 - content and data (alias)
and normalizes them to canonical attributes used across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4, UUID
import time


def _now_timestamp() -> int:
    return int(time.time())


@dataclass
class SocialSignal:
    # canonical attributes
    signal_id: UUID = field(default_factory=uuid4)
    signal_type: str = "generic"
    content: Any = None  # canonical content field
    source_id: Optional[int] = None
    target_id: Optional[int] = None
    urgency: float = 0.5
    requires_response: bool = False
    idempotency_key: Optional[str] = None
    partition_key: Optional[str] = None
    correlation_id: Optional[str] = None
    processing_attempts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: int = field(default_factory=_now_timestamp)
    # backward-compat alias container for any extra data
    meta: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        signal_type: Optional[str] = None,
        content: Any = None,
        data: Any = None,
        source_id: Optional[int] = None,
        source_node_id: Optional[int] = None,
        target_id: Optional[int] = None,
        target_node_id: Optional[int] = None,
        urgency: float = 0.5,
        requires_response: bool = False,
        idempotency_key: Optional[str] = None,
        partition_key: Optional[str] = None,
        correlation_id: Optional[str] = None,
        signal_id: Optional[UUID] = None,
        timestamp: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Accept many common constructor argument names and normalize them.

        Examples of accepted constructor kwargs seen across code/tests:
            - source_node_id (alias for source_id)
            - target_node_id (alias for target_id)
            - data (alias for content)
            - source_id, target_id, content
        """
        # canonical mapping
        self.signal_id = signal_id or uuid4()
        self.signal_type = signal_type or "generic"
        # prefer explicit content param, otherwise accept data
        self.content = content if content is not None else data
        # accept alias names
        self.source_id = source_id if source_id is not None else source_node_id
        self.target_id = target_id if target_id is not None else target_node_id
        self.urgency = float(urgency) if urgency is not None else 0.5
        self.requires_response = bool(requires_response)
        self.idempotency_key = idempotency_key
        self.partition_key = partition_key
        self.correlation_id = correlation_id
        self.processing_attempts = []
        self.timestamp = int(timestamp) if timestamp is not None else _now_timestamp()
        self.meta = meta or {}
        # keep any extra kwargs for forwards compatibility
        if kwargs:
            # store unexpected extra keys to meta for debugging / migration safety
            self.meta.setdefault("_extra_kwargs", {}).update(kwargs)

    def mark_attempt(self, node_id: int, correlation_id: Optional[str] = None) -> None:
        self.processing_attempts.append(
            {"node_id": node_id, "timestamp": _now_timestamp(), "correlation_id": correlation_id}
        )

    def __repr__(self) -> str:
        return (
            f"SocialSignal(type={self.signal_type!r}, id={str(self.signal_id)[:8]}, "
            f"src={self.source_id!r}, tgt={self.target_id!r}, urgency={self.urgency!r})"
        )
