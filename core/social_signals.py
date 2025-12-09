"""Social signal infrastructure for node-to-node communication.

This module provides structured signals with production features including:
- Idempotency keys
- Correlation IDs for distributed tracing
- Partition keys for ordering guarantees
- Schema versioning for evolution
- Retry tracking
"""

import uuid
from typing import Any

from core.time_manager import get_timestamp


class SocialSignal:
    """Structured signal for node-to-node communication with production features"""

    def __init__(
        self,
        content: Any,
        signal_type: str,
        urgency: float,
        source_id: int,
        requires_response: bool = False,
        idempotency_key: str | None = None,
        partition_key: str | None = None,
        correlation_id: str | None = None,
        schema_version: str = "1.0",
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.signal_type = (
            signal_type  # 'memory', 'query', 'warning', 'resource'
        )
        self.urgency = urgency  # 0.0 to 1.0
        self.source_id = source_id
        self.timestamp = get_timestamp()
        self.requires_response = requires_response
        self.response = None

        # Production features
        self.idempotency_key = (
            idempotency_key
            or f"{source_id}_{signal_type}_{uuid.uuid4().hex[:8]}"
        )
        self.partition_key = (
            partition_key or f"{source_id}_{signal_type}"
        )  # For ordering guarantees
        self.correlation_id = correlation_id or str(
            uuid.uuid4()
        )  # For distributed tracing
        self.schema_version = schema_version  # For schema evolution
        self.retry_count = 0  # Track retry attempts
        self.created_at = (
            get_timestamp()
        )  # Creation timestamp for age calculation
        self.processing_attempts = []  # Track processing history
