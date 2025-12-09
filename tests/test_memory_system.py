"""Unit tests for the memory system module.

Tests Memory, MemoryType, Classification, and ShortMemoryStore classes.
"""

import time
import pytest
from core.memory_system import (
    Memory,
    MemoryType,
    Classification,
    ShortMemoryStore,
)


class TestMemoryType:
    """Test MemoryType enum."""

    def test_memory_types_exist(self):
        """Test all memory type values."""
        assert MemoryType.REWARD == "reward"
        assert MemoryType.SHARED == "shared"
        assert MemoryType.PREDICTION == "prediction"
        assert MemoryType.PATTERN == "pattern"
        assert MemoryType.SHORT == "short"

    def test_memory_type_string_conversion(self):
        """Test memory type can be created from string."""
        mt = MemoryType("reward")
        assert mt == MemoryType.REWARD


class TestClassification:
    """Test Classification enum."""

    def test_classification_levels(self):
        """Test all classification values."""
        assert Classification.PUBLIC == "public"
        assert Classification.PROTECTED == "protected"
        assert Classification.PRIVATE == "private"
        assert Classification.CONFIDENTIAL == "confidential"

    def test_classification_string_conversion(self):
        """Test classification can be created from string."""
        cls = Classification("private")
        assert cls == Classification.PRIVATE


class TestMemory:
    """Test Memory dataclass."""

    def test_memory_creation_basic(self):
        """Test basic memory creation."""
        mem = Memory(
            content="test content",
            importance=0.8,
            timestamp=int(time.time()),
            memory_type=MemoryType.REWARD,
        )
        assert mem.content == "test content"
        assert mem.importance == 0.8
        assert mem.memory_type == MemoryType.REWARD
        assert mem.classification == Classification.PUBLIC

    def test_memory_with_string_enum(self):
        """Test memory creation with string enum values."""
        mem = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type="reward",
            classification="private",
        )
        assert mem.memory_type == MemoryType.REWARD
        assert mem.classification == Classification.PRIVATE

    def test_memory_timestamp_default(self):
        """Test memory gets default timestamp if invalid."""
        mem = Memory(
            content="test",
            importance=0.5,
            timestamp=0,
            memory_type=MemoryType.SHARED,
        )
        assert mem.timestamp > 0

    def test_memory_importance_clamp(self):
        """Test importance is clamped to non-negative."""
        mem = Memory(
            content="test",
            importance=-1.0,
            timestamp=int(time.time()),
            memory_type=MemoryType.PATTERN,
        )
        assert mem.importance == 0.0

    def test_memory_emotional_valence_clamp(self):
        """Test emotional valence is clamped to [-1.0, 1.0]."""
        mem1 = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHARED,
            emotional_valence=2.0,
        )
        assert mem1.emotional_valence == 1.0

        mem2 = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHARED,
            emotional_valence=-2.0,
        )
        assert mem2.emotional_valence == -1.0

    def test_memory_is_short(self):
        """Test is_short method."""
        mem_short = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )
        assert mem_short.is_short() is True

        mem_long = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.REWARD,
        )
        assert mem_long.is_short() is False

    def test_memory_expiry(self):
        """Test memory expiry functionality."""
        now = int(time.time())
        mem = Memory(
            content="test",
            importance=0.5,
            timestamp=now - 100,  # 100 seconds ago
            memory_type=MemoryType.SHORT,
            retention_limit=50,  # expires after 50 seconds
        )
        assert mem.is_expired(now) is True

        mem2 = Memory(
            content="test",
            importance=0.5,
            timestamp=now - 10,
            memory_type=MemoryType.SHORT,
            retention_limit=50,
        )
        assert mem2.is_expired(now) is False

    def test_memory_decay(self):
        """Test memory importance decay."""
        mem = Memory(
            content="test",
            importance=1.0,
            timestamp=int(time.time()),
            memory_type=MemoryType.PATTERN,
            decay_rate=0.9,
        )
        initial_importance = mem.importance
        mem.age()
        assert mem.importance < initial_importance
        assert mem.importance == pytest.approx(0.9, rel=0.01)

    def test_memory_decay_with_emotion(self):
        """Test memory decay is affected by emotional valence."""
        mem_emotional = Memory(
            content="test",
            importance=1.0,
            timestamp=int(time.time()),
            memory_type=MemoryType.REWARD,
            decay_rate=0.9,
            emotional_valence=0.9,
        )

        mem_neutral = Memory(
            content="test",
            importance=1.0,
            timestamp=int(time.time()),
            memory_type=MemoryType.REWARD,
            decay_rate=0.9,
            emotional_valence=0.0,
        )

        mem_emotional.age()
        mem_neutral.age()

        # Emotional memories decay slower
        assert mem_emotional.importance > mem_neutral.importance

    def test_memory_access_public(self):
        """Test accessing public memory."""
        mem = Memory(
            content="public data",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHARED,
            classification=Classification.PUBLIC,
            source_node=1,
        )

        # Any node can access public memory
        content = mem.access(accessor_id=2)
        assert content == "public data"
        assert mem.access_count == 1

    def test_memory_access_private(self):
        """Test accessing private memory."""
        mem = Memory(
            content="private data",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHARED,
            classification=Classification.PRIVATE,
            source_node=1,
        )

        # Owner can access
        content_owner = mem.access(accessor_id=1)
        assert content_owner == "private data"

        # Non-owner gets redacted
        content_other = mem.access(accessor_id=2)
        assert content_other == "[REDACTED]"

    def test_memory_access_confidential(self):
        """Test accessing confidential memory."""
        mem = Memory(
            content="confidential data",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHARED,
            classification=Classification.CONFIDENTIAL,
            source_node=1,
        )

        # Owner can access
        content_owner = mem.access(accessor_id=1)
        assert content_owner == "confidential data"

        # Non-owner gets redacted
        content_other = mem.access(accessor_id=2)
        assert content_other == "[REDACTED - CONFIDENTIAL]"

    def test_memory_access_protected(self):
        """Test accessing protected memory."""
        long_content = "x" * 200
        mem = Memory(
            content=long_content,
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHARED,
            classification=Classification.PROTECTED,
            source_node=1,
        )

        # Owner can access full content
        content_owner = mem.access(accessor_id=1)
        assert content_owner == long_content

        # Non-owner gets limited/summary access
        content_other = mem.access(accessor_id=2)
        # Protected content shows summary for long content
        assert "[SUMMARY:" in content_other or "[LIMITED:" in content_other
        assert len(content_other) < len(long_content)

    def test_memory_to_dict(self):
        """Test memory serialization."""
        mem = Memory(
            content="test",
            importance=0.7,
            timestamp=12345,
            memory_type=MemoryType.REWARD,
            emotional_valence=0.5,
            source_node=1,
        )

        data = mem.to_dict()
        assert data["content"] == "test"
        assert data["importance"] == 0.7
        assert data["timestamp"] == 12345
        assert data["memory_type"] == "reward"
        assert data["emotional_valence"] == 0.5

    def test_memory_to_dict_with_redaction(self):
        """Test memory serialization with redaction."""
        mem = Memory(
            content="private data",
            importance=0.7,
            timestamp=12345,
            memory_type=MemoryType.SHARED,
            classification=Classification.PRIVATE,
            source_node=1,
        )

        data = mem.to_dict(redact_for=2)
        assert data["content"] == "[REDACTED]"

    def test_memory_update_content(self):
        """Test updating memory content."""
        mem = Memory(
            content="original",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.PATTERN,
        )

        mem.update_content("updated", importance_delta=0.2)
        assert mem.content == "updated"
        assert mem.importance == 0.7

    def test_memory_audit_log(self):
        """Test audit logging on access."""
        mem = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHARED,
        )

        mem.access(accessor_id=1)
        mem.access(accessor_id=2)

        assert len(mem.audit_log) == 2
        assert "accessed_by_1_at_" in mem.audit_log[0]
        assert "accessed_by_2_at_" in mem.audit_log[1]


class TestShortMemoryStore:
    """Test ShortMemoryStore class."""

    def test_store_creation(self):
        """Test creating a short memory store."""
        store = ShortMemoryStore(capacity_mb=2.0)
        assert store.capacity_mb == 2.0
        stats = store.stats()
        assert stats["used_mb"] == 0.0
        assert stats["count"] == 0

    def test_store_put_and_get(self):
        """Test storing and retrieving memories."""
        store = ShortMemoryStore()
        mem = Memory(
            content="test data",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )

        store.put(key=1, mem=mem)
        retrieved = store.get(key=1)

        assert retrieved is not None
        assert retrieved.content == "test data"

    def test_store_rejects_non_short_memory(self):
        """Test store rejects non-SHORT memory types."""
        store = ShortMemoryStore()
        mem = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.REWARD,  # Not SHORT
        )

        with pytest.raises(ValueError):
            store.put(key=1, mem=mem)

    def test_store_lru_eviction(self):
        """Test LRU eviction when capacity exceeded."""
        store = ShortMemoryStore(capacity_mb=0.001)  # Very small

        # Add memories until eviction occurs
        for i in range(10):
            mem = Memory(
                content=f"data_{i}" * 100,  # Make it bigger
                importance=0.5,
                timestamp=int(time.time()),
                memory_type=MemoryType.SHORT,
            )
            store.put(key=i, mem=mem)

        stats = store.stats()
        # Should have evicted some memories
        assert stats["count"] < 10
        assert stats["used_mb"] <= store.capacity_mb

    def test_store_lru_ordering(self):
        """Test LRU ordering on access."""
        store = ShortMemoryStore(capacity_mb=0.001)

        mem1 = Memory(
            content="A" * 50,
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )
        mem2 = Memory(
            content="B" * 50,
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )

        store.put(key=1, mem=mem1)
        store.put(key=2, mem=mem2)

        # Access mem1 to make it most recent
        store.get(key=1)

        # Add a large memory to trigger eviction
        mem3 = Memory(
            content="C" * 200,
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )
        store.put(key=3, mem=mem3)

        # mem1 should still be there (recently accessed)
        # mem2 might be evicted (least recently used)
        assert store.get(key=1) is not None

    def test_store_remove(self):
        """Test removing memory from store."""
        store = ShortMemoryStore()
        mem = Memory(
            content="test",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )

        store.put(key=1, mem=mem)
        assert store.get(key=1) is not None

        store.remove(key=1)
        assert store.get(key=1) is None

    def test_store_clear(self):
        """Test clearing the store."""
        store = ShortMemoryStore()

        for i in range(5):
            mem = Memory(
                content=f"test_{i}",
                importance=0.5,
                timestamp=int(time.time()),
                memory_type=MemoryType.SHORT,
            )
            store.put(key=i, mem=mem)

        stats_before = store.stats()
        assert stats_before["count"] == 5

        store.clear()

        stats_after = store.stats()
        assert stats_after["count"] == 0
        assert stats_after["used_mb"] == 0.0

    def test_store_size_estimation(self):
        """Test size estimation for different content types."""
        store = ShortMemoryStore()

        # String content
        mem_str = Memory(
            content="test string",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )
        store.put(key=1, mem=mem_str)
        assert mem_str.size_mb > 0

        # Bytes content
        mem_bytes = Memory(
            content=b"test bytes",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )
        store.put(key=2, mem=mem_bytes)
        assert mem_bytes.size_mb > 0

    def test_store_update_existing_key(self):
        """Test updating an existing key updates LRU position."""
        store = ShortMemoryStore()

        mem1 = Memory(
            content="first",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )
        mem2 = Memory(
            content="second",
            importance=0.5,
            timestamp=int(time.time()),
            memory_type=MemoryType.SHORT,
        )

        store.put(key=1, mem=mem1)
        stats1 = store.stats()

        store.put(key=1, mem=mem2)  # Update same key
        retrieved = store.get(key=1)

        assert retrieved.content == "second"
        stats2 = store.stats()
        assert stats2["count"] == stats1["count"]  # Count unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
