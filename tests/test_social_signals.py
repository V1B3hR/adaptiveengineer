"""Unit tests for the social signals module.

Tests SocialSignal class with production features.
"""

import pytest
from core.social_signals import SocialSignal


class TestSocialSignal:
    """Test SocialSignal class."""
    
    def test_signal_creation_basic(self):
        """Test basic signal creation."""
        signal = SocialSignal(
            content="test message",
            signal_type="memory",
            urgency=0.8,
            source_id=1
        )
        
        assert signal.content == "test message"
        assert signal.signal_type == "memory"
        assert signal.urgency == 0.8
        assert signal.source_id == 1
        assert signal.requires_response is False
        assert signal.response is None
        assert signal.id is not None
        assert signal.timestamp > 0
    
    def test_signal_types(self):
        """Test different signal types."""
        types = ["memory", "query", "warning", "resource"]
        
        for sig_type in types:
            signal = SocialSignal(
                content="test",
                signal_type=sig_type,
                urgency=0.5,
                source_id=1
            )
            assert signal.signal_type == sig_type
    
    def test_signal_urgency_range(self):
        """Test urgency values (should be 0.0 to 1.0)."""
        signal_low = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.0,
            source_id=1
        )
        assert signal_low.urgency == 0.0
        
        signal_high = SocialSignal(
            content="test",
            signal_type="warning",
            urgency=1.0,
            source_id=1
        )
        assert signal_high.urgency == 1.0
        
        signal_mid = SocialSignal(
            content="test",
            signal_type="query",
            urgency=0.5,
            source_id=1
        )
        assert signal_mid.urgency == 0.5
    
    def test_signal_requires_response(self):
        """Test signals that require responses."""
        signal = SocialSignal(
            content="request",
            signal_type="query",
            urgency=0.7,
            source_id=1,
            requires_response=True
        )
        
        assert signal.requires_response is True
        assert signal.response is None
        
        # Simulate response
        signal.response = "response data"
        assert signal.response == "response data"
    
    def test_signal_idempotency_key(self):
        """Test idempotency key generation."""
        signal1 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        # Auto-generated idempotency key
        assert signal1.idempotency_key is not None
        assert "1_memory" in signal1.idempotency_key
        
        # Custom idempotency key
        signal2 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1,
            idempotency_key="custom_key"
        )
        assert signal2.idempotency_key == "custom_key"
    
    def test_signal_partition_key(self):
        """Test partition key for ordering guarantees."""
        signal1 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        # Auto-generated partition key
        assert signal1.partition_key == "1_memory"
        
        # Custom partition key
        signal2 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1,
            partition_key="custom_partition"
        )
        assert signal2.partition_key == "custom_partition"
    
    def test_signal_correlation_id(self):
        """Test correlation ID for distributed tracing."""
        signal1 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        # Auto-generated correlation ID
        assert signal1.correlation_id is not None
        
        # Custom correlation ID (for request chaining)
        custom_corr_id = "trace-12345"
        signal2 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1,
            correlation_id=custom_corr_id
        )
        assert signal2.correlation_id == custom_corr_id
    
    def test_signal_schema_version(self):
        """Test schema versioning for evolution."""
        signal1 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        # Default schema version
        assert signal1.schema_version == "1.0"
        
        # Custom schema version
        signal2 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1,
            schema_version="2.0"
        )
        assert signal2.schema_version == "2.0"
    
    def test_signal_retry_tracking(self):
        """Test retry count tracking."""
        signal = SocialSignal(
            content="test",
            signal_type="query",
            urgency=0.7,
            source_id=1
        )
        
        assert signal.retry_count == 0
        
        # Simulate retries
        signal.retry_count += 1
        assert signal.retry_count == 1
        
        signal.retry_count += 1
        assert signal.retry_count == 2
    
    def test_signal_creation_timestamp(self):
        """Test creation timestamp tracking."""
        signal = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        assert signal.created_at > 0
        assert signal.timestamp > 0
        # created_at and timestamp should be close
        assert abs(signal.created_at - signal.timestamp) < 1.0
    
    def test_signal_processing_attempts(self):
        """Test processing attempts tracking."""
        signal = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        assert signal.processing_attempts == []
        
        # Simulate processing attempts
        signal.processing_attempts.append({"node": 1, "result": "success"})
        signal.processing_attempts.append({"node": 2, "result": "failed"})
        
        assert len(signal.processing_attempts) == 2
        assert signal.processing_attempts[0]["result"] == "success"
    
    def test_signal_unique_ids(self):
        """Test that each signal gets a unique ID."""
        signal1 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        signal2 = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        
        # Each signal should have a unique ID
        assert signal1.id != signal2.id
    
    def test_signal_content_types(self):
        """Test signals with different content types."""
        # String content
        signal_str = SocialSignal(
            content="string message",
            signal_type="memory",
            urgency=0.5,
            source_id=1
        )
        assert isinstance(signal_str.content, str)
        
        # Dict content
        signal_dict = SocialSignal(
            content={"key": "value", "data": [1, 2, 3]},
            signal_type="query",
            urgency=0.7,
            source_id=1
        )
        assert isinstance(signal_dict.content, dict)
        assert signal_dict.content["key"] == "value"
        
        # Numeric content
        signal_num = SocialSignal(
            content=42,
            signal_type="resource",
            urgency=0.3,
            source_id=1
        )
        assert signal_num.content == 42
    
    def test_signal_source_tracking(self):
        """Test source node tracking."""
        sources = [1, 5, 10, 100]
        
        for source in sources:
            signal = SocialSignal(
                content="test",
                signal_type="memory",
                urgency=0.5,
                source_id=source
            )
            assert signal.source_id == source
            assert str(source) in signal.idempotency_key
            assert str(source) in signal.partition_key
    
    def test_signal_production_features_integration(self):
        """Test integration of all production features."""
        signal = SocialSignal(
            content={"action": "backup", "priority": "high"},
            signal_type="warning",
            urgency=0.9,
            source_id=42,
            requires_response=True,
            idempotency_key="backup-op-001",
            partition_key="critical-ops",
            correlation_id="trace-xyz-789",
            schema_version="2.1"
        )
        
        # Verify all features are set correctly
        assert signal.content["action"] == "backup"
        assert signal.signal_type == "warning"
        assert signal.urgency == 0.9
        assert signal.source_id == 42
        assert signal.requires_response is True
        assert signal.idempotency_key == "backup-op-001"
        assert signal.partition_key == "critical-ops"
        assert signal.correlation_id == "trace-xyz-789"
        assert signal.schema_version == "2.1"
        assert signal.retry_count == 0
        assert signal.processing_attempts == []
        
        # Simulate processing
        signal.processing_attempts.append({
            "timestamp": signal.timestamp,
            "node": 5,
            "result": "acknowledged"
        })
        signal.response = {"status": "started", "eta": 300}
        
        assert len(signal.processing_attempts) == 1
        assert signal.response["status"] == "started"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
