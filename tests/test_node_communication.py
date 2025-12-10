"""
Tests for node communication module.
"""

import pytest
import time
from collections import deque

from core.node_communication import NodeCommunication
from core.social_signals import SocialSignal
from core.time_manager import get_timestamp


class TestNodeCommunication:
    """Test NodeCommunication class."""
    
    @pytest.fixture
    def comm(self):
        """Create a NodeCommunication instance for testing."""
        return NodeCommunication(node_id=1)
    
    @pytest.fixture
    def sample_signal(self):
        """Create a sample signal for testing."""
        return SocialSignal(
            content={"message": "test"},
            signal_type="test",
            urgency=0.5,
            source_id=2,
            requires_response=False
        )
    
    def test_initialization(self, comm):
        """Test NodeCommunication initialization."""
        assert comm.node_id == 1
        assert isinstance(comm.communication_queue, deque)
        assert isinstance(comm.signal_history, deque)
        assert comm.communication_queue.maxlen == 20
        assert comm.signal_history.maxlen == 100
    
    def test_add_signal_to_queue(self, comm, sample_signal):
        """Test adding signal to queue."""
        result = comm.add_signal_to_queue(sample_signal)
        assert result is True
        assert len(comm.communication_queue) == 1
        assert len(comm.signal_history) == 1
    
    def test_duplicate_signal_rejection(self, comm, sample_signal):
        """Test that duplicate signals are rejected."""
        # Add signal first time
        result1 = comm.add_signal_to_queue(sample_signal)
        assert result1 is True
        
        # Try to add same signal again
        result2 = comm.add_signal_to_queue(sample_signal)
        assert result2 is False
        assert len(comm.communication_queue) == 1
    
    def test_record_sent_signal(self, comm, sample_signal):
        """Test recording sent signals."""
        comm.record_sent_signal(sample_signal)
        assert len(comm.signal_history) == 1
        assert comm.signal_history[0]['action'] == 'sent'
    
    def test_validate_signal_schema(self, comm, sample_signal):
        """Test signal schema validation."""
        assert comm._validate_signal_schema(sample_signal) is True
    
    def test_record_signal_processed(self, comm, sample_signal):
        """Test recording processed signals."""
        comm.record_signal_processed(sample_signal)
        assert len(comm.processed_signals) > 0
    
    def test_record_signal_attempt(self, comm, sample_signal):
        """Test recording signal processing attempts."""
        comm.record_signal_attempt(sample_signal, success=True)
        comm.record_signal_attempt(sample_signal, success=False)
        
        signal_id = f"{sample_signal.source_id}_{sample_signal.signal_type}_{sample_signal.timestamp}"
        assert signal_id in comm.signal_attempts
        assert comm.signal_attempts[signal_id]['attempts'] == 2
        assert comm.signal_attempts[signal_id]['successes'] == 1
        assert comm.signal_attempts[signal_id]['failures'] == 1
    
    def test_partition_queue(self, comm, sample_signal):
        """Test partition queue functionality."""
        comm.add_to_partition_queue(0, sample_signal)
        assert 0 in comm.partition_queues
        assert len(comm.partition_queues[0]) == 1
    
    def test_get_queue_metrics(self, comm, sample_signal):
        """Test getting queue metrics."""
        comm.add_signal_to_queue(sample_signal)
        metrics = comm.get_queue_metrics()
        
        assert isinstance(metrics, dict)
        assert 'queue_size' in metrics
        assert 'queue_capacity' in metrics
        assert 'queue_utilization' in metrics
        assert metrics['queue_size'] == 1
    
    def test_get_signal_history(self, comm, sample_signal):
        """Test getting signal history."""
        comm.add_signal_to_queue(sample_signal)
        history = comm.get_signal_history(limit=5)
        
        assert isinstance(history, list)
        assert len(history) == 1
        assert 'signal' in history[0]
        assert 'timestamp' in history[0]
        assert 'action' in history[0]
    
    def test_clear_old_data(self, comm):
        """Test clearing old data."""
        # Add some old signals
        for i in range(5):
            sig = SocialSignal(
                content={},
                signal_type="test",
                urgency=0.5,
                source_id=i
            )
            sig.timestamp = get_timestamp() - 7200  # 2 hours ago
            comm.signal_history.append({
                'signal': sig,
                'timestamp': sig.timestamp,
                'action': 'received'
            })
        
        # Clear data older than 1 hour
        comm.clear_old_data(max_age=3600)
        
        # All old signals should be cleared
        assert len(comm.signal_history) == 0
    
    def test_processed_signals_limit(self, comm):
        """Test that processed signals set doesn't grow unbounded."""
        # Add many processed signals
        for i in range(1500):
            sig = SocialSignal(
                content={},
                signal_type="test",
                urgency=0.5,
                source_id=i
            )
            comm.record_signal_processed(sig)
        
        # Should be limited to around 1000 (500 after cleanup)
        assert len(comm.processed_signals) <= 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
