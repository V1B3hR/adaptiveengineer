"""
Tests for adaptive defense mechanisms.
"""

import pytest
from core.adaptive_defense import (
    AdaptiveDefenseSystem,
    ThreatType,
    ThreatEvent,
    DefenseAction,
    HealingAction,
)


class TestAdaptiveDefenseSystem:
    """Test adaptive defense system."""

    def test_system_creation(self):
        """Test creating adaptive defense system."""
        system = AdaptiveDefenseSystem(node_id=1)
        assert system.node_id == 1
        assert system.threats_detected == 0

    def test_detect_threat(self):
        """Test threat detection."""
        system = AdaptiveDefenseSystem(node_id=1)

        event = system.detect_threat(
            threat_type=ThreatType.DDOS,
            source="192.168.1.100",
            severity=0.8,
            confidence=0.9,
        )

        assert event is not None
        assert event.threat_type == ThreatType.DDOS

    def test_get_metrics(self):
        """Test metrics retrieval."""
        system = AdaptiveDefenseSystem(node_id=1)
        metrics = system.get_defense_metrics()
        assert metrics is not None
        assert "threats_detected" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
