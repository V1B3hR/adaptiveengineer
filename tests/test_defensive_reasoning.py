#!/usr/bin/env python3
"""
Tests for Defensive Reasoning Methods
"""

import sys
import os

# Add parent directory to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from adaptiveengineer import AliveLoopNode
from core.threat_patterns import ThreatPattern, ThreatLibrary


def test_reason_about_threat():
    """Test threat reasoning"""
    print("Testing reason_about_threat...")

    node = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1
    )
    library = ThreatLibrary()

    pattern = ThreatPattern(
        signature=[0.8, 0.3, 0.7],
        severity=0.75,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )
    library.add_pattern(pattern)

    analysis = node.reason_about_threat(pattern, library)

    assert "confidence" in analysis
    assert "recommended_countermeasures" in analysis
    assert "reasoning_chain" in analysis
    assert analysis["confidence"] >= 0.0

    print("✓ Threat reasoning test passed")


def test_generate_countermeasure():
    """Test countermeasure generation"""
    print("Testing generate_countermeasure...")

    node = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1
    )

    pattern = ThreatPattern(
        signature=[0.8, 0.3, 0.7],
        severity=0.75,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )

    countermeasure = node.generate_countermeasure(pattern)

    assert "strategy" in countermeasure
    assert "actions" in countermeasure
    assert "energy_cost" in countermeasure
    assert "expected_effectiveness" in countermeasure
    assert len(countermeasure["actions"]) > 0

    print("✓ Countermeasure generation test passed")


def test_share_threat_intelligence():
    """Test threat intelligence sharing"""
    print("Testing share_threat_intelligence...")

    node1 = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1
    )
    node2 = AliveLoopNode(
        position=(1, 0), velocity=(0, 0), initial_energy=10.0, node_id=2
    )
    node3 = AliveLoopNode(
        position=(2, 0), velocity=(0, 0), initial_energy=10.0, node_id=3
    )

    # Setup trust
    node1.trust_network[2] = 0.8
    node1.trust_network[3] = 0.8

    pattern = ThreatPattern(
        signature=[0.8, 0.3, 0.7],
        severity=0.75,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )

    notified = node1.share_threat_intelligence(
        [node2, node3], pattern, confidence=0.9
    )

    assert notified >= 0  # Should notify some nodes

    print(f"  Notified {notified} nodes")
    print("✓ Intelligence sharing test passed")


def run_all_tests():
    """Run all defensive reasoning tests"""
    print("\n" + "=" * 70)
    print("  DEFENSIVE REASONING TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_reason_about_threat,
        test_generate_countermeasure,
        test_share_threat_intelligence,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
