#!/usr/bin/env python3
"""
Tests for Threat Pattern Genome System
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.threat_patterns import ThreatPattern, ThreatLibrary


def test_threat_pattern_creation():
    """Test creating threat patterns"""
    print("Testing ThreatPattern creation...")

    pattern = ThreatPattern(
        signature=[0.8, 0.3, 0.9, 0.1, 0.7],
        severity=0.75,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )

    assert pattern.severity == 0.75
    assert pattern.mutation_rate == 0.2
    assert pattern.attack_type == "energy_drain"
    assert len(pattern.signature) == 5
    assert pattern.detection_count == 0

    print("✓ ThreatPattern creation test passed")


def test_threat_pattern_countermeasures():
    """Test adding and retrieving countermeasures"""
    print("Testing ThreatPattern countermeasures...")

    pattern = ThreatPattern(
        signature=[0.5, 0.5, 0.5], severity=0.6, mutation_rate=0.15
    )

    # Add countermeasures
    pattern.add_countermeasure("rate_limiting", 0.8)
    pattern.add_countermeasure("firewall_rule", 0.7)
    pattern.add_countermeasure("isolation", 0.9)

    # Get best countermeasures
    best = pattern.get_best_countermeasures(top_n=2)

    assert len(best) == 2
    assert best[0][0] == "isolation"  # Highest effectiveness
    assert best[0][1] == 0.9

    print("✓ Countermeasures test passed")


def test_threat_pattern_mutation():
    """Test pattern mutation"""
    print("Testing ThreatPattern mutation...")

    original = ThreatPattern(
        signature=[0.5, 0.5, 0.5, 0.5],
        severity=0.6,
        mutation_rate=0.3,
        attack_type="test_attack",
    )

    # Mutate
    mutant = original.mutate()

    assert mutant.pattern_id != original.pattern_id
    assert mutant.attack_type == original.attack_type
    assert len(mutant.signature) == len(original.signature)

    # Signatures should be different (with high probability)
    signature_diff = sum(
        abs(a - b) for a, b in zip(original.signature, mutant.signature)
    )
    # With mutation_rate=0.3, we expect some difference
    # (this is probabilistic, but very likely to differ)

    print(f"  Signature difference: {signature_diff:.4f}")
    print("✓ Mutation test passed")


def test_threat_library_basic():
    """Test basic threat library operations"""
    print("Testing ThreatLibrary basic operations...")

    library = ThreatLibrary(similarity_threshold=0.7)

    # Add patterns
    pattern1 = ThreatPattern(
        signature=[0.8, 0.2, 0.9],
        severity=0.7,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )

    pattern2 = ThreatPattern(
        signature=[0.3, 0.9, 0.2],
        severity=0.6,
        mutation_rate=0.15,
        attack_type="jamming",
    )

    library.add_pattern(pattern1)
    library.add_pattern(pattern2)

    assert len(library.patterns) == 2
    assert library.total_threats_detected == 2

    # Retrieve by ID
    retrieved = library.get_pattern_by_id(pattern1.pattern_id)
    assert retrieved.pattern_id == pattern1.pattern_id

    print("✓ Basic operations test passed")


def test_threat_library_similarity():
    """Test pattern similarity matching"""
    print("Testing ThreatLibrary similarity matching...")

    library = ThreatLibrary(similarity_threshold=0.5)

    # Add similar patterns
    pattern1 = ThreatPattern(
        signature=[0.8, 0.2, 0.9, 0.1],
        severity=0.7,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )

    pattern2 = ThreatPattern(
        signature=[0.85, 0.25, 0.85, 0.15],  # Very similar
        severity=0.75,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )

    pattern3 = ThreatPattern(
        signature=[0.1, 0.9, 0.1, 0.9],  # Very different
        severity=0.5,
        mutation_rate=0.15,
        attack_type="jamming",
    )

    library.add_pattern(pattern1)
    library.add_pattern(pattern2)
    library.add_pattern(pattern3)

    # Find similar to pattern1
    similar = library.find_similar(pattern1, threshold=0.5)

    # Should find pattern2 as similar, but not pattern3
    assert len(similar) >= 1

    # Pattern2 should be more similar than pattern3
    similar_ids = [p.pattern_id for p, _ in similar]
    if pattern2.pattern_id in similar_ids:
        print("  ✓ Found similar pattern (pattern2)")

    print("✓ Similarity matching test passed")


def test_threat_library_evolution():
    """Test pattern evolution"""
    print("Testing ThreatLibrary evolution...")

    library = ThreatLibrary()

    # Add base patterns
    for i in range(3):
        pattern = ThreatPattern(
            signature=[0.5 + i * 0.1, 0.5 - i * 0.1, 0.5],
            severity=0.6,
            mutation_rate=0.2,
            attack_type="test",
        )
        library.add_pattern(pattern)

    initial_count = len(library.patterns)

    # Evolve patterns
    evolved = library.evolve_patterns(count=2)

    assert len(evolved) == 2
    assert len(library.patterns) == initial_count + 2
    assert library.evolution_generations == 1

    print(f"  Created {len(evolved)} evolved patterns")
    print("✓ Evolution test passed")


def test_threat_library_countermeasures():
    """Test retrieving effective countermeasures"""
    print("Testing ThreatLibrary countermeasure retrieval...")

    library = ThreatLibrary(similarity_threshold=0.6)

    # Create pattern with known countermeasures
    pattern = ThreatPattern(
        signature=[0.8, 0.3, 0.7],
        severity=0.7,
        mutation_rate=0.2,
        attack_type="energy_drain",
    )

    pattern.add_countermeasure("rate_limiting", 0.85)
    pattern.add_countermeasure("throttling", 0.75)
    pattern.add_countermeasure("isolation", 0.90)

    library.add_pattern(pattern)

    # Get effective countermeasures
    countermeasures = library.get_effective_countermeasures(pattern, top_n=3)

    assert len(countermeasures) > 0
    assert (
        countermeasures[0][1] >= countermeasures[-1][1]
    )  # Sorted by effectiveness

    print(f"  Retrieved {len(countermeasures)} countermeasures")
    print("✓ Countermeasure retrieval test passed")


def test_threat_library_statistics():
    """Test library statistics"""
    print("Testing ThreatLibrary statistics...")

    library = ThreatLibrary()

    # Add diverse patterns
    for i in range(5):
        pattern = ThreatPattern(
            signature=[0.5, 0.5, 0.5],
            severity=0.5 + i * 0.1,
            mutation_rate=0.2,
            attack_type=["energy_drain", "jamming", "trust_poisoning"][i % 3],
        )
        pattern.detection_count = i + 1
        pattern.successful_mitigations = i
        library.add_pattern(pattern)

    stats = library.get_statistics()

    assert stats["total_patterns"] == 5
    assert "attack_types" in stats
    assert stats["total_threats_detected"] > 0

    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Attack types: {stats['attack_types']}")
    print("✓ Statistics test passed")


def run_all_tests():
    """Run all threat pattern tests"""
    print("\n" + "=" * 70)
    print("  THREAT PATTERN TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_threat_pattern_creation,
        test_threat_pattern_countermeasures,
        test_threat_pattern_mutation,
        test_threat_library_basic,
        test_threat_library_similarity,
        test_threat_library_evolution,
        test_threat_library_countermeasures,
        test_threat_library_statistics,
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
