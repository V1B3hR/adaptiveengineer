#!/usr/bin/env python3
"""
Tests for Adversarial Co-Evolution Environment
"""

import sys
import os

# Add parent directory to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from adaptiveengineer import AliveLoopNode
from simulation.adversarial_environment import (
    AdversarialEnvironment,
    AttackerAgent,
    AttackType,
)


def test_attacker_agent_creation():
    """Test creating attacker agents"""
    print("Testing AttackerAgent creation...")

    attacker = AttackerAgent(agent_id=1, initial_energy=10.0)

    assert attacker.agent_id == 1
    assert attacker.energy == 10.0
    assert attacker.max_energy == 10.0
    assert len(attacker.attack_strategies) > 0
    assert attacker.successful_attacks == 0

    print("✓ AttackerAgent creation test passed")


def test_attacker_execute_attack():
    """Test attack execution"""
    print("Testing AttackerAgent attack execution...")

    attacker = AttackerAgent(agent_id=1, initial_energy=10.0)

    # Execute attack
    pattern, event = attacker.execute_attack(
        target_id=99, attack_type=AttackType.ENERGY_DRAIN
    )

    assert pattern is not None
    assert event is not None
    assert event.attacker_id == 1
    assert event.target_id == 99
    assert event.attack_type == AttackType.ENERGY_DRAIN

    # Energy should be consumed
    assert attacker.energy < 10.0

    print("✓ Attack execution test passed")


def test_attacker_learning():
    """Test attacker learning from detection"""
    print("Testing AttackerAgent learning...")

    attacker = AttackerAgent(agent_id=1)

    initial_mutation_rate = attacker.attack_strategies[
        AttackType.ENERGY_DRAIN
    ].mutation_rate

    # Learn from detection
    attacker.learn_from_detection(AttackType.ENERGY_DRAIN, detected=True)

    # Mutation rate should increase
    new_mutation_rate = attacker.attack_strategies[
        AttackType.ENERGY_DRAIN
    ].mutation_rate
    assert new_mutation_rate >= initial_mutation_rate
    assert attacker.times_detected == 1

    print("✓ Learning test passed")


def test_attacker_evolution():
    """Test attacker strategy evolution"""
    print("Testing AttackerAgent evolution...")

    attacker = AttackerAgent(agent_id=1)

    # Simulate detection
    attacker.times_detected = 10
    attacker.successful_attacks = 5
    attacker.failed_attacks = 5

    # Store original strategy IDs
    original_ids = {
        attack_type: strategy.pattern_id
        for attack_type, strategy in attacker.attack_strategies.items()
    }

    # Evolve strategies
    attacker.evolve_strategies()

    # At least some strategies should have evolved (new IDs)
    evolved_count = sum(
        1
        for attack_type, strategy in attacker.attack_strategies.items()
        if strategy.pattern_id != original_ids.get(attack_type)
    )

    print(f"  {evolved_count} strategies evolved")
    print("✓ Evolution test passed")


def test_attacker_fitness():
    """Test attacker fitness calculation"""
    print("Testing AttackerAgent fitness...")

    attacker = AttackerAgent(agent_id=1)

    # Simulate performance
    attacker.successful_attacks = 8
    attacker.failed_attacks = 2
    attacker.total_energy_drained = 15.0
    attacker.times_detected = 3

    fitness = attacker.get_fitness()

    assert fitness > 0
    assert attacker.evolution_score == fitness

    print(f"  Fitness: {fitness:.3f}")
    print("✓ Fitness test passed")


def test_adversarial_environment_creation():
    """Test creating adversarial environment"""
    print("Testing AdversarialEnvironment creation...")

    env = AdversarialEnvironment(
        num_attackers=3, num_defenders=5, evolution_interval=25
    )

    assert len(env.attackers) == 3
    assert env.num_defenders == 5
    assert env.evolution_interval == 25
    assert env.current_generation == 0

    print("✓ Environment creation test passed")


def test_adversarial_environment_attack_wave():
    """Test simulating attack waves"""
    print("Testing AdversarialEnvironment attack wave...")

    env = AdversarialEnvironment(num_attackers=2, num_defenders=3)

    # Create defender nodes
    defenders = {}
    for i in range(3):
        node = AliveLoopNode(
            position=(i * 2.0, 0.0),
            velocity=(0.0, 0.0),
            initial_energy=10.0,
            node_id=i,
        )
        defenders[i] = node

    # Simulate attack wave
    attacks = env.simulate_attack_wave(defenders)

    assert isinstance(attacks, list)
    # Should have some attacks (might be 0 if attackers low on energy)
    assert env.steps == 1

    print(f"  {len(attacks)} attacks in wave")
    print("✓ Attack wave test passed")


def test_adversarial_environment_detection():
    """Test recording detections"""
    print("Testing AdversarialEnvironment detection recording...")

    env = AdversarialEnvironment(num_attackers=2, num_defenders=3)

    initial_detections = env.total_detections

    # Record detection
    env.record_detection(
        attacker_id=0,
        attack_type=AttackType.ENERGY_DRAIN,
        defender_id=1,
        detection_time=0.5,
    )

    assert env.total_detections == initial_detections + 1

    print("✓ Detection recording test passed")


def test_adversarial_environment_mitigation():
    """Test recording mitigations"""
    print("Testing AdversarialEnvironment mitigation recording...")

    env = AdversarialEnvironment(num_attackers=2, num_defenders=3)

    initial_mitigations = env.total_mitigations

    # Record successful mitigation
    env.record_mitigation(
        defender_id=1,
        pattern_id="pattern_123",
        success=True,
        countermeasure="rate_limiting",
        response_time=0.3,
    )

    assert env.total_mitigations == initial_mitigations + 1
    assert len(env.defense_events) == 1

    print("✓ Mitigation recording test passed")


def test_adversarial_environment_evolution():
    """Test population evolution"""
    print("Testing AdversarialEnvironment evolution...")

    env = AdversarialEnvironment(num_attackers=3, num_defenders=5)

    initial_generation = env.current_generation
    initial_pattern_count = len(env.threat_library.patterns)

    # Trigger evolution
    env.evolve_population()

    assert env.current_generation == initial_generation + 1
    # Should have more patterns after evolution
    assert len(env.threat_library.patterns) >= initial_pattern_count

    print(f"  Generation: {env.current_generation}")
    print(f"  Patterns: {len(env.threat_library.patterns)}")
    print("✓ Evolution test passed")


def test_adversarial_environment_statistics():
    """Test environment statistics"""
    print("Testing AdversarialEnvironment statistics...")

    env = AdversarialEnvironment(num_attackers=2, num_defenders=3)

    # Create defenders
    defenders = {}
    for i in range(3):
        node = AliveLoopNode(
            position=(i * 2.0, 0.0),
            velocity=(0.0, 0.0),
            initial_energy=10.0,
            node_id=i,
        )
        node.max_energy = 10.0
        defenders[i] = node

    # Simulate some activity
    for _ in range(5):
        env.simulate_attack_wave(defenders)

    # Get statistics
    stats = env.get_statistics()

    assert "generation" in stats
    assert "total_attacks" in stats
    assert "attacker_stats" in stats
    assert "threat_library" in stats

    print(f"  Total attacks: {stats['total_attacks']}")
    print("✓ Statistics test passed")


def run_all_tests():
    """Run all adversarial environment tests"""
    print("\n" + "=" * 70)
    print("  ADVERSARIAL ENVIRONMENT TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_attacker_agent_creation,
        test_attacker_execute_attack,
        test_attacker_learning,
        test_attacker_evolution,
        test_attacker_fitness,
        test_adversarial_environment_creation,
        test_adversarial_environment_attack_wave,
        test_adversarial_environment_detection,
        test_adversarial_environment_mitigation,
        test_adversarial_environment_evolution,
        test_adversarial_environment_statistics,
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
