#!/usr/bin/env python3
"""
Cyber-Defense Demonstration - Adversarial Co-Evolution

Demonstrates:
- Threat pattern detection and learning
- Adversarial arms race between attackers and defenders
- Threat intelligence sharing among trusted nodes
- Defensive reasoning and countermeasure generation
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from adaptiveengineer import AliveLoopNode
from core.threat_patterns import ThreatPattern, ThreatLibrary
from simulation.adversarial_environment import (
    AdversarialEnvironment,
    AttackType,
)


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    print("\n🛡️  CYBER-DEFENSE DEMONSTRATION 🛡️")
    print("Adversarial Co-Evolution Environment\n")

    # Initialize defender nodes
    print_section("Initializing Defender Nodes")

    defenders = {}
    for i in range(5):
        node = AliveLoopNode(
            position=(i * 2.0, 0.0),
            velocity=(0.0, 0.0),
            initial_energy=10.0,
            node_id=i,
        )
        node.max_energy = 10.0
        defenders[i] = node
        print(
            f"✓ Defender Node {i}: Position {node.position}, Energy {node.energy:.2f}"
        )

    # Setup trust network
    print("\nEstablishing trust network...")
    for node_id, node in defenders.items():
        for other_id in defenders.keys():
            if other_id != node_id:
                node.trust_network[other_id] = 0.7  # Start with moderate trust
    print("✓ Trust network established")

    # Initialize adversarial environment
    print_section("Initializing Adversarial Environment")

    env = AdversarialEnvironment(
        num_attackers=3, num_defenders=len(defenders), evolution_interval=20
    )

    for defender_id in defenders.keys():
        env.register_defender(defender_id)

    print(f"✓ Created environment with {env.num_attackers} attackers")
    print(f"✓ Threat library initialized")

    # Run simulation
    print_section("Running Adversarial Simulation")

    detection_count = 0
    mitigation_count = 0
    shared_intelligence = 0

    for step in range(50):
        print(f"\n--- Step {step + 1}/50 ---")

        # Simulate attack wave
        attacks = env.simulate_attack_wave(defenders)

        if attacks:
            print(f"⚔️  {len(attacks)} attacks launched")

            for pattern, event in attacks:
                if event.success:
                    # Attack succeeded - drain energy from target
                    if event.target_id in defenders:
                        target = defenders[event.target_id]
                        target.energy = max(
                            0.0, target.energy - event.energy_drained
                        )
                        print(
                            f"   💥 Attack on Node {event.target_id}: "
                            f"drained {event.energy_drained:.2f} energy "
                            f"(remaining: {target.energy:.2f})"
                        )

                        # Defender analyzes threat
                        analysis = target.reason_about_threat(
                            pattern, env.threat_library
                        )

                        if analysis["confidence"] > 0.5:
                            detection_count += 1
                            print(
                                f"   🔍 Node {event.target_id} detected threat: "
                                f"confidence={analysis['confidence']:.2f}"
                            )

                            # Record detection
                            env.record_detection(
                                event.attacker_id,
                                AttackType(event.attack_type),
                                event.target_id,
                                detection_time=0.5,
                            )

                            # Generate countermeasure
                            countermeasure = target.generate_countermeasure(
                                pattern
                            )

                            if countermeasure["constraints_satisfied"]:
                                # Apply countermeasure
                                target.energy -= countermeasure["energy_cost"]

                                mitigation_success = (
                                    countermeasure["expected_effectiveness"]
                                    > 0.6
                                )

                                if mitigation_success:
                                    mitigation_count += 1
                                    print(
                                        f"   🛡️  Countermeasure applied: "
                                        f"{countermeasure['actions'][0]}"
                                    )

                                    # Update threat pattern with effectiveness
                                    pattern.add_countermeasure(
                                        countermeasure["actions"][0],
                                        countermeasure[
                                            "expected_effectiveness"
                                        ],
                                    )

                                # Record mitigation
                                env.record_mitigation(
                                    event.target_id,
                                    pattern.pattern_id,
                                    mitigation_success,
                                    countermeasure["actions"][0],
                                    response_time=0.3,
                                )

                            # Share threat intelligence with trusted nodes
                            other_defenders = [
                                n
                                for nid, n in defenders.items()
                                if nid != event.target_id
                            ]

                            shared = target.share_threat_intelligence(
                                other_defenders,
                                pattern,
                                confidence=analysis["confidence"],
                            )

                            if shared > 0:
                                shared_intelligence += shared
                                print(
                                    f"   📡 Shared intelligence with {shared} nodes"
                                )

        # Recharge defenders (simulate recovery)
        for node in defenders.values():
            if node.energy < node.max_energy:
                node.energy = min(node.max_energy, node.energy + 0.5)

        # Evolution checkpoint
        if (step + 1) % 10 == 0:
            print("\n" + env.visualize_arms_race())

    # Final statistics
    print_section("Final Statistics")

    stats = env.get_statistics()

    print(f"📊 Overall Performance:")
    print(f"   Total Attacks: {stats['total_attacks']}")
    print(f"   Successful Attacks: {stats['successful_attacks']}")
    print(f"   Attack Success Rate: {stats['attack_success_rate']*100:.1f}%")
    print(f"   ")
    print(f"   Detections: {stats['total_detections']}")
    print(f"   Detection Rate: {stats['detection_rate']*100:.1f}%")
    print(f"   Mitigations: {stats['total_mitigations']}")
    print(f"   Mitigation Rate: {stats['mitigation_rate']*100:.1f}%")
    print(f"   ")
    print(f"   Intelligence Sharing Events: {shared_intelligence}")

    print(f"\n🧬 Evolution:")
    print(
        f"   Threat Patterns in Library: {stats['threat_library']['total_patterns']}"
    )
    print(
        f"   Evolution Generations: {stats['threat_library']['evolution_generations']}"
    )
    print(
        f"   Average Attacker Fitness: {stats['attacker_stats']['avg_fitness']:.3f}"
    )

    print(f"\n🛡️  Defender Metrics:")
    for defender_id, node in defenders.items():
        print(
            f"   Node {defender_id}: Energy={node.energy:.2f}/{node.max_energy:.2f}, "
            f"Trust Network Size={len(node.trust_network)}"
        )

    # Show learned countermeasures
    print(f"\n📚 Learned Countermeasures:")
    threat_stats = env.threat_library.get_statistics()
    for attack_type in AttackType:
        patterns = env.threat_library.get_patterns_by_type(attack_type.value)
        if patterns:
            print(f"\n   {attack_type.value}:")
            for pattern in patterns[:2]:  # Show top 2
                best_cms = pattern.get_best_countermeasures(top_n=2)
                if best_cms:
                    for cm, effectiveness in best_cms:
                        print(
                            f"      • {cm}: {effectiveness*100:.1f}% effective"
                        )

    print("\n✅ Cyber-Defense Demonstration Complete!\n")
    print("Key Observations:")
    print("1. Defenders learned to detect and respond to threats")
    print("2. Threat patterns evolved to evade detection")
    print("3. Intelligence sharing improved collective defense")
    print("4. Countermeasures adapted based on effectiveness\n")


if __name__ == "__main__":
    main()
