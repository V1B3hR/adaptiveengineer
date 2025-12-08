#!/usr/bin/env python3
"""
Hybrid Defense Swarm Demonstration - Mobile Cyber-Defense

Demonstrates:
- Swarm robots defending infrastructure
- Mobile defense agents patrolling network
- Coordination against distributed attacks
- Adaptive positioning based on threat intelligence
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptiveengineer import AliveLoopNode
from core.threat_patterns import ThreatPattern, ThreatLibrary
from simulation.adversarial_environment import (
    AdversarialEnvironment,
    AttackType
)
from plugins.swarm_robotics import (
    SwarmRoboticsPlugin,
    FormationType,
    SwarmBehavior
)


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def visualize_hybrid_state(defenders, threats, title="Hybrid Defense State"):
    """Visualize defenders and threats"""
    print(f"\n{title}:")
    grid_size = 25
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Mark defenders
    for node_id, node in defenders.items():
        x = int(node.position[0]) % grid_size
        y = int(node.position[1]) % grid_size
        
        if 0 <= x < grid_size and 0 <= y < grid_size:
            if node.energy < 0.3:
                grid[y][x] = 'd'  # Low energy defender
            else:
                grid[y][x] = 'D'  # Active defender
    
    # Mark threat zones
    for threat_pos in threats:
        x = int(threat_pos[0]) % grid_size
        y = int(threat_pos[1]) % grid_size
        
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[y][x] = 'X'  # Threat
    
    print("  +" + "-" * grid_size + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * grid_size + "+")
    print("  Legend: D=Defender, d=Low Energy, X=Threat\n")


def main():
    print("\n🛡️🤖 HYBRID DEFENSE SWARM DEMONSTRATION 🤖🛡️")
    print("Mobile Cyber-Defense with Swarm Coordination\n")
    
    # Initialize mobile defense swarm
    print_section("Initializing Mobile Defense Swarm")
    
    num_defenders = 6
    defenders = {}
    
    for i in range(num_defenders):
        node = AliveLoopNode(
            position=(
                np.random.uniform(5, 20),
                np.random.uniform(5, 20)
            ),
            velocity=(0.0, 0.0),
            initial_energy=10.0,
            node_id=i
        )
        node.max_energy = 10.0
        defenders[i] = node
        print(f"✓ Defense Agent {i}: Position ({node.position[0]:.1f}, {node.position[1]:.1f})")
    
    # Setup trust network for threat intelligence sharing
    print("\nEstablishing defense network...")
    for node_id, node in defenders.items():
        for other_id in defenders.keys():
            if other_id != node_id:
                node.trust_network[other_id] = 0.8  # High trust within team
    print("✓ Defense network established")
    
    # Initialize swarm robotics plugin
    swarm_plugin = SwarmRoboticsPlugin(
        formation_type=FormationType.CIRCLE,
        behavior_mode=SwarmBehavior.PERIMETER_DEFENSE,
        perception_radius=6.0
    )
    
    # Set protected area center
    protected_center = np.array([12.0, 12.0])
    swarm_plugin.set_swarm_center(protected_center)
    
    for node in defenders.values():
        swarm_plugin.initialize(node)
    
    print(f"✓ Swarm plugin configured: PERIMETER_DEFENSE mode")
    print(f"✓ Protected center: ({protected_center[0]}, {protected_center[1]})")
    
    # Initialize adversarial environment
    print_section("Initializing Threat Environment")
    
    env = AdversarialEnvironment(
        num_attackers=4,
        num_defenders=num_defenders,
        evolution_interval=25
    )
    
    for defender_id in defenders.keys():
        env.register_defender(defender_id)
    
    print(f"✓ Created threat environment with {env.num_attackers} attackers")
    
    # Phase 1: Perimeter Defense Setup
    print_section("Phase 1: Establishing Perimeter Defense")
    print("Deploying defense agents around protected infrastructure...\n")
    
    for step in range(20):
        for node_id, node in defenders.items():
            # Update neighbors
            neighbors = []
            for other_id, other in defenders.items():
                if other_id != node_id:
                    distance = np.linalg.norm(node.position - other.position)
                    if distance < swarm_plugin.perception_radius:
                        neighbors.append(other)
            
            node._swarm_neighbors = neighbors
            
            # Update swarm position
            swarm_plugin.update(node, delta_time=0.1)
            
            # Move
            node.position = node.position + node.velocity * 0.1
            movement_cost = np.linalg.norm(node.velocity) * 0.01
            node.energy = max(0.0, node.energy - movement_cost)
    
    print("✓ Perimeter defense established")
    visualize_hybrid_state(defenders, [protected_center], "Initial Defense Perimeter")
    
    # Phase 2: Coordinated Defense Against Attacks
    print_section("Phase 2: Defending Against Coordinated Attacks")
    
    threat_locations = []
    detection_count = 0
    coordinated_responses = 0
    
    for step in range(50):
        # Simulate attack wave
        attacks = env.simulate_attack_wave(defenders)
        
        if attacks:
            print(f"\n⚔️  Step {step + 1}: {len(attacks)} attacks detected")
            
            for pattern, event in attacks:
                if event.success:
                    # Process attack
                    if event.target_id in defenders:
                        target = defenders[event.target_id]
                        target.energy = max(0.0, target.energy - event.energy_drained)
                        
                        # Record threat location
                        threat_loc = target.position.copy()
                        threat_locations.append(threat_loc)
                        
                        print(f"   💥 Agent {event.target_id} under attack: "
                              f"{event.attack_type.value}")
                        
                        # Analyze threat
                        analysis = target.reason_about_threat(
                            pattern,
                            env.threat_library
                        )
                        
                        if analysis['confidence'] > 0.5:
                            detection_count += 1
                            print(f"   🔍 Threat detected with confidence {analysis['confidence']:.2f}")
                            
                            # Record detection
                            env.record_detection(
                                event.attacker_id,
                                AttackType(event.attack_type),
                                event.target_id,
                                detection_time=0.3
                            )
                            
                            # Generate and apply countermeasure
                            countermeasure = target.generate_countermeasure(pattern)
                            
                            if countermeasure['constraints_satisfied']:
                                target.energy -= countermeasure['energy_cost']
                                
                                mitigation_success = (
                                    countermeasure['expected_effectiveness'] > 0.6
                                )
                                
                                if mitigation_success:
                                    print(f"   🛡️  Countermeasure: {countermeasure['actions'][0]}")
                                    
                                    pattern.add_countermeasure(
                                        countermeasure['actions'][0],
                                        countermeasure['expected_effectiveness']
                                    )
                                
                                env.record_mitigation(
                                    event.target_id,
                                    pattern.pattern_id,
                                    mitigation_success,
                                    countermeasure['actions'][0],
                                    response_time=0.2
                                )
                            
                            # Share threat intelligence
                            other_defenders = [
                                n for nid, n in defenders.items()
                                if nid != event.target_id
                            ]
                            
                            shared = target.share_threat_intelligence(
                                other_defenders,
                                pattern,
                                confidence=analysis['confidence']
                            )
                            
                            if shared > 0:
                                coordinated_responses += 1
                                print(f"   📡 Coordinated {shared} nearby defenders")
                                
                                # Reposition swarm toward threat
                                # Update swarm center toward threat
                                threat_direction = threat_loc - protected_center
                                swarm_plugin.set_swarm_center(
                                    protected_center + 0.3 * threat_direction
                                )
        
        # Update swarm positions (patrol and response)
        for node_id, node in defenders.items():
            if node.energy > 0.1:  # Only active nodes
                neighbors = []
                for other_id, other in defenders.items():
                    if other_id != node_id and other.energy > 0.1:
                        distance = np.linalg.norm(node.position - other.position)
                        if distance < swarm_plugin.perception_radius:
                            neighbors.append(other)
                
                node._swarm_neighbors = neighbors
                swarm_plugin.update(node, delta_time=0.1)
                
                # Move
                node.position = node.position + node.velocity * 0.1
                movement_cost = np.linalg.norm(node.velocity) * 0.01
                node.energy = max(0.0, node.energy - movement_cost)
        
        # Recharge defenders periodically
        if step % 10 == 0:
            for node in defenders.values():
                if node.energy < node.max_energy:
                    node.energy = min(node.max_energy, node.energy + 1.0)
        
        # Check for compromised defenders
        compromised = [nid for nid, node in defenders.items() if node.energy < 0.2]
        if compromised:
            print(f"   ⚠️  Warning: Agents {compromised} at critical energy!")
            
            # Switch to emergency regroup if multiple compromised
            if len(compromised) >= 2:
                swarm_plugin.set_behavior_mode(SwarmBehavior.EMERGENCY_REGROUP)
                print(f"   🚨 EMERGENCY REGROUP activated")
                
                # Regroup for 10 steps
                for regroup_step in range(10):
                    for node_id, node in defenders.items():
                        if node.energy > 0.1:
                            swarm_plugin.update(node, delta_time=0.1)
                            node.position = node.position + node.velocity * 0.1
                
                # Return to perimeter defense
                swarm_plugin.set_behavior_mode(SwarmBehavior.PERIMETER_DEFENSE)
                swarm_plugin.set_swarm_center(protected_center)
    
    # Final visualization
    visualize_hybrid_state(defenders, threat_locations[-5:], "Final Defense State")
    
    # Final statistics
    print_section("Final Statistics")
    
    env_stats = env.get_statistics()
    swarm_stats = swarm_plugin.get_statistics()
    
    print(f"📊 Hybrid Defense Performance:")
    print(f"\n🛡️  Cyber-Defense Metrics:")
    print(f"   Total Attacks: {env_stats['total_attacks']}")
    print(f"   Detections: {env_stats['total_detections']}")
    print(f"   Detection Rate: {env_stats['detection_rate']*100:.1f}%")
    print(f"   Mitigations: {env_stats['total_mitigations']}")
    print(f"   Mitigation Rate: {env_stats['mitigation_rate']*100:.1f}%")
    print(f"   Coordinated Responses: {coordinated_responses}")
    
    print(f"\n🤖 Swarm Coordination Metrics:")
    print(f"   Formation: {swarm_stats['formation_type']}")
    print(f"   Behavior: {swarm_stats['behavior_mode']}")
    print(f"   Hardware Failures: {swarm_stats['hardware_failures']}")
    
    print(f"\n🧬 Threat Evolution:")
    print(f"   Threat Patterns: {env_stats['threat_library']['total_patterns']}")
    print(f"   Evolution Generations: {env_stats['threat_library']['evolution_generations']}")
    
    print(f"\n🤖 Defense Agent Status:")
    for agent_id, node in defenders.items():
        status = "ACTIVE" if node.energy > 0.3 else "DEGRADED"
        if node.energy < 0.1:
            status = "CRITICAL"
        print(f"   Agent {agent_id}: Energy={node.energy:.2f}, "
              f"Position=({node.position[0]:.1f}, {node.position[1]:.1f}), "
              f"Status={status}")
    
    print("\n✅ Hybrid Defense Swarm Demonstration Complete!\n")
    print("Key Observations:")
    print("1. Mobile defenders autonomously patrolled perimeter")
    print("2. Swarm repositioned dynamically based on threat location")
    print("3. Threat intelligence shared for coordinated response")
    print("4. Emergency regroup activated when multiple agents compromised")
    print("5. Combined swarm coordination with cyber-defense reasoning\n")


if __name__ == "__main__":
    main()
