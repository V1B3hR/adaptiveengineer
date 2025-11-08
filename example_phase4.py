"""
Phase 4 Example: Autonomy, Adaptive Defenses, and Positive Collaboration

Demonstrates:
1. Autonomy with self-repair and ethical escalation
2. Adaptive self-healing cyber defenses
3. Biological-inspired swarm coordination
4. Evolving adversary simulation
"""

import logging
import sys
import time
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('phase4_example')

# Import Phase 4 components
from core.autonomy_engine import (
    AutonomyEngine, RepairStrategy, EscalationReason
)
from core.adaptive_defense import (
    AdaptiveDefenseSystem, ThreatType, DefenseAction, HealingAction
)
from core.swarm_defense import (
    SwarmDefenseSystem, ThreatLevel, SwarmRole
)
from core.adversary_simulation import (
    AdversarySimulation, AdversaryType
)


def demonstrate_autonomy():
    """Demonstrate autonomous self-repair and ethical escalation"""
    logger.info("="*70)
    logger.info("PHASE 4 DEMONSTRATION: Autonomy, Adaptive Defenses, and Collaboration")
    logger.info("="*70)
    
    logger.info("\n1. Demonstrating Autonomy and Self-Repair...")
    logger.info("   " + "-"*66)
    
    # Create autonomy engine
    logger.info("\n   1.1 Initialize autonomy engine...")
    autonomy = AutonomyEngine(
        node_id=1,
        ethics_threshold=0.7,
        privacy_threshold=0.8,
        auto_repair_enabled=True
    )
    logger.info("       ✓ Autonomy engine initialized")
    
    # Simulate service degradation and repair
    logger.info("\n   1.2 Detect service degradation and assess repair need...")
    service_name = "web_service"
    health_score = 0.3  # Degraded
    metrics = {
        'error_rate': 0.6,
        'response_time': 2.5,
        'uptime': 0.3
    }
    
    strategy = autonomy.assess_repair_need(service_name, health_score, metrics)
    logger.info(f"       ✓ Repair assessment: {strategy}")
    
    # Attempt autonomous repair
    logger.info("\n   1.3 Attempt autonomous self-repair...")
    repair = autonomy.attempt_self_repair(service_name, strategy)
    logger.info(f"       ✓ Repair {repair.status}: {repair.strategy}")
    logger.info(f"         Confidence: {repair.confidence:.2f}")
    
    # Test ethical escalation
    logger.info("\n   1.4 Test ethical escalation...")
    action = "access_user_database"
    context = {
        'accesses_pii': True,
        'accesses_user_data': True,
        'exports_data': True
    }
    
    should_escalate, reason = autonomy.should_escalate(action, context)
    if should_escalate:
        escalation = autonomy.escalate_action(action, reason, context, risk_level=0.9)
        logger.info(f"       ⚠ Action escalated: {reason}")
        logger.info(f"         Risk level: {escalation.risk_level:.2f}")
    
    # Show autonomy metrics
    metrics = autonomy.get_autonomy_metrics()
    logger.info(f"\n       Autonomy Metrics:")
    logger.info(f"       - Repairs attempted: {metrics['repairs_attempted']}")
    logger.info(f"       - Success rate: {metrics['success_rate']:.2%}")
    logger.info(f"       - Escalations: {metrics['escalations_made']}")


def demonstrate_adaptive_defense():
    """Demonstrate adaptive self-healing cyber defenses"""
    logger.info("\n\n2. Demonstrating Adaptive Self-Healing Defenses...")
    logger.info("   " + "-"*66)
    
    # Create defense system
    logger.info("\n   2.1 Initialize adaptive defense system...")
    defense = AdaptiveDefenseSystem(
        node_id=1,
        auto_response_enabled=True,
        response_threshold=0.7,
        audit_all=True
    )
    logger.info("       ✓ Adaptive defense system initialized")
    
    # Simulate various threats
    logger.info("\n   2.2 Detect and respond to threats automatically...")
    
    threats = [
        (ThreatType.DDOS, "192.168.1.100", 0.8, 0.9),
        (ThreatType.MALWARE, "suspicious_process.exe", 0.9, 0.85),
        (ThreatType.INTRUSION, "10.0.0.50", 0.7, 0.8),
    ]
    
    for threat_type, source, severity, confidence in threats:
        threat = defense.detect_threat(threat_type, source, severity, confidence)
        logger.info(f"       ⚠ Threat: {threat_type} from {source} (severity={severity:.2f})")
    
    # Show automated responses
    logger.info(f"\n       ✓ Automated responses executed:")
    logger.info(f"         - Blocked IPs: {len(defense.blocked_ips)}")
    logger.info(f"         - Quarantined items: {len(defense.quarantined_items)}")
    logger.info(f"         - Threats blocked: {defense.threats_blocked}/{defense.threats_detected}")
    
    # Demonstrate self-healing
    logger.info("\n   2.3 Perform self-healing actions...")
    
    healing = defense.heal_system(
        HealingAction.ROLLBACK,
        "firewall_config",
        "Suspicious configuration change detected"
    )
    logger.info(f"       ✓ Healing: {healing.healing_action} on {healing.target}")
    logger.info(f"         Success: {healing.success}, Recovery time: {healing.recovery_time:.3f}s")
    
    # Demonstrate adaptive learning
    logger.info("\n   2.4 Adapt thresholds based on feedback...")
    defense.adapt_thresholds(ThreatType.DDOS, 'false_positive')
    logger.info(f"       ✓ Adapted DDOS threshold: {defense.threat_thresholds[ThreatType.DDOS]:.2f}")
    
    # Show audit log
    logger.info("\n   2.5 Audit log (recent actions)...")
    audit_entries = defense.get_audit_log(limit=3)
    for entry in audit_entries:
        logger.info(f"       - {entry['type']}: {entry['action']} on {entry['target']}")


def demonstrate_swarm_defense():
    """Demonstrate biological-inspired swarm coordination"""
    logger.info("\n\n3. Demonstrating Swarm Defense (Digital White Blood Cells)...")
    logger.info("   " + "-"*66)
    
    # Create swarm defense system
    logger.info("\n   3.1 Initialize swarm defense system...")
    swarm = SwarmDefenseSystem(
        network_size=10,
        agent_count=20,
        coordination_range=5.0
    )
    logger.info("       ✓ Swarm initialized with 20 agents")
    
    # Show agent distribution
    metrics = swarm.get_swarm_metrics()
    logger.info("       Agent roles:")
    for role, count in metrics['role_distribution'].items():
        logger.info(f"         - {role}: {count}")
    
    # Simulate threat detection and swarm response
    logger.info("\n   3.2 Detect threats and coordinate swarm response...")
    
    threats = [
        ((3.0, 4.0), ThreatLevel.HIGH),
        ((7.0, 2.0), ThreatLevel.MEDIUM),
        ((5.0, 8.0), ThreatLevel.CRITICAL),
    ]
    
    for location, threat_level in threats:
        zone = swarm.detect_threat_swarm(location, threat_level)
        logger.info(f"       ⚠ {threat_level.value.upper()} threat at {location}")
        logger.info(f"         Agents responding: {len(zone.agents_responding)}")
        time.sleep(0.1)  # Brief pause for demonstration
    
    # Update swarm (agents move and follow pheromones)
    logger.info("\n   3.3 Update swarm state (agents patrol and follow pheromones)...")
    swarm.update_agent_positions(delta_time=1.0)
    logger.info("       ✓ Agents updated positions")
    
    # Show swarm metrics
    metrics = swarm.get_swarm_metrics()
    logger.info(f"\n       Swarm Metrics:")
    logger.info(f"       - Threats detected: {metrics['threats_detected']}")
    logger.info(f"       - Threats neutralized: {metrics['threats_neutralized']}")
    logger.info(f"       - Neutralization rate: {metrics['neutralization_rate']:.2%}")
    logger.info(f"       - Responses coordinated: {metrics['responses_coordinated']}")
    logger.info(f"       - Pheromone trails: {metrics['pheromone_trails']}")
    logger.info(f"       - Threat patterns stored: {metrics['threat_patterns_stored']}")
    
    # Visualize swarm state
    logger.info(swarm.visualize_swarm_state())


def demonstrate_evolving_adversaries():
    """Demonstrate evolving adversary simulation"""
    logger.info("\n4. Demonstrating Evolving Adversary Simulation...")
    logger.info("   " + "-"*66)
    
    # Create adversary simulation
    logger.info("\n   4.1 Initialize adversary simulation...")
    simulation = AdversarySimulation(population_size=5)
    logger.info("       ✓ Initialized 5 adversaries")
    
    # Show initial population
    logger.info("\n       Initial population:")
    for adv in simulation.adversaries.values():
        metrics = adv.get_metrics()
        logger.info(f"         - {metrics['adversary_id']}: {metrics['type']} "
                   f"(soph={metrics['sophistication']:.2f})")
    
    # Simulate attacks over multiple generations
    logger.info("\n   4.2 Simulate attacks over 3 generations...")
    
    targets = ['server1', 'server2', 'server3']
    defense_level = 0.5  # Medium defense
    
    for gen in range(1, 4):
        logger.info(f"\n       Generation {gen}:")
        
        # Simulate attacks
        attempts = simulation.simulate_attacks(targets, defense_level, rounds=10)
        
        # Show generation results
        sim_metrics = simulation.get_simulation_metrics()
        logger.info(f"         - Total attacks: {sim_metrics['total_attacks']}")
        logger.info(f"         - Successful attacks: {sim_metrics['total_successful']}")
        logger.info(f"         - Avg sophistication: {sim_metrics['avg_sophistication']:.2f}")
        logger.info(f"         - Avg fitness: {sim_metrics['avg_fitness']:.2f}")
        
        # Evolve population (except last generation)
        if gen < 3:
            simulation.evolve_population(selection_rate=0.5)
            logger.info("         ✓ Population evolved")
    
    # Show final adversary capabilities
    logger.info("\n   4.3 Final adversary capabilities (after evolution)...")
    for adv in simulation.adversaries.values():
        metrics = adv.get_metrics()
        logger.info(f"       - {metrics['adversary_id']}:")
        logger.info(f"         Type: {metrics['type']}, Generation: {metrics['generation']}")
        logger.info(f"         Sophistication: {metrics['sophistication']:.2f}, Fitness: {metrics['fitness']:.2f}")
        logger.info(f"         Success rate: {metrics['success_rate']:.2%}")
        logger.info(f"         Learned defenses: {metrics['learned_defenses']}")


def demonstrate_integrated_system():
    """Demonstrate all Phase 4 systems working together"""
    logger.info("\n\n5. Integrated Demonstration: Full Phase 4 System...")
    logger.info("   " + "-"*66)
    
    logger.info("\n   5.1 Initialize integrated defense system...")
    
    # Create all components
    autonomy = AutonomyEngine(node_id=1, auto_repair_enabled=True)
    defense = AdaptiveDefenseSystem(node_id=1, auto_response_enabled=True)
    swarm = SwarmDefenseSystem(network_size=10, agent_count=15)
    simulation = AdversarySimulation(population_size=3)
    
    logger.info("       ✓ All systems initialized")
    
    # Simulate integrated scenario
    logger.info("\n   5.2 Scenario: Coordinated attack by evolving adversaries...")
    
    # Adversaries attack
    targets = ['web_service', 'database', 'api_gateway']
    attempts = simulation.simulate_attacks(targets, defense_level=0.6, rounds=5)
    
    logger.info(f"       ⚠ {len(attempts)} attacks detected")
    
    # Defense systems respond
    for attempt in attempts[:3]:  # Show first 3
        # Adaptive defense detects and blocks
        if attempt.detected:
            threat_type = ThreatType.INTRUSION
            defense.detect_threat(
                threat_type,
                attempt.adversary_id,
                severity=0.8,
                confidence=0.85
            )
    
    # Swarm coordinates response to critical threats
    critical_location = (5.0, 5.0)
    swarm.detect_threat_swarm(critical_location, ThreatLevel.HIGH)
    
    # Autonomy engine repairs affected services
    for target in targets[:2]:
        health = random.uniform(0.3, 0.6)
        strategy = autonomy.assess_repair_need(target, health, {'error_rate': 0.5})
        if strategy:
            autonomy.attempt_self_repair(target, strategy)
    
    logger.info("\n       ✓ Coordinated defense response executed")
    
    # Show combined metrics
    logger.info("\n   5.3 Combined system metrics...")
    
    auto_metrics = autonomy.get_autonomy_metrics()
    defense_metrics = defense.get_defense_metrics()
    swarm_metrics = swarm.get_swarm_metrics()
    sim_metrics = simulation.get_simulation_metrics()
    
    logger.info(f"\n       Autonomy:")
    logger.info(f"         - Repairs: {auto_metrics['repairs_attempted']} "
               f"({auto_metrics['success_rate']:.1%} success)")
    logger.info(f"         - Escalations: {auto_metrics['escalations_made']}")
    
    logger.info(f"\n       Adaptive Defense:")
    logger.info(f"         - Threats: {defense_metrics['threats_detected']} detected, "
               f"{defense_metrics['threats_blocked']} blocked")
    logger.info(f"         - Block rate: {defense_metrics['block_rate']:.1%}")
    logger.info(f"         - Healings: {defense_metrics['healings_performed']}")
    
    logger.info(f"\n       Swarm Defense:")
    logger.info(f"         - Agents: {swarm_metrics['agent_count']}")
    logger.info(f"         - Threats neutralized: {swarm_metrics['threats_neutralized']}/{swarm_metrics['threats_detected']}")
    logger.info(f"         - Responses coordinated: {swarm_metrics['responses_coordinated']}")
    
    logger.info(f"\n       Adversaries:")
    logger.info(f"         - Population: {sim_metrics['population_size']}, "
               f"Gen: {sim_metrics['generation']}")
    logger.info(f"         - Attacks: {sim_metrics['total_attacks']} "
               f"({sim_metrics['total_successful']} successful)")
    logger.info(f"         - Avg sophistication: {sim_metrics['avg_sophistication']:.2f}")
    
    logger.info("\n       ✓ Integrated system demonstrated successfully!")


def main():
    """Run Phase 4 demonstration"""
    try:
        # Demonstrate each Phase 4 capability
        demonstrate_autonomy()
        demonstrate_adaptive_defense()
        demonstrate_swarm_defense()
        demonstrate_evolving_adversaries()
        demonstrate_integrated_system()
        
        logger.info("\n" + "="*70)
        logger.info("PHASE 4 DEMONSTRATION COMPLETE")
        logger.info("="*70)
        logger.info("\nKey Capabilities Demonstrated:")
        logger.info("  ✓ Autonomy with self-repair and ethical escalation")
        logger.info("  ✓ Adaptive self-healing cyber defenses")
        logger.info("  ✓ Biological-inspired swarm coordination")
        logger.info("  ✓ Evolving adversary simulation")
        logger.info("  ✓ Fully integrated autonomous defense system")
        logger.info("\nPhase 4 implementation is complete and operational!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
