"""
Phase 4 Example: The Sentient Polity & Governance Layer

Demonstrates:
1. Autonomy with self-repair and ethical escalation
2. Adaptive self-healing cyber defenses
3. Biological-inspired swarm coordination
4. Evolving adversary simulation
5. NEW: Reputation Ledger & Meritocratic Progression
6. NEW: Contract Net Protocol & Collaborative Tasking
7. NEW: Council of Professors Governance
8. NEW: Constitutional Framework
9. NEW: Real-World Integration Bridge

Usage:
    python3 example/example_phase4.py
"""

import logging
import sys
import os
import time
import random

# Add parent directory to path to allow imports from root
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("phase4_example")

# Import Phase 4 components (existing)
from core.autonomy_engine import (
    AutonomyEngine,
    RepairStrategy,
    EscalationReason,
)
from core.adaptive_defense import (
    AdaptiveDefenseSystem,
    ThreatType,
    DefenseAction,
    HealingAction,
)
from core.swarm_defense import SwarmDefenseSystem, ThreatLevel, SwarmRole
from core.adversary_simulation import AdversarySimulation, AdversaryType

# Import Phase 4 NEW components (Polity & Governance)
from core.reputation_ledger import ReputationLedger, PrivilegeTier
from core.contract_net import ContractNetProtocol, TaskRequirements
from core.governance import CouncilOfProfessors
from core.constitutional_framework import ConstitutionalFramework
from core.real_world_bridge import RealWorldBridge, IntegrationSystem


def demonstrate_autonomy():
    """Demonstrate autonomous self-repair and ethical escalation"""
    logger.info("=" * 70)
    logger.info(
        "PHASE 4 DEMONSTRATION: Autonomy, Adaptive Defenses, and Collaboration"
    )
    logger.info("=" * 70)

    logger.info("\n1. Demonstrating Autonomy and Self-Repair...")
    logger.info("   " + "-" * 66)

    # Create autonomy engine
    logger.info("\n   1.1 Initialize autonomy engine...")
    autonomy = AutonomyEngine(
        node_id=1,
        ethics_threshold=0.7,
        privacy_threshold=0.8,
        auto_repair_enabled=True,
    )
    logger.info("       ✓ Autonomy engine initialized")

    # Simulate service degradation and repair
    logger.info(
        "\n   1.2 Detect service degradation and assess repair need..."
    )
    service_name = "web_service"
    health_score = 0.3  # Degraded
    metrics = {"error_rate": 0.6, "response_time": 2.5, "uptime": 0.3}

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
        "accesses_pii": True,
        "accesses_user_data": True,
        "exports_data": True,
    }

    should_escalate, reason = autonomy.should_escalate(action, context)
    if should_escalate:
        escalation = autonomy.escalate_action(
            action, reason, context, risk_level=0.9
        )
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
    logger.info("   " + "-" * 66)

    # Create defense system
    logger.info("\n   2.1 Initialize adaptive defense system...")
    defense = AdaptiveDefenseSystem(
        node_id=1,
        auto_response_enabled=True,
        response_threshold=0.7,
        audit_all=True,
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
        threat = defense.detect_threat(
            threat_type, source, severity, confidence
        )
        logger.info(
            f"       ⚠ Threat: {threat_type} from {source} (severity={severity:.2f})"
        )

    # Show automated responses
    logger.info(f"\n       ✓ Automated responses executed:")
    logger.info(f"         - Blocked IPs: {len(defense.blocked_ips)}")
    logger.info(
        f"         - Quarantined items: {len(defense.quarantined_items)}"
    )
    logger.info(
        f"         - Threats blocked: {defense.threats_blocked}/{defense.threats_detected}"
    )

    # Demonstrate self-healing
    logger.info("\n   2.3 Perform self-healing actions...")

    healing = defense.heal_system(
        HealingAction.ROLLBACK,
        "firewall_config",
        "Suspicious configuration change detected",
    )
    logger.info(
        f"       ✓ Healing: {healing.healing_action} on {healing.target}"
    )
    logger.info(
        f"         Success: {healing.success}, Recovery time: {healing.recovery_time:.3f}s"
    )

    # Demonstrate adaptive learning
    logger.info("\n   2.4 Adapt thresholds based on feedback...")
    defense.adapt_thresholds(ThreatType.DDOS, "false_positive")
    logger.info(
        f"       ✓ Adapted DDOS threshold: {defense.threat_thresholds[ThreatType.DDOS]:.2f}"
    )

    # Show audit log
    logger.info("\n   2.5 Audit log (recent actions)...")
    audit_entries = defense.get_audit_log(limit=3)
    for entry in audit_entries:
        logger.info(
            f"       - {entry['type']}: {entry['action']} on {entry['target']}"
        )


def demonstrate_swarm_defense():
    """Demonstrate biological-inspired swarm coordination"""
    logger.info(
        "\n\n3. Demonstrating Swarm Defense (Digital White Blood Cells)..."
    )
    logger.info("   " + "-" * 66)

    # Create swarm defense system
    logger.info("\n   3.1 Initialize swarm defense system...")
    swarm = SwarmDefenseSystem(
        network_size=10, agent_count=20, coordination_range=5.0
    )
    logger.info("       ✓ Swarm initialized with 20 agents")

    # Show agent distribution
    metrics = swarm.get_swarm_metrics()
    logger.info("       Agent roles:")
    for role, count in metrics["role_distribution"].items():
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
        logger.info(
            f"       ⚠ {threat_level.value.upper()} threat at {location}"
        )
        logger.info(
            f"         Agents responding: {len(zone.agents_responding)}"
        )
        time.sleep(0.1)  # Brief pause for demonstration

    # Update swarm (agents move and follow pheromones)
    logger.info(
        "\n   3.3 Update swarm state (agents patrol and follow pheromones)..."
    )
    swarm.update_agent_positions(delta_time=1.0)
    logger.info("       ✓ Agents updated positions")

    # Show swarm metrics
    metrics = swarm.get_swarm_metrics()
    logger.info(f"\n       Swarm Metrics:")
    logger.info(f"       - Threats detected: {metrics['threats_detected']}")
    logger.info(
        f"       - Threats neutralized: {metrics['threats_neutralized']}"
    )
    logger.info(
        f"       - Neutralization rate: {metrics['neutralization_rate']:.2%}"
    )
    logger.info(
        f"       - Responses coordinated: {metrics['responses_coordinated']}"
    )
    logger.info(f"       - Pheromone trails: {metrics['pheromone_trails']}")
    logger.info(
        f"       - Threat patterns stored: {metrics['threat_patterns_stored']}"
    )

    # Visualize swarm state
    logger.info(swarm.visualize_swarm_state())


def demonstrate_evolving_adversaries():
    """Demonstrate evolving adversary simulation"""
    logger.info("\n4. Demonstrating Evolving Adversary Simulation...")
    logger.info("   " + "-" * 66)

    # Create adversary simulation
    logger.info("\n   4.1 Initialize adversary simulation...")
    simulation = AdversarySimulation(population_size=5)
    logger.info("       ✓ Initialized 5 adversaries")

    # Show initial population
    logger.info("\n       Initial population:")
    for adv in simulation.adversaries.values():
        metrics = adv.get_metrics()
        logger.info(
            f"         - {metrics['adversary_id']}: {metrics['type']} "
            f"(soph={metrics['sophistication']:.2f})"
        )

    # Simulate attacks over multiple generations
    logger.info("\n   4.2 Simulate attacks over 3 generations...")

    targets = ["server1", "server2", "server3"]
    defense_level = 0.5  # Medium defense

    for gen in range(1, 4):
        logger.info(f"\n       Generation {gen}:")

        # Simulate attacks
        attempts = simulation.simulate_attacks(
            targets, defense_level, rounds=10
        )

        # Show generation results
        sim_metrics = simulation.get_simulation_metrics()
        logger.info(
            f"         - Total attacks: {sim_metrics['total_attacks']}"
        )
        logger.info(
            f"         - Successful attacks: {sim_metrics['total_successful']}"
        )
        logger.info(
            f"         - Avg sophistication: {sim_metrics['avg_sophistication']:.2f}"
        )
        logger.info(
            f"         - Avg fitness: {sim_metrics['avg_fitness']:.2f}"
        )

        # Evolve population (except last generation)
        if gen < 3:
            simulation.evolve_population(selection_rate=0.5)
            logger.info("         ✓ Population evolved")

    # Show final adversary capabilities
    logger.info("\n   4.3 Final adversary capabilities (after evolution)...")
    for adv in simulation.adversaries.values():
        metrics = adv.get_metrics()
        logger.info(f"       - {metrics['adversary_id']}:")
        logger.info(
            f"         Type: {metrics['type']}, Generation: {metrics['generation']}"
        )
        logger.info(
            f"         Sophistication: {metrics['sophistication']:.2f}, Fitness: {metrics['fitness']:.2f}"
        )
        logger.info(f"         Success rate: {metrics['success_rate']:.2%}")
        logger.info(
            f"         Learned defenses: {metrics['learned_defenses']}"
        )


def demonstrate_integrated_system():
    """Demonstrate all Phase 4 systems working together"""
    logger.info("\n\n5. Integrated Demonstration: Full Phase 4 System...")
    logger.info("   " + "-" * 66)

    logger.info("\n   5.1 Initialize integrated defense system...")

    # Create all components
    autonomy = AutonomyEngine(node_id=1, auto_repair_enabled=True)
    defense = AdaptiveDefenseSystem(node_id=1, auto_response_enabled=True)
    swarm = SwarmDefenseSystem(network_size=10, agent_count=15)
    simulation = AdversarySimulation(population_size=3)

    logger.info("       ✓ All systems initialized")

    # Simulate integrated scenario
    logger.info(
        "\n   5.2 Scenario: Coordinated attack by evolving adversaries..."
    )

    # Adversaries attack
    targets = ["web_service", "database", "api_gateway"]
    attempts = simulation.simulate_attacks(
        targets, defense_level=0.6, rounds=5
    )

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
                confidence=0.85,
            )

    # Swarm coordinates response to critical threats
    critical_location = (5.0, 5.0)
    swarm.detect_threat_swarm(critical_location, ThreatLevel.HIGH)

    # Autonomy engine repairs affected services
    for target in targets[:2]:
        health = random.uniform(0.3, 0.6)
        strategy = autonomy.assess_repair_need(
            target, health, {"error_rate": 0.5}
        )
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
    logger.info(
        f"         - Repairs: {auto_metrics['repairs_attempted']} "
        f"({auto_metrics['success_rate']:.1%} success)"
    )
    logger.info(f"         - Escalations: {auto_metrics['escalations_made']}")

    logger.info(f"\n       Adaptive Defense:")
    logger.info(
        f"         - Threats: {defense_metrics['threats_detected']} detected, "
        f"{defense_metrics['threats_blocked']} blocked"
    )
    logger.info(f"         - Block rate: {defense_metrics['block_rate']:.1%}")
    logger.info(
        f"         - Healings: {defense_metrics['healings_performed']}"
    )

    logger.info(f"\n       Swarm Defense:")
    logger.info(f"         - Agents: {swarm_metrics['agent_count']}")
    logger.info(
        f"         - Threats neutralized: {swarm_metrics['threats_neutralized']}/{swarm_metrics['threats_detected']}"
    )
    logger.info(
        f"         - Responses coordinated: {swarm_metrics['responses_coordinated']}"
    )

    logger.info(f"\n       Adversaries:")
    logger.info(
        f"         - Population: {sim_metrics['population_size']}, "
        f"Gen: {sim_metrics['generation']}"
    )
    logger.info(
        f"         - Attacks: {sim_metrics['total_attacks']} "
        f"({sim_metrics['total_successful']} successful)"
    )
    logger.info(
        f"         - Avg sophistication: {sim_metrics['avg_sophistication']:.2f}"
    )

    logger.info("\n       ✓ Integrated system demonstrated successfully!")


def demonstrate_reputation_and_meritocracy():
    """Demonstrate reputation ledger and meritocratic progression"""
    logger.info(
        "\n\n6. Demonstrating Reputation Ledger & Meritocratic Progression..."
    )
    logger.info("   " + "-" * 66)

    # Initialize reputation ledger
    logger.info("\n   6.1 Initialize reputation ledger...")
    ledger = ReputationLedger(byzantine_tolerance=0.33)
    ledger.add_validator("validator_1")
    ledger.add_validator("validator_2")
    logger.info("       ✓ Reputation ledger initialized with 2 validators")

    # Register agents
    logger.info("\n   6.2 Register agents and track progression...")
    agents = []
    for i in range(5):
        agent_id = f"agent_{i+1}"
        ledger.register_agent(agent_id)
        agents.append(agent_id)
    logger.info(f"       ✓ {len(agents)} agents registered")

    # Simulate reputation progression
    logger.info("\n   6.3 Simulate reputation progression through tiers...")

    # Agent 1: Quick progression to EMERITUS
    for action_type, amount in [
        ("task_completion", 30),
        ("incident_resolution", 20),
        ("reliable_info", 10),
        ("task_completion", 40),
        ("task_completion", 50),
        ("squad_leadership", 30),
        ("task_completion", 100),
        ("strategy_proposal", 50),
    ]:
        ledger.record_reputation_change(
            agents[0], amount, action_type, "validator_1"
        )

    # Agent 2: Progress to VETERAN
    for _ in range(10):
        ledger.record_reputation_change(
            agents[1], 60, "task_completion", "validator_2"
        )

    # Agent 3: Reach TRUSTED_PEER
    for _ in range(5):
        ledger.record_reputation_change(
            agents[2], 25, "task_completion", "validator_1"
        )

    # Agent 4: Some failures (negative reputation)
    for i in range(3):
        ledger.record_reputation_change(
            agents[3], 20, "task_completion", "validator_1"
        )
        ledger.record_reputation_change(
            agents[3], -10, "task_completion", "validator_2"
        )

    # Show progression
    logger.info("\n       Agent Progression:")
    for agent_id in agents:
        rep = ledger.get_reputation(agent_id)
        logger.info(
            f"       - {agent_id}: {rep.total_reputation:.0f} reputation "
            f"→ {rep.privilege_tier.name} (Tier {rep.privilege_tier.value})"
        )
        if rep.privilege_tier.value > 0:
            capabilities = []
            if rep.has_capability("join_task"):
                capabilities.append("join_task")
            if rep.has_capability("lead_squad"):
                capabilities.append("lead_squad")
            if rep.has_capability("propose_strategy"):
                capabilities.append("propose_strategy")
            logger.info(f"         Capabilities: {', '.join(capabilities)}")

    # Show statistics
    logger.info("\n   6.4 Reputation ledger statistics...")
    stats = ledger.get_statistics()
    logger.info(f"       Total agents: {stats['total_agents']}")
    logger.info(f"       Total records: {stats['total_records']}")
    logger.info(f"       Avg reputation: {stats['avg_reputation']:.1f}")
    logger.info("       Tier distribution:")
    for tier, count in stats["tier_distribution"].items():
        logger.info(f"         - {tier}: {count} agents")

    # Verify ledger integrity
    logger.info("\n   6.5 Verify ledger integrity...")
    if ledger.verify_ledger_integrity():
        logger.info(
            "       ✓ Ledger integrity verified - all signatures valid"
        )

    return ledger


def demonstrate_contract_net_protocol():
    """Demonstrate contract net protocol and collaborative tasking"""
    logger.info(
        "\n\n7. Demonstrating Contract Net Protocol & Collaborative Tasking..."
    )
    logger.info("   " + "-" * 66)

    # Initialize with reputation ledger
    logger.info("\n   7.1 Initialize contract net protocol...")
    ledger = ReputationLedger()
    contract_net = ContractNetProtocol(
        default_bidding_time=5.0, reputation_ledger=ledger  # Shorter for demo
    )

    # Register agents with different capabilities
    agents = {}
    all_skills = [
        {"security"},
        {"networking"},
        {"database"},
        {"security", "networking"},
        {"security", "database"},
        {"networking", "database"},
    ]
    for i in range(6):
        agent_id = f"agent_{i+1}"
        ledger.register_agent(agent_id)
        # Give them different reputation levels
        ledger.record_reputation_change(
            agent_id, 50 * (i + 1), "task_completion"
        )
        agents[agent_id] = {"skills": all_skills[i], "energy": 50 + i * 10}

    logger.info(f"       ✓ Contract net initialized with {len(agents)} agents")

    # Propose tasks
    logger.info("\n   7.2 Propose tasks to the marketplace...")

    tasks = []

    # Task 1: Security patch (high-value)
    task1 = contract_net.propose_task(
        description="Apply critical security patch to database servers",
        requirements=TaskRequirements(
            required_skills={"security", "database"},
            min_reputation=50,
            min_energy=30,
        ),
        reward=100.0,
        bidding_time=3.0,
    )
    tasks.append(task1.task_id)
    logger.info(
        f"       ✓ Task proposed: {task1.description} (reward={task1.reward})"
    )

    # Task 2: Network optimization (medium-value)
    task2 = contract_net.propose_task(
        description="Optimize network routing for better latency",
        requirements=TaskRequirements(
            required_skills={"networking"}, min_reputation=30, min_energy=20
        ),
        reward=60.0,
        bidding_time=3.0,
    )
    tasks.append(task2.task_id)
    logger.info(
        f"       ✓ Task proposed: {task2.description} (reward={task2.reward})"
    )

    # Agents bid on tasks
    logger.info("\n   7.3 Agents bid on tasks...")

    bids_submitted = 0
    for agent_id, agent_data in agents.items():
        rep = ledger.get_reputation(agent_id)

        # Bid on task 1 if qualified
        task = contract_net.get_task(task1.task_id)
        if task.requirements.is_qualified(
            agent_data["skills"],
            rep.total_reputation,
            rep.privilege_tier.value,
            None,
            agent_data["energy"],
        ):
            bid = contract_net.submit_bid(
                task_id=task1.task_id,
                bidder_id=agent_id,
                bid_amount=80.0 + random.uniform(-10, 10),
                estimated_time=300 + random.uniform(-50, 50),
                confidence=0.7 + random.uniform(0, 0.2),
                bidder_skills=agent_data["skills"],
                bidder_reputation=rep.total_reputation,
                bidder_energy=agent_data["energy"],
            )
            if bid:
                bids_submitted += 1
                logger.info(
                    f"       - {agent_id} bid on Task 1 (score={bid.bid_score:.2f})"
                )

        # Bid on task 2 if qualified
        task = contract_net.get_task(task2.task_id)
        if task.requirements.is_qualified(
            agent_data["skills"],
            rep.total_reputation,
            rep.privilege_tier.value,
            None,
            agent_data["energy"],
        ):
            bid = contract_net.submit_bid(
                task_id=task2.task_id,
                bidder_id=agent_id,
                bid_amount=50.0 + random.uniform(-10, 10),
                estimated_time=200 + random.uniform(-50, 50),
                confidence=0.8 + random.uniform(0, 0.1),
                bidder_skills=agent_data["skills"],
                bidder_reputation=rep.total_reputation,
                bidder_energy=agent_data["energy"],
            )
            if bid:
                bids_submitted += 1
                logger.info(
                    f"       - {agent_id} bid on Task 2 (score={bid.bid_score:.2f})"
                )

    logger.info(f"       ✓ {bids_submitted} bids submitted")

    # Award contracts
    logger.info("\n   7.4 Award contracts to best bidders...")

    for task_id in tasks:
        winner = contract_net.award_contract(task_id, auto_select_best=True)
        if winner:
            logger.info(f"       ✓ Task {task_id[-8:]} awarded to {winner}")
            contract_net.start_task(task_id, winner)

    # Simulate task execution
    logger.info("\n   7.5 Simulate task execution...")

    for task_id in tasks:
        task = contract_net.get_task(task_id)
        if task.awarded_to:
            # Report progress
            contract_net.report_progress(
                task_id, task.awarded_to, 0.5, "Task in progress"
            )
            time.sleep(0.1)
            contract_net.report_progress(
                task_id, task.awarded_to, 0.9, "Nearly complete"
            )
            time.sleep(0.1)
            # Complete task
            contract_net.complete_task(task_id, task.awarded_to)
            logger.info(
                f"       ✓ Task {task_id[-8:]} completed by {task.awarded_to}"
            )

    # Show statistics
    logger.info("\n   7.6 Contract net statistics...")
    stats = contract_net.get_statistics()
    logger.info(f"       Tasks announced: {stats['total_tasks_announced']}")
    logger.info(f"       Tasks completed: {stats['total_tasks_completed']}")
    logger.info(f"       Completion rate: {stats['completion_rate']:.1%}")
    logger.info(f"       Avg bids per task: {stats['avg_bids_per_task']:.1f}")


def demonstrate_council_of_professors():
    """Demonstrate Council of Professors governance"""
    logger.info("\n\n8. Demonstrating Council of Professors Governance...")
    logger.info("   " + "-" * 66)

    # Initialize Council
    logger.info("\n   8.1 Assemble Council of Professors...")
    council = CouncilOfProfessors()
    logger.info("       ✓ Council assembled with 3 professors:")
    logger.info("         - Systemic Pathologist (failure analysis)")
    logger.info("         - Strategic Immunologist (vulnerability monitoring)")
    logger.info("         - Evolutionary Biologist (genetic health)")

    # Systemic Pathologist: Analyze failures
    logger.info("\n   8.2 Systemic Pathologist analyzes failures...")

    failures = [
        (
            "f1",
            "task_failure",
            {"insufficient_resources": True, "critical_service": True},
        ),
        (
            "f2",
            "security_breach",
            {"weak_authentication": True, "widespread_impact": True},
        ),
        ("f3", "task_failure", {"insufficient_resources": True}),  # Recurring
    ]

    for failure_id, failure_type, context in failures:
        analysis = council.pathologist.analyze_failure(
            failure_id=failure_id,
            agent_id=f"agent_{random.randint(1,5)}",
            failure_type=failure_type,
            context=context,
        )
        logger.info(
            f"       ✓ Failure {failure_id} analyzed: {analysis.root_cause}"
        )

        # Publish lesson learned
        lesson = council.pathologist.publish_lesson_learned(failure_id)
        if lesson:
            logger.info(
                f"         Lesson published: {lesson.what_not_to_do[:50]}..."
            )

    logger.info(f"       Total failures analyzed: {len(failures)}")
    logger.info(
        f"       Lessons published: {council.pathologist.total_lessons_published}"
    )

    # Strategic Immunologist: Monitor vulnerabilities
    logger.info("\n   8.3 Strategic Immunologist monitors vulnerabilities...")

    # Report same vulnerability multiple times to trigger antibody
    for i in range(6):
        vuln = council.immunologist.monitor_vulnerability(
            vuln_id="vuln_auth_001",
            vulnerability_type="authentication_weakness",
            description="Weak password policy allowing brute force",
            severity=0.8,
            affected_components=["auth_service", "api_gateway"],
        )

    logger.info(f"       ✓ Vulnerability reported {vuln.frequency} times")

    if vuln.vuln_id in council.immunologist.chronic_vulnerabilities:
        logger.info(
            "       ⚠ Chronic vulnerability identified - antibody injected"
        )
        # Check if antibody was created
        antibodies = [
            a
            for a in council.immunologist.antibodies.values()
            if a.target_vulnerability == vuln.vuln_id
        ]
        if antibodies:
            antibody = antibodies[0]
            logger.info(
                f"       ✓ Antibody type: {antibody.antibody_type.value}"
            )
            # Deploy antibody
            council.immunologist.deploy_antibody(antibody.antibody_id)
            logger.info(f"       ✓ Antibody deployed")

    # Evolutionary Biologist: Assess genetic health
    logger.info("\n   8.4 Evolutionary Biologist assesses genetic health...")

    health = council.biologist.assess_genetic_health(
        population_size=50,
        genetic_diversity=0.25,  # Low diversity
        avg_fitness=0.6,
        stagnation_level=0.8,  # High stagnation
        mutation_rate=0.05,
    )

    logger.info(f"       ✓ Genetic health assessed:")
    logger.info(
        f"         - Diversity: {health.genetic_diversity:.2f} (threshold: 0.30)"
    )
    logger.info(
        f"         - Stagnation: {health.stagnation_level:.2f} (threshold: 0.70)"
    )
    logger.info(f"         - Avg fitness: {health.avg_fitness:.2f}")
    logger.info(f"       Recommendation: {health.recommendation}")

    # Show council statistics
    logger.info("\n   8.5 Council of Professors statistics...")
    stats = council.get_statistics()
    logger.info(
        f"       Pathologist: {stats['pathologist']['total_failures_analyzed']} failures analyzed"
    )
    logger.info(
        f"       Immunologist: {stats['immunologist']['total_antibodies_injected']} antibodies injected"
    )
    logger.info(
        f"       Biologist: {stats['biologist']['total_interventions']} interventions performed"
    )


def demonstrate_constitutional_framework():
    """Demonstrate constitutional framework enforcement"""
    logger.info("\n\n9. Demonstrating Constitutional Framework...")
    logger.info("   " + "-" * 66)

    # Initialize framework
    logger.info("\n   9.1 Establish constitutional framework...")
    constitution = ConstitutionalFramework(
        critical_services={"web_service", "database", "auth_service"},
        system_components={"main_server", "backup_server", "load_balancer"},
    )
    logger.info("       ✓ Constitutional framework established")
    logger.info("         - 3 immutable laws defined")
    logger.info("         - 3 critical services registered")
    logger.info("         - 3 system components registered")

    # Test various actions
    logger.info(
        "\n   9.2 Evaluate actions against constitutional framework..."
    )

    test_cases = [
        {
            "name": "Safe configuration change",
            "context": {
                "affects_services": False,
                "has_backup": True,
                "resource_efficiency": 0.8,
            },
            "expected": "APPROVED",
        },
        {
            "name": "Destructive operation without backup",
            "context": {
                "is_destructive": True,
                "has_backup": False,
                "compromises_system_integrity": True,
            },
            "expected": "DENIED",
        },
        {
            "name": "Critical service restart (scheduled)",
            "context": {
                "affects_services": True,
                "affected_services": ["web_service"],
                "causes_downtime": True,
                "scheduled_maintenance": True,
            },
            "expected": "CAUTION",
        },
        {
            "name": "Unscheduled critical service downtime",
            "context": {
                "affects_services": True,
                "affected_services": ["database", "auth_service"],
                "causes_downtime": True,
                "scheduled_maintenance": False,
            },
            "expected": "DENIED",
        },
    ]

    for i, test in enumerate(test_cases):
        evaluation = constitution.evaluate_action(
            action_id=f"action_{i+1}",
            action_description=test["name"],
            action_context=test["context"],
        )

        status = "✓ COMPLIANT" if evaluation.is_compliant else "✗ VIOLATION"
        logger.info(f"\n       {status}: {test['name']}")
        logger.info(
            f"         Compliance score: {evaluation.compliance_score:.2f}"
        )
        if evaluation.violations:
            for law, reason in evaluation.violations:
                logger.info(f"         Violates {law.name}: {reason}")
        if evaluation.warnings:
            logger.info(f"         Warnings: {', '.join(evaluation.warnings)}")
        logger.info(f"         Recommendation: {evaluation.recommendation}")

    # Show statistics
    logger.info("\n   9.3 Constitutional framework statistics...")
    stats = constitution.get_statistics()
    logger.info(
        f"       Actions evaluated: {stats['total_actions_evaluated']}"
    )
    logger.info(f"       Compliance rate: {stats['compliance_rate']:.1%}")
    logger.info(
        f"       Violations prevented: {stats['total_violations_prevented']}"
    )
    logger.info("       Violations by law:")
    for law, count in stats["violations_by_law"].items():
        if count > 0:
            logger.info(f"         - {law}: {count}")


def demonstrate_real_world_bridge():
    """Demonstrate real-world integration bridge"""
    logger.info("\n\n10. Demonstrating Real-World Integration Bridge...")
    logger.info("   " + "-" * 66)

    # Initialize bridge
    logger.info("\n   10.1 Initialize real-world integration bridge...")
    constitution = ConstitutionalFramework()
    bridge = RealWorldBridge(
        require_approval_for_high_risk=True,
        dry_run_mode=True,  # Safe for demo
        constitutional_framework=constitution,
    )
    logger.info("       ✓ Bridge initialized (DRY RUN MODE)")

    # Register adapters
    logger.info("\n   10.2 Register integration adapters...")

    adapters = [
        IntegrationSystem.ANSIBLE,
        IntegrationSystem.KUBERNETES,
        IntegrationSystem.AWS,
        IntegrationSystem.DOCKER,
    ]

    for system in adapters:
        bridge.register_adapter(
            system=system,
            config={"endpoint": f"https://{system.value}.example.com"},
            dry_run_mode=True,
        )
        logger.info(f"       ✓ {system.value} adapter registered")

    # Translate agent decisions to commands
    logger.info(
        "\n   10.3 Translate agent decisions to real-world commands..."
    )

    decisions = [
        (
            "patch_server",
            {"server_id": "web-01", "patch_name": "security-2024-001"},
        ),
        (
            "scale_service",
            {
                "service_name": "api-service",
                "replicas": 5,
                "original_replicas": 2,
            },
        ),
        ("restart_service", {"service_name": "worker-service", "timeout": 30}),
    ]

    all_commands = []
    for decision_type, decision_data in decisions:
        decision_id = f"decision_{decision_type}_{int(time.time()*1000)}"
        commands = bridge.translate_decision(
            decision_id=decision_id,
            decision_type=decision_type,
            decision_data=decision_data,
            agent_id="agent_1",
        )
        all_commands.extend(commands)
        for cmd in commands:
            logger.info(
                f"       ✓ Command created: {cmd.system.value}:{cmd.operation}"
            )
            logger.info(f"         Risk level: {cmd.risk_level.value}")
            logger.info(f"         Requires approval: {cmd.requires_approval}")

    # Execute commands
    logger.info("\n   10.4 Execute commands...")

    for cmd in all_commands:
        if cmd.requires_approval:
            # Approve high-risk commands
            bridge.approve_command(cmd.command_id, "human_operator")
            logger.info(f"       ✓ Command {cmd.command_id[-8:]} approved")

        # Execute
        success = bridge.execute_command(cmd.command_id)
        if success:
            logger.info(
                f"       ✓ Command {cmd.command_id[-8:]} executed (DRY RUN)"
            )

    # Show statistics
    logger.info("\n   10.5 Real-world bridge statistics...")
    stats = bridge.get_statistics()
    logger.info(f"       Commands created: {stats['total_commands_created']}")
    logger.info(
        f"       Commands executed: {stats['total_commands_executed']}"
    )
    logger.info(f"       Success rate: {stats['success_rate']:.1%}")
    logger.info(f"       Adapters registered: {len(stats['adapters'])}")


def main():
    """Run Phase 4 demonstration"""
    try:
        # Demonstrate existing Phase 4 capabilities
        demonstrate_autonomy()
        demonstrate_adaptive_defense()
        demonstrate_swarm_defense()
        demonstrate_evolving_adversaries()
        demonstrate_integrated_system()

        # Demonstrate NEW Phase 4 Polity & Governance capabilities
        demonstrate_reputation_and_meritocracy()
        demonstrate_contract_net_protocol()
        demonstrate_council_of_professors()
        demonstrate_constitutional_framework()
        demonstrate_real_world_bridge()

        logger.info("\n" + "=" * 70)
        logger.info(
            "PHASE 4: THE SENTIENT POLITY & GOVERNANCE LAYER - COMPLETE"
        )
        logger.info("=" * 70)
        logger.info("\nKey Capabilities Demonstrated:")
        logger.info("\n  EXISTING PHASE 4 FEATURES:")
        logger.info("  ✓ Autonomy with self-repair and ethical escalation")
        logger.info("  ✓ Adaptive self-healing cyber defenses")
        logger.info("  ✓ Biological-inspired swarm coordination")
        logger.info("  ✓ Evolving adversary simulation")
        logger.info("  ✓ Fully integrated autonomous defense system")
        logger.info("\n  NEW PHASE 4 POLITY & GOVERNANCE:")
        logger.info(
            "  ✓ Reputation Ledger & Meritocratic Progression (4 Tiers)"
        )
        logger.info("  ✓ Contract Net Protocol & Market-Based Tasking")
        logger.info("  ✓ Council of Professors Governance (3 Professors)")
        logger.info("  ✓ Constitutional Framework (3 Immutable Laws)")
        logger.info("  ✓ Real-World Integration Bridge (Multi-system)")
        logger.info(
            "\n🎉 Phase 4: The Sentient Polity is fully operational! 🎉"
        )

    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
