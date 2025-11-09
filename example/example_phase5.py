"""
Phase 5 Example: Advanced Features, Openness, and Large-Scale Simulation

Demonstrates:
1. Openness & Complexity (AL Principle #5) - Adaptation to unpredictable environments
2. Human-in-the-Loop Ethics, Privacy, and Compliance
3. Large-Scale Simulation & Field Deployment

Usage:
    python3 example/example_phase5.py
"""

import logging
import sys
import os
import time
import random

# Add parent directory to path to allow imports from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('phase5_example')

# Import Phase 5 components
from core.openness_engine import (
    OpennessEngine, EnvironmentType, AdaptationStrategy, OrganizationPattern
)
from core.human_loop import (
    HumanLoopSystem, ApprovalStatus, ActionSensitivity, ComplianceFramework
)
from core.large_scale_simulation import (
    LargeScaleSimulation, SimulationMode, NodeState
)


def demonstrate_openness():
    """Demonstrate openness and environmental adaptation"""
    logger.info("="*70)
    logger.info("PHASE 5 DEMONSTRATION: Advanced Features, Openness, and Large-Scale")
    logger.info("="*70)
    
    logger.info("\n1. Demonstrating Openness & Environmental Adaptation...")
    logger.info("   " + "-"*66)
    
    # Create openness engine
    logger.info("\n   1.1 Initialize openness engine...")
    openness = OpennessEngine(
        node_id=1,
        adaptation_threshold=0.6,
        exploration_rate=0.2
    )
    logger.info("       ✓ Openness engine initialized")
    
    # Simulate different environmental conditions
    logger.info("\n   1.2 Sense and adapt to various environments...")
    
    environments = [
        ("Stable Production", 0.2, 0.3, 0.1, 0.2, 0.8),
        ("Dynamic Market", 0.6, 0.5, 0.3, 0.3, 0.7),
        ("Chaotic Crisis", 0.9, 0.8, 0.4, 0.7, 0.5),
        ("Novel Situation", 0.5, 0.6, 0.9, 0.4, 0.6),
        ("Adversarial Attack", 0.7, 0.7, 0.5, 0.9, 0.4)
    ]
    
    for name, volatility, complexity, novelty, threat, resources in environments:
        logger.info(f"\n       Environment: {name}")
        
        # Sense environment
        env = openness.sense_environment(
            volatility=volatility,
            complexity=complexity,
            novelty=novelty,
            threat_level=threat,
            resource_availability=resources
        )
        
        logger.info(f"         Type: {env.environment_type.value}")
        logger.info(f"         Stability score: {env.get_stability_score():.2f}")
        
        # Check if adaptation needed
        should_adapt = openness.should_adapt(env)
        logger.info(f"         Adaptation needed: {should_adapt}")
        
        # Adapt if necessary
        if should_adapt:
            result = openness.adapt_to_environment(env)
            logger.info(f"         Strategy used: {result.strategy_used.value}")
            logger.info(f"         Success: {result.success}")
            logger.info(f"         Changes made: {len(result.changes_made)}")
            if result.new_organization:
                logger.info(f"         New organization: {result.new_organization.value}")
    
    # Show openness metrics
    logger.info("\n   1.3 Openness metrics...")
    metrics = openness.get_openness_metrics()
    logger.info(f"       Environments encountered: {metrics['environments_encountered']}")
    logger.info(f"       Adaptations performed: {metrics['adaptations_performed']}")
    logger.info(f"       Success rate: {metrics['adaptation_success_rate']:.2%}")
    logger.info(f"       Reorganizations: {metrics['reorganizations_performed']}")
    logger.info(f"       Patterns learned: {metrics['patterns_learned']}")
    
    # Show learned patterns
    if metrics['patterns_learned'] > 0:
        logger.info("\n       Learned patterns:")
        patterns = openness.get_learned_patterns()
        for pattern in patterns[:3]:  # Show first 3
            logger.info(f"         - {pattern['environment_type']}: "
                       f"{pattern['encounter_count']} encounters")
            if pattern['successful_strategies']:
                logger.info(f"           Successful: {pattern['successful_strategies']}")


def demonstrate_human_loop():
    """Demonstrate human-in-the-loop oversight and compliance"""
    logger.info("\n\n2. Demonstrating Human-in-the-Loop & Compliance...")
    logger.info("   " + "-"*66)
    
    # Create human loop system
    logger.info("\n   2.1 Initialize human-in-the-loop system...")
    human_loop = HumanLoopSystem(
        node_id=1,
        auto_approve_low_sensitivity=True,
        approval_timeout=300.0,
        compliance_frameworks=[
            ComplianceFramework.GDPR,
            ComplianceFramework.SOC2
        ]
    )
    logger.info("       ✓ Human-in-the-loop system initialized")
    logger.info(f"       Compliance frameworks: GDPR, SOC2")
    
    # Test various approval scenarios
    logger.info("\n   2.2 Test approval workflows...")
    
    # Low sensitivity - auto-approved
    logger.info("\n       Scenario 1: Low sensitivity action (auto-approved)")
    request1 = human_loop.request_approval(
        action="monitor_cpu_usage",
        description="Monitor CPU usage metrics",
        sensitivity=ActionSensitivity.LOW,
        context={'system_metric': True},
        risk_level=0.1
    )
    logger.info(f"         Status: {request1.status.value}")
    
    # High sensitivity - requires approval
    logger.info("\n       Scenario 2: High sensitivity action (requires approval)")
    request2 = human_loop.request_approval(
        action="access_user_data",
        description="Access user personal data for analysis",
        sensitivity=ActionSensitivity.HIGH,
        context={
            'accesses_pii': True,
            'has_consent': True,
            'has_purpose': True
        },
        risk_level=0.8
    )
    logger.info(f"         Status: {request2.status.value}")
    logger.info(f"         Request ID: {request2.request_id}")
    
    # Simulate human approval
    logger.info("\n       Simulating human approval...")
    human_loop.approve_request(request2.request_id, approver="admin@example.com")
    status = human_loop.check_approval_status(request2.request_id)
    logger.info(f"         New status: {status.value}")
    
    # Test privacy boundaries
    logger.info("\n   2.3 Test privacy boundaries...")
    human_loop.set_privacy_boundary('access_pii', True)
    human_loop.set_privacy_boundary('export_data', False)
    
    can_access = human_loop.check_privacy_boundary('access_pii')
    can_export = human_loop.check_privacy_boundary('export_data')
    logger.info(f"       Can access PII: {can_access}")
    logger.info(f"       Can export data: {can_export}")
    
    # Test override rules
    logger.info("\n   2.4 Test override rules...")
    human_loop.set_override_rule('emergency_shutdown', True)
    human_loop.set_override_rule('delete_database', False)
    
    request3 = human_loop.request_approval(
        action="emergency_shutdown",
        description="Emergency system shutdown",
        sensitivity=ActionSensitivity.CRITICAL,
        context={},
        risk_level=0.9
    )
    logger.info(f"       Emergency shutdown: {request3.status.value}")
    
    # Check compliance
    logger.info("\n   2.5 Check compliance...")
    compliance = human_loop.check_compliance(
        action="process_user_data",
        context={
            'accesses_pii': True,
            'has_consent': True,
            'has_purpose': True,
            'is_audited': True,
            'is_secure': True
        }
    )
    logger.info("       Compliance results:")
    for framework, compliant in compliance.items():
        logger.info(f"         {framework.value}: {'✓' if compliant else '✗'}")
    
    # Create decision explanation
    logger.info("\n   2.6 Create explainable decision...")
    explanation = human_loop.explain_decision(
        decision_id="decision_001",
        action="block_suspicious_ip",
        final_decision="block",
        reasoning_steps=[
            "Detected unusual traffic pattern",
            "IP has history of malicious activity",
            "Traffic matches known attack signature",
            "Decision: Block IP address"
        ],
        factors_considered={
            'traffic_volume': 0.9,
            'reputation_score': 0.8,
            'pattern_match': 0.95
        },
        alternatives_evaluated=['monitor', 'rate_limit', 'block'],
        confidence=0.92
    )
    logger.info(f"       Decision explained: {explanation.decision_id}")
    logger.info(f"       Confidence: {explanation.confidence:.2%}")
    logger.info(f"       Reasoning steps: {len(explanation.reasoning_steps)}")
    
    # Show human loop metrics
    logger.info("\n   2.7 Human-in-the-loop metrics...")
    metrics = human_loop.get_human_loop_metrics()
    logger.info(f"       Approval requests: {metrics['approval_requests_made']}")
    logger.info(f"       Approvals granted: {metrics['approvals_granted']}")
    logger.info(f"       Approvals rejected: {metrics['approvals_rejected']}")
    logger.info(f"       Approval rate: {metrics['approval_rate']:.2%}")
    logger.info(f"       Audit entries: {metrics['audit_entries']}")
    logger.info(f"       Explanations recorded: {metrics['explanations_recorded']}")


def demonstrate_large_scale_simulation():
    """Demonstrate large-scale simulation testbed"""
    logger.info("\n\n3. Demonstrating Large-Scale Simulation...")
    logger.info("   " + "-"*66)
    
    # Test with different scales
    scales = [
        (100, "Small deployment"),
        (500, "Medium deployment"),
        (1000, "Large deployment")
    ]
    
    for node_count, description in scales:
        logger.info(f"\n   3.{scales.index((node_count, description)) + 1} {description} ({node_count} nodes)...")
        
        # Create simulation
        simulation = LargeScaleSimulation(
            node_count=node_count,
            mode=SimulationMode.PRODUCTION_LIKE,
            area_size=(100.0, 100.0),
            enable_feedback=True
        )
        
        # Initialize nodes
        simulation.initialize_nodes()
        
        # Run simulation
        logger.info(f"       Running simulation...")
        simulation.start_simulation()
        
        # Simulate for a few steps
        for _ in range(5):
            simulation.simulate_step(delta_time=1.0)
        
        simulation.stop_simulation()
        
        # Get metrics
        metrics = simulation.get_current_metrics()
        logger.info(f"       Total nodes: {metrics.total_nodes}")
        logger.info(f"       Average health: {metrics.avg_health:.2f}")
        logger.info(f"       Average load: {metrics.avg_load:.2f}")
        logger.info(f"       Events generated: {metrics.events_generated}")
        logger.info(f"       Node states:")
        for state, count in metrics.nodes_by_state.items():
            logger.info(f"         {state.value}: {count}")
    
    # Demonstrate adversarial simulation
    logger.info("\n   3.4 Adversarial simulation (testing defense resilience)...")
    adv_simulation = LargeScaleSimulation(
        node_count=200,
        mode=SimulationMode.ADVERSARIAL,
        enable_feedback=True
    )
    
    adv_simulation.initialize_nodes()
    adv_simulation.start_simulation()
    
    logger.info("       Running adversarial simulation...")
    for step in range(10):
        adv_simulation.simulate_step(delta_time=1.0)
    
    adv_simulation.stop_simulation()
    
    # Show adversarial metrics
    adv_metrics = adv_simulation.get_current_metrics()
    logger.info(f"\n       Adversarial simulation results:")
    logger.info(f"       Attacks detected: {adv_metrics.total_attacks_detected}")
    logger.info(f"       Attacks blocked: {adv_metrics.total_attacks_blocked}")
    
    if adv_metrics.total_attacks_detected > 0:
        block_rate = (adv_metrics.total_attacks_blocked / 
                     adv_metrics.total_attacks_detected)
        logger.info(f"       Block rate: {block_rate:.2%}")
    
    logger.info(f"       Failures: {adv_metrics.total_failures}")
    logger.info(f"       Recoveries: {adv_metrics.total_recoveries}")
    logger.info(f"       Average health: {adv_metrics.avg_health:.2f}")
    
    # Show health distribution
    health_dist = adv_simulation.get_node_health_distribution()
    logger.info(f"\n       Health distribution:")
    for level, count in health_dist.items():
        percentage = (count / adv_metrics.total_nodes) * 100
        logger.info(f"         {level}: {count} ({percentage:.1f}%)")
    
    # Show continuous improvement
    feedback = adv_simulation.get_feedback_history()
    if feedback:
        logger.info(f"\n       Continuous improvement:")
        logger.info(f"       Iterations: {len(feedback)}")
        if len(feedback) >= 2:
            first = feedback[0]
            last = feedback[-1]
            health_improvement = last['avg_health'] - first['avg_health']
            logger.info(f"       Health improvement: {health_improvement:+.2f}")


def demonstrate_integrated_phase5():
    """Demonstrate all Phase 5 systems working together"""
    logger.info("\n\n4. Integrated Phase 5 Demonstration...")
    logger.info("   " + "-"*66)
    
    logger.info("\n   4.1 Initialize integrated Phase 5 system...")
    
    # Create all components
    openness = OpennessEngine(node_id=1, adaptation_threshold=0.6)
    human_loop = HumanLoopSystem(
        node_id=1,
        compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOC2]
    )
    simulation = LargeScaleSimulation(
        node_count=100,
        mode=SimulationMode.FIELD_DEPLOYMENT,
        enable_feedback=True
    )
    
    logger.info("       ✓ All Phase 5 systems initialized")
    
    # Integrated scenario
    logger.info("\n   4.2 Scenario: Field deployment with environmental adaptation...")
    
    # Initialize simulation
    simulation.initialize_nodes()
    simulation.start_simulation()
    
    # Sense environment from simulation state
    sim_metrics = simulation.get_current_metrics()
    volatility = 1.0 - sim_metrics.avg_health
    complexity = sim_metrics.avg_load
    novelty = 0.3
    threat = 0.2
    resources = sim_metrics.avg_health
    
    env = openness.sense_environment(
        volatility=volatility,
        complexity=complexity,
        novelty=novelty,
        threat_level=threat,
        resource_availability=resources
    )
    
    logger.info(f"       Environment detected: {env.environment_type.value}")
    
    # Check if adaptation needed
    if openness.should_adapt(env):
        logger.info("       Adaptation required...")
        
        # Request approval for adaptation
        request = human_loop.request_approval(
            action="system_adaptation",
            description=f"Adapt to {env.environment_type.value} environment",
            sensitivity=ActionSensitivity.MEDIUM,
            context={
                'environment_type': env.environment_type.value,
                'volatility': env.volatility,
                'is_audited': True,
                'is_secure': True
            },
            risk_level=0.5
        )
        
        logger.info(f"       Approval status: {request.status.value}")
        
        if request.status in [ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED]:
            # Perform adaptation
            result = openness.adapt_to_environment(env)
            logger.info(f"       Adaptation performed: {result.strategy_used.value}")
            logger.info(f"       Success: {result.success}")
            
            # Create explanation
            explanation = human_loop.explain_decision(
                decision_id="adapt_001",
                action="system_adaptation",
                final_decision=f"Applied {result.strategy_used.value}",
                reasoning_steps=[
                    f"Detected {env.environment_type.value} environment",
                    f"Instability score: {1.0 - env.get_stability_score():.2f}",
                    f"Selected {result.strategy_used.value} strategy",
                    f"Applied {len(result.changes_made)} changes"
                ],
                factors_considered={
                    'volatility': env.volatility,
                    'complexity': env.complexity,
                    'novelty': env.novelty
                },
                alternatives_evaluated=[s.value for s in AdaptationStrategy],
                confidence=result.confidence
            )
            logger.info(f"       Decision explained and audited")
    
    # Run simulation with adaptations
    logger.info("\n       Running simulation...")
    for _ in range(5):
        simulation.simulate_step(delta_time=1.0)
    
    simulation.stop_simulation()
    
    # Show final results
    logger.info("\n   4.3 Final integrated metrics...")
    
    # Openness metrics
    open_metrics = openness.get_openness_metrics()
    logger.info(f"       Openness:")
    logger.info(f"         Adaptations: {open_metrics['adaptations_performed']}")
    logger.info(f"         Success rate: {open_metrics['adaptation_success_rate']:.2%}")
    
    # Human loop metrics
    hl_metrics = human_loop.get_human_loop_metrics()
    logger.info(f"       Human-in-the-loop:")
    logger.info(f"         Approval requests: {hl_metrics['approval_requests_made']}")
    logger.info(f"         Audit entries: {hl_metrics['audit_entries']}")
    
    # Simulation metrics
    sim_metrics = simulation.get_current_metrics()
    logger.info(f"       Simulation:")
    logger.info(f"         Nodes: {sim_metrics.total_nodes}")
    logger.info(f"         Average health: {sim_metrics.avg_health:.2f}")
    logger.info(f"         Events: {sim_metrics.events_generated}")


def main():
    """Main demonstration function"""
    try:
        # Run all demonstrations
        demonstrate_openness()
        demonstrate_human_loop()
        demonstrate_large_scale_simulation()
        demonstrate_integrated_phase5()
        
        logger.info("\n" + "="*70)
        logger.info("PHASE 5 DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info("\nPhase 5 Implementation Summary:")
        logger.info("✓ Openness & Complexity (AL Principle #5)")
        logger.info("  - Environmental adaptation and reorganization")
        logger.info("  - Learning from unpredictable conditions")
        logger.info("✓ Human-in-the-Loop Ethics, Privacy, and Compliance")
        logger.info("  - Approval workflows and override mechanisms")
        logger.info("  - Full audit trail and explainable decisions")
        logger.info("  - Multi-framework compliance (GDPR, SOC2, etc.)")
        logger.info("✓ Large-Scale Simulation & Field Deployment")
        logger.info("  - Testbed for hundreds/thousands of nodes")
        logger.info("  - Production-like and adversarial testing")
        logger.info("  - Continuous improvement from feedback")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
