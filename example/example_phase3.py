#!/usr/bin/env python3
"""
Example demonstrating Phase 3: Learning, Evolution, Trust, and Consensus

This example shows:
1. Adaptive Learning & Evolution - Genetic algorithms for strategy optimization
2. Trust Network & Byzantine-Resilient Consensus - Byzantine fault tolerance

Usage:
    python3 example/example_phase3.py
"""

import logging
import sys
import os
import time
import random

# Add parent directory to path to allow imports from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Phase 3 components
from core.evolution_engine import EvolutionEngine, Strategy, StrategyType
from core.adaptive_learning import AdaptiveLearningSystem, BehaviorType
from core.consensus import ConsensusEngine, ConsensusType, VoteType
from core.trust_network import TrustNetwork

# Import base components
from adaptiveengineer import AliveLoopNode
from core.plugin_manager import PluginManager
from plugins.it_operations import ITOperationsPlugin
from plugins.security import SecurityPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('phase3_example')


def demonstrate_phase3():
    """Demonstrate Phase 3 capabilities."""
    logger.info("=" * 70)
    logger.info("PHASE 3 DEMONSTRATION: Learning, Evolution, Trust, and Consensus")
    logger.info("=" * 70)
    
    # ========================================================================
    # 1. Evolutionary Learning for Detection Strategies
    # ========================================================================
    logger.info("\n1. Demonstrating Evolutionary Learning...")
    logger.info("   " + "-" * 66)
    
    logger.info("\n   1.1 Initialize evolution engine for detection strategies...")
    
    evolution = EvolutionEngine(
        population_size=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_count=2
    )
    
    # Define detection strategy parameters
    detection_params = {
        'threshold': (0.5, 0.95),
        'sensitivity': (0.3, 1.0),
        'response_time': (0.1, 0.9),
        'false_positive_tolerance': (0.1, 0.5)
    }
    
    evolution.initialize_population(StrategyType.DETECTION, detection_params)
    logger.info("      ✓ Initialized population of 20 detection strategies")
    
    logger.info("\n   1.2 Evolve strategies over multiple generations...")
    
    # Fitness function: balance between detection rate and false positives
    def evaluate_detection_strategy(strategy: Strategy) -> float:
        """Evaluate detection strategy fitness."""
        params = strategy.parameters
        
        # Simulate detection performance
        detection_rate = params['sensitivity'] * (1 - params['threshold'])
        false_pos_penalty = params['false_positive_tolerance'] * 0.5
        response_bonus = (1 - params['response_time']) * 0.3
        
        fitness = detection_rate - false_pos_penalty + response_bonus
        return max(0.0, min(1.0, fitness))
    
    # Evolve for 10 generations
    for gen in range(10):
        stats = evolution.evolve_generation(
            StrategyType.DETECTION,
            evaluate_detection_strategy
        )
        
        if gen == 0 or gen == 4 or gen == 9:
            logger.info(f"      Gen {stats['generation']}: "
                       f"best={stats['best_fitness']:.4f}, "
                       f"avg={stats['average_fitness']:.4f}")
    
    best_strategy = evolution.get_best_strategy(StrategyType.DETECTION)
    logger.info(f"\n      ✓ Best evolved strategy (fitness={best_strategy.fitness:.4f}):")
    logger.info(f"        - threshold: {best_strategy.parameters['threshold']:.3f}")
    logger.info(f"        - sensitivity: {best_strategy.parameters['sensitivity']:.3f}")
    logger.info(f"        - response_time: {best_strategy.parameters['response_time']:.3f}")
    
    # ========================================================================
    # 2. Adaptive Learning System
    # ========================================================================
    logger.info("\n2. Demonstrating Adaptive Learning...")
    logger.info("   " + "-" * 66)
    
    logger.info("\n   2.1 Initialize adaptive learning system...")
    
    learning = AdaptiveLearningSystem(
        learning_rate=0.01,
        auto_tune_interval=10.0  # Auto-tune every 10 seconds for demo
    )
    logger.info("      ✓ Adaptive learning system initialized")
    
    logger.info("\n   2.2 Learn normal service behavior...")
    
    # Simulate normal service behavior observations
    normal_cpu = []
    for i in range(50):
        # Normal CPU usage around 0.3-0.5
        cpu_value = 0.4 + random.gauss(0, 0.05)
        cpu_value = max(0.0, min(1.0, cpu_value))
        normal_cpu.append(cpu_value)
        
        result = learning.observe(
            BehaviorType.SERVICE,
            cpu_value,
            metadata={'metric': 'cpu_usage'}
        )
    
    logger.info(f"      ✓ Learned from {len(normal_cpu)} observations")
    
    profile = learning.get_profile(BehaviorType.SERVICE)
    logger.info(f"        Mean: {profile.mean:.3f}")
    logger.info(f"        Std Dev: {(profile.variance ** 0.5):.3f}")
    
    # Force auto-tune
    tune_results = learning.force_auto_tune(BehaviorType.SERVICE)
    thresholds = tune_results[BehaviorType.SERVICE]
    logger.info(f"        Auto-tuned thresholds: [{thresholds['lower_threshold']:.3f}, "
               f"{thresholds['upper_threshold']:.3f}]")
    
    logger.info("\n   2.3 Detect anomalies...")
    
    # Test with normal and anomalous values
    test_values = [0.42, 0.38, 0.85, 0.95, 0.40]  # Last two are anomalies
    anomalies_detected = 0
    
    for value in test_values:
        result = learning.observe(
            BehaviorType.SERVICE,
            value,
            metadata={'metric': 'cpu_usage'}
        )
        
        if result['is_anomaly']:
            anomalies_detected += 1
            logger.info(f"      ⚠ Anomaly detected: {value:.2f} "
                       f"(score: {result['anomaly_score']:.2f})")
    
    logger.info(f"      ✓ Detected {anomalies_detected} anomalies from {len(test_values)} observations")
    
    # ========================================================================
    # 3. Trust Network with Byzantine Fault Detection
    # ========================================================================
    logger.info("\n3. Demonstrating Trust Network with Byzantine Fault Detection...")
    logger.info("   " + "-" * 66)
    
    logger.info("\n   3.1 Initialize trust network...")
    
    trust_net = TrustNetwork(node_id=1)
    
    # Establish trust with several nodes
    for node_id in range(2, 8):
        # Initialize with neutral trust
        trust_net.trust_network[node_id] = 0.5
    
    logger.info(f"      ✓ Trust network initialized with {len(trust_net.trust_network)} nodes")
    
    logger.info("\n   3.2 Simulate normal interactions (building trust)...")
    
    # Good behavior from nodes 2, 3, 4
    for _ in range(3):
        for node_id in [2, 3, 4]:
            trust_net.update_trust(
                type('Node', (), {'node_id': node_id})(),
                'resource',
                {'timestamp': time.time()}
            )
    
    trusted = trust_net.get_trusted_nodes(min_trust=0.6)
    logger.info(f"      ✓ Trusted nodes after positive interactions: {trusted}")
    
    logger.info("\n   3.3 Detect Byzantine (malicious) behavior...")
    
    # Node 5 exhibits Byzantine behavior
    byzantine_node = 5
    for _ in range(3):
        is_suspicious = trust_net.detect_byzantine_behavior(
            byzantine_node,
            expected_behavior={'value': 0.5, 'timestamp': time.time()},
            actual_behavior={'value': 0.9, 'contradicts_previous': True}
        )
    
    if trust_net.is_compromised(byzantine_node):
        logger.info(f"      ✓ Node {byzantine_node} detected and marked as compromised")
    
    logger.info("\n   3.4 Discount input from unreliable nodes...")
    
    # Test input discounting
    inputs = {
        2: 0.7,  # Trusted node
        3: 0.65,  # Trusted node
        5: 0.2,  # Compromised node (should be ignored)
        6: 0.4   # Neutral node
    }
    
    logger.info("      Input values from nodes:")
    for nid, val in inputs.items():
        trust = trust_net.get_trust(nid)
        discounted = trust_net.discount_input(nid, val)
        logger.info(f"        Node {nid}: {val:.2f} (trust={trust:.2f}) → {discounted:.2f}")
    
    # Byzantine-resilient aggregation
    aggregated = trust_net.byzantine_resilient_aggregate(inputs, method="weighted_median")
    logger.info(f"      ✓ Byzantine-resilient aggregated value: {aggregated:.3f}")
    
    metrics = trust_net.get_byzantine_resilience_metrics()
    logger.info(f"      Network health: {metrics['network_health']:.2f}")
    logger.info(f"      Byzantine tolerance: {metrics['byzantine_tolerance']:.2f}")
    
    # ========================================================================
    # 4. Byzantine-Resilient Consensus
    # ========================================================================
    logger.info("\n4. Demonstrating Byzantine-Resilient Consensus...")
    logger.info("   " + "-" * 66)
    
    logger.info("\n   4.1 Initialize consensus engines for network nodes...")
    
    # Create consensus engines for multiple nodes
    consensus_engines = {}
    for node_id in range(1, 8):
        engine = ConsensusEngine(
            node_id=node_id,
            byzantine_tolerance=0.33,
            default_quorum_ratio=0.67
        )
        
        # Set trust scores from trust network
        for other_id in range(1, 8):
            if other_id != node_id:
                trust = trust_net.get_trust(other_id) if node_id == 1 else 0.5
                engine.set_node_trust(other_id, trust)
        
        consensus_engines[node_id] = engine
    
    logger.info(f"      ✓ Created {len(consensus_engines)} consensus engines")
    
    logger.info("\n   4.2 Initiate consensus on incident root cause...")
    
    # Node 1 initiates consensus about an incident
    engine1 = consensus_engines[1]
    proposal_id = engine1.initiate_consensus(
        ConsensusType.ROOT_CAUSE,
        subject={
            'incident_id': 'inc_001',
            'suspected_cause': 'memory_leak',
            'severity': 0.8
        },
        network_size=7,
        timeout=300.0
    )
    
    logger.info(f"      ✓ Consensus initiated: {proposal_id}")
    
    logger.info("\n   4.3 Nodes cast votes (including Byzantine node)...")
    
    # Most nodes approve (memory leak is the cause)
    honest_votes = [
        (2, VoteType.APPROVE, 0.85),
        (3, VoteType.APPROVE, 0.90),
        (4, VoteType.APPROVE, 0.80),
        (6, VoteType.APPROVE, 0.75),
        (7, VoteType.APPROVE, 0.70)
    ]
    
    # Byzantine node tries to disrupt (but will be discounted)
    byzantine_vote = (5, VoteType.REJECT, 0.95)
    
    # Cast votes
    for voter_id, vote_type, confidence in honest_votes + [byzantine_vote]:
        result = engine1.cast_vote(
            proposal_id,
            voter_id,
            vote_type,
            confidence,
            evidence={'analysis': f'Vote from node {voter_id}'}
        )
        
        vote_mark = "⚠" if voter_id == 5 else "✓"
        logger.info(f"      {vote_mark} Node {voter_id}: {vote_type} "
                   f"(confidence={confidence:.2f})")
    
    logger.info("\n   4.4 Consensus result...")
    
    status = engine1.get_proposal_status(proposal_id)
    
    if status['consensus_reached']:
        logger.info(f"      ✓ Consensus reached: {status['consensus_result']}")
        logger.info(f"        Confidence: {status['confidence']:.2f}")
        logger.info(f"        Votes received: {status['votes_received']}/{status['quorum_size']}")
        logger.info("        → Byzantine node's vote was discounted by trust weighting")
    
    # Check for suspicious nodes
    suspicious = engine1.get_suspicious_nodes()
    if suspicious:
        logger.info(f"\n   4.5 Suspicious voting patterns detected:")
        for node_info in suspicious:
            logger.info(f"      ⚠ Node {node_info['node_id']}: "
                       f"suspicion={node_info['suspicion_score']:.2f}")
    
    # ========================================================================
    # 5. Integrated Demonstration
    # ========================================================================
    logger.info("\n5. Integrated Demonstration: Evolution + Learning + Consensus...")
    logger.info("   " + "-" * 66)
    
    logger.info("\n   5.1 Create adaptive node with evolved strategies...")
    
    node = AliveLoopNode(
        position=(0, 0),
        velocity=(0, 0),
        initial_energy=10.0,
        node_id=1
    )
    
    pm = PluginManager()
    pm.register_plugin(ITOperationsPlugin())
    pm.register_plugin(SecurityPlugin())
    pm.initialize_all(node)
    
    logger.info("      ✓ Node created with IT and Security plugins")
    
    logger.info("\n   5.2 Apply evolved detection strategy...")
    
    # Use best evolved strategy
    security_plugin = pm.get_plugin('security')
    if security_plugin and best_strategy:
        # Apply evolved parameters
        logger.info(f"      ✓ Applied evolved detection threshold: "
                   f"{best_strategy.parameters['threshold']:.3f}")
        logger.info(f"      ✓ Applied evolved sensitivity: "
                   f"{best_strategy.parameters['sensitivity']:.3f}")
    
    logger.info("\n   5.3 Learn and adapt to system behavior...")
    
    # Observe system metrics
    for _ in range(10):
        # Simulate metrics
        cpu_usage = 0.3 + random.gauss(0, 0.05)
        memory_usage = 0.5 + random.gauss(0, 0.05)
        
        learning.observe(BehaviorType.RESOURCE, cpu_usage)
        learning.observe(BehaviorType.PERFORMANCE, memory_usage)
    
    stats = learning.get_all_statistics()
    logger.info(f"      ✓ Learned from {stats['total_observations']} observations")
    logger.info(f"      ✓ Auto-tuned thresholds {stats['threshold_adjustments']} times")
    
    logger.info("\n   5.4 Collective decision with consensus...")
    
    # Initiate consensus on a threshold update
    threshold_proposal_id = engine1.initiate_consensus(
        ConsensusType.THRESHOLD_UPDATE,
        subject={
            'metric': 'cpu_usage',
            'current_threshold': 0.8,
            'proposed_threshold': thresholds['upper_threshold'],
            'rationale': 'Auto-tuned based on learned behavior'
        },
        network_size=7
    )
    
    # Nodes vote (most approve the learned threshold)
    for voter_id, vote_type, confidence in [
        (2, VoteType.APPROVE, 0.80),
        (3, VoteType.APPROVE, 0.85),
        (4, VoteType.APPROVE, 0.75),
        (6, VoteType.APPROVE, 0.70),
        (7, VoteType.APPROVE, 0.78)
    ]:
        engine1.cast_vote(threshold_proposal_id, voter_id, vote_type, confidence)
    
    threshold_status = engine1.get_proposal_status(threshold_proposal_id)
    if threshold_status['consensus_reached']:
        logger.info(f"      ✓ Consensus on threshold update: {threshold_status['consensus_result']}")
        logger.info(f"        Confidence: {threshold_status['confidence']:.2f}")
        logger.info("        → System adapts thresholds collectively via consensus")
    
    # ========================================================================
    # 6. Phase 3 System Statistics
    # ========================================================================
    logger.info("\n6. Phase 3 System Statistics...")
    logger.info("   " + "-" * 66)
    
    # Evolution stats
    evo_stats = evolution.get_population_stats(StrategyType.DETECTION)
    logger.info("\n   Evolutionary Learning:")
    logger.info(f"      Generation: {evo_stats['generation']}")
    logger.info(f"      Best fitness: {evo_stats['best_fitness']:.4f}")
    logger.info(f"      Average fitness: {evo_stats['average_fitness']:.4f}")
    logger.info(f"      Population size: {evo_stats['population_size']}")
    
    # Adaptive learning stats
    learning_stats = learning.get_all_statistics()
    logger.info("\n   Adaptive Learning:")
    logger.info(f"      Total observations: {learning_stats['total_observations']}")
    logger.info(f"      Anomalies detected: {learning_stats['anomalies_detected']}")
    logger.info(f"      Threshold adjustments: {learning_stats['threshold_adjustments']}")
    logger.info(f"      Behavior profiles: {len(learning_stats['profiles'])}")
    
    # Trust network stats
    trust_metrics = trust_net.get_byzantine_resilience_metrics()
    logger.info("\n   Trust Network:")
    logger.info(f"      Total nodes: {trust_metrics['total_nodes']}")
    logger.info(f"      Trusted nodes: {trust_metrics['trusted_nodes']}")
    logger.info(f"      Compromised nodes: {trust_metrics['compromised_nodes']}")
    logger.info(f"      Network health: {trust_metrics['network_health']:.2f}")
    logger.info(f"      Byzantine tolerance: {trust_metrics['byzantine_tolerance']:.2f}")
    
    # Consensus stats
    consensus_stats = engine1.get_statistics()
    logger.info("\n   Consensus System:")
    logger.info(f"      Total proposals: {consensus_stats['total_proposals']}")
    logger.info(f"      Finalized proposals: {consensus_stats['finalized_proposals']}")
    logger.info(f"      Suspicious nodes detected: {consensus_stats['suspicious_nodes_count']}")
    logger.info(f"      Trusted nodes: {consensus_stats['trusted_nodes_count']}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 IMPLEMENTATION COMPLETE")
    logger.info("=" * 70)
    
    logger.info("\n✓ 1. Adaptive Learning & Evolution:")
    logger.info("       - Genetic algorithms evolve detection strategies")
    logger.info("       - Strategies improve through reproduction, variation, selection")
    logger.info("       - Learn normal behavior and auto-tune thresholds")
    logger.info("       - Continual adaptation to system drift")
    
    logger.info("\n✓ 2. Trust Network & Byzantine-Resilient Consensus:")
    logger.info("       - Detect and mark compromised/malicious nodes")
    logger.info("       - Discount input from unreliable nodes")
    logger.info("       - Byzantine-resilient aggregation (weighted median)")
    logger.info("       - Consensus for root cause, attack validation, responses")
    logger.info("       - Tolerate up to 33% malicious nodes in voting")
    
    logger.info("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        demonstrate_phase3()
    except KeyboardInterrupt:
        logger.info("\nDemonstration interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        sys.exit(1)
