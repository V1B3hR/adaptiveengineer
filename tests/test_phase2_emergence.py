#!/usr/bin/env python3
"""
Tests for Phase 2: Emergence & Adaptation

Tests all major Phase 2 components:
1. Behavior strategies (trees, FSMs, roles)
2. Swarm intelligence (pheromones, supply chains, coordination)
3. Predictive homeostasis (failure prediction, resource management)
4. Integration and emergence
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.behavior_strategy import (
    BehaviorStrategy,
    BehaviorNode,
    FiniteStateMachine,
    FSMTransition,
    BehaviorStrategyFactory,
    AgentRole,
    NodeType,
    StateType,
    BehaviorStatus,
)
from core.swarm_intelligence import (
    SwarmIntelligenceManager,
    PheromoneType,
    StructuredPheromone,
)
from core.predictive_homeostasis import (
    PredictiveHomeostasisSystem,
    FailureType,
    StressType,
)
from plugins.phase2_emergence import Phase2EmergencePlugin
from adaptiveengineer import AliveLoopNode
from core.plugin_manager import PluginManager

import time


def test_behavior_strategies():
    """Test behavior strategy creation and execution"""
    print("\n" + "=" * 70)
    print("Testing Behavior Strategies...")
    print("=" * 70)

    # Test role-specific strategy creation
    factory = BehaviorStrategyFactory()

    scout = factory.create_scout_strategy()
    assert scout.role == AgentRole.SCOUT
    assert scout.behavior_tree is not None
    assert scout.fsm is not None
    assert scout.speed_modifier > 1.0  # Scouts are fast
    print("✓ Scout strategy created with high speed")

    harvester = factory.create_harvester_strategy()
    assert harvester.role == AgentRole.HARVESTER
    assert harvester.energy_efficiency > 1.0  # Harvesters are efficient
    print("✓ Harvester strategy created with high efficiency")

    guardian = factory.create_guardian_strategy()
    assert guardian.role == AgentRole.GUARDIAN
    assert guardian.defensive_strength > 1.0  # Guardians are defensive
    print("✓ Guardian strategy created with high defense")

    # Test specialization computation
    scout_spec = scout.compute_specialization()
    generalist = factory.create_generalist_strategy()
    generalist_spec = generalist.compute_specialization()

    assert scout_spec > generalist_spec  # Scout more specialized
    print(
        f"✓ Specialization: Scout={scout_spec:.3f}, Generalist={generalist_spec:.3f}"
    )

    # Test behavior tree execution
    context = {
        "conditions": {
            "check_energy": lambda ctx, params: ctx.get("energy", 1.0)
            > params.get("threshold", 0.5)
        },
        "actions": {
            "explore": lambda ctx, params: True,
            "scan": lambda ctx, params: True,
        },
        "energy": 0.8,
    }

    if scout.behavior_tree:
        status = scout.behavior_tree.execute(context)
        assert status in [
            BehaviorStatus.SUCCESS,
            BehaviorStatus.FAILURE,
            BehaviorStatus.RUNNING,
        ]
        print(f"✓ Behavior tree executed: {status.value}")

    # Test FSM state transitions
    fsm_context = {
        "conditions": {
            "detect_threat": lambda ctx, params: ctx.get("threat_level", 0)
            > 0.7,
            "threat_cleared": lambda ctx, params: ctx.get("threat_level", 0)
            < 0.3,
            "low_energy": lambda ctx, params: ctx.get("energy", 1.0) < 0.2,
        },
        "threat_level": 0.8,
        "energy": 0.5,
    }

    if scout.fsm:
        initial_state = scout.fsm.current_state
        new_state, changed = scout.fsm.update(fsm_context)
        print(
            f"✓ FSM transition: {initial_state.value} -> {new_state.value} (changed: {changed})"
        )

    # Test serialization
    scout_dict = scout.to_dict()
    scout_restored = BehaviorStrategy.from_dict(scout_dict)
    assert scout_restored.role == scout.role
    assert scout_restored.strategy_id == scout.strategy_id
    print("✓ Strategy serialization/deserialization works")

    print("\n✓ Behavior strategy tests passed")
    return True


def test_swarm_intelligence():
    """Test swarm intelligence and stigmergy"""
    print("\n" + "=" * 70)
    print("Testing Swarm Intelligence...")
    print("=" * 70)

    manager = SwarmIntelligenceManager()

    # Test pheromone deposition
    pheromone1 = manager.deposit_pheromone(
        node_id="node_1",
        pheromone_type=PheromoneType.THREAT,
        depositor_id=1,
        depositor_role="scout",
        signature="sql_injection",
        confidence=0.85,
        priority=0.9,
    )

    assert pheromone1.pheromone_type == PheromoneType.THREAT
    assert pheromone1.confidence == 0.85
    print("✓ Pheromone deposited with structured information")

    # Test pheromone sensing
    pheromones = manager.sense_pheromones(
        node_id="node_1",
        pheromone_types=[PheromoneType.THREAT],
        min_intensity=0.1,
    )

    assert len(pheromones) == 1
    assert pheromones[0].pheromone_id == pheromone1.pheromone_id
    print("✓ Pheromone sensing works")

    # Test pheromone reinforcement
    success = manager.reinforce_pheromone(
        pheromone1.pheromone_id, agent_id=2, amount=0.2
    )
    assert success
    assert pheromone1.reinforcement_count == 1
    assert 2 in pheromone1.contributors
    print("✓ Pheromone reinforcement (stigmergy) works")

    # Test supply chain establishment
    chain_success = manager.establish_supply_chain(
        chain_id="chain_1",
        nodes=["node_1", "node_2", "node_3", "node_4"],
        resource_type="data",
    )

    assert chain_success
    chain = manager.get_supply_chain("chain_1")
    assert chain is not None
    assert len(chain) == 3  # 4 nodes = 3 links
    print(f"✓ Supply chain established with {len(chain)} links")

    # Test supply chain healing
    heal_success = manager.heal_supply_chain(
        chain_id="chain_1", broken_position=1, new_node="node_5"
    )

    assert heal_success
    stats = manager.get_statistics()
    assert stats["chains_healed"] == 1
    print("✓ Supply chain healing works")

    # Test coordination
    coord_id = manager.start_coordination(
        task_type="incident_response", target_node="node_1"
    )

    assert coord_id is not None

    manager.join_coordination(coord_id, agent_id=1, role="scout")
    manager.join_coordination(coord_id, agent_id=2, role="guardian")
    manager.join_coordination(coord_id, agent_id=3, role="healer")

    coordination = manager.coordinations[coord_id]
    assert coordination.total_agents() == 3
    assert coordination.is_diverse()  # Multiple roles
    print(f"✓ Swarm coordination with {coordination.total_agents()} agents")

    # Complete coordination
    metrics = manager.complete_coordination(coord_id)
    assert metrics is not None
    assert metrics["role_diversity"] == True
    print("✓ Coordination completion tracked")

    # Test pheromone decay
    initial_intensity = pheromone1.intensity
    manager.update(delta_time=5.0)
    assert pheromone1.intensity < initial_intensity
    print("✓ Pheromone decay over time")

    final_stats = manager.get_statistics()
    print(f"\n  Total pheromones: {final_stats['total_pheromones']}")
    print(f"  Supply chains: {final_stats['supply_chains']}")
    print(f"  Coordinated actions: {final_stats['coordinated_actions']}")

    print("\n✓ Swarm intelligence tests passed")
    return True


def test_predictive_homeostasis():
    """Test predictive homeostasis and resilience"""
    print("\n" + "=" * 70)
    print("Testing Predictive Homeostasis...")
    print("=" * 70)

    system = PredictiveHomeostasisSystem(prediction_window=60.0)

    # Test metric recording
    for i in range(20):
        system.record_metric("latency_edge_A", 50.0 + i * 5.0)
        system.record_metric("cpu_node_B", 0.5 + i * 0.02)

    assert len(system.metric_history["latency_edge_A"]) == 20
    print("✓ Metric recording works")

    # Test correlation detection
    correlation = system.detect_correlation("latency_edge_A", "cpu_node_B")
    assert -1.0 <= correlation <= 1.0
    print(f"✓ Correlation detected: {correlation:.3f}")

    # Test learning failure precursors
    precursor_id = system.learn_failure_precursor(
        failure_type=FailureType.CASCADING_OVERLOAD,
        indicators={"latency_edge_A": "> 100", "cpu_node_B": "> 0.9"},
        time_before_failure=30.0,
    )

    assert precursor_id is not None
    assert FailureType.CASCADING_OVERLOAD in system.precursors
    print(f"✓ Failure precursor learned: {precursor_id}")

    # Test failure prediction
    current_metrics = {"latency_edge_A": 105.0, "cpu_node_B": 0.92}

    # Make prediction (initially low confidence, but should detect pattern)
    predictions = system.predict_failure(
        current_metrics, confidence_threshold=0.0
    )

    # Update precursor confidence
    if predictions:
        precursor = system.precursors[FailureType.CASCADING_OVERLOAD][0]
        precursor.update_statistics(was_true_positive=True)
        precursor.update_statistics(was_true_positive=True)
        precursor.update_statistics(was_true_positive=True)

        # Try again with higher confidence
        predictions = system.predict_failure(
            current_metrics, confidence_threshold=0.5
        )
        assert len(predictions) > 0
        print(f"✓ Predicted {len(predictions)} potential failure(s)")
        print(f"  Type: {predictions[0]['failure_type']}")
        print(f"  Confidence: {predictions[0]['confidence']:.2f}")

    # Test resource allocation
    allocation = system.allocate_resources(
        entity_id="agent_1", node_id="node_1", priority=0.7, min_allocation=0.1
    )

    assert allocation.entity_id == "agent_1"
    assert allocation.cpu_allocation > 0.1
    print(f"✓ Resource allocated: CPU={allocation.cpu_allocation:.2f}")

    # Test adaptive reallocation
    reallocation = system.adaptive_reallocation(
        stress_type=StressType.CPU_PRESSURE, affected_nodes={"node_1"}
    )

    print(
        f"✓ Adaptive reallocation: {len(reallocation['reallocations'])} entities affected"
    )

    # Test agent migration
    migration_success = system.migrate_agent(
        agent_id="agent_1",
        from_node="node_1",
        to_node="node_2",
        reason="load_balancing",
    )

    assert migration_success
    assert allocation.node_id == "node_2"  # Updated
    print("✓ Agent migration successful")

    # Test system stress detection
    node_metrics = {
        "node_1": {"cpu_load": 0.90, "memory_usage": 0.85},
        "node_2": {"cpu_load": 0.88, "memory_usage": 0.80},
        "node_3": {"cpu_load": 0.87, "memory_usage": 0.75},
        "node_4": {"cpu_load": 0.50, "memory_usage": 0.50},
    }

    stress_states = system.detect_system_stress(node_metrics)
    assert len(stress_states) > 0
    print(f"✓ Detected {len(stress_states)} system stress condition(s)")

    for stress in stress_states:
        print(
            f"  {stress.stress_type.value}: {len(stress.affected_nodes)} nodes, severity={stress.severity:.2f}"
        )

    # Get statistics
    stats = system.get_statistics()
    print(f"\n  Learned precursors: {stats['learned_precursors']}")
    print(f"  Predictions made: {stats['predictions_made']}")
    print(f"  Agents migrated: {stats['agents_migrated']}")
    print(f"  Active stress conditions: {stats['active_stress_conditions']}")

    print("\n✓ Predictive homeostasis tests passed")
    return True


def test_phase2_plugin_integration():
    """Test Phase 2 plugin integration"""
    print("\n" + "=" * 70)
    print("Testing Phase 2 Plugin Integration...")
    print("=" * 70)

    # Create agent with Phase 2 plugin
    agent = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1
    )

    manager = PluginManager()
    phase2_plugin = Phase2EmergencePlugin(
        plugin_id="phase2_test", config={"initial_role": "scout"}
    )

    manager.register_plugin(phase2_plugin)
    manager.initialize_all(agent)

    assert phase2_plugin.node is not None
    assert phase2_plugin.assigned_role == AgentRole.SCOUT
    print(
        f"✓ Plugin initialized with role: {phase2_plugin.assigned_role.value}"
    )

    # Test state variables
    state_vars = phase2_plugin.state_variables
    assert "role" in state_vars
    assert "specialization_score" in state_vars
    assert "emergence_level" in state_vars
    print(f"✓ State variables defined: {len(state_vars)} variables")

    # Test pheromone deposition action
    success = phase2_plugin.execute_action(
        "deposit_pheromone",
        {
            "type": "threat",
            "signature": "test_threat",
            "confidence": 0.8,
            "priority": 0.9,
        },
    )

    assert success
    pheromones_deposited = phase2_plugin.state_variables[
        "pheromones_deposited"
    ].value
    assert pheromones_deposited > 0
    print("✓ Pheromone deposition action works")

    # Test supply chain establishment action
    success = phase2_plugin.execute_action(
        "establish_supply_chain",
        {"nodes": ["node_1", "node_2", "node_3"], "resource_type": "data"},
    )

    assert success
    assert len(phase2_plugin.supply_chain_ids) > 0
    print("✓ Supply chain establishment action works")

    # Test failure prediction action
    success = phase2_plugin.execute_action(
        "predict_failure",
        {
            "metrics": {"cpu_node_1": 0.95, "memory_node_1": 0.90},
            "confidence_threshold": 0.0,  # Low threshold for testing
        },
    )

    # May or may not predict (depends on learned precursors), but should execute
    print(f"✓ Failure prediction action executed (success: {success})")

    # Test state update
    phase2_plugin.update_state(delta_time=1.0)

    emergence_level = phase2_plugin.state_variables["emergence_level"].value
    print(f"✓ State updated, emergence level: {emergence_level:.3f}")

    # Test emergence summary
    summary = phase2_plugin.get_emergence_summary()
    assert "role" in summary
    assert "swarm_intelligence" in summary
    assert "predictive_homeostasis" in summary
    print(f"✓ Emergence summary generated")

    print("\n✓ Phase 2 plugin integration tests passed")
    return True


def test_emergent_behaviors():
    """Test emergent collective behaviors"""
    print("\n" + "=" * 70)
    print("Testing Emergent Collective Behaviors...")
    print("=" * 70)

    # Create small population with diverse roles
    agents_data = []

    for i, role in enumerate(
        [AgentRole.SCOUT, AgentRole.HARVESTER, AgentRole.GUARDIAN]
    ):
        agent = AliveLoopNode(
            position=(0, 0),
            velocity=(0, 0),
            initial_energy=10.0,
            node_id=i + 1,
        )

        manager = PluginManager()
        plugin = Phase2EmergencePlugin(
            plugin_id=f"phase2_{i+1}", config={"initial_role": role.value}
        )

        manager.register_plugin(plugin)
        manager.initialize_all(agent)

        agents_data.append({"agent": agent, "plugin": plugin, "role": role})

    print(
        f"✓ Created population of {len(agents_data)} agents with diverse roles"
    )

    # Test role diversity
    roles = [a["role"] for a in agents_data]
    unique_roles = len(set(roles))
    assert unique_roles == 3
    print(f"✓ Role diversity: {unique_roles} different roles")

    # Test collective pheromone field
    shared_manager = agents_data[0]["plugin"].swarm_manager

    for agent_data in agents_data:
        plugin = agent_data["plugin"]
        plugin.execute_action(
            "deposit_pheromone",
            {
                "type": "resource",
                "signature": "collective_resource",
                "confidence": 0.8,
            },
        )

    # All agents should sense the collective pheromone field
    stats = shared_manager.get_statistics()
    print(
        f"✓ Collective pheromone field: {stats['total_pheromones']} pheromones"
    )

    # Test coordinated action
    # Use the first agent's manager as shared manager for coordination
    first_plugin = agents_data[0]["plugin"]
    coord_id = first_plugin.swarm_manager.start_coordination(
        task_type="collective_test", target_node="test_node"
    )

    # Manually join coordination (since each has separate manager)
    for agent_data in agents_data:
        agent_id = int(agent_data["agent"].node_id)
        first_plugin.swarm_manager.join_coordination(
            coord_id, agent_id, agent_data["role"].value
        )

    coordination = first_plugin.swarm_manager.coordinations[coord_id]
    assert coordination.total_agents() == len(agents_data)
    assert coordination.is_diverse()  # Has multiple roles
    print(
        f"✓ Coordinated action with {coordination.total_agents()} agents (diverse: {coordination.is_diverse()})"
    )

    # Test specialization scores
    specializations = [
        a["plugin"].behavior_strategy.compute_specialization()
        for a in agents_data
    ]

    avg_specialization = sum(specializations) / len(specializations)
    print(f"✓ Average specialization: {avg_specialization:.3f}")

    # Test emergence levels
    for agent_data in agents_data:
        agent_data["plugin"].update_state(delta_time=1.0)

    emergence_levels = [
        a["plugin"].state_variables["emergence_level"].value
        for a in agents_data
    ]

    avg_emergence = sum(emergence_levels) / len(emergence_levels)
    print(f"✓ Average emergence level: {avg_emergence:.3f}")

    if avg_emergence > 0.3:
        print("  → Collective intelligence emerging!")

    print("\n✓ Emergent behavior tests passed")
    return True


def run_all_tests():
    """Run all Phase 2 tests"""
    print("=" * 70)
    print("PHASE 2: EMERGENCE & ADAPTATION - TEST SUITE")
    print("=" * 70)

    tests = [
        ("Behavior Strategies", test_behavior_strategies),
        ("Swarm Intelligence", test_swarm_intelligence),
        ("Predictive Homeostasis", test_predictive_homeostasis),
        ("Plugin Integration", test_phase2_plugin_integration),
        ("Emergent Behaviors", test_emergent_behaviors),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✓ ALL TESTS PASSED ✓")
        return True
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
