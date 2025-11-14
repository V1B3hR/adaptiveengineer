#!/usr/bin/env python3
"""
Tests for Phase 1 Advanced Features:
- Living Graph Environment
- Advanced Sensory & Communication Protocol
- Agent Lifecycle Management
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptiveengineer import AliveLoopNode
from core.living_graph import LivingGraph, NodeType, EdgeType, HealthStatus
from core.advanced_communication import (
    AdvancedSensorySystem,
    AdvancedCommunicationProtocol,
    Priority,
    SensoryMode
)
from core.agent_lifecycle import AgentLifecycleManager, BirthConditions, DeathConditions


def test_living_graph_creation():
    """Test basic living graph creation"""
    print("Testing Living Graph creation...")
    
    graph = LivingGraph(seed=42)
    
    # Add nodes
    server_id = graph.add_node(NodeType.SERVER, "TestServer")
    db_id = graph.add_node(NodeType.DATABASE, "TestDB")
    
    assert len(graph.nodes) == 2
    assert server_id in graph.nodes
    assert db_id in graph.nodes
    
    # Add edge
    edge_id = graph.add_edge(server_id, db_id, EdgeType.SERVICE_DEPENDENCY)
    assert len(graph.edges) == 1
    assert edge_id in graph.edges
    
    # Check neighbors
    neighbors = graph.get_neighbors(server_id)
    assert db_id in neighbors
    
    print("✓ Living Graph creation test passed")


def test_graph_dynamics():
    """Test graph dynamics and stress propagation"""
    print("\nTesting graph dynamics...")
    
    graph = LivingGraph(seed=42)
    server_id = graph.add_node(NodeType.SERVER, "TestServer")
    
    # Simulate DDoS
    graph.simulate_ddos_attack(server_id, intensity=0.8)
    
    server_state = graph.get_node_state(server_id)
    assert server_state['active_threat_score'] > 0.5
    assert server_state['stress_level'] > 0.5
    
    # Update graph
    for _ in range(5):
        graph.update(delta_time=1.0)
    
    print("✓ Graph dynamics test passed")


def test_advanced_sensory_system():
    """Test advanced sensory system"""
    print("\nTesting advanced sensory system...")
    
    agent = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
    sensory = AdvancedSensorySystem(agent)
    
    # Test proprioception
    internal_state = sensory.proprioception()
    assert internal_state['sensory_mode'] == SensoryMode.PROPRIOCEPTION.value
    assert 'energy_level' in internal_state
    assert 'health' in internal_state
    assert internal_state['energy_level'] == 10.0
    
    # Test local sensing with graph
    graph = LivingGraph(seed=42)
    node_id = graph.add_node(NodeType.SERVER, "TestServer")
    
    local_state = sensory.local_environmental_sensing(graph, node_id)
    assert local_state['sensory_mode'] == SensoryMode.LOCAL.value
    assert 'current_node' in local_state
    assert local_state['current_node']['node_type'] == 'server'
    
    # Test graph-level query
    query_result = sensory.graph_level_awareness(
        graph, 
        query_type="get_summary",
        query_params={}
    )
    assert query_result is not None
    assert query_result['sensory_mode'] == SensoryMode.GRAPH_LEVEL.value
    assert 'summary' in query_result
    
    print("✓ Advanced sensory system test passed")


def test_advanced_communication():
    """Test advanced communication protocol"""
    print("\nTesting advanced communication protocol...")
    
    agent1 = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
    agent2 = AliveLoopNode(position=(2, 0), velocity=(0, 0), initial_energy=10.0, node_id=2)
    
    comm1 = AdvancedCommunicationProtocol(agent1)
    comm2 = AdvancedCommunicationProtocol(agent2)
    
    # Test pheromone
    pheromone_id = comm1.deposit_pheromone(
        node_id="test_node",
        content="Test pheromone"
    )
    assert pheromone_id is not None
    assert comm1.metrics['pheromones_deposited'] == 1
    
    # Test signal
    signal_id = comm1.send_signal(
        recipient_ids=[2],
        message_type="test",
        payload={"data": "test data"},
        priority=Priority.NORMAL
    )
    assert signal_id is not None
    assert comm1.metrics['signals_sent'] == 1
    
    # Receive signal
    signal = comm1.signal_outbox.popleft()
    accepted = comm2.receive_signal(signal)
    assert accepted
    assert comm2.metrics['signals_received'] == 1
    
    # Test gossip
    gossip_id = comm1.initiate_gossip(
        gossip_type="test_gossip",
        content={"info": "test info"}
    )
    assert gossip_id is not None
    assert comm1.metrics['gossip_originated'] == 1
    
    print("✓ Advanced communication test passed")


def test_agent_lifecycle():
    """Test agent lifecycle management"""
    print("\nTesting agent lifecycle...")
    
    lifecycle = AgentLifecycleManager(
        birth_conditions=BirthConditions(
            min_energy_available=0.5,
            min_population=0,
            max_population=10,
            spawn_cooldown=0.0
        ),
        death_conditions=DeathConditions(
            zero_energy_threshold=0.0,
            death_probability_low_energy=1.0  # Guaranteed death when low energy
        )
    )
    
    # Create graph for spawning
    graph = LivingGraph(seed=42)
    node_id = graph.add_node(NodeType.SERVER, "TestServer")
    graph.update(delta_time=1.0)
    
    # Test spawning
    agent = lifecycle.spawn_agent(
        agent_class=AliveLoopNode,
        spawn_location_id=node_id,
        initial_energy=10.0,
        position=(0, 0),
        velocity=(0, 0),
        environment=graph
    )
    
    assert agent is not None
    assert lifecycle.total_births == 1
    assert len(lifecycle.living_agents) == 1
    
    # Test death condition
    agent.energy = 0.0
    should_die, cause = lifecycle.check_death_conditions(agent)
    assert should_die
    assert cause is not None
    
    # Process death
    lifecycle.update(delta_time=1.0)
    assert lifecycle.total_deaths == 1
    assert len(lifecycle.living_agents) == 0
    
    # Check statistics
    stats = lifecycle.get_population_stats()
    assert stats['total_births'] == 1
    assert stats['total_deaths'] == 1
    assert stats['current_population'] == 0
    
    print("✓ Agent lifecycle test passed")


def test_integration():
    """Integration test of all Phase 1 features"""
    print("\nTesting Phase 1 integration...")
    
    # Create environment
    graph = LivingGraph(seed=42)
    server_id = graph.add_node(NodeType.SERVER, "Server-01")
    db_id = graph.add_node(NodeType.DATABASE, "DB-01")
    graph.add_edge(server_id, db_id, EdgeType.SERVICE_DEPENDENCY)
    
    # Create lifecycle manager
    lifecycle = AgentLifecycleManager(
        birth_conditions=BirthConditions(
            min_energy_available=0.5,
            min_population=0,
            max_population=5,
            spawn_cooldown=0.0
        ),
        death_conditions=DeathConditions(
            zero_energy_threshold=0.0
        )
    )
    
    # Spawn agent
    agent = lifecycle.spawn_agent(
        agent_class=AliveLoopNode,
        spawn_location_id=server_id,
        initial_energy=10.0,
        position=(0, 0),
        velocity=(0, 0),
        environment=graph
    )
    
    assert agent is not None
    
    # Add communication and sensory systems
    agent.communication_protocol = AdvancedCommunicationProtocol(agent)
    agent.sensory_system = AdvancedSensorySystem(agent)
    
    # Agent senses environment
    sensory_state = agent.sensory_system.get_full_sensory_state(graph, server_id)
    assert 'proprioception' in sensory_state
    assert 'local' in sensory_state
    
    # Agent deposits pheromone
    pheromone_id = agent.communication_protocol.deposit_pheromone(
        node_id=server_id,
        content="Agent active at this location"
    )
    assert pheromone_id is not None
    
    # Simulate some time
    for _ in range(5):
        graph.update(delta_time=1.0)
        agent.energy -= 0.5
        agent._time += 1
    
    # Check agent is still alive
    assert agent.energy > 0
    assert len(lifecycle.living_agents) == 1
    
    # Drain energy to trigger death
    agent.energy = 0.0
    lifecycle.update(delta_time=1.0)
    
    # Check agent died
    assert len(lifecycle.living_agents) == 0
    assert lifecycle.total_deaths == 1
    
    print("✓ Phase 1 integration test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("PHASE 1 ADVANCED FEATURES - TEST SUITE")
    print("=" * 70)
    
    test_living_graph_creation()
    test_graph_dynamics()
    test_advanced_sensory_system()
    test_advanced_communication()
    test_agent_lifecycle()
    test_integration()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
