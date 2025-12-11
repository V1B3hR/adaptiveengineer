#!/usr/bin/env python3
"""
Example demonstrating Phase 1 Advanced Features:
- Living Graph Environment
- Advanced Sensory & Communication Protocol
- Agent Lifecycle Management

This example shows how agents interact with a dynamic graph-based IT system,
use sophisticated communication protocols, and have natural birth/death cycles.

Usage:
    python3 example/example_living_graph.py
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from adaptiveengineer import AliveLoopNode
from core.living_graph import LivingGraph, NodeType, EdgeType, HealthStatus
from core.advanced_communication import (
    AdvancedSensorySystem,
    AdvancedCommunicationProtocol,
    Priority,
)
from core.agent_lifecycle import (
    AgentLifecycleManager,
    BirthConditions,
    DeathConditions,
)

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("living_graph_example")


def demonstrate_living_graph():
    """Demonstrate the Living Graph environment"""
    logger.info("=" * 70)
    logger.info("PHASE 1 ADVANCED FEATURES: Living Graph Environment")
    logger.info("=" * 70)

    # Create living graph
    graph = LivingGraph(seed=42)

    # Build a simple IT infrastructure
    logger.info("\n1. Building IT infrastructure graph...")

    # Add nodes
    web_server = graph.add_node(
        NodeType.SERVER, "WebServer-01", position=(0, 0)
    )
    app_server = graph.add_node(
        NodeType.SERVER, "AppServer-01", position=(5, 0)
    )
    db_server = graph.add_node(
        NodeType.DATABASE, "Database-01", position=(10, 0)
    )
    load_balancer = graph.add_node(
        NodeType.SERVICE, "LoadBalancer", position=(0, 5)
    )
    api_gateway = graph.add_node(
        NodeType.API_ENDPOINT, "APIGateway", position=(5, 5)
    )

    logger.info(f"   Created {len(graph.nodes)} nodes:")
    for node in graph.nodes.values():
        logger.info(f"     - {node.name} ({node.node_type.value})")

    # Add edges (dependencies)
    graph.add_edge(load_balancer, web_server, EdgeType.NETWORK_CONNECTIVITY)
    graph.add_edge(web_server, app_server, EdgeType.SERVICE_DEPENDENCY)
    graph.add_edge(app_server, db_server, EdgeType.SERVICE_DEPENDENCY)
    graph.add_edge(api_gateway, app_server, EdgeType.DATA_FLOW)

    logger.info(f"\n   Created {len(graph.edges)} edges (dependencies)")

    # Simulate normal operation
    logger.info("\n2. Simulating normal operation...")
    for i in range(5):
        graph.update(delta_time=1.0)

    summary = graph.get_graph_summary()
    logger.info(f"   Graph state after 5 time steps:")
    logger.info(
        f"     - Nodes: {summary['total_nodes']}, Edges: {summary['total_edges']}"
    )
    logger.info(f"     - Avg CPU: {summary['avg_cpu_load']:.2%}")
    logger.info(f"     - Avg Memory: {summary['avg_memory_usage']:.2%}")
    logger.info(f"     - Avg Stress: {summary['avg_stress_level']:.2%}")

    # Simulate DDoS attack
    logger.info("\n3. Simulating DDoS attack on web server...")
    graph.simulate_ddos_attack(web_server, intensity=0.8)

    for i in range(3):
        graph.update(delta_time=1.0)

    web_state = graph.get_node_state(web_server)
    logger.info(f"   Web server state after attack:")
    logger.info(f"     - CPU Load: {web_state['cpu_load']:.2%}")
    logger.info(f"     - Stress: {web_state['stress_level']:.2%}")
    logger.info(f"     - Threat Score: {web_state['active_threat_score']:.2%}")
    logger.info(f"     - Health: {web_state['health_status']}")

    # Simulate resource exhaustion
    logger.info("\n4. Simulating resource exhaustion on database...")
    graph.simulate_resource_exhaustion(db_server, resource="memory")

    for i in range(3):
        graph.update(delta_time=1.0)

    db_state = graph.get_node_state(db_server)
    logger.info(f"   Database state after exhaustion:")
    logger.info(f"     - Memory Usage: {db_state['memory_usage']:.2%}")
    logger.info(f"     - Stress: {db_state['stress_level']:.2%}")

    # Show cascading failures
    logger.info("\n5. Observing cascading effects...")
    app_state = graph.get_node_state(app_server)
    logger.info(f"   App server (depends on database):")
    logger.info(f"     - Stress increased to: {app_state['stress_level']:.2%}")

    return graph


def demonstrate_advanced_communication():
    """Demonstrate advanced communication protocols"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 ADVANCED FEATURES: Advanced Communication Protocol")
    logger.info("=" * 70)

    # Create agents
    logger.info("\n1. Creating agents with advanced communication...")
    agent1 = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1
    )
    agent2 = AliveLoopNode(
        position=(2, 0), velocity=(0, 0), initial_energy=15.0, node_id=2
    )
    agent3 = AliveLoopNode(
        position=(4, 0), velocity=(0, 0), initial_energy=15.0, node_id=3
    )

    # Add communication protocols
    agent1.communication_protocol = AdvancedCommunicationProtocol(agent1)
    agent2.communication_protocol = AdvancedCommunicationProtocol(agent2)
    agent3.communication_protocol = AdvancedCommunicationProtocol(agent3)

    # Add sensory systems
    agent1.sensory_system = AdvancedSensorySystem(agent1)
    agent2.sensory_system = AdvancedSensorySystem(agent2)
    agent3.sensory_system = AdvancedSensorySystem(agent3)

    logger.info("   Created 3 agents with communication protocols")

    # Demonstrate proprioception
    logger.info("\n2. Demonstrating Proprioception (self-sensing)...")
    self_state = agent1.sensory_system.proprioception()
    logger.info(f"   Agent 1 internal state:")
    logger.info(f"     - Energy: {self_state['energy_level']:.1f}")
    logger.info(f"     - Health: {self_state['health']:.1f}")
    logger.info(f"     - Trust Score: {self_state['trust_score']:.2f}")
    logger.info(f"     - Anxiety: {self_state['anxiety']:.1f}")

    # Demonstrate pheromones
    logger.info("\n3. Demonstrating Pheromones (ambient broadcast)...")

    # Agent 1 deposits pheromones
    pheromone_store = {}  # Simple global pheromone store

    node_id = "test_node_1"
    pheromone_id = agent1.communication_protocol.deposit_pheromone(
        node_id=node_id,
        content="Trace of anomalous process detected",
        intensity=0.8,
    )

    logger.info(f"   Agent 1 deposited pheromone: '{pheromone_id[:8]}...'")
    logger.info(f"   Content: 'Trace of anomalous process detected'")

    # Demonstrate signals
    logger.info("\n4. Demonstrating Signals (targeted messaging)...")

    signal_id = agent1.communication_protocol.send_signal(
        recipient_ids=[2, 3],
        message_type="threat_alert",
        payload={
            "threat_type": "SQL_INJECTION",
            "location": "Node-DB-01",
            "confidence": 0.95,
            "severity": "critical",
        },
        priority=Priority.CRITICAL,
        requires_ack=True,
    )

    logger.info(f"   Agent 1 sent critical signal to agents 2 and 3")
    logger.info(f"   Signal ID: {signal_id[:8]}...")
    logger.info(
        f"   Message: SQL_INJECTION threat detected with 95% confidence"
    )

    # Agents receive the signal
    while agent1.communication_protocol.signal_outbox:
        signal = agent1.communication_protocol.signal_outbox.popleft()
        agent2.communication_protocol.receive_signal(signal)
        agent3.communication_protocol.receive_signal(signal)

    logger.info(
        f"   Agent 2 received {len(agent2.communication_protocol.signal_inbox)} signal(s)"
    )
    logger.info(
        f"   Agent 3 received {len(agent3.communication_protocol.signal_inbox)} signal(s)"
    )

    # Demonstrate gossip protocol
    logger.info(
        "\n5. Demonstrating Gossip Protocol (decentralized propagation)..."
    )

    gossip_id = agent1.communication_protocol.initiate_gossip(
        gossip_type="threat_signature",
        content={
            "signature": "malware_variant_x123",
            "detection_method": "behavioral_analysis",
            "effectiveness": 0.87,
        },
    )

    logger.info(f"   Agent 1 initiated gossip about new threat signature")
    logger.info(f"   Gossip ID: {gossip_id[:8]}...")

    # Propagate gossip
    propagated = agent1.communication_protocol.propagate_gossip(
        [agent2, agent3]
    )
    logger.info(f"   Propagated to {propagated} neighbor(s)")

    # Show metrics
    logger.info("\n6. Communication metrics:")
    for agent in [agent1, agent2, agent3]:
        metrics = agent.communication_protocol.get_metrics()
        logger.info(f"   Agent {agent.node_id}:")
        logger.info(f"     - Signals sent: {metrics['signals_sent']}")
        logger.info(f"     - Signals received: {metrics['signals_received']}")
        logger.info(
            f"     - Gossip originated: {metrics['gossip_originated']}"
        )
        logger.info(
            f"     - Gossip propagated: {metrics['gossip_propagated']}"
        )

    return [agent1, agent2, agent3]


def demonstrate_agent_lifecycle(graph):
    """Demonstrate agent lifecycle management"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 ADVANCED FEATURES: Agent Lifecycle Management")
    logger.info("=" * 70)

    # Create lifecycle manager
    logger.info("\n1. Initializing agent lifecycle manager...")

    lifecycle_manager = AgentLifecycleManager(
        birth_conditions=BirthConditions(
            min_energy_available=0.5,  # Lowered for demo
            max_stress_threshold=0.7,
            min_population=0,  # Allow initial spawning
            max_population=10,
            spawn_cooldown=0.1,  # Shorter cooldown for demo
        ),
        death_conditions=DeathConditions(
            zero_energy_threshold=0.0,
            zero_health_threshold=0.0,
            death_probability_low_energy=0.1,
            max_lifespan=100.0,
        ),
    )

    logger.info("   Lifecycle manager initialized")
    logger.info(
        f"   Max population: {lifecycle_manager.birth_conditions.max_population}"
    )
    logger.info(
        f"   Max lifespan: {lifecycle_manager.death_conditions.max_lifespan} time steps"
    )

    # Spawn initial population
    logger.info("\n2. Spawning initial population...")

    # Find a healthy node to spawn on
    healthy_nodes = graph.query_nodes(
        health_status=HealthStatus.OK, max_stress=0.5
    )
    if healthy_nodes:
        spawn_location = healthy_nodes[0]

        # Spawn several agents
        agents = []
        for i in range(5):
            agent = lifecycle_manager.spawn_agent(
                agent_class=AliveLoopNode,
                spawn_location_id=spawn_location,
                initial_energy=np.random.uniform(8.0, 12.0),
                position=(i * 2, 0),
                velocity=(0, 0),
                environment=graph,
            )
            if agent:
                agents.append(agent)
                logger.info(
                    f"   Spawned agent {agent.node_id} with energy {agent.energy:.1f}"
                )

        logger.info(f"\n   Total agents spawned: {len(agents)}")
        logger.info(
            f"   Current population: {len(lifecycle_manager.living_agents)}"
        )

    # Simulate life and death
    logger.info("\n3. Simulating agent lifecycle (50 time steps)...")

    for t in range(50):
        # Update graph
        graph.update(delta_time=1.0)

        # Update agents
        for agent in list(lifecycle_manager.living_agents.values()):
            # Agents consume energy
            agent.energy -= 0.2
            agent._time = t

            # Occasionally give some agents low energy
            if t == 25 and agent.node_id == 1:
                agent.energy = 0.5
                logger.info(f"   Time {t}: Agent 1 energy critical!")

            if t == 40 and agent.node_id == 2:
                agent.energy = 0.0
                logger.info(f"   Time {t}: Agent 2 energy depleted!")

        # Check for deaths
        lifecycle_manager.update(delta_time=1.0)

        # Report deaths
        if len(lifecycle_manager.living_agents) < len(agents):
            logger.info(
                f"   Time {t}: Population decreased to {len(lifecycle_manager.living_agents)}"
            )

    # Show final statistics
    logger.info("\n4. Final lifecycle statistics:")
    stats = lifecycle_manager.get_population_stats()
    logger.info(f"   Total births: {stats['total_births']}")
    logger.info(f"   Total deaths: {stats['total_deaths']}")
    logger.info(f"   Current population: {stats['current_population']}")
    logger.info(f"   Death causes: {stats['death_causes']}")

    if lifecycle_manager.dead_agents:
        avg_lifespan = lifecycle_manager.get_average_lifespan()
        logger.info(f"   Average lifespan: {avg_lifespan:.1f} time steps")

    return lifecycle_manager


def main():
    """Run all demonstrations"""
    logger.info("\n" + "=" * 70)
    logger.info(
        "PHASE 1: THE PRIMORDIAL SOUP - ADVANCED FEATURES DEMONSTRATION"
    )
    logger.info("=" * 70)

    # 1. Living Graph
    graph = demonstrate_living_graph()

    # 2. Advanced Communication
    agents = demonstrate_advanced_communication()

    # 3. Agent Lifecycle
    lifecycle_manager = demonstrate_agent_lifecycle(graph)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\n✓ Phase 1 Advanced Features Implemented:")
    logger.info("  1. Living Graph Environment:")
    logger.info("     - Dynamic IT system graph with nodes and edges")
    logger.info("     - Resource dynamics and stress simulation")
    logger.info("     - Cascading failures through dependencies")
    logger.info("     - Environmental stressors (DDoS, exhaustion)")
    logger.info("\n  2. Advanced Communication Protocol:")
    logger.info("     - Proprioception (self-sensing)")
    logger.info("     - Local environmental sensing")
    logger.info("     - Pheromones (ambient broadcast)")
    logger.info("     - Signals (targeted messaging)")
    logger.info("     - Gossip protocol (decentralized)")
    logger.info("     - Cryptographic message signing")
    logger.info("\n  3. Agent Lifecycle:")
    logger.info("     - Birth conditions and spawning")
    logger.info("     - Death conditions (energy, health)")
    logger.info("     - Population management")
    logger.info("     - Lifecycle tracking and metrics")
    logger.info("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
