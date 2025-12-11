"""
Performance benchmarks for critical system components.
"""

import pytest
import numpy as np
from adaptiveengineer import AliveLoopNode
from core.plugin_manager import PluginManager
from plugins.communication import CommunicationPlugin
from plugins.security import SecurityPlugin
from core.memory_system import Memory, MemoryType, Classification

# SwarmCoordinator will be imported when needed


class TestAgentBenchmarks:
    """Benchmark tests for agent operations."""

    def test_agent_creation(self, benchmark):
        """Benchmark agent creation time."""

        def create_agent():
            return AliveLoopNode(
                position=(0, 0, 0),
                velocity=(0, 0, 0),
                initial_energy=10.0,
                node_id=1,
            )

        result = benchmark(create_agent)
        assert result is not None

    def test_agent_position_update(self, benchmark):
        """Benchmark agent position updates."""
        node = AliveLoopNode(
            position=(0, 0, 0),
            velocity=(1, 1, 1),
            initial_energy=10.0,
            node_id=1,
        )

        def update_position():
            node.position = node.position + node.velocity * 0.1

        benchmark(update_position)


class TestMemoryBenchmarks:
    """Benchmark tests for memory operations."""

    def test_memory_creation(self, benchmark):
        """Benchmark memory object creation."""

        def create_memory():
            return Memory(
                memory_type=MemoryType.EVENT,
                data={"event": "test", "timestamp": 123.0},
                classification=Classification.IMPORTANT,
            )

        result = benchmark(create_memory)
        assert result is not None

    def test_memory_storage(self, benchmark):
        """Benchmark memory storage operations."""
        node = AliveLoopNode(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=1,
        )

        def store_memory():
            memory = Memory(
                memory_type=MemoryType.EVENT,
                data={"event": "test"},
                classification=Classification.NORMAL,
            )
            node.memory.append(memory)

        benchmark(store_memory)


class TestSwarmBenchmarks:
    """Benchmark tests for swarm operations."""

    def test_multi_agent_creation(self, benchmark):
        """Benchmark creation of multiple agents."""

        def create_agents():
            agents = []
            for i in range(50):
                node = AliveLoopNode(
                    position=(i * 0.1, i * 0.1, 0),
                    velocity=(0, 0, 0),
                    initial_energy=10.0,
                    node_id=i,
                )
                agents.append(node)
            return agents

        result = benchmark(create_agents)
        assert len(result) == 50

    def test_trust_network_update(self, benchmark):
        """Benchmark trust network updates."""
        node = AliveLoopNode(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=1,
        )

        def update_trust():
            node.trust_network_system.update_trust(
                target_node_id=2, delta=0.1, context="test_interaction"
            )

        benchmark(update_trust)


class TestCommunicationBenchmarks:
    """Benchmark tests for communication system."""

    def test_signal_creation(self, benchmark):
        """Benchmark signal creation."""
        sender = AliveLoopNode(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=1,
        )
        receiver = AliveLoopNode(
            position=(1, 1, 1),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=2,
        )

        def create_signal():
            sender.send_signal(
                target_nodes=[receiver],
                signal_type="test_signal",
                content={"message": "test"},
            )

        benchmark(create_signal)

    def test_signal_processing(self, benchmark):
        """Benchmark signal processing."""
        from core.social_signals import SocialSignal

        node = AliveLoopNode(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=1,
        )

        # Create a signal to process
        signal = SocialSignal(
            signal_type="resource_offer",
            source_node_id=2,
            target_node_id=1,
            data={"resource": "energy", "amount": 1.0},
        )

        def process_signal():
            node.receive_signal(signal)

        benchmark(process_signal)


class TestSpatialBenchmarks:
    """Benchmark tests for spatial operations."""

    def test_distance_calculation(self, benchmark):
        """Benchmark distance calculation between agents."""
        node1 = AliveLoopNode(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=1,
        )
        node2 = AliveLoopNode(
            position=(10, 10, 10),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=2,
        )

        def calculate_distance():
            return np.linalg.norm(node1.position - node2.position)

        result = benchmark(calculate_distance)
        assert result > 0

    def test_neighbor_search(self, benchmark):
        """Benchmark neighbor search operations."""
        # Create a grid of agents
        agents = []
        for i in range(10):
            for j in range(10):
                node = AliveLoopNode(
                    position=(i, j, 0),
                    velocity=(0, 0, 0),
                    initial_energy=10.0,
                    node_id=i * 10 + j,
                )
                agents.append(node)

        search_pos = np.array([5.0, 5.0, 0.0])
        search_radius = 3.0

        def find_neighbors():
            neighbors = []
            for agent in agents:
                dist = np.linalg.norm(agent.position - search_pos)
                if dist <= search_radius:
                    neighbors.append(agent)
            return neighbors

        result = benchmark(find_neighbors)
        assert len(result) > 0


class TestPluginBenchmarks:
    """Benchmark tests for plugin system."""

    def test_plugin_registration(self, benchmark):
        """Benchmark plugin registration."""

        def register_plugins():
            manager = PluginManager()
            manager.register_plugin(CommunicationPlugin())
            manager.register_plugin(SecurityPlugin())
            return manager

        result = benchmark(register_plugins)
        assert result is not None

    def test_plugin_initialization(self, benchmark):
        """Benchmark plugin initialization."""
        node = AliveLoopNode(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=1,
        )

        def init_plugins():
            manager = PluginManager()
            manager.register_plugin(CommunicationPlugin())
            manager.initialize_all(node)

        benchmark(init_plugins)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
