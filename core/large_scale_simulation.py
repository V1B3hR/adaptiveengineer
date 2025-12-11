"""
Large-Scale Simulation System: Testbed for Hundreds/Thousands of Nodes

Enables full-scale evaluation in production-like and adversarial environments.
Supports continuous improvement based on real-world feedback with efficient
simulation of large node populations.
"""

import logging
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

logger = logging.getLogger("large_scale_simulation")


class SimulationMode(str, Enum):
    """Simulation execution modes"""

    PRODUCTION_LIKE = "production_like"
    ADVERSARIAL = "adversarial"
    STRESS_TEST = "stress_test"
    CHAOS_ENGINEERING = "chaos_engineering"
    FIELD_DEPLOYMENT = "field_deployment"


class NodeState(str, Enum):
    """States a simulated node can be in"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    COMPROMISED = "compromised"


class EventType(str, Enum):
    """Types of simulation events"""

    NODE_STARTED = "node_started"
    NODE_STOPPED = "node_stopped"
    NODE_FAILED = "node_failed"
    NODE_RECOVERED = "node_recovered"
    ATTACK_DETECTED = "attack_detected"
    ATTACK_BLOCKED = "attack_blocked"
    COLLABORATION = "collaboration"
    ADAPTATION = "adaptation"
    REORGANIZATION = "reorganization"


@dataclass
class SimulatedNode:
    """Represents a node in large-scale simulation"""

    node_id: int
    state: NodeState
    position: Tuple[float, float]
    health: float  # 0.0 to 1.0
    energy: float  # 0.0 to 1.0
    load: float  # 0.0 to 1.0 (workload)
    connections: List[int]  # Connected node IDs
    role: str
    started_at: float
    failures: int = 0
    recoveries: int = 0
    attacks_detected: int = 0
    attacks_blocked: int = 0

    def update_health(self, delta: float):
        """Update health with bounds checking"""
        self.health = max(0.0, min(1.0, self.health + delta))
        if self.health < 0.3:
            self.state = NodeState.DEGRADED
        elif self.health < 0.1:
            self.state = NodeState.FAILED
        elif self.state in [NodeState.DEGRADED, NodeState.RECOVERING]:
            if self.health > 0.7:
                self.state = NodeState.HEALTHY


@dataclass
class SimulationEvent:
    """Event that occurred during simulation"""

    timestamp: float
    event_type: EventType
    node_id: Optional[int]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "node_id": self.node_id,
            "details": self.details,
        }


@dataclass
class SimulationMetrics:
    """Aggregated metrics from simulation"""

    total_nodes: int
    simulation_time: float
    events_generated: int
    nodes_by_state: Dict[NodeState, int]
    avg_health: float
    avg_load: float
    total_failures: int
    total_recoveries: int
    total_attacks_detected: int
    total_attacks_blocked: int
    avg_connections_per_node: float
    improvement_rate: float  # Rate of improvement over time


class LargeScaleSimulation:
    """
    Manages large-scale simulation of hundreds to thousands of nodes.

    Key capabilities:
    - Efficient simulation of large node populations
    - Various simulation modes (production, adversarial, stress, chaos)
    - Event generation and tracking
    - Performance metrics and analytics
    - Continuous improvement feedback loops
    - Scalable architecture
    """

    def __init__(
        self,
        node_count: int,
        mode: SimulationMode = SimulationMode.PRODUCTION_LIKE,
        area_size: Tuple[float, float] = (100.0, 100.0),
        enable_feedback: bool = True,
    ):
        """
        Initialize large-scale simulation.

        Args:
            node_count: Number of nodes to simulate
            mode: Simulation mode
            area_size: Size of simulation area (width, height)
            enable_feedback: Enable continuous improvement feedback
        """
        self.node_count = node_count
        self.mode = mode
        self.area_size = area_size
        self.enable_feedback = enable_feedback

        # Node storage
        self.nodes: Dict[int, SimulatedNode] = {}

        # Event tracking
        self.events: List[SimulationEvent] = []
        self.max_events = 100000  # Prevent memory overflow

        # Simulation state
        self.simulation_start_time = time.time()
        self.simulation_time = 0.0
        self.is_running = False

        # Metrics
        self.total_failures = 0
        self.total_recoveries = 0
        self.total_attacks = 0
        self.total_blocks = 0

        # Feedback mechanism for continuous improvement
        self.feedback_history: List[Dict[str, Any]] = []
        self.improvement_iterations = 0

        logger.info(
            f"Large-scale simulation initialized: {node_count} nodes, "
            f"mode={mode.value}, area={area_size}"
        )

    def initialize_nodes(self):
        """Initialize all simulated nodes"""
        logger.info(f"Initializing {self.node_count} nodes...")

        start_time = time.time()

        # Determine roles distribution
        roles = self._distribute_roles()

        for node_id in range(self.node_count):
            # Random position in area
            position = (
                random.uniform(0, self.area_size[0]),
                random.uniform(0, self.area_size[1]),
            )

            # Create node
            node = SimulatedNode(
                node_id=node_id,
                state=NodeState.HEALTHY,
                position=position,
                health=random.uniform(0.7, 1.0),
                energy=random.uniform(0.5, 1.0),
                load=random.uniform(0.1, 0.5),
                connections=[],
                role=roles[node_id],
                started_at=time.time(),
            )

            self.nodes[node_id] = node

            # Log event
            self._log_event(
                EventType.NODE_STARTED,
                node_id,
                {"role": node.role, "position": position},
            )

        # Establish connections between nodes
        self._establish_connections()

        init_time = time.time() - start_time
        logger.info(
            f"Initialized {self.node_count} nodes in {init_time:.2f}s "
            f"({self.node_count/init_time:.0f} nodes/sec)"
        )

    def _distribute_roles(self) -> List[str]:
        """Distribute roles across nodes"""
        roles = []

        # Different distributions based on mode
        if self.mode == SimulationMode.PRODUCTION_LIKE:
            # Typical production distribution
            distributions = [
                ("worker", 0.6),
                ("coordinator", 0.2),
                ("monitor", 0.1),
                ("defender", 0.1),
            ]
        elif self.mode == SimulationMode.ADVERSARIAL:
            # More defenders in adversarial mode
            distributions = [
                ("worker", 0.4),
                ("coordinator", 0.2),
                ("monitor", 0.2),
                ("defender", 0.2),
            ]
        else:
            # Balanced distribution for other modes
            distributions = [
                ("worker", 0.5),
                ("coordinator", 0.2),
                ("monitor", 0.15),
                ("defender", 0.15),
            ]

        # Assign roles based on distribution
        for role, percentage in distributions:
            count = int(self.node_count * percentage)
            roles.extend([role] * count)

        # Fill remaining with workers
        while len(roles) < self.node_count:
            roles.append("worker")

        # Shuffle for randomness
        random.shuffle(roles)

        return roles

    def _establish_connections(self):
        """Establish connections between nodes based on proximity"""
        # For large scale, use spatial hashing for efficiency
        connection_range = min(self.area_size[0], self.area_size[1]) / 10.0
        avg_connections_target = 5  # Target average connections per node

        for node_id, node in self.nodes.items():
            # Find nearby nodes
            nearby = self._find_nearby_nodes(node, connection_range)

            # Connect to some nearby nodes
            max_connections = random.randint(3, 8)
            connections = random.sample(
                nearby, min(len(nearby), max_connections)
            )

            node.connections = connections

    def _find_nearby_nodes(
        self, node: SimulatedNode, max_distance: float
    ) -> List[int]:
        """Find nodes within max_distance of given node"""
        nearby = []

        for other_id, other_node in self.nodes.items():
            if other_id == node.node_id:
                continue

            # Calculate distance
            dx = node.position[0] - other_node.position[0]
            dy = node.position[1] - other_node.position[1]
            distance = (dx * dx + dy * dy) ** 0.5

            if distance <= max_distance:
                nearby.append(other_id)

        return nearby

    def start_simulation(self):
        """Start the simulation"""
        self.is_running = True
        self.simulation_start_time = time.time()
        logger.info("Simulation started")

    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        logger.info("Simulation stopped")

    def simulate_step(self, delta_time: float = 1.0):
        """
        Simulate one time step.

        Args:
            delta_time: Time step duration (seconds)
        """
        if not self.is_running:
            return

        self.simulation_time += delta_time

        # Update all nodes
        for node in self.nodes.values():
            self._update_node(node, delta_time)

        # Generate events based on mode
        if self.mode == SimulationMode.ADVERSARIAL:
            self._simulate_attacks(delta_time)
        elif self.mode == SimulationMode.CHAOS_ENGINEERING:
            self._simulate_chaos(delta_time)
        elif self.mode == SimulationMode.STRESS_TEST:
            self._simulate_stress(delta_time)

        # Apply continuous improvement if enabled
        if self.enable_feedback and self.simulation_time % 10.0 < delta_time:
            self._apply_continuous_improvement()

    def _update_node(self, node: SimulatedNode, delta_time: float):
        """Update single node state"""
        # Natural health recovery
        if node.state == NodeState.DEGRADED:
            node.update_health(0.01 * delta_time)
            if node.health > 0.7:
                node.state = NodeState.RECOVERING

        elif node.state == NodeState.RECOVERING:
            node.update_health(0.02 * delta_time)
            if node.health > 0.9:
                node.state = NodeState.HEALTHY
                node.recoveries += 1
                self.total_recoveries += 1
                self._log_event(
                    EventType.NODE_RECOVERED,
                    node.node_id,
                    {"health": node.health},
                )

        # Energy decay based on load
        node.energy = max(0.0, node.energy - node.load * 0.01 * delta_time)

        # Random health fluctuations (wear and tear)
        node.update_health(random.uniform(-0.005, 0.005) * delta_time)

    def _simulate_attacks(self, delta_time: float):
        """Simulate attacks in adversarial mode"""
        # Random attacks on nodes
        attack_probability = 0.001 * delta_time  # 0.1% per second

        for node in self.nodes.values():
            if random.random() < attack_probability:
                self.total_attacks += 1
                node.attacks_detected += 1

                # Some attacks are blocked
                if random.random() < 0.7:  # 70% block rate
                    node.attacks_blocked += 1
                    self.total_blocks += 1
                    self._log_event(
                        EventType.ATTACK_BLOCKED,
                        node.node_id,
                        {"total_attacks": node.attacks_detected},
                    )
                else:
                    # Attack succeeded - damage node
                    node.update_health(-0.2)
                    if node.state == NodeState.FAILED:
                        node.failures += 1
                        self.total_failures += 1
                        self._log_event(
                            EventType.NODE_FAILED,
                            node.node_id,
                            {"cause": "attack"},
                        )
                    else:
                        self._log_event(
                            EventType.ATTACK_DETECTED,
                            node.node_id,
                            {"damage": 0.2},
                        )

    def _simulate_chaos(self, delta_time: float):
        """Simulate chaos engineering scenarios"""
        # Random failures
        failure_probability = 0.0005 * delta_time

        for node in self.nodes.values():
            if (
                node.state != NodeState.FAILED
                and random.random() < failure_probability
            ):
                node.state = NodeState.FAILED
                node.health = 0.1
                node.failures += 1
                self.total_failures += 1
                self._log_event(
                    EventType.NODE_FAILED, node.node_id, {"cause": "chaos"}
                )

    def _simulate_stress(self, delta_time: float):
        """Simulate stress test conditions"""
        # Increase load on all nodes
        for node in self.nodes.values():
            node.load = min(1.0, node.load + 0.01 * delta_time)

            # High load causes degradation
            if node.load > 0.8:
                node.update_health(-0.01 * delta_time)

    def _apply_continuous_improvement(self):
        """Apply continuous improvement based on feedback"""
        self.improvement_iterations += 1

        # Calculate current performance
        metrics = self.get_current_metrics()

        # Store feedback
        feedback = {
            "iteration": self.improvement_iterations,
            "timestamp": time.time(),
            "avg_health": metrics.avg_health,
            "failure_rate": (
                metrics.total_failures / self.node_count
                if self.node_count > 0
                else 0
            ),
            "block_rate": (
                metrics.total_attacks_blocked / max(1, self.total_attacks)
            ),
        }
        self.feedback_history.append(feedback)

        # Apply improvements based on metrics
        if metrics.avg_health < 0.6:
            # Low health - increase recovery rate
            self._apply_improvement("increase_recovery_rate")

        if metrics.total_failures > self.node_count * 0.1:
            # High failure rate - improve resilience
            self._apply_improvement("improve_resilience")

        logger.debug(
            f"Continuous improvement iteration {self.improvement_iterations}: "
            f"health={metrics.avg_health:.2f}, "
            f"failures={metrics.total_failures}"
        )

    def _apply_improvement(self, improvement_type: str):
        """Apply specific improvement to nodes"""
        if improvement_type == "increase_recovery_rate":
            for node in self.nodes.values():
                if node.state in [NodeState.DEGRADED, NodeState.RECOVERING]:
                    node.update_health(0.05)

        elif improvement_type == "improve_resilience":
            for node in self.nodes.values():
                if node.health < 0.5:
                    node.update_health(0.1)

    def _log_event(
        self,
        event_type: EventType,
        node_id: Optional[int],
        details: Dict[str, Any],
    ):
        """Log simulation event"""
        # Prevent memory overflow for very long simulations
        if len(self.events) >= self.max_events:
            # Keep only recent events
            self.events = self.events[-self.max_events // 2 :]

        event = SimulationEvent(
            timestamp=time.time(),
            event_type=event_type,
            node_id=node_id,
            details=details,
        )
        self.events.append(event)

    def get_current_metrics(self) -> SimulationMetrics:
        """Get current simulation metrics"""
        # Count nodes by state
        nodes_by_state = defaultdict(int)
        health_values = []
        load_values = []
        connection_counts = []
        total_attacks_detected = 0
        total_attacks_blocked = 0

        for node in self.nodes.values():
            nodes_by_state[node.state] += 1
            health_values.append(node.health)
            load_values.append(node.load)
            connection_counts.append(len(node.connections))
            total_attacks_detected += node.attacks_detected
            total_attacks_blocked += node.attacks_blocked

        # Calculate averages
        avg_health = statistics.mean(health_values) if health_values else 0.0
        avg_load = statistics.mean(load_values) if load_values else 0.0
        avg_connections = (
            statistics.mean(connection_counts) if connection_counts else 0.0
        )

        # Calculate improvement rate
        improvement_rate = 0.0
        if len(self.feedback_history) >= 2:
            recent = self.feedback_history[-1]["avg_health"]
            previous = self.feedback_history[-2]["avg_health"]
            improvement_rate = recent - previous

        return SimulationMetrics(
            total_nodes=self.node_count,
            simulation_time=self.simulation_time,
            events_generated=len(self.events),
            nodes_by_state=dict(nodes_by_state),
            avg_health=avg_health,
            avg_load=avg_load,
            total_failures=self.total_failures,
            total_recoveries=self.total_recoveries,
            total_attacks_detected=total_attacks_detected,
            total_attacks_blocked=total_attacks_blocked,
            avg_connections_per_node=avg_connections,
            improvement_rate=improvement_rate,
        )

    def get_node_health_distribution(self) -> Dict[str, int]:
        """Get distribution of node health levels"""
        distribution = {
            "critical": 0,  # 0.0-0.3
            "low": 0,  # 0.3-0.5
            "medium": 0,  # 0.5-0.7
            "good": 0,  # 0.7-0.9
            "excellent": 0,  # 0.9-1.0
        }

        for node in self.nodes.values():
            if node.health < 0.3:
                distribution["critical"] += 1
            elif node.health < 0.5:
                distribution["low"] += 1
            elif node.health < 0.7:
                distribution["medium"] += 1
            elif node.health < 0.9:
                distribution["good"] += 1
            else:
                distribution["excellent"] += 1

        return distribution

    def get_events(
        self, event_type: Optional[EventType] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get simulation events.

        Args:
            event_type: Filter by event type
            limit: Maximum events to return

        Returns:
            List of events as dictionaries
        """
        events = self.events

        # Filter by type if specified
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Get most recent
        events = events[-limit:]

        return [e.to_dict() for e in events]

    def get_feedback_history(self) -> List[Dict[str, Any]]:
        """Get continuous improvement feedback history"""
        return list(self.feedback_history)

    def export_results(self) -> Dict[str, Any]:
        """Export complete simulation results"""
        metrics = self.get_current_metrics()

        return {
            "simulation_config": {
                "node_count": self.node_count,
                "mode": self.mode.value,
                "area_size": self.area_size,
                "feedback_enabled": self.enable_feedback,
            },
            "metrics": {
                "total_nodes": metrics.total_nodes,
                "simulation_time": metrics.simulation_time,
                "events_generated": metrics.events_generated,
                "avg_health": metrics.avg_health,
                "avg_load": metrics.avg_load,
                "total_failures": metrics.total_failures,
                "total_recoveries": metrics.total_recoveries,
                "total_attacks_detected": metrics.total_attacks_detected,
                "total_attacks_blocked": metrics.total_attacks_blocked,
                "improvement_rate": metrics.improvement_rate,
            },
            "node_states": {
                state.value: count
                for state, count in metrics.nodes_by_state.items()
            },
            "health_distribution": self.get_node_health_distribution(),
            "improvement_iterations": self.improvement_iterations,
        }
