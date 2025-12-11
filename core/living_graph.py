"""
Living Graph Environment - Dynamic IT System Simulation

This module implements a dynamic, graph-based environment that models modern IT systems.
The graph represents hardware, software, and logical components with realistic dependencies,
resource dynamics, and cascading failures.
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

logger = logging.getLogger("living_graph")


class NodeType(str, Enum):
    """Types of system components in the graph"""

    # Hardware nodes
    SERVER = "server"
    VM = "vm"
    ROUTER = "router"
    SWITCH = "switch"

    # Software nodes
    SERVICE = "service"
    APPLICATION = "application"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"

    # Logical nodes
    SUBNET = "subnet"
    SECURITY_GROUP = "security_group"
    CLOUD_REGION = "cloud_region"


class EdgeType(str, Enum):
    """Types of relationships between nodes"""

    NETWORK_CONNECTIVITY = "network_connectivity"
    SERVICE_DEPENDENCY = "service_dependency"
    DATA_FLOW = "data_flow"
    HOSTED_ON = "hosted_on"


class HealthStatus(str, Enum):
    """Health status of nodes"""

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class NodeAttributes:
    """Dynamic attributes for graph nodes"""

    # Resource attributes
    cpu_load: float = 0.0  # 0.0 to 1.0
    memory_usage: float = 0.0  # 0.0 to 1.0
    disk_usage: float = 0.0  # 0.0 to 1.0

    # Health attributes
    health_status: HealthStatus = HealthStatus.OK
    uptime: float = 0.0  # seconds
    last_failure_time: Optional[float] = None

    # Security attributes
    security_patch_level: int = 0
    open_ports: Set[int] = field(default_factory=set)
    active_threat_score: float = 0.0  # 0.0 to 1.0
    vulnerability_count: int = 0

    # Energy/stress attributes
    stress_level: float = 0.0  # 0.0 to 1.0
    energy_provided: float = 1.0  # Energy available to agents on this node
    agent_capacity: int = 10  # Max agents this node can host
    current_agents: int = 0

    # Metadata
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class EdgeAttributes:
    """Dynamic attributes for graph edges"""

    # Network attributes
    latency_ms: float = 0.0
    bandwidth_mbps: float = 1000.0
    bandwidth_utilization: float = 0.0  # 0.0 to 1.0
    packet_drop_rate: float = 0.0  # 0.0 to 1.0

    # Security attributes
    firewall_rule_status: str = "allow"  # allow, deny, inspect
    encrypted: bool = False

    # Flow attributes
    active: bool = True
    last_activity: float = field(default_factory=time.time)

    # Metadata
    created_at: float = field(default_factory=time.time)


@dataclass
class GraphNode:
    """A node in the living graph"""

    node_id: str
    node_type: NodeType
    name: str
    attributes: NodeAttributes
    position: Tuple[float, float] = (0.0, 0.0)  # For visualization
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_stress(self, delta: float):
        """Update stress level based on resource usage"""
        resource_stress = (
            self.attributes.cpu_load * 0.4
            + self.attributes.memory_usage * 0.3
            + self.attributes.disk_usage * 0.2
            + self.attributes.active_threat_score * 0.1
        )
        self.attributes.stress_level = max(
            0.0,
            min(
                1.0,
                self.attributes.stress_level + delta + resource_stress * 0.1,
            ),
        )

    def can_host_agent(self) -> bool:
        """Check if node can host another agent"""
        return (
            self.attributes.current_agents < self.attributes.agent_capacity
            and self.attributes.health_status == HealthStatus.OK
            and self.attributes.stress_level < 0.8
        )

    def provide_energy(self) -> float:
        """Calculate energy this node provides to agents"""
        if self.attributes.health_status == HealthStatus.FAILED:
            return 0.0

        # High-CPU nodes provide more energy but increase stress
        base_energy = self.attributes.energy_provided
        cpu_bonus = self.attributes.cpu_load * 0.5  # More CPU = more energy
        stress_penalty = (
            self.attributes.stress_level * 0.3
        )  # But more stress reduces it

        return max(0.0, base_energy + cpu_bonus - stress_penalty)


@dataclass
class GraphEdge:
    """An edge in the living graph"""

    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    attributes: EdgeAttributes
    metadata: Dict[str, Any] = field(default_factory=dict)


class LivingGraph:
    """
    Dynamic, graph-based environment representing an IT system.

    The graph is mutable and responds to events, resource usage, and failures.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the living graph.

        Args:
            seed: Random seed for reproducibility
        """
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(
            set
        )  # node_id -> set of connected node_ids
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(
            set
        )  # For incoming edges

        self.simulation_time = 0.0
        self.event_history: List[Dict[str, Any]] = []

        if seed is not None:
            np.random.seed(seed)

        logger.info("Living graph initialized")

    def add_node(
        self,
        node_type: NodeType,
        name: str,
        position: Optional[Tuple[float, float]] = None,
        attributes: Optional[NodeAttributes] = None,
    ) -> str:
        """
        Add a node to the graph.

        Args:
            node_type: Type of the node
            name: Human-readable name
            position: Position for visualization
            attributes: Node attributes (defaults to NodeAttributes())

        Returns:
            node_id: Unique identifier for the node
        """
        node_id = str(uuid.uuid4())

        if position is None:
            # Random position for visualization
            position = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))

        if attributes is None:
            attributes = NodeAttributes()

        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            attributes=attributes,
            position=position,
        )

        self.nodes[node_id] = node
        logger.debug(f"Added node {name} ({node_type}) with id {node_id}")

        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        attributes: Optional[EdgeAttributes] = None,
    ) -> str:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            attributes: Edge attributes (defaults to EdgeAttributes())

        Returns:
            edge_id: Unique identifier for the edge
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist")

        edge_id = str(uuid.uuid4())

        if attributes is None:
            attributes = EdgeAttributes()

        edge = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            attributes=attributes,
        )

        self.edges[edge_id] = edge
        self.adjacency[source_id].add(target_id)
        self.reverse_adjacency[target_id].add(source_id)

        logger.debug(
            f"Added edge from {source_id} to {target_id} ({edge_type})"
        )

        return edge_id

    def remove_node(self, node_id: str):
        """Remove a node and all its edges from the graph"""
        if node_id not in self.nodes:
            return

        # Remove all edges connected to this node
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove from adjacency lists
        if node_id in self.adjacency:
            del self.adjacency[node_id]
        if node_id in self.reverse_adjacency:
            del self.reverse_adjacency[node_id]

        # Remove node
        del self.nodes[node_id]
        logger.debug(f"Removed node {node_id}")

    def remove_edge(self, edge_id: str):
        """Remove an edge from the graph"""
        if edge_id not in self.edges:
            return

        edge = self.edges[edge_id]
        self.adjacency[edge.source_id].discard(edge.target_id)
        self.reverse_adjacency[edge.target_id].discard(edge.source_id)

        del self.edges[edge_id]
        logger.debug(f"Removed edge {edge_id}")

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors (outgoing edges) of a node"""
        return list(self.adjacency.get(node_id, set()))

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all dependencies (incoming edges) of a node"""
        return list(self.reverse_adjacency.get(node_id, set()))

    def update(self, delta_time: float = 1.0):
        """
        Update graph state - simulate resource dynamics and failures.

        Args:
            delta_time: Time step for simulation
        """
        self.simulation_time += delta_time

        # Update all nodes
        for node in self.nodes.values():
            self._update_node(node, delta_time)

        # Check for cascading failures
        self._propagate_failures()

        # Update all edges
        for edge in self.edges.values():
            self._update_edge(edge, delta_time)

    def _update_node(self, node: GraphNode, delta_time: float):
        """Update a single node's state"""
        # Natural resource variation
        node.attributes.cpu_load += np.random.normal(0, 0.02)
        node.attributes.cpu_load = max(0.0, min(1.0, node.attributes.cpu_load))

        node.attributes.memory_usage += np.random.normal(0, 0.01)
        node.attributes.memory_usage = max(
            0.0, min(1.0, node.attributes.memory_usage)
        )

        # Update stress
        node.update_stress(0.0)

        # High stress can degrade health
        if node.attributes.stress_level > 0.9 and np.random.random() < 0.01:
            if node.attributes.health_status == HealthStatus.OK:
                node.attributes.health_status = HealthStatus.DEGRADED
                logger.warning(f"Node {node.name} degraded due to high stress")

        # Degraded nodes can fail
        if (
            node.attributes.health_status == HealthStatus.DEGRADED
            and np.random.random() < 0.005
        ):
            node.attributes.health_status = HealthStatus.FAILED
            node.attributes.last_failure_time = self.simulation_time
            logger.error(f"Node {node.name} failed")
            self._record_event(
                "node_failure",
                {"node_id": node.node_id, "node_name": node.name},
            )

        # Update uptime
        if node.attributes.health_status != HealthStatus.FAILED:
            node.attributes.uptime += delta_time

        node.attributes.last_updated = self.simulation_time

    def _update_edge(self, edge: GraphEdge, delta_time: float):
        """Update a single edge's state"""
        # Natural network variation
        edge.attributes.latency_ms += np.random.normal(0, 1.0)
        edge.attributes.latency_ms = max(0.0, edge.attributes.latency_ms)

        edge.attributes.bandwidth_utilization += np.random.normal(0, 0.02)
        edge.attributes.bandwidth_utilization = max(
            0.0, min(1.0, edge.attributes.bandwidth_utilization)
        )

        # High utilization can cause packet drops
        if edge.attributes.bandwidth_utilization > 0.9:
            edge.attributes.packet_drop_rate = min(
                0.5, edge.attributes.packet_drop_rate + 0.01
            )
        else:
            edge.attributes.packet_drop_rate = max(
                0.0, edge.attributes.packet_drop_rate - 0.01
            )

    def _propagate_failures(self):
        """Simulate cascading failures through dependencies"""
        for node_id, node in list(self.nodes.items()):
            if node.attributes.health_status == HealthStatus.FAILED:
                # Propagate stress to dependent nodes
                for dependent_id in self.get_neighbors(node_id):
                    if dependent_id in self.nodes:
                        dependent = self.nodes[dependent_id]
                        dependent.attributes.stress_level = min(
                            1.0, dependent.attributes.stress_level + 0.2
                        )
                        logger.debug(
                            f"Failure of {node.name} increased stress on {dependent.name}"
                        )

    def simulate_ddos_attack(
        self, target_node_id: str, intensity: float = 0.8
    ):
        """
        Simulate a DDoS attack on a node.

        Args:
            target_node_id: Node to attack
            intensity: Attack intensity (0.0 to 1.0)
        """
        if target_node_id not in self.nodes:
            return

        node = self.nodes[target_node_id]

        # Saturate incoming edges
        for source_id in self.get_dependencies(target_node_id):
            for edge_id, edge in self.edges.items():
                if (
                    edge.source_id == source_id
                    and edge.target_id == target_node_id
                ):
                    edge.attributes.bandwidth_utilization = min(
                        1.0, edge.attributes.bandwidth_utilization + intensity
                    )
                    edge.attributes.packet_drop_rate = min(
                        0.9, edge.attributes.packet_drop_rate + intensity * 0.5
                    )

        # Increase node stress and CPU
        node.attributes.stress_level = min(
            1.0, node.attributes.stress_level + intensity
        )
        node.attributes.cpu_load = min(
            1.0, node.attributes.cpu_load + intensity * 0.7
        )
        node.attributes.active_threat_score = min(
            1.0, node.attributes.active_threat_score + intensity
        )

        self._record_event(
            "ddos_attack",
            {
                "target_node_id": target_node_id,
                "target_name": node.name,
                "intensity": intensity,
            },
        )

        logger.warning(
            f"DDoS attack on {node.name} with intensity {intensity}"
        )

    def simulate_resource_exhaustion(
        self, node_id: str, resource: str = "cpu"
    ):
        """
        Simulate resource exhaustion on a node.

        Args:
            node_id: Node to exhaust
            resource: Resource type ('cpu', 'memory', 'disk')
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        if resource == "cpu":
            node.attributes.cpu_load = 1.0
        elif resource == "memory":
            node.attributes.memory_usage = 1.0
        elif resource == "disk":
            node.attributes.disk_usage = 1.0

        node.attributes.stress_level = min(
            1.0, node.attributes.stress_level + 0.5
        )

        self._record_event(
            "resource_exhaustion",
            {"node_id": node_id, "node_name": node.name, "resource": resource},
        )

        logger.warning(f"Resource exhaustion ({resource}) on {node.name}")

    def add_vulnerability(self, node_id: str, severity: float = 0.5):
        """
        Add a vulnerability to a node.

        Args:
            node_id: Node to add vulnerability to
            severity: Vulnerability severity (0.0 to 1.0)
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        node.attributes.vulnerability_count += 1
        node.attributes.active_threat_score = min(
            1.0, node.attributes.active_threat_score + severity * 0.3
        )

        self._record_event(
            "vulnerability_added",
            {"node_id": node_id, "node_name": node.name, "severity": severity},
        )

        logger.info(
            f"Added vulnerability (severity {severity}) to {node.name}"
        )

    def _record_event(self, event_type: str, data: Dict[str, Any]):
        """Record an event in history"""
        event = {
            "time": self.simulation_time,
            "type": event_type,
            "data": data,
        }
        self.event_history.append(event)

    def get_node_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a node for agent sensing"""
        if node_id not in self.nodes:
            return None

        node = self.nodes[node_id]
        return {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "name": node.name,
            "cpu_load": node.attributes.cpu_load,
            "memory_usage": node.attributes.memory_usage,
            "disk_usage": node.attributes.disk_usage,
            "health_status": node.attributes.health_status.value,
            "stress_level": node.attributes.stress_level,
            "active_threat_score": node.attributes.active_threat_score,
            "energy_provided": node.provide_energy(),
            "can_host_agent": node.can_host_agent(),
        }

    def get_edge_state(
        self, source_id: str, target_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the current state of an edge for agent sensing"""
        for edge in self.edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                return {
                    "edge_id": edge.edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type.value,
                    "latency_ms": edge.attributes.latency_ms,
                    "bandwidth_utilization": edge.attributes.bandwidth_utilization,
                    "packet_drop_rate": edge.attributes.packet_drop_rate,
                    "active": edge.attributes.active,
                }
        return None

    def query_nodes(
        self,
        node_type: Optional[NodeType] = None,
        health_status: Optional[HealthStatus] = None,
        max_stress: Optional[float] = None,
    ) -> List[str]:
        """
        Query nodes by criteria.

        Args:
            node_type: Filter by node type
            health_status: Filter by health status
            max_stress: Filter by maximum stress level

        Returns:
            List of matching node IDs
        """
        results = []
        for node_id, node in self.nodes.items():
            if node_type and node.node_type != node_type:
                continue
            if (
                health_status
                and node.attributes.health_status != health_status
            ):
                continue
            if (
                max_stress is not None
                and node.attributes.stress_level > max_stress
            ):
                continue
            results.append(node_id)
        return results

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the graph state"""
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)

        node_types = defaultdict(int)
        health_counts = defaultdict(int)

        avg_cpu = 0.0
        avg_memory = 0.0
        avg_stress = 0.0
        avg_threat = 0.0

        for node in self.nodes.values():
            node_types[node.node_type.value] += 1
            health_counts[node.attributes.health_status.value] += 1
            avg_cpu += node.attributes.cpu_load
            avg_memory += node.attributes.memory_usage
            avg_stress += node.attributes.stress_level
            avg_threat += node.attributes.active_threat_score

        if total_nodes > 0:
            avg_cpu /= total_nodes
            avg_memory /= total_nodes
            avg_stress /= total_nodes
            avg_threat /= total_nodes

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_types": dict(node_types),
            "health_status": dict(health_counts),
            "avg_cpu_load": avg_cpu,
            "avg_memory_usage": avg_memory,
            "avg_stress_level": avg_stress,
            "avg_threat_score": avg_threat,
            "simulation_time": self.simulation_time,
            "total_events": len(self.event_history),
        }
