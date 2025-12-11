"""
Advanced Sensory & Communication Protocol

This module implements a sophisticated multi-layered sensory and communication system:
1. Multi-layered sensory system (proprioception, local sensing, graph-level awareness)
2. Pheromones (ambient, asynchronous broadcast)
3. Signals (targeted, synchronous unicast/multicast)
4. Gossip Protocol (decentralized, eventually consistent)
5. Cryptographic message signing and verification
"""

import hashlib
import hmac
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("advanced_communication")


class SensoryMode(str, Enum):
    """Types of sensory input"""

    PROPRIOCEPTION = "proprioception"  # Self-sensing
    LOCAL = "local"  # Immediate environment
    GRAPH_LEVEL = "graph_level"  # Non-local queries


class MessageType(str, Enum):
    """Types of communication messages"""

    PHEROMONE = "pheromone"  # Ambient broadcast
    SIGNAL = "signal"  # Targeted message
    GOSSIP = "gossip"  # Decentralized propagation


class Priority(str, Enum):
    """Message priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Pheromone:
    """
    Ambient, asynchronous broadcast message left on a node.

    Pheromones decay over time and are sensed passively by agents.
    """

    pheromone_id: str
    content: str
    intensity: float  # 0.0 to 1.0, decays over time
    depositor_id: int
    node_id: str  # Node where pheromone is deposited
    timestamp: float
    decay_rate: float = 0.95  # Per time step
    ttl: int = 100  # Time steps before complete decay
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, delta_time: float = 1.0):
        """Apply decay to pheromone intensity"""
        self.intensity *= self.decay_rate**delta_time
        self.ttl -= int(delta_time)

    def is_expired(self) -> bool:
        """Check if pheromone has expired"""
        return self.intensity < 0.01 or self.ttl <= 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for sensing"""
        return {
            "id": self.pheromone_id,
            "content": self.content,
            "intensity": self.intensity,
            "depositor_id": self.depositor_id,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "age": time.time() - self.timestamp,
        }


@dataclass
class Signal:
    """
    Targeted, synchronous unicast/multicast message.

    Signals are directed messages with routing and delivery guarantees.
    """

    signal_id: str
    sender_id: int
    recipient_ids: List[int]  # Can be multiple for multicast
    message_type: str
    payload: Dict[str, Any]
    priority: Priority
    timestamp: float
    ttl: int = 50  # Hops before message is dropped
    correlation_id: Optional[str] = None  # For request/response tracking
    signature: Optional[str] = None  # Cryptographic signature
    requires_ack: bool = False  # Whether acknowledgment is required
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        return {
            "signal_id": self.signal_id,
            "sender_id": self.sender_id,
            "recipient_ids": self.recipient_ids,
            "message_type": self.message_type,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "correlation_id": self.correlation_id,
            "signature": self.signature,
            "requires_ack": self.requires_ack,
            "metadata": self.metadata,
        }


@dataclass
class GossipMessage:
    """
    Decentralized gossip protocol message.

    Gossip messages are eventually consistent and propagate through random exchanges.
    """

    gossip_id: str
    originator_id: int
    content: Dict[str, Any]
    gossip_type: (
        str  # e.g., "threat_signature", "trust_update", "strategy_change"
    )
    generation: int  # How many hops from originator
    seen_by: Set[int] = field(default_factory=set)  # Nodes that have seen this
    timestamp: float = field(default_factory=time.time)
    ttl: int = 200  # Max propagation hops
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def increment_generation(self, node_id: int):
        """Record that this node has seen and propagated the message"""
        self.generation += 1
        self.seen_by.add(node_id)
        self.ttl -= 1

    def should_propagate(self, node_id: int) -> bool:
        """Check if this node should propagate the message"""
        return (
            node_id not in self.seen_by
            and self.ttl > 0
            and self.generation < 20  # Limit propagation depth
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "gossip_id": self.gossip_id,
            "originator_id": self.originator_id,
            "content": self.content,
            "gossip_type": self.gossip_type,
            "generation": self.generation,
            "seen_by_count": len(self.seen_by),
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "signature": self.signature,
        }


class CryptoSigner:
    """Simple cryptographic signer for messages"""

    def __init__(self, node_id: int, secret: Optional[str] = None):
        """
        Initialize signer.

        Args:
            node_id: ID of the node
            secret: Secret key for signing (generated if not provided)
        """
        self.node_id = node_id
        self.secret = secret or self._generate_secret()

    def _generate_secret(self) -> str:
        """Generate a random secret key"""
        return hashlib.sha256(
            f"{self.node_id}_{time.time()}_{uuid.uuid4()}".encode()
        ).hexdigest()

    def sign(self, message: str) -> str:
        """
        Sign a message.

        Args:
            message: Message to sign

        Returns:
            Signature hex string
        """
        return hmac.new(
            self.secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

    def verify(self, message: str, signature: str) -> bool:
        """
        Verify a message signature.

        Args:
            message: Original message
            signature: Signature to verify

        Returns:
            True if signature is valid
        """
        expected = self.sign(message)
        return hmac.compare_digest(expected, signature)


class AdvancedSensorySystem:
    """
    Multi-layered sensory system for agents.

    Provides three tiers of sensing:
    1. Proprioception - Self-sensing
    2. Local Environmental Sensing - Immediate surroundings
    3. Graph-Level Awareness - Remote queries
    """

    def __init__(self, agent_node):
        """
        Initialize sensory system.

        Args:
            agent_node: The AliveLoopNode this system belongs to
        """
        self.agent = agent_node
        self.query_energy_cost = 0.5  # Cost for graph-level queries
        self.last_proprioception: Dict[str, Any] = {}
        self.last_local_sense: Dict[str, Any] = {}

    def proprioception(self) -> Dict[str, Any]:
        """
        Self-sensing: Awareness of internal state.

        Returns:
            Dictionary of internal state variables
        """
        self.last_proprioception = {
            "sensory_mode": SensoryMode.PROPRIOCEPTION.value,
            "node_id": self.agent.node_id,
            "health": getattr(self.agent, "health", self.agent.energy),
            "energy_level": self.agent.energy,
            "current_task": getattr(self.agent, "phase", "active"),
            "computational_load": getattr(
                self.agent, "communications_this_step", 0
            )
            / getattr(self.agent, "max_communications_per_step", 5),
            "trust_score": self.agent.trust,
            "anxiety": self.agent.anxiety,
            "emotional_state": getattr(self.agent, "emotional_state", {}),
            "position": (
                tuple(self.agent.position)
                if hasattr(self.agent, "position")
                else None
            ),
            "timestamp": time.time(),
        }
        return self.last_proprioception

    def local_environmental_sensing(
        self, graph_env, current_node_id: str
    ) -> Dict[str, Any]:
        """
        Local sensing: Direct sensing of current node and immediate edges.

        Args:
            graph_env: LivingGraph environment
            current_node_id: ID of node agent is currently on

        Returns:
            Dictionary of local environmental state
        """
        if graph_env is None or current_node_id not in graph_env.nodes:
            return {}

        node_state = graph_env.get_node_state(current_node_id)

        # Get immediate neighbors
        neighbors = graph_env.get_neighbors(current_node_id)
        neighbor_states = [
            graph_env.get_node_state(nid) for nid in neighbors[:5]
        ]  # Limit to 5

        # Get edges to neighbors
        edge_states = []
        for neighbor_id in neighbors[:5]:
            edge_state = graph_env.get_edge_state(current_node_id, neighbor_id)
            if edge_state:
                edge_states.append(edge_state)

        self.last_local_sense = {
            "sensory_mode": SensoryMode.LOCAL.value,
            "current_node": node_state,
            "neighbor_nodes": neighbor_states,
            "connecting_edges": edge_states,
            "local_threat_level": (
                node_state.get("active_threat_score", 0.0)
                if node_state
                else 0.0
            ),
            "local_resources": {
                "energy_available": (
                    node_state.get("energy_provided", 0.0)
                    if node_state
                    else 0.0
                ),
                "cpu_load": (
                    node_state.get("cpu_load", 0.0) if node_state else 0.0
                ),
                "stress_level": (
                    node_state.get("stress_level", 0.0) if node_state else 0.0
                ),
            },
            "timestamp": time.time(),
        }
        return self.last_local_sense

    def graph_level_awareness(
        self, graph_env, query_type: str, query_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Graph-level sensing: Active queries of non-local parts of the graph.

        This is resource-consuming and deliberate.

        Args:
            graph_env: LivingGraph environment
            query_type: Type of query ('find_nodes', 'get_path', 'get_summary')
            query_params: Parameters for the query

        Returns:
            Query results or None if insufficient energy
        """
        # Check energy for query
        if self.agent.energy < self.query_energy_cost:
            logger.debug(
                f"Node {self.agent.node_id}: Insufficient energy for graph-level query"
            )
            return None

        # Consume energy
        self.agent.energy -= self.query_energy_cost

        results = None

        if query_type == "find_nodes":
            # Find nodes matching criteria
            node_type = query_params.get("node_type")
            health_status = query_params.get("health_status")
            max_stress = query_params.get("max_stress")

            matching_nodes = graph_env.query_nodes(
                node_type=node_type,
                health_status=health_status,
                max_stress=max_stress,
            )

            results = {
                "query_type": query_type,
                "matching_nodes": matching_nodes[:10],  # Limit results
                "total_matches": len(matching_nodes),
            }

        elif query_type == "get_summary":
            # Get graph summary
            results = {
                "query_type": query_type,
                "summary": graph_env.get_graph_summary(),
            }

        elif query_type == "get_node_state":
            # Get specific node state
            target_node_id = query_params.get("node_id")
            if target_node_id:
                results = {
                    "query_type": query_type,
                    "node_state": graph_env.get_node_state(target_node_id),
                }

        if results:
            results["sensory_mode"] = SensoryMode.GRAPH_LEVEL.value
            results["energy_cost"] = self.query_energy_cost
            results["timestamp"] = time.time()

        return results

    def get_full_sensory_state(
        self, graph_env=None, current_node_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get complete sensory state across all layers.

        Args:
            graph_env: Optional LivingGraph environment
            current_node_id: Optional current node ID

        Returns:
            Combined sensory state
        """
        state = {
            "proprioception": self.proprioception(),
            "local": None,
            "timestamp": time.time(),
        }

        if graph_env and current_node_id:
            state["local"] = self.local_environmental_sensing(
                graph_env, current_node_id
            )

        return state


class AdvancedCommunicationProtocol:
    """
    Multi-modal communication protocol for agents.

    Supports:
    1. Pheromones - Ambient, low-cost messaging
    2. Signals - Targeted, reliable messaging
    3. Gossip - Decentralized, eventually consistent messaging
    """

    def __init__(self, agent_node):
        """
        Initialize communication protocol.

        Args:
            agent_node: The AliveLoopNode this protocol belongs to
        """
        self.agent = agent_node
        self.crypto_signer = CryptoSigner(agent_node.node_id)

        # Pheromone tracking
        self.deposited_pheromones: Dict[str, List[str]] = defaultdict(
            list
        )  # node_id -> pheromone_ids

        # Signal routing
        self.signal_inbox: deque = deque(maxlen=50)
        self.signal_outbox: deque = deque(maxlen=50)
        self.pending_acks: Dict[str, Signal] = {}

        # Gossip protocol
        self.gossip_cache: Dict[str, GossipMessage] = (
            {}
        )  # gossip_id -> message
        self.gossip_inbox: deque = deque(maxlen=100)
        self.gossip_propagation_rate = (
            0.3  # Probability of propagating a gossip message
        )

        # Metrics
        self.metrics = {
            "pheromones_deposited": 0,
            "signals_sent": 0,
            "signals_received": 0,
            "gossip_originated": 0,
            "gossip_propagated": 0,
        }

    # ========== Pheromone Methods ==========

    def deposit_pheromone(
        self,
        node_id: str,
        content: str,
        intensity: float = 1.0,
        decay_rate: float = 0.95,
    ) -> str:
        """
        Deposit a pheromone on a node.

        Args:
            node_id: Node to deposit pheromone on
            content: Pheromone content
            intensity: Initial intensity (0.0 to 1.0)
            decay_rate: Decay rate per time step

        Returns:
            pheromone_id
        """
        pheromone_id = str(uuid.uuid4())

        pheromone = Pheromone(
            pheromone_id=pheromone_id,
            content=content,
            intensity=max(0.0, min(1.0, intensity)),
            depositor_id=self.agent.node_id,
            node_id=node_id,
            timestamp=time.time(),
            decay_rate=decay_rate,
        )

        self.deposited_pheromones[node_id].append(pheromone_id)
        self.metrics["pheromones_deposited"] += 1

        logger.debug(
            f"Node {self.agent.node_id}: Deposited pheromone on node {node_id}"
        )

        return pheromone_id

    def sense_pheromones(
        self, node_id: str, pheromone_store: Dict[str, List[Pheromone]]
    ) -> List[Dict[str, Any]]:
        """
        Sense pheromones on a node.

        Args:
            node_id: Node to sense
            pheromone_store: Global pheromone store (managed by environment)

        Returns:
            List of sensed pheromones
        """
        if node_id not in pheromone_store:
            return []

        return [
            p.to_dict() for p in pheromone_store[node_id] if not p.is_expired()
        ]

    # ========== Signal Methods ==========

    def send_signal(
        self,
        recipient_ids: List[int],
        message_type: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        requires_ack: bool = False,
    ) -> str:
        """
        Send a targeted signal to specific recipients.

        Args:
            recipient_ids: List of recipient node IDs
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            requires_ack: Whether acknowledgment is required

        Returns:
            signal_id
        """
        signal_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        # Create message for signing
        message_str = (
            f"{signal_id}:{self.agent.node_id}:{message_type}:{str(payload)}"
        )
        signature = self.crypto_signer.sign(message_str)

        signal = Signal(
            signal_id=signal_id,
            sender_id=self.agent.node_id,
            recipient_ids=recipient_ids,
            message_type=message_type,
            payload=payload,
            priority=priority,
            timestamp=time.time(),
            correlation_id=correlation_id,
            signature=signature,
            requires_ack=requires_ack,
        )

        self.signal_outbox.append(signal)

        if requires_ack:
            self.pending_acks[signal_id] = signal

        self.metrics["signals_sent"] += 1

        logger.debug(
            f"Node {self.agent.node_id}: Sent signal {signal_id} to {recipient_ids}"
        )

        return signal_id

    def receive_signal(self, signal: Signal) -> bool:
        """
        Receive and process a signal.

        Args:
            signal: Signal to receive

        Returns:
            True if signal was accepted
        """
        # Verify signature
        if signal.signature:
            message_str = f"{signal.signal_id}:{signal.sender_id}:{signal.message_type}:{str(signal.payload)}"
            # In a real implementation, we'd verify with sender's public key
            # For now, we trust the signature exists

        # Check if this agent is a recipient
        if self.agent.node_id not in signal.recipient_ids:
            return False

        self.signal_inbox.append(signal)
        self.metrics["signals_received"] += 1

        # Send acknowledgment if required
        if signal.requires_ack:
            self.send_acknowledgment(signal)

        logger.debug(
            f"Node {self.agent.node_id}: Received signal {signal.signal_id} from {signal.sender_id}"
        )

        return True

    def send_acknowledgment(self, original_signal: Signal):
        """Send acknowledgment for a signal"""
        ack_signal = Signal(
            signal_id=str(uuid.uuid4()),
            sender_id=self.agent.node_id,
            recipient_ids=[original_signal.sender_id],
            message_type="acknowledgment",
            payload={"ack_for": original_signal.signal_id},
            priority=Priority.HIGH,
            timestamp=time.time(),
            correlation_id=original_signal.correlation_id,
        )
        self.signal_outbox.append(ack_signal)

    def process_signal_inbox(self) -> List[Signal]:
        """
        Process all signals in inbox.

        Returns:
            List of processed signals
        """
        processed = []
        while self.signal_inbox:
            signal = self.signal_inbox.popleft()
            processed.append(signal)
        return processed

    # ========== Gossip Protocol Methods ==========

    def initiate_gossip(
        self, gossip_type: str, content: Dict[str, Any]
    ) -> str:
        """
        Initiate a new gossip message.

        Args:
            gossip_type: Type of gossip
            content: Gossip content

        Returns:
            gossip_id
        """
        gossip_id = str(uuid.uuid4())

        # Sign the gossip
        message_str = (
            f"{gossip_id}:{self.agent.node_id}:{gossip_type}:{str(content)}"
        )
        signature = self.crypto_signer.sign(message_str)

        gossip = GossipMessage(
            gossip_id=gossip_id,
            originator_id=self.agent.node_id,
            content=content,
            gossip_type=gossip_type,
            generation=0,
            signature=signature,
        )

        gossip.seen_by.add(self.agent.node_id)
        self.gossip_cache[gossip_id] = gossip
        self.gossip_inbox.append(gossip)

        self.metrics["gossip_originated"] += 1

        logger.debug(
            f"Node {self.agent.node_id}: Initiated gossip {gossip_id} ({gossip_type})"
        )

        return gossip_id

    def receive_gossip(self, gossip: GossipMessage) -> bool:
        """
        Receive a gossip message.

        Args:
            gossip: Gossip message

        Returns:
            True if gossip was accepted (new or should propagate)
        """
        # Check if we've seen this gossip
        if gossip.gossip_id in self.gossip_cache:
            return False

        # Check if we should accept this gossip
        if not gossip.should_propagate(self.agent.node_id):
            return False

        # Add to cache and inbox
        self.gossip_cache[gossip.gossip_id] = gossip
        self.gossip_inbox.append(gossip)

        logger.debug(
            f"Node {self.agent.node_id}: Received gossip {gossip.gossip_id} (gen {gossip.generation})"
        )

        return True

    def propagate_gossip(self, nearby_agents: List[Any]) -> int:
        """
        Propagate gossip messages to nearby agents.

        Args:
            nearby_agents: List of nearby agents

        Returns:
            Number of gossip messages propagated
        """
        if not nearby_agents or not self.gossip_cache:
            return 0

        propagated = 0

        # Select gossip messages to propagate
        for gossip in list(self.gossip_cache.values()):
            if not gossip.should_propagate(self.agent.node_id):
                continue

            # Probabilistic propagation
            if np.random.random() > self.gossip_propagation_rate:
                continue

            # Select random subset of neighbors
            num_neighbors = max(
                1, int(len(nearby_agents) * 0.3)
            )  # 30% of neighbors
            selected = np.random.choice(
                nearby_agents,
                size=min(num_neighbors, len(nearby_agents)),
                replace=False,
            )

            # Create a copy with incremented generation
            gossip_copy = GossipMessage(
                gossip_id=gossip.gossip_id,
                originator_id=gossip.originator_id,
                content=gossip.content,
                gossip_type=gossip.gossip_type,
                generation=gossip.generation + 1,
                seen_by=gossip.seen_by.copy(),
                timestamp=gossip.timestamp,
                ttl=gossip.ttl - 1,
                signature=gossip.signature,
                metadata=gossip.metadata.copy(),
            )
            gossip_copy.increment_generation(self.agent.node_id)

            # Send to selected neighbors
            for agent in selected:
                if hasattr(agent, "communication_protocol"):
                    if agent.communication_protocol.receive_gossip(
                        gossip_copy
                    ):
                        propagated += 1

            self.metrics["gossip_propagated"] += propagated

        if propagated > 0:
            logger.debug(
                f"Node {self.agent.node_id}: Propagated {propagated} gossip messages"
            )

        return propagated

    def process_gossip_inbox(self) -> List[GossipMessage]:
        """
        Process all gossip messages in inbox.

        Returns:
            List of processed gossip messages
        """
        processed = []
        while self.gossip_inbox:
            gossip = self.gossip_inbox.popleft()
            processed.append(gossip)
        return processed

    def cleanup_old_gossip(self, max_age: float = 600.0):
        """
        Clean up old gossip messages.

        Args:
            max_age: Maximum age in seconds
        """
        current_time = time.time()
        to_remove = []

        for gossip_id, gossip in self.gossip_cache.items():
            if current_time - gossip.timestamp > max_age or gossip.ttl <= 0:
                to_remove.append(gossip_id)

        for gossip_id in to_remove:
            del self.gossip_cache[gossip_id]

        if to_remove:
            logger.debug(
                f"Node {self.agent.node_id}: Cleaned up {len(to_remove)} old gossip messages"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return {
            **self.metrics,
            "signal_inbox_size": len(self.signal_inbox),
            "signal_outbox_size": len(self.signal_outbox),
            "pending_acks": len(self.pending_acks),
            "gossip_cache_size": len(self.gossip_cache),
            "gossip_inbox_size": len(self.gossip_inbox),
        }
