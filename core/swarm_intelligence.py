"""
Swarm Intelligence & Stigmergy for Phase 2: Emergence & Adaptation

This module implements coordinated swarm intelligence through indirect coordination
via environmental modifications (stigmergy).

Key Features:
- Advanced pheromone system with structured messages
- Emergent supply chains through pheromone trails
- Coordinated incident response with multi-agent strategies
- Self-organizing collective behaviors
"""

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PheromoneType(str, Enum):
    """Types of structured pheromones for stigmergic coordination"""
    THREAT = "threat"                    # Danger signal
    RESOURCE = "resource"                # Resource availability
    HELP_NEEDED = "help_needed"          # Request for assistance
    PROCESSING_NEEDED = "processing_needed"  # Data/task needs processing
    TRANSPORT_NEEDED = "transport_needed"    # Item needs transport
    DESTINATION = "destination"          # Delivery target
    SUPPRESSION = "suppression"          # Slow threat spread
    REPAIR = "repair"                    # Repair needed
    EXPLORATION = "exploration"          # Unexplored area
    PATROL = "patrol"                    # Regular patrol route


@dataclass
class StructuredPheromone:
    """
    Advanced pheromone with rich, structured information.
    
    Enables complex indirect coordination through the environment.
    """
    pheromone_id: str
    pheromone_type: PheromoneType
    node_id: str  # Where deposited
    depositor_id: int
    depositor_role: str  # Scout, Harvester, etc.
    
    # Structured payload
    signature: Optional[str] = None      # Threat signature, resource type, etc.
    confidence: float = 0.5              # Confidence in information (0-1)
    priority: float = 0.5                # Urgency (0-1)
    quantity: float = 1.0                # Amount/intensity
    
    # Trail information for supply chains
    source_node: Optional[str] = None    # Origin node
    target_node: Optional[str] = None    # Destination node
    sequence_number: int = 0             # Position in multi-step chain
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    ttl: int = 100  # Time steps before decay
    intensity: float = 1.0  # Current strength (decays over time)
    decay_rate: float = 0.95
    
    # Multi-agent coordination
    contributors: Set[int] = field(default_factory=set)  # Agents that reinforced this
    reinforcement_count: int = 0
    
    def decay(self, delta_time: float = 1.0):
        """Apply temporal decay"""
        self.intensity *= (self.decay_rate ** delta_time)
        self.ttl -= int(delta_time)
    
    def reinforce(self, agent_id: int, amount: float = 0.1):
        """Reinforce pheromone strength (positive stigmergy)"""
        self.intensity = min(2.0, self.intensity + amount)
        self.contributors.add(agent_id)
        self.reinforcement_count += 1
    
    def is_expired(self) -> bool:
        """Check if pheromone has expired"""
        return self.intensity < 0.01 or self.ttl <= 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for sensing/communication"""
        return {
            "pheromone_id": self.pheromone_id,
            "type": self.pheromone_type.value,
            "node_id": self.node_id,
            "depositor_id": self.depositor_id,
            "depositor_role": self.depositor_role,
            "signature": self.signature,
            "confidence": self.confidence,
            "priority": self.priority,
            "quantity": self.quantity,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "sequence_number": self.sequence_number,
            "intensity": self.intensity,
            "age": time.time() - self.timestamp,
            "reinforcement_count": self.reinforcement_count,
            "contributors": list(self.contributors)
        }


@dataclass
class SupplyChainLink:
    """
    A link in an emergent supply chain.
    
    Supply chains emerge from pheromone trails left by multiple agents.
    """
    link_id: str
    from_node: str
    to_node: str
    resource_type: str
    
    # Chain metadata
    chain_id: str  # Multiple links form a chain
    position: int  # Position in chain (0 = start)
    
    # Performance metrics
    throughput: float = 0.0  # Resources/time
    latency: float = 0.0     # Average transit time
    reliability: float = 1.0  # Success rate
    
    # Pheromone trail strength
    trail_strength: float = 1.0
    last_reinforcement: float = field(default_factory=time.time)
    
    # Health
    active: bool = True
    failure_count: int = 0
    
    def update_performance(self, success: bool, transit_time: float, amount: float):
        """Update link performance metrics"""
        if success:
            # Update throughput (exponential moving average)
            self.throughput = 0.9 * self.throughput + 0.1 * amount
            self.latency = 0.9 * self.latency + 0.1 * transit_time
            self.reliability = min(1.0, self.reliability + 0.05)
            self.failure_count = max(0, self.failure_count - 1)
        else:
            self.reliability = max(0.0, self.reliability - 0.1)
            self.failure_count += 1
            
            # Deactivate if too many failures
            if self.failure_count > 5:
                self.active = False
    
    def reinforce_trail(self, amount: float = 0.1):
        """Strengthen pheromone trail"""
        self.trail_strength = min(2.0, self.trail_strength + amount)
        self.last_reinforcement = time.time()
    
    def decay_trail(self, delta_time: float = 1.0):
        """Decay pheromone trail over time"""
        self.trail_strength *= (0.98 ** delta_time)
        
        # Deactivate if trail becomes too weak
        if self.trail_strength < 0.1:
            self.active = False


@dataclass
class SwarmCoordinationState:
    """
    State for coordinated swarm actions.
    
    Tracks multi-agent coordination for complex tasks.
    """
    coordination_id: str
    task_type: str  # "incident_response", "supply_chain", "defense", etc.
    
    # Participating agents
    coordinators: Set[int] = field(default_factory=set)
    scouts: Set[int] = field(default_factory=set)
    harvesters: Set[int] = field(default_factory=set)
    guardians: Set[int] = field(default_factory=set)
    healers: Set[int] = field(default_factory=set)
    
    # Task state
    target_node: Optional[str] = None
    progress: float = 0.0  # 0 to 1
    phase: str = "initializing"  # Task-specific phases
    
    # Performance
    start_time: float = field(default_factory=time.time)
    efficiency: float = 1.0
    
    def add_agent(self, agent_id: int, role: str):
        """Add agent to coordination"""
        role_lower = role.lower()
        if "scout" in role_lower:
            self.scouts.add(agent_id)
        elif "harvest" in role_lower:
            self.harvesters.add(agent_id)
        elif "guardian" in role_lower or "guard" in role_lower:
            self.guardians.add(agent_id)
        elif "heal" in role_lower:
            self.healers.add(agent_id)
        elif "coord" in role_lower:
            self.coordinators.add(agent_id)
    
    def total_agents(self) -> int:
        """Total participating agents"""
        return (len(self.coordinators) + len(self.scouts) + 
                len(self.harvesters) + len(self.guardians) + len(self.healers))
    
    def is_diverse(self) -> bool:
        """Check if swarm has diverse roles"""
        role_count = sum([
            len(self.scouts) > 0,
            len(self.harvesters) > 0,
            len(self.guardians) > 0,
            len(self.healers) > 0
        ])
        return role_count >= 2


class SwarmIntelligenceManager:
    """
    Manages swarm intelligence and stigmergic coordination.
    
    Enables emergent collective behaviors through environmental modification.
    """
    
    def __init__(self):
        # Pheromone field (node_id -> list of pheromones)
        self.pheromones: Dict[str, List[StructuredPheromone]] = defaultdict(list)
        
        # Supply chains (chain_id -> list of links)
        self.supply_chains: Dict[str, List[SupplyChainLink]] = {}
        
        # Active coordinations (coordination_id -> state)
        self.coordinations: Dict[str, SwarmCoordinationState] = {}
        
        # Performance metrics
        self.pheromone_deposits = 0
        self.pheromone_reinforcements = 0
        self.chains_established = 0
        self.chains_healed = 0
        self.coordinated_actions = 0
        
        logger.info("SwarmIntelligenceManager initialized")
    
    def deposit_pheromone(
        self,
        node_id: str,
        pheromone_type: PheromoneType,
        depositor_id: int,
        depositor_role: str,
        **kwargs
    ) -> StructuredPheromone:
        """
        Deposit a structured pheromone at a node.
        
        Args:
            node_id: Where to deposit
            pheromone_type: Type of pheromone
            depositor_id: Agent ID
            depositor_role: Agent role
            **kwargs: Additional pheromone attributes
            
        Returns:
            Created pheromone
        """
        pheromone = StructuredPheromone(
            pheromone_id=str(uuid.uuid4()),
            pheromone_type=pheromone_type,
            node_id=node_id,
            depositor_id=depositor_id,
            depositor_role=depositor_role,
            **kwargs
        )
        
        self.pheromones[node_id].append(pheromone)
        self.pheromone_deposits += 1
        
        logger.debug(f"Agent {depositor_id} ({depositor_role}) deposited {pheromone_type.value} at {node_id}")
        return pheromone
    
    def sense_pheromones(
        self,
        node_id: str,
        pheromone_types: Optional[List[PheromoneType]] = None,
        min_intensity: float = 0.1
    ) -> List[StructuredPheromone]:
        """
        Sense pheromones at a node.
        
        Args:
            node_id: Node to sense
            pheromone_types: Filter by types (None = all)
            min_intensity: Minimum intensity threshold
            
        Returns:
            List of sensed pheromones
        """
        pheromones = self.pheromones.get(node_id, [])
        
        # Filter by type and intensity
        results = [
            p for p in pheromones
            if p.intensity >= min_intensity and
            (pheromone_types is None or p.pheromone_type in pheromone_types)
        ]
        
        # Sort by priority and intensity
        results.sort(key=lambda p: (p.priority, p.intensity), reverse=True)
        return results
    
    def reinforce_pheromone(self, pheromone_id: str, agent_id: int, amount: float = 0.1):
        """Reinforce existing pheromone (positive stigmergy)"""
        for node_pheromones in self.pheromones.values():
            for pheromone in node_pheromones:
                if pheromone.pheromone_id == pheromone_id:
                    pheromone.reinforce(agent_id, amount)
                    self.pheromone_reinforcements += 1
                    return True
        return False
    
    def establish_supply_chain(
        self,
        chain_id: str,
        nodes: List[str],
        resource_type: str
    ) -> bool:
        """
        Establish a supply chain from pheromone trails.
        
        Args:
            chain_id: Unique chain identifier
            nodes: Ordered list of nodes in chain
            resource_type: Type of resource being transported
            
        Returns:
            True if established successfully
        """
        if len(nodes) < 2:
            return False
        
        links = []
        for i in range(len(nodes) - 1):
            link = SupplyChainLink(
                link_id=str(uuid.uuid4()),
                from_node=nodes[i],
                to_node=nodes[i + 1],
                resource_type=resource_type,
                chain_id=chain_id,
                position=i
            )
            links.append(link)
        
        self.supply_chains[chain_id] = links
        self.chains_established += 1
        
        logger.info(f"Established supply chain {chain_id}: {len(links)} links for {resource_type}")
        return True
    
    def get_supply_chain(self, chain_id: str) -> Optional[List[SupplyChainLink]]:
        """Get supply chain by ID"""
        return self.supply_chains.get(chain_id)
    
    def heal_supply_chain(self, chain_id: str, broken_position: int, new_node: str) -> bool:
        """
        Heal a broken supply chain by finding alternate route.
        
        Args:
            chain_id: Chain to heal
            broken_position: Position of broken link
            new_node: Replacement node
            
        Returns:
            True if healed
        """
        chain = self.supply_chains.get(chain_id)
        if not chain or broken_position >= len(chain):
            return False
        
        # Deactivate broken link
        chain[broken_position].active = False
        
        # Create new link with replacement node
        if broken_position > 0:
            from_node = chain[broken_position - 1].to_node
        else:
            from_node = chain[0].from_node
        
        if broken_position < len(chain) - 1:
            to_node = chain[broken_position + 1].from_node
        else:
            to_node = chain[-1].to_node
        
        new_link = SupplyChainLink(
            link_id=str(uuid.uuid4()),
            from_node=from_node if broken_position > 0 else new_node,
            to_node=new_node if broken_position < len(chain) - 1 else to_node,
            resource_type=chain[broken_position].resource_type,
            chain_id=chain_id,
            position=broken_position
        )
        
        # Insert new link
        chain[broken_position] = new_link
        self.chains_healed += 1
        
        logger.info(f"Healed supply chain {chain_id} at position {broken_position}")
        return True
    
    def start_coordination(
        self,
        task_type: str,
        target_node: Optional[str] = None
    ) -> str:
        """
        Start a coordinated swarm action.
        
        Args:
            task_type: Type of coordinated task
            target_node: Optional target node
            
        Returns:
            Coordination ID
        """
        coordination_id = str(uuid.uuid4())
        
        state = SwarmCoordinationState(
            coordination_id=coordination_id,
            task_type=task_type,
            target_node=target_node
        )
        
        self.coordinations[coordination_id] = state
        logger.info(f"Started coordination {coordination_id} for {task_type}")
        return coordination_id
    
    def join_coordination(self, coordination_id: str, agent_id: int, role: str) -> bool:
        """Add agent to coordination"""
        if coordination_id not in self.coordinations:
            return False
        
        self.coordinations[coordination_id].add_agent(agent_id, role)
        return True
    
    def update_coordination_progress(self, coordination_id: str, progress: float, phase: str):
        """Update coordination progress"""
        if coordination_id in self.coordinations:
            self.coordinations[coordination_id].progress = progress
            self.coordinations[coordination_id].phase = phase
    
    def complete_coordination(self, coordination_id: str) -> Optional[Dict[str, Any]]:
        """
        Complete coordination and return metrics.
        
        Returns:
            Dictionary of coordination metrics
        """
        if coordination_id not in self.coordinations:
            return None
        
        state = self.coordinations.pop(coordination_id)
        duration = time.time() - state.start_time
        
        self.coordinated_actions += 1
        
        metrics = {
            "coordination_id": coordination_id,
            "task_type": state.task_type,
            "duration": duration,
            "total_agents": state.total_agents(),
            "role_diversity": state.is_diverse(),
            "efficiency": state.efficiency,
            "progress": state.progress
        }
        
        logger.info(f"Completed coordination {coordination_id}: {metrics}")
        return metrics
    
    def update(self, delta_time: float = 1.0):
        """
        Update swarm intelligence state.
        
        Decays pheromones, updates supply chains, etc.
        """
        # Decay and clean up pheromones
        for node_id in list(self.pheromones.keys()):
            active = []
            for pheromone in self.pheromones[node_id]:
                pheromone.decay(delta_time)
                if not pheromone.is_expired():
                    active.append(pheromone)
            
            if active:
                self.pheromones[node_id] = active
            else:
                del self.pheromones[node_id]
        
        # Decay supply chain trails
        for chain in self.supply_chains.values():
            for link in chain:
                link.decay_trail(delta_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm intelligence statistics"""
        total_pheromones = sum(len(p) for p in self.pheromones.values())
        active_chains = sum(
            1 for chain in self.supply_chains.values()
            if any(link.active for link in chain)
        )
        
        return {
            "total_pheromones": total_pheromones,
            "pheromone_nodes": len(self.pheromones),
            "pheromone_deposits": self.pheromone_deposits,
            "pheromone_reinforcements": self.pheromone_reinforcements,
            "supply_chains": len(self.supply_chains),
            "active_chains": active_chains,
            "chains_established": self.chains_established,
            "chains_healed": self.chains_healed,
            "active_coordinations": len(self.coordinations),
            "coordinated_actions": self.coordinated_actions
        }
