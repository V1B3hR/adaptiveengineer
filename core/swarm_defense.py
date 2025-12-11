"""
Swarm Defense System - Biological-inspired collaborative response

This module implements biological-inspired algorithms for coordinated incident response:
- "Digital white-blood-cells" swarm coordination
- Immune system-inspired threat detection and containment
- Ant colony optimization for incident response
- Swarm intelligence for distributed detection and recovery
"""

import logging
import time
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import random

logger = logging.getLogger("swarm_defense")


class SwarmRole(str, Enum):
    """Roles in the swarm (inspired by immune system)"""

    SCOUT = "scout"  # Detect and patrol
    DEFENDER = "defender"  # Active defense
    HEALER = "healer"  # Recovery and repair
    MEMORY_CELL = "memory_cell"  # Pattern recognition
    COORDINATOR = "coordinator"  # Organize response


class ThreatLevel(str, Enum):
    """Threat severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PheromoneType(str, Enum):
    """Pheromone types (inspired by ant colonies)"""

    THREAT = "threat"  # Marks threat location
    SAFE = "safe"  # Marks safe areas
    HELP = "help"  # Request assistance
    CLEARED = "cleared"  # Threat neutralized


@dataclass
class SwarmAgent:
    """Individual agent in the swarm"""

    agent_id: int
    role: SwarmRole
    position: Tuple[float, float]  # Virtual position in network
    energy: float = 1.0
    health: float = 1.0
    last_action: Optional[str] = None
    threats_detected: int = 0
    threats_neutralized: int = 0


@dataclass
class Pheromone:
    """Chemical signal for swarm communication"""

    pheromone_type: PheromoneType
    position: Tuple[float, float]
    strength: float  # 0.0 to 1.0
    timestamp: float
    source_agent: int
    decay_rate: float = 0.1  # Per time unit

    def current_strength(self) -> float:
        """Calculate current strength after decay"""
        age = time.time() - self.timestamp
        return max(0.0, self.strength * math.exp(-self.decay_rate * age))


@dataclass
class ThreatZone:
    """Area with detected threat"""

    zone_id: str
    center: Tuple[float, float]
    radius: float
    threat_level: ThreatLevel
    first_detected: float
    agents_responding: Set[int] = field(default_factory=set)
    containment_progress: float = 0.0  # 0.0 to 1.0
    neutralized: bool = False


@dataclass
class SwarmResponse:
    """Coordinated swarm response to incident"""

    response_id: str
    timestamp: float
    threat_zone: ThreatZone
    agents_deployed: List[int]
    strategy: str
    success: bool
    duration: float


class SwarmDefenseSystem:
    """
    Biological-inspired swarm defense system.

    Uses principles from:
    - Immune systems (white blood cells, memory cells)
    - Ant colonies (pheromones, trail following)
    - Swarm intelligence (collective decision making)
    """

    def __init__(
        self,
        network_size: int = 10,
        agent_count: int = 20,
        coordination_range: float = 5.0,
    ):
        """
        Initialize swarm defense system.

        Args:
            network_size: Size of virtual network space
            agent_count: Number of agents in the swarm
            coordination_range: Range for agent coordination
        """
        self.network_size = network_size
        self.agent_count = agent_count
        self.coordination_range = coordination_range

        # Initialize swarm agents
        self.agents: Dict[int, SwarmAgent] = {}
        self._initialize_agents()

        # Pheromone trails
        self.pheromones: List[Pheromone] = []

        # Threat tracking
        self.threat_zones: Dict[str, ThreatZone] = {}
        self.response_history: deque = deque(maxlen=100)

        # Pattern memory (immune memory)
        self.threat_patterns: List[Dict[str, Any]] = []

        # Metrics
        self.collective_threats_detected = 0
        self.collective_threats_neutralized = 0
        self.responses_coordinated = 0

        logger.info(
            f"Swarm defense system initialized with {agent_count} agents "
            f"in {network_size}x{network_size} network"
        )

    def _initialize_agents(self):
        """Initialize swarm agents with diverse roles"""
        role_distribution = {
            SwarmRole.SCOUT: 0.3,  # 30% scouts
            SwarmRole.DEFENDER: 0.3,  # 30% defenders
            SwarmRole.HEALER: 0.2,  # 20% healers
            SwarmRole.MEMORY_CELL: 0.1,  # 10% memory cells
            SwarmRole.COORDINATOR: 0.1,  # 10% coordinators
        }

        roles = []
        for role, ratio in role_distribution.items():
            count = int(self.agent_count * ratio)
            roles.extend([role] * count)

        # Fill remaining with scouts
        while len(roles) < self.agent_count:
            roles.append(SwarmRole.SCOUT)

        # Create agents
        for i in range(self.agent_count):
            position = (
                random.uniform(0, self.network_size),
                random.uniform(0, self.network_size),
            )

            self.agents[i] = SwarmAgent(
                agent_id=i, role=roles[i], position=position
            )

    def detect_threat_swarm(
        self,
        location: Tuple[float, float],
        threat_level: ThreatLevel,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> ThreatZone:
        """
        Detect threat and initiate swarm response.

        Args:
            location: Location of threat in network space
            threat_level: Severity of threat
            indicators: Optional threat indicators

        Returns:
            ThreatZone created for this threat
        """
        zone_id = f"zone_{int(time.time()*1000)}"

        # Determine threat radius based on level
        radius_map = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 2.0,
            ThreatLevel.HIGH: 3.0,
            ThreatLevel.CRITICAL: 5.0,
        }

        threat_zone = ThreatZone(
            zone_id=zone_id,
            center=location,
            radius=radius_map[threat_level],
            threat_level=threat_level,
            first_detected=time.time(),
        )

        self.threat_zones[zone_id] = threat_zone
        self.collective_threats_detected += 1

        # Release threat pheromone
        self._release_pheromone(
            PheromoneType.THREAT,
            location,
            strength=0.8,
            source_agent=-1,  # System detection
        )

        logger.warning(f"Swarm detected threat at {location}: {threat_level}")

        # Trigger swarm response
        self._initiate_swarm_response(threat_zone)

        return threat_zone

    def _release_pheromone(
        self,
        pheromone_type: PheromoneType,
        position: Tuple[float, float],
        strength: float,
        source_agent: int,
    ):
        """Release a pheromone signal"""
        pheromone = Pheromone(
            pheromone_type=pheromone_type,
            position=position,
            strength=strength,
            timestamp=time.time(),
            source_agent=source_agent,
        )

        self.pheromones.append(pheromone)

        # Clean up old pheromones
        self._decay_pheromones()

    def _decay_pheromones(self):
        """Remove fully decayed pheromones"""
        self.pheromones = [
            p for p in self.pheromones if p.current_strength() > 0.01
        ]

    def _initiate_swarm_response(self, threat_zone: ThreatZone):
        """
        Initiate coordinated swarm response to threat.

        Inspired by immune system:
        1. Scouts detect and mark threat (like dendritic cells)
        2. Defenders mobilize to contain (like T-cells)
        3. Healers repair damage (like tissue repair)
        4. Memory cells record pattern (like B-cells)
        """
        response_id = f"response_{threat_zone.zone_id}"
        start_time = time.time()

        # Find nearby agents
        nearby_agents = self._find_nearby_agents(
            threat_zone.center, radius=self.coordination_range
        )

        # Recruit additional agents based on threat level
        required_agents = self._calculate_required_agents(
            threat_zone.threat_level
        )

        if len(nearby_agents) < required_agents:
            # Call for help with pheromone
            self._release_pheromone(
                PheromoneType.HELP,
                threat_zone.center,
                strength=0.9,
                source_agent=-1,
            )

            # Recruit more agents
            nearby_agents = self._recruit_agents(
                threat_zone.center, required_agents - len(nearby_agents)
            )

        # Assign agents to threat zone
        for agent_id in nearby_agents:
            threat_zone.agents_responding.add(agent_id)

        # Execute coordinated strategy
        strategy = self._select_swarm_strategy(threat_zone)
        success = self._execute_swarm_strategy(
            threat_zone, strategy, nearby_agents
        )

        duration = time.time() - start_time

        response = SwarmResponse(
            response_id=response_id,
            timestamp=time.time(),
            threat_zone=threat_zone,
            agents_deployed=list(nearby_agents),
            strategy=strategy,
            success=success,
            duration=duration,
        )

        self.response_history.append(response)
        self.responses_coordinated += 1

        if success:
            self.collective_threats_neutralized += 1
            threat_zone.neutralized = True
            threat_zone.containment_progress = 1.0

            # Release cleared pheromone
            self._release_pheromone(
                PheromoneType.CLEARED,
                threat_zone.center,
                strength=0.7,
                source_agent=-1,
            )

            # Store pattern in immune memory
            self._store_threat_pattern(threat_zone, response)

            logger.info(
                f"Swarm successfully neutralized threat {threat_zone.zone_id} "
                f"with {len(nearby_agents)} agents in {duration:.2f}s"
            )
        else:
            logger.warning(
                f"Swarm failed to neutralize threat {threat_zone.zone_id}"
            )

    def _find_nearby_agents(
        self, location: Tuple[float, float], radius: float
    ) -> List[int]:
        """Find agents within radius of location"""
        nearby = []

        for agent_id, agent in self.agents.items():
            distance = self._calculate_distance(agent.position, location)
            if distance <= radius:
                nearby.append(agent_id)

        return nearby

    def _calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between positions"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _calculate_required_agents(self, threat_level: ThreatLevel) -> int:
        """Calculate number of agents needed for threat level"""
        agent_map = {
            ThreatLevel.LOW: 2,
            ThreatLevel.MEDIUM: 4,
            ThreatLevel.HIGH: 7,
            ThreatLevel.CRITICAL: 12,
        }
        return agent_map[threat_level]

    def _recruit_agents(
        self, location: Tuple[float, float], count: int
    ) -> List[int]:
        """Recruit additional agents to location (following pheromone trails)"""
        # Find available agents not currently responding
        available = []
        for agent_id, agent in self.agents.items():
            # Check if agent is not already responding to another threat
            responding = any(
                agent_id in zone.agents_responding
                for zone in self.threat_zones.values()
                if not zone.neutralized
            )
            if not responding:
                available.append(agent_id)

        # Sort by distance to location
        available.sort(
            key=lambda aid: self._calculate_distance(
                self.agents[aid].position, location
            )
        )

        # Recruit closest agents
        recruited = available[:count]

        # Move agents toward location (simulate movement)
        for agent_id in recruited:
            agent = self.agents[agent_id]
            # Move 50% toward location
            new_x = agent.position[0] + 0.5 * (location[0] - agent.position[0])
            new_y = agent.position[1] + 0.5 * (location[1] - agent.position[1])
            agent.position = (new_x, new_y)

        return recruited

    def _select_swarm_strategy(self, threat_zone: ThreatZone) -> str:
        """
        Select swarm strategy based on threat and available agents.

        Inspired by:
        - Immune system: surround and neutralize
        - Ant colonies: pheromone trails for coordination
        - Swarm intelligence: distributed decision making
        """
        agent_count = len(threat_zone.agents_responding)

        if threat_zone.threat_level == ThreatLevel.CRITICAL:
            return "surround_and_isolate"
        elif threat_zone.threat_level == ThreatLevel.HIGH:
            return "coordinated_attack"
        elif agent_count >= 5:
            return "distributed_containment"
        else:
            return "focused_response"

    def _execute_swarm_strategy(
        self, threat_zone: ThreatZone, strategy: str, agent_ids: List[int]
    ) -> bool:
        """Execute coordinated swarm strategy"""

        # Update agent actions
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            agent.last_action = f"{strategy}:{threat_zone.zone_id}"

            # Update agent metrics based on role
            if agent.role == SwarmRole.SCOUT:
                agent.threats_detected += 1
            elif agent.role in [SwarmRole.DEFENDER, SwarmRole.HEALER]:
                agent.threats_neutralized += 1

        # Simulate strategy execution
        if strategy == "surround_and_isolate":
            # Position agents in circle around threat
            success_rate = 0.90
        elif strategy == "coordinated_attack":
            # Agents attack from multiple angles
            success_rate = 0.85
        elif strategy == "distributed_containment":
            # Agents form distributed perimeter
            success_rate = 0.80
        else:  # focused_response
            # Agents focus on single point
            success_rate = 0.75

        # Success depends on agent count and strategy
        effectiveness = min(
            1.0,
            len(agent_ids)
            / self._calculate_required_agents(threat_zone.threat_level),
        )
        success_chance = success_rate * effectiveness

        # Simulate containment progress
        threat_zone.containment_progress = min(1.0, effectiveness * 0.8)

        return random.random() < success_chance

    def _store_threat_pattern(
        self, threat_zone: ThreatZone, response: SwarmResponse
    ):
        """
        Store threat pattern in immune memory.

        Like adaptive immune system, remember patterns for faster future response.
        """
        pattern = {
            "threat_level": threat_zone.threat_level,
            "agents_needed": len(response.agents_deployed),
            "strategy": response.strategy,
            "success": response.success,
            "duration": response.duration,
            "timestamp": time.time(),
        }

        self.threat_patterns.append(pattern)

        # Assign memory cell agents
        memory_agents = [
            a for a in self.agents.values() if a.role == SwarmRole.MEMORY_CELL
        ]

        for agent in memory_agents[:2]:  # Use 2 memory cells per pattern
            agent.last_action = f"store_pattern:{threat_zone.zone_id}"

    def recall_threat_pattern(
        self, threat_level: ThreatLevel
    ) -> Optional[Dict[str, Any]]:
        """
        Recall similar threat pattern from memory.

        Returns best matching pattern for faster response.
        """
        matching = [
            p
            for p in self.threat_patterns
            if p["threat_level"] == threat_level and p["success"]
        ]

        if not matching:
            return None

        # Return most recent successful pattern
        return matching[-1]

    def update_agent_positions(self, delta_time: float):
        """
        Update agent positions (patrol and pheromone following).

        Inspired by ant colony optimization:
        - Agents follow pheromone trails
        - Random exploration when no trails
        - Return to patrol when idle
        """
        for agent in self.agents.values():
            # Check if agent is currently responding
            responding = any(
                agent.agent_id in zone.agents_responding
                for zone in self.threat_zones.values()
                if not zone.neutralized
            )

            if responding:
                continue  # Agent is busy

            # Scouts patrol randomly
            if agent.role == SwarmRole.SCOUT:
                # Random walk
                dx = random.uniform(-0.5, 0.5) * delta_time
                dy = random.uniform(-0.5, 0.5) * delta_time

                new_x = max(0, min(self.network_size, agent.position[0] + dx))
                new_y = max(0, min(self.network_size, agent.position[1] + dy))

                agent.position = (new_x, new_y)

            # Other agents follow pheromone trails
            elif agent.role in [SwarmRole.DEFENDER, SwarmRole.HEALER]:
                target = self._find_strongest_pheromone(
                    agent.position, [PheromoneType.THREAT, PheromoneType.HELP]
                )

                if target:
                    # Move toward pheromone
                    dx = 0.3 * (target[0] - agent.position[0]) * delta_time
                    dy = 0.3 * (target[1] - agent.position[1]) * delta_time

                    agent.position = (
                        agent.position[0] + dx,
                        agent.position[1] + dy,
                    )

    def _find_strongest_pheromone(
        self,
        position: Tuple[float, float],
        pheromone_types: List[PheromoneType],
    ) -> Optional[Tuple[float, float]]:
        """Find position of strongest nearby pheromone"""
        best_pheromone = None
        best_strength = 0.0

        for pheromone in self.pheromones:
            if pheromone.pheromone_type not in pheromone_types:
                continue

            distance = self._calculate_distance(position, pheromone.position)
            if distance > self.coordination_range:
                continue

            # Strength decreases with distance
            effective_strength = pheromone.current_strength() / (1 + distance)

            if effective_strength > best_strength:
                best_strength = effective_strength
                best_pheromone = pheromone

        return best_pheromone.position if best_pheromone else None

    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get metrics about swarm operations"""
        active_threats = sum(
            1 for zone in self.threat_zones.values() if not zone.neutralized
        )

        agents_responding = sum(
            len(zone.agents_responding)
            for zone in self.threat_zones.values()
            if not zone.neutralized
        )

        role_counts = {}
        for role in SwarmRole:
            role_counts[role.value] = sum(
                1 for agent in self.agents.values() if agent.role == role
            )

        neutralization_rate = (
            self.collective_threats_neutralized
            / self.collective_threats_detected
            if self.collective_threats_detected > 0
            else 0.0
        )

        return {
            "agent_count": len(self.agents),
            "active_threats": active_threats,
            "agents_responding": agents_responding,
            "threats_detected": self.collective_threats_detected,
            "threats_neutralized": self.collective_threats_neutralized,
            "neutralization_rate": neutralization_rate,
            "responses_coordinated": self.responses_coordinated,
            "pheromone_trails": len(self.pheromones),
            "threat_patterns_stored": len(self.threat_patterns),
            "role_distribution": role_counts,
        }

    def visualize_swarm_state(self) -> str:
        """Generate text visualization of swarm state"""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(
            f"SWARM STATE (Network: {self.network_size}x{self.network_size})"
        )
        lines.append(f"{'='*60}")

        # Agent summary
        lines.append(f"\nAgents: {len(self.agents)}")
        for role in SwarmRole:
            count = sum(1 for a in self.agents.values() if a.role == role)
            lines.append(f"  {role.value}: {count}")

        # Active threats
        active = [z for z in self.threat_zones.values() if not z.neutralized]
        lines.append(f"\nActive Threats: {len(active)}")
        for zone in active:
            lines.append(
                f"  {zone.zone_id}: {zone.threat_level.value} "
                f"({len(zone.agents_responding)} agents responding, "
                f"{zone.containment_progress*100:.0f}% contained)"
            )

        # Pheromone trails
        lines.append(f"\nPheromone Trails: {len(self.pheromones)}")
        pheromone_counts = {}
        for p in self.pheromones:
            pheromone_counts[p.pheromone_type.value] = (
                pheromone_counts.get(p.pheromone_type.value, 0) + 1
            )
        for ptype, count in pheromone_counts.items():
            lines.append(f"  {ptype}: {count}")

        lines.append(f"{'='*60}\n")

        return "\n".join(lines)
