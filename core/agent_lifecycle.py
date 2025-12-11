"""
Agent Lifecycle Management

This module defines the conditions and mechanisms for agent birth (instantiation)
and death (removal) in the simulation.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_lifecycle")


class LifecycleState(str, Enum):
    """Agent lifecycle states"""

    UNBORN = "unborn"  # Agent slot exists but not yet instantiated
    ALIVE = "alive"  # Agent is active
    DYING = "dying"  # Agent is in death process
    DEAD = "dead"  # Agent has been removed


class DeathCause(str, Enum):
    """Reasons for agent death"""

    ENERGY_DEPLETION = "energy_depletion"
    HEALTH_FAILURE = "health_failure"
    STRESS_OVERLOAD = "stress_overload"
    EXTERNAL_TERMINATION = "external_termination"
    NATURAL_EXPIRATION = "natural_expiration"


@dataclass
class BirthConditions:
    """Conditions that must be met for agent birth"""

    min_energy_available: float = (
        5.0  # Environment must have sufficient energy
    )
    min_health_threshold: float = 0.3  # Spawning location must be healthy
    max_stress_threshold: float = (
        0.7  # Spawning location can't be too stressed
    )
    require_stable_environment: bool = True  # Environment should be stable
    min_population: int = 0  # Minimum population before new births
    max_population: int = 100  # Maximum population cap
    spawn_cooldown: float = 10.0  # Minimum time between spawns (seconds)


@dataclass
class DeathConditions:
    """Conditions that trigger agent death"""

    zero_energy_threshold: float = (
        0.0  # Energy at or below this triggers death
    )
    zero_health_threshold: float = (
        0.0  # Health at or below this triggers death
    )
    critical_stress_threshold: float = (
        1.0  # Stress at or above this can trigger death
    )
    max_lifespan: Optional[float] = (
        None  # Maximum age in seconds (None = unlimited)
    )
    death_probability_low_energy: float = (
        0.01  # Chance of death per step when energy is low
    )
    death_probability_high_stress: float = (
        0.005  # Chance of death per step when stress is high
    )


class AgentLifecycleManager:
    """
    Manages agent birth and death in the simulation.

    Handles:
    - Birth conditions and spawning
    - Death conditions and removal
    - Population tracking
    - Lifecycle events and logging
    """

    def __init__(
        self,
        birth_conditions: Optional[BirthConditions] = None,
        death_conditions: Optional[DeathConditions] = None,
    ):
        """
        Initialize lifecycle manager.

        Args:
            birth_conditions: Conditions for agent birth
            death_conditions: Conditions for agent death
        """
        self.birth_conditions = birth_conditions or BirthConditions()
        self.death_conditions = death_conditions or DeathConditions()

        # Population tracking
        self.living_agents: Dict[int, Any] = {}  # node_id -> agent
        self.dead_agents: List[Dict[str, Any]] = []  # History of dead agents
        self.birth_history: List[Dict[str, Any]] = []  # History of births

        # Metrics
        self.total_births = 0
        self.total_deaths = 0
        self.death_causes: Dict[str, int] = defaultdict(int)

        # Spawn tracking
        self.last_spawn_time = 0.0
        self.next_agent_id = 1

        logger.info("Agent lifecycle manager initialized")

    def register_agent(self, agent):
        """
        Register a living agent.

        Args:
            agent: Agent to register
        """
        self.living_agents[agent.node_id] = agent
        logger.debug(f"Registered agent {agent.node_id}")

    def unregister_agent(self, agent_id: int):
        """
        Unregister an agent (after death).

        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.living_agents:
            del self.living_agents[agent_id]
            logger.debug(f"Unregistered agent {agent_id}")

    def check_birth_conditions(
        self, environment=None, spawn_location_id: Optional[str] = None
    ) -> bool:
        """
        Check if conditions are met for a new agent birth.

        Args:
            environment: LivingGraph environment
            spawn_location_id: Optional specific location to spawn

        Returns:
            True if birth conditions are met
        """
        current_time = time.time()

        # Check population limits
        current_population = len(self.living_agents)
        if current_population < self.birth_conditions.min_population:
            return False  # Not enough agents yet
        if current_population >= self.birth_conditions.max_population:
            logger.debug("Birth blocked: population at maximum")
            return False  # Too many agents

        # Check spawn cooldown
        if (
            current_time - self.last_spawn_time
            < self.birth_conditions.spawn_cooldown
        ):
            return False  # Too soon after last spawn

        # Check environment conditions
        if environment and spawn_location_id:
            location_state = environment.get_node_state(spawn_location_id)
            if location_state:
                # Check energy availability
                if (
                    location_state.get("energy_provided", 0.0)
                    < self.birth_conditions.min_energy_available
                ):
                    logger.debug(
                        f"Birth blocked: insufficient energy at location {spawn_location_id}"
                    )
                    return False

                # Check location health
                health_status = location_state.get("health_status", "unknown")
                if health_status == "failed":
                    logger.debug(
                        f"Birth blocked: failed health status at location {spawn_location_id}"
                    )
                    return False

                # Check stress level
                if (
                    location_state.get("stress_level", 0.0)
                    > self.birth_conditions.max_stress_threshold
                ):
                    logger.debug(
                        f"Birth blocked: high stress at location {spawn_location_id}"
                    )
                    return False

                # Check if location can host another agent
                if not location_state.get("can_host_agent", False):
                    logger.debug(
                        f"Birth blocked: location {spawn_location_id} cannot host more agents"
                    )
                    return False

        return True

    def spawn_agent(
        self,
        agent_class,
        spawn_location_id: Optional[str] = None,
        initial_energy: float = 10.0,
        **kwargs,
    ) -> Optional[Any]:
        """
        Spawn a new agent if conditions are met.

        Args:
            agent_class: Class to instantiate (e.g., AliveLoopNode)
            spawn_location_id: Location to spawn at
            initial_energy: Initial energy for agent
            **kwargs: Additional arguments for agent constructor

        Returns:
            New agent instance or None if spawn failed
        """
        if not self.check_birth_conditions(
            kwargs.get("environment"), spawn_location_id
        ):
            return None

        # Create agent
        agent_id = self.next_agent_id
        self.next_agent_id += 1

        try:
            agent = agent_class(
                position=kwargs.get("position", (0, 0)),
                velocity=kwargs.get("velocity", (0, 0)),
                initial_energy=initial_energy,
                node_id=agent_id,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "position",
                        "velocity",
                        "initial_energy",
                        "environment",
                    ]
                },
            )

            # Register agent
            self.register_agent(agent)

            # Record birth
            birth_event = {
                "agent_id": agent_id,
                "spawn_location_id": spawn_location_id,
                "timestamp": time.time(),
                "initial_energy": initial_energy,
                "population_size": len(self.living_agents),
            }
            self.birth_history.append(birth_event)
            self.total_births += 1
            self.last_spawn_time = time.time()

            logger.info(
                f"Agent {agent_id} born at location {spawn_location_id}"
            )

            return agent

        except Exception as e:
            logger.error(f"Failed to spawn agent: {e}")
            return None

    def check_death_conditions(
        self, agent
    ) -> tuple[bool, Optional[DeathCause]]:
        """
        Check if an agent meets death conditions.

        Args:
            agent: Agent to check

        Returns:
            (should_die, cause) tuple
        """
        # Check energy depletion
        if (
            hasattr(agent, "energy")
            and agent.energy <= self.death_conditions.zero_energy_threshold
        ):
            logger.info(
                f"Agent {agent.node_id} death condition: energy depletion ({agent.energy:.2f})"
            )
            return True, DeathCause.ENERGY_DEPLETION

        # Check health failure
        if hasattr(agent, "health"):
            if agent.health <= self.death_conditions.zero_health_threshold:
                logger.info(
                    f"Agent {agent.node_id} death condition: health failure ({agent.health:.2f})"
                )
                return True, DeathCause.HEALTH_FAILURE

        # Check stress overload
        if hasattr(agent, "anxiety") or hasattr(agent, "stress_level"):
            stress = getattr(
                agent, "stress_level", getattr(agent, "anxiety", 0.0)
            )
            if stress >= self.death_conditions.critical_stress_threshold:
                # Probabilistic death from high stress
                import numpy as np

                if (
                    np.random.random()
                    < self.death_conditions.death_probability_high_stress
                ):
                    logger.info(
                        f"Agent {agent.node_id} death condition: stress overload ({stress:.2f})"
                    )
                    return True, DeathCause.STRESS_OVERLOAD

        # Check low energy (probabilistic)
        if hasattr(agent, "energy") and agent.energy < 2.0:
            import numpy as np

            if (
                np.random.random()
                < self.death_conditions.death_probability_low_energy
            ):
                logger.info(
                    f"Agent {agent.node_id} death condition: low energy ({agent.energy:.2f})"
                )
                return True, DeathCause.ENERGY_DEPLETION

        # Check maximum lifespan
        if self.death_conditions.max_lifespan is not None:
            if hasattr(agent, "_time"):
                age = agent._time
                if age >= self.death_conditions.max_lifespan:
                    logger.info(
                        f"Agent {agent.node_id} death condition: natural expiration (age {age:.0f})"
                    )
                    return True, DeathCause.NATURAL_EXPIRATION

        return False, None

    def process_agent_death(self, agent, cause: DeathCause):
        """
        Process an agent's death.

        Args:
            agent: Agent that is dying
            cause: Cause of death
        """
        agent_id = agent.node_id

        # Record death
        death_event = {
            "agent_id": agent_id,
            "cause": cause.value,
            "timestamp": time.time(),
            "final_energy": getattr(agent, "energy", 0.0),
            "final_health": getattr(
                agent, "health", getattr(agent, "energy", 0.0)
            ),
            "age": getattr(agent, "_time", 0.0),
            "position": (
                tuple(agent.position) if hasattr(agent, "position") else None
            ),
            "memory_count": (
                len(agent.memory) if hasattr(agent, "memory") else 0
            ),
            "trust_network_size": (
                len(agent.trust_network)
                if hasattr(agent, "trust_network")
                else 0
            ),
        }

        self.dead_agents.append(death_event)
        self.total_deaths += 1
        self.death_causes[cause.value] += 1

        # Unregister agent
        self.unregister_agent(agent_id)

        logger.info(f"Agent {agent_id} died from {cause.value}")

    def update(self, delta_time: float = 1.0):
        """
        Update lifecycle manager - check all agents for death conditions.

        Args:
            delta_time: Time step
        """
        agents_to_process = list(self.living_agents.values())

        for agent in agents_to_process:
            should_die, cause = self.check_death_conditions(agent)
            if should_die and cause:
                self.process_agent_death(agent, cause)

    def get_population_stats(self) -> Dict[str, Any]:
        """Get current population statistics"""
        return {
            "current_population": len(self.living_agents),
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "net_population_change": self.total_births - self.total_deaths,
            "death_causes": dict(self.death_causes),
            "birth_rate": self.total_births
            / max(
                1,
                time.time()
                - (
                    self.birth_history[0]["timestamp"]
                    if self.birth_history
                    else time.time()
                ),
            ),
            "death_rate": self.total_deaths
            / max(
                1,
                time.time()
                - (
                    self.dead_agents[0]["timestamp"]
                    if self.dead_agents
                    else time.time()
                ),
            ),
        }

    def get_agent_lifespans(self) -> List[float]:
        """Get list of agent lifespans (for dead agents)"""
        return [event["age"] for event in self.dead_agents]

    def get_average_lifespan(self) -> float:
        """Calculate average lifespan of dead agents"""
        lifespans = self.get_agent_lifespans()
        return sum(lifespans) / len(lifespans) if lifespans else 0.0

    def force_agent_death(
        self,
        agent_id: int,
        cause: DeathCause = DeathCause.EXTERNAL_TERMINATION,
    ):
        """
        Force an agent's death (for external termination).

        Args:
            agent_id: ID of agent to terminate
            cause: Cause of death
        """
        if agent_id in self.living_agents:
            agent = self.living_agents[agent_id]
            self.process_agent_death(agent, cause)

    def reset(self):
        """Reset lifecycle manager state"""
        self.living_agents.clear()
        self.dead_agents.clear()
        self.birth_history.clear()
        self.total_births = 0
        self.total_deaths = 0
        self.death_causes.clear()
        self.last_spawn_time = 0.0
        logger.info("Lifecycle manager reset")
