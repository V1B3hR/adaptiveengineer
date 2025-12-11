"""
Phase 2 Emergence & Adaptation Plugin

Integrates all Phase 2 features:
- Advanced evolutionary mechanics with behavior strategies
- Swarm intelligence and stigmergy
- Predictive homeostasis and systemic resilience

This plugin enables the emergence of true collective intelligence.
"""

import logging
from typing import Any, Dict, List, Optional

from core.plugin_base import PluginBase, StateVariable
from core.behavior_strategy import (
    BehaviorStrategy,
    BehaviorStrategyFactory,
    AgentRole,
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

logger = logging.getLogger(__name__)


class Phase2EmergencePlugin(PluginBase):
    """
    Plugin for Phase 2: Emergence & Adaptation features.

    Enables:
    - Evolving behavior strategies (not just parameters)
    - Agent role specialization (Scouts, Harvesters, Guardians)
    - Coordinated swarm intelligence with stigmergy
    - Predictive failure prevention
    - Adaptive resource management
    """

    def __init__(
        self,
        plugin_id: str = "phase2_emergence",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Phase 2 Emergence plugin."""
        super().__init__(plugin_id, config)

        self.node = None
        self.swarm_manager = SwarmIntelligenceManager()
        self.homeostasis = PredictiveHomeostasisSystem()

        # Agent's current strategy
        self.behavior_strategy: Optional[BehaviorStrategy] = None
        self.assigned_role: AgentRole = AgentRole.GENERALIST

        # Coordination state
        self.active_coordinations: List[str] = []
        self.supply_chain_ids: List[str] = []

    def get_plugin_type(self) -> str:
        """Return plugin type."""
        return "Phase2Emergence"

    def initialize(self, node: Any) -> None:
        """
        Initialize plugin with host node.

        Args:
            node: Host AliveLoopNode instance
        """
        self.node = node

        # Assign initial role based on config or default
        initial_role = self.config.get("initial_role", "generalist")
        self.assigned_role = AgentRole(initial_role)

        # Create initial behavior strategy
        self.behavior_strategy = self._create_initial_strategy(
            self.assigned_role
        )

        # Initialize state variables
        self.state_variables = self.get_state_schema()

        # Allocate resources for this agent
        node_id = str(getattr(node, "node_id", "unknown"))
        agent_id = str(node_id)
        priority = self._calculate_priority()

        self.homeostasis.allocate_resources(
            entity_id=agent_id, node_id=node_id, priority=priority
        )

        logger.info(
            f"Phase 2 Emergence plugin initialized for node {node_id} "
            f"with role {self.assigned_role.value}"
        )

    def _create_initial_strategy(self, role: AgentRole) -> BehaviorStrategy:
        """Create initial behavior strategy for role"""
        factory = BehaviorStrategyFactory()

        if role == AgentRole.SCOUT:
            return factory.create_scout_strategy()
        elif role == AgentRole.HARVESTER:
            return factory.create_harvester_strategy()
        elif role == AgentRole.GUARDIAN:
            return factory.create_guardian_strategy()
        else:
            return factory.create_generalist_strategy()

    def _calculate_priority(self) -> float:
        """Calculate agent priority based on role and state"""
        if not self.node:
            return 0.5

        # Base priority on role
        role_priorities = {
            AgentRole.SCOUT: 0.6,
            AgentRole.HARVESTER: 0.5,
            AgentRole.GUARDIAN: 0.7,
            AgentRole.COORDINATOR: 0.8,
            AgentRole.HEALER: 0.7,
            AgentRole.ANALYZER: 0.6,
            AgentRole.GENERALIST: 0.5,
        }

        base_priority = role_priorities.get(self.assigned_role, 0.5)

        # Adjust based on health and energy
        health = getattr(self.node, "health", 1.0)
        energy = getattr(self.node, "energy", 10.0) / 10.0  # Normalize

        # Critical agents get higher priority when healthy
        adjustment = (health + energy) / 2.0

        return max(0.1, min(1.0, base_priority * adjustment))

    def get_state_schema(self) -> Dict[str, StateVariable]:
        """
        Define Phase 2 Emergence state variables.

        Returns:
            Dictionary of state variable definitions
        """
        return {
            # Role and specialization (stored as numeric for compatibility)
            "role": StateVariable(
                name="role",
                value=0.0,  # Will store role index
                min_value=0.0,
                max_value=10.0,
                metadata={
                    "description": "Agent specialization role (numeric)",
                    "role_name": self.assigned_role.value,
                },
            ),
            "specialization_score": StateVariable(
                name="specialization_score",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                metadata={"description": "How specialized vs generalist"},
            ),
            "cooperation_score": StateVariable(
                name="cooperation_score",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                metadata={"description": "Cooperation effectiveness"},
            ),
            # Swarm coordination
            "active_coordinations": StateVariable(
                name="active_coordinations",
                value=0.0,
                metadata={"description": "Number of active coordinations"},
            ),
            "pheromones_deposited": StateVariable(
                name="pheromones_deposited",
                value=0.0,
                metadata={"description": "Pheromones deposited"},
            ),
            "supply_chains_active": StateVariable(
                name="supply_chains_active",
                value=0.0,
                metadata={"description": "Active supply chains"},
            ),
            # Predictive capabilities
            "predictions_made": StateVariable(
                name="predictions_made",
                value=0.0,
                metadata={"description": "Failure predictions made"},
            ),
            "failures_prevented": StateVariable(
                name="failures_prevented",
                value=0.0,
                metadata={"description": "Failures successfully prevented"},
            ),
            # Resource management
            "resource_efficiency": StateVariable(
                name="resource_efficiency",
                value=1.0,
                min_value=0.0,
                max_value=2.0,
                metadata={"description": "Resource usage efficiency"},
            ),
            # Overall emergence score
            "emergence_level": StateVariable(
                name="emergence_level",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                metadata={
                    "description": "Level of emergent collective intelligence"
                },
            ),
        }

    def update_state(self, delta_time: float) -> None:
        """
        Update state variables and systems.

        Args:
            delta_time: Time elapsed since last update
        """
        if not self.node or not self.behavior_strategy:
            return

        # Update swarm intelligence
        self.swarm_manager.update(delta_time)

        # Update state variables (convert role to numeric)
        role_index = list(AgentRole).index(self.assigned_role)
        self.update_state_variable("role", float(role_index))

        # Compute specialization
        spec_score = self.behavior_strategy.compute_specialization()
        self.update_state_variable("specialization_score", spec_score)

        self.update_state_variable(
            "cooperation_score", self.behavior_strategy.cooperation_score
        )
        self.update_state_variable(
            "active_coordinations", float(len(self.active_coordinations))
        )
        self.update_state_variable(
            "supply_chains_active", float(len(self.supply_chain_ids))
        )

        # Update homeostasis metrics
        stats = self.homeostasis.get_statistics()
        self.update_state_variable(
            "predictions_made", float(stats["predictions_made"])
        )
        self.update_state_variable(
            "failures_prevented", float(stats["failures_prevented"])
        )

        # Update resource efficiency
        agent_id = str(getattr(self.node, "node_id", "unknown"))
        if agent_id in self.homeostasis.allocations:
            allocation = self.homeostasis.allocations[agent_id]
            allocation.update_efficiency()
            self.update_state_variable(
                "resource_efficiency", allocation.efficiency
            )

        # Calculate emergence level
        emergence = self._calculate_emergence_level()
        self.update_state_variable("emergence_level", emergence)

    def _calculate_emergence_level(self) -> float:
        """
        Calculate overall emergence level (0-1).

        Higher values indicate more sophisticated emergent behaviors.
        """
        factors = []

        # Specialization (0.25 weight)
        spec_score = self.state_variables["specialization_score"].value
        factors.append(spec_score * 0.25)

        # Cooperation (0.25 weight)
        coop_score = self.state_variables["cooperation_score"].value
        factors.append(coop_score * 0.25)

        # Active coordinations (0.2 weight)
        coord_score = min(1.0, len(self.active_coordinations) / 3.0)
        factors.append(coord_score * 0.2)

        # Supply chains (0.15 weight)
        chain_score = min(1.0, len(self.supply_chain_ids) / 2.0)
        factors.append(chain_score * 0.15)

        # Predictive success (0.15 weight)
        stats = self.homeostasis.get_statistics()
        if stats["predictions_made"] > 0:
            pred_score = stats["prediction_accuracy"]
        else:
            pred_score = 0.0
        factors.append(pred_score * 0.15)

        return sum(factors)

    def process_signal(self, signal: Any) -> Optional[Any]:
        """
        Process Phase 2 signals.

        Args:
            signal: Incoming signal

        Returns:
            Optional response signal
        """
        # Handle coordination invitations, pheromone sensing, etc.
        return None

    def get_actions(self) -> List[str]:
        """Return available Phase 2 actions."""
        return [
            "deposit_pheromone",
            "sense_pheromones",
            "join_coordination",
            "establish_supply_chain",
            "predict_failure",
            "migrate_self",
            "adaptive_reallocate",
            "execute_behavior_tree",
            "update_fsm_state",
        ]

    def execute_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """
        Execute a Phase 2 action.

        Args:
            action_type: Type of action
            params: Action parameters

        Returns:
            True if action succeeded
        """
        if not self.node:
            return False

        node_id = str(getattr(self.node, "node_id", "unknown"))

        if action_type == "deposit_pheromone":
            return self._deposit_pheromone(node_id, params)

        elif action_type == "sense_pheromones":
            return self._sense_pheromones(node_id, params)

        elif action_type == "join_coordination":
            return self._join_coordination(params)

        elif action_type == "establish_supply_chain":
            return self._establish_supply_chain(params)

        elif action_type == "predict_failure":
            return self._predict_failure(params)

        elif action_type == "migrate_self":
            return self._migrate_self(params)

        elif action_type == "adaptive_reallocate":
            return self._adaptive_reallocate(params)

        elif action_type == "execute_behavior_tree":
            return self._execute_behavior_tree(params)

        elif action_type == "update_fsm_state":
            return self._update_fsm_state(params)

        return False

    def _deposit_pheromone(self, node_id: str, params: Dict[str, Any]) -> bool:
        """Deposit a structured pheromone"""
        pheromone_type = params.get("type", "exploration")

        try:
            ptype = PheromoneType(pheromone_type)
        except ValueError:
            return False

        pheromone = self.swarm_manager.deposit_pheromone(
            node_id=node_id,
            pheromone_type=ptype,
            depositor_id=int(getattr(self.node, "node_id", 0)),
            depositor_role=self.assigned_role.value,
            signature=params.get("signature"),
            confidence=params.get("confidence", 0.5),
            priority=params.get("priority", 0.5),
            quantity=params.get("quantity", 1.0),
        )

        # Update counter
        current = self.state_variables["pheromones_deposited"].value
        self.update_state_variable("pheromones_deposited", current + 1)

        return True

    def _sense_pheromones(self, node_id: str, params: Dict[str, Any]) -> bool:
        """Sense pheromones at current location"""
        pheromone_types = params.get("types")
        if pheromone_types:
            pheromone_types = [PheromoneType(t) for t in pheromone_types]

        pheromones = self.swarm_manager.sense_pheromones(
            node_id=node_id,
            pheromone_types=pheromone_types,
            min_intensity=params.get("min_intensity", 0.1),
        )

        # Store sensed pheromones for decision making
        if hasattr(self.node, "sensed_pheromones"):
            self.node.sensed_pheromones = pheromones

        return len(pheromones) > 0

    def _join_coordination(self, params: Dict[str, Any]) -> bool:
        """Join a swarm coordination"""
        coordination_id = params.get("coordination_id")
        if not coordination_id:
            return False

        agent_id = int(getattr(self.node, "node_id", 0))

        # Check if coordination exists in this manager's swarm_manager
        # If not, it might be in another agent's manager - skip for now
        if coordination_id not in self.swarm_manager.coordinations:
            # Try to join anyway - it might be created later or in shared manager
            return False

        success = self.swarm_manager.join_coordination(
            coordination_id, agent_id, self.assigned_role.value
        )

        if success:
            self.active_coordinations.append(coordination_id)

        return success

    def _establish_supply_chain(self, params: Dict[str, Any]) -> bool:
        """Establish a supply chain"""
        nodes = params.get("nodes", [])
        resource_type = params.get("resource_type", "data")

        if len(nodes) < 2:
            return False

        chain_id = f"chain_{int(getattr(self.node, 'node_id', 0))}_{len(self.supply_chain_ids)}"

        success = self.swarm_manager.establish_supply_chain(
            chain_id=chain_id, nodes=nodes, resource_type=resource_type
        )

        if success:
            self.supply_chain_ids.append(chain_id)

        return success

    def _predict_failure(self, params: Dict[str, Any]) -> bool:
        """Make failure predictions"""
        current_metrics = params.get("metrics", {})

        predictions = self.homeostasis.predict_failure(
            current_metrics=current_metrics,
            confidence_threshold=params.get("confidence_threshold", 0.6),
        )

        # Store predictions for action
        if hasattr(self.node, "failure_predictions"):
            self.node.failure_predictions = predictions

        return len(predictions) > 0

    def _migrate_self(self, params: Dict[str, Any]) -> bool:
        """Migrate self to another node"""
        to_node = params.get("to_node")
        reason = params.get("reason", "load_balancing")

        if not to_node:
            return False

        agent_id = str(getattr(self.node, "node_id", "unknown"))
        from_node = str(getattr(self.node, "current_node", "unknown"))

        return self.homeostasis.migrate_agent(
            agent_id=agent_id,
            from_node=from_node,
            to_node=to_node,
            reason=reason,
        )

    def _adaptive_reallocate(self, params: Dict[str, Any]) -> bool:
        """Trigger adaptive resource reallocation"""
        stress_type_str = params.get("stress_type", "cpu_pressure")
        affected_nodes = set(params.get("affected_nodes", []))

        try:
            stress_type = StressType(stress_type_str)
        except ValueError:
            return False

        result = self.homeostasis.adaptive_reallocation(
            stress_type=stress_type, affected_nodes=affected_nodes
        )

        return len(result.get("reallocations", [])) > 0

    def _execute_behavior_tree(self, params: Dict[str, Any]) -> bool:
        """Execute agent's behavior tree"""
        if (
            not self.behavior_strategy
            or not self.behavior_strategy.behavior_tree
        ):
            return False

        context = params.get("context", {})

        # Add agent state to context
        context["agent"] = self.node
        context["role"] = self.assigned_role.value

        status = self.behavior_strategy.behavior_tree.execute(context)
        return status.value == "success"

    def _update_fsm_state(self, params: Dict[str, Any]) -> bool:
        """Update finite state machine"""
        if not self.behavior_strategy or not self.behavior_strategy.fsm:
            return False

        context = params.get("context", {})

        # Add agent state to context
        context["agent"] = self.node
        context["role"] = self.assigned_role.value

        new_state, changed = self.behavior_strategy.fsm.update(context)
        return changed

    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get comprehensive Phase 2 emergence summary"""
        swarm_stats = self.swarm_manager.get_statistics()
        homeostasis_stats = self.homeostasis.get_statistics()

        return {
            "role": self.assigned_role.value,
            "specialization_score": self.state_variables[
                "specialization_score"
            ].value,
            "cooperation_score": self.state_variables[
                "cooperation_score"
            ].value,
            "emergence_level": self.state_variables["emergence_level"].value,
            "swarm_intelligence": swarm_stats,
            "predictive_homeostasis": homeostasis_stats,
            "active_coordinations": len(self.active_coordinations),
            "supply_chains": len(self.supply_chain_ids),
        }
