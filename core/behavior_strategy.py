"""
Behavior Strategy System for Phase 2: Emergence & Adaptation

This module implements advanced evolutionary mechanics that allow agents to evolve
entire behavior trees and finite-state machines, not just simple parameters.

Key Features:
- Behavior Trees and Finite-State Machines as evolvable strategies
- Agent role specialization (Scouts, Harvesters, Guardians, etc.)
- Genetic encoding of complex strategies
- Fitness pressures for role diversity and cooperation
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import json
import time

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent specialization roles that emerge through evolution"""

    SCOUT = "scout"  # Fast-moving explorers, high mobility
    HARVESTER = "harvester"  # Resource gatherers, efficient collectors
    GUARDIAN = "guardian"  # Defensive specialists, threat responders
    COORDINATOR = "coordinator"  # Communication hubs, organize swarms
    HEALER = "healer"  # Support agents, restore health/energy
    ANALYZER = "analyzer"  # Pattern recognition, threat assessment
    GENERALIST = "generalist"  # Balanced, adaptable to various tasks


class NodeType(str, Enum):
    """Behavior tree node types"""

    # Control nodes
    SEQUENCE = "sequence"  # Execute children in order, fail on first failure
    SELECTOR = "selector"  # Try children in order, succeed on first success
    PARALLEL = "parallel"  # Execute all children simultaneously

    # Decorator nodes
    INVERTER = "inverter"  # Invert child result
    REPEATER = "repeater"  # Repeat child N times

    # Leaf nodes (actions/conditions)
    CONDITION = "condition"  # Check a condition
    ACTION = "action"  # Perform an action


class BehaviorStatus(str, Enum):
    """Execution status for behavior tree nodes"""

    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class StateType(str, Enum):
    """FSM state types"""

    IDLE = "idle"
    PATROL = "patrol"
    GATHER = "gather"
    DEFEND = "defend"
    RETREAT = "retreat"
    COORDINATE = "coordinate"
    HEAL = "heal"
    ANALYZE = "analyze"


@dataclass
class BehaviorNode:
    """
    A node in a behavior tree.

    Can represent control flow (sequence, selector) or leaf actions/conditions.
    Evolvable through genetic operators.
    """

    node_id: str
    node_type: NodeType
    name: str
    children: List["BehaviorNode"] = field(default_factory=list)

    # For condition/action nodes
    condition_fn: Optional[str] = None  # Serializable function name
    action_fn: Optional[str] = None  # Serializable function name
    parameters: Dict[str, float] = field(default_factory=dict)

    # Execution state
    status: BehaviorStatus = BehaviorStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for genetic encoding"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "children": [c.to_dict() for c in self.children],
            "condition_fn": self.condition_fn,
            "action_fn": self.action_fn,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorNode":
        """Deserialize from dictionary"""
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            children=children,
            condition_fn=data.get("condition_fn"),
            action_fn=data.get("action_fn"),
            parameters=data.get("parameters", {}),
        )

    def execute(self, context: Dict[str, Any]) -> BehaviorStatus:
        """
        Execute this behavior node with given context.

        Args:
            context: Execution context (agent state, environment, etc.)

        Returns:
            BehaviorStatus indicating success/failure/running
        """
        if self.node_type == NodeType.SEQUENCE:
            # Execute all children in sequence
            for child in self.children:
                status = child.execute(context)
                if status != BehaviorStatus.SUCCESS:
                    self.status = status
                    return status
            self.status = BehaviorStatus.SUCCESS
            return self.status

        elif self.node_type == NodeType.SELECTOR:
            # Try children until one succeeds
            for child in self.children:
                status = child.execute(context)
                if status == BehaviorStatus.SUCCESS:
                    self.status = status
                    return status
            self.status = BehaviorStatus.FAILURE
            return self.status

        elif self.node_type == NodeType.PARALLEL:
            # Execute all children, succeed if any succeed
            results = [child.execute(context) for child in self.children]
            if any(r == BehaviorStatus.SUCCESS for r in results):
                self.status = BehaviorStatus.SUCCESS
            elif all(r == BehaviorStatus.FAILURE for r in results):
                self.status = BehaviorStatus.FAILURE
            else:
                self.status = BehaviorStatus.RUNNING
            return self.status

        elif self.node_type == NodeType.CONDITION:
            # Evaluate condition
            if self.condition_fn and self.condition_fn in context.get(
                "conditions", {}
            ):
                result = context["conditions"][self.condition_fn](
                    context, self.parameters
                )
                self.status = (
                    BehaviorStatus.SUCCESS
                    if result
                    else BehaviorStatus.FAILURE
                )
            else:
                self.status = BehaviorStatus.FAILURE
            return self.status

        elif self.node_type == NodeType.ACTION:
            # Execute action
            if self.action_fn and self.action_fn in context.get("actions", {}):
                result = context["actions"][self.action_fn](
                    context, self.parameters
                )
                self.status = (
                    BehaviorStatus.SUCCESS
                    if result
                    else BehaviorStatus.FAILURE
                )
            else:
                self.status = BehaviorStatus.FAILURE
            return self.status

        return BehaviorStatus.FAILURE


@dataclass
class FSMTransition:
    """Transition in a finite-state machine"""

    from_state: StateType
    to_state: StateType
    condition: str  # Condition function name
    priority: float = 0.5  # For conflict resolution
    parameters: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "condition": self.condition,
            "priority": self.priority,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FSMTransition":
        return cls(
            from_state=StateType(data["from_state"]),
            to_state=StateType(data["to_state"]),
            condition=data["condition"],
            priority=data.get("priority", 0.5),
            parameters=data.get("parameters", {}),
        )


@dataclass
class FiniteStateMachine:
    """
    A finite-state machine for agent behavior.

    Defines states and transitions between them based on conditions.
    Evolvable through genetic operators.
    """

    fsm_id: str
    current_state: StateType
    transitions: List[FSMTransition] = field(default_factory=list)
    state_actions: Dict[StateType, str] = field(
        default_factory=dict
    )  # State -> action name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fsm_id": self.fsm_id,
            "current_state": self.current_state.value,
            "transitions": [t.to_dict() for t in self.transitions],
            "state_actions": {
                k.value: v for k, v in self.state_actions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FiniteStateMachine":
        return cls(
            fsm_id=data["fsm_id"],
            current_state=StateType(data["current_state"]),
            transitions=[
                FSMTransition.from_dict(t) for t in data.get("transitions", [])
            ],
            state_actions={
                StateType(k): v
                for k, v in data.get("state_actions", {}).items()
            },
        )

    def update(self, context: Dict[str, Any]) -> Tuple[StateType, bool]:
        """
        Update FSM state based on transitions.

        Args:
            context: Execution context with conditions

        Returns:
            Tuple of (new_state, changed)
        """
        # Find valid transitions from current state
        valid_transitions = [
            t for t in self.transitions if t.from_state == self.current_state
        ]

        # Sort by priority (highest first)
        valid_transitions.sort(key=lambda t: t.priority, reverse=True)

        # Try each transition
        for transition in valid_transitions:
            if transition.condition in context.get("conditions", {}):
                if context["conditions"][transition.condition](
                    context, transition.parameters
                ):
                    # Transition is valid
                    old_state = self.current_state
                    self.current_state = transition.to_state
                    return self.current_state, old_state != self.current_state

        # No transition occurred
        return self.current_state, False

    def execute_state_action(self, context: Dict[str, Any]) -> bool:
        """Execute the action associated with current state"""
        if self.current_state in self.state_actions:
            action_name = self.state_actions[self.current_state]
            if action_name in context.get("actions", {}):
                return context["actions"][action_name](context, {})
        return False


@dataclass
class BehaviorStrategy:
    """
    Complete behavior strategy for an agent.

    Combines behavior tree, FSM, role specialization, and parameters.
    This is what evolves through genetic algorithms.
    """

    strategy_id: str
    role: AgentRole
    behavior_tree: Optional[BehaviorNode] = None
    fsm: Optional[FiniteStateMachine] = None

    # Role-specific parameters (evolved)
    speed_modifier: float = 1.0  # Movement speed multiplier
    energy_efficiency: float = 1.0  # Energy consumption multiplier
    detection_range: float = 1.0  # Sensor range multiplier
    communication_range: float = 1.0  # Signal range multiplier
    defensive_strength: float = 1.0  # Defense capability multiplier
    offensive_strength: float = 1.0  # Attack capability multiplier

    # Performance tracking
    fitness: float = 0.0
    generation: int = 0
    evaluations: int = 0
    parent_ids: List[str] = field(default_factory=list)

    # Specialization metrics (computed)
    specialization_score: float = 0.0  # How specialized vs generalist
    cooperation_score: float = 0.0  # How well it cooperates

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for genetic encoding"""
        return {
            "strategy_id": self.strategy_id,
            "role": self.role.value,
            "behavior_tree": (
                self.behavior_tree.to_dict() if self.behavior_tree else None
            ),
            "fsm": self.fsm.to_dict() if self.fsm else None,
            "speed_modifier": self.speed_modifier,
            "energy_efficiency": self.energy_efficiency,
            "detection_range": self.detection_range,
            "communication_range": self.communication_range,
            "defensive_strength": self.defensive_strength,
            "offensive_strength": self.offensive_strength,
            "fitness": self.fitness,
            "generation": self.generation,
            "evaluations": self.evaluations,
            "parent_ids": self.parent_ids,
            "specialization_score": self.specialization_score,
            "cooperation_score": self.cooperation_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorStrategy":
        """Deserialize from dictionary"""
        bt_data = data.get("behavior_tree")
        fsm_data = data.get("fsm")

        return cls(
            strategy_id=data["strategy_id"],
            role=AgentRole(data["role"]),
            behavior_tree=BehaviorNode.from_dict(bt_data) if bt_data else None,
            fsm=FiniteStateMachine.from_dict(fsm_data) if fsm_data else None,
            speed_modifier=data.get("speed_modifier", 1.0),
            energy_efficiency=data.get("energy_efficiency", 1.0),
            detection_range=data.get("detection_range", 1.0),
            communication_range=data.get("communication_range", 1.0),
            defensive_strength=data.get("defensive_strength", 1.0),
            offensive_strength=data.get("offensive_strength", 1.0),
            fitness=data.get("fitness", 0.0),
            generation=data.get("generation", 0),
            evaluations=data.get("evaluations", 0),
            parent_ids=data.get("parent_ids", []),
            specialization_score=data.get("specialization_score", 0.0),
            cooperation_score=data.get("cooperation_score", 0.0),
        )

    def compute_specialization(self) -> float:
        """
        Compute how specialized this strategy is.

        Returns a score from 0 (generalist) to 1 (highly specialized).
        Specialization means high values in some parameters and low in others.
        """
        params = [
            self.speed_modifier,
            self.energy_efficiency,
            self.detection_range,
            self.communication_range,
            self.defensive_strength,
            self.offensive_strength,
        ]

        # Compute variance - high variance means specialized
        mean = sum(params) / len(params)
        variance = sum((p - mean) ** 2 for p in params) / len(params)

        # Normalize to 0-1 range (variance for [0,2] range is max ~1)
        self.specialization_score = min(1.0, variance)
        return self.specialization_score


class BehaviorStrategyFactory:
    """Factory for creating initial behavior strategies for different roles"""

    @staticmethod
    def create_scout_strategy() -> BehaviorStrategy:
        """Create a scout strategy optimized for exploration"""
        strategy_id = str(uuid.uuid4())

        # Scout behavior tree: Explore -> Detect -> Signal
        root = BehaviorNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.SEQUENCE,
            name="scout_behavior",
            children=[
                BehaviorNode(
                    node_id=str(uuid.uuid4()),
                    node_type=NodeType.CONDITION,
                    name="has_energy",
                    condition_fn="check_energy",
                    parameters={"threshold": 0.3},
                ),
                BehaviorNode(
                    node_id=str(uuid.uuid4()),
                    node_type=NodeType.ACTION,
                    name="move_to_unexplored",
                    action_fn="explore",
                    parameters={"speed_multiplier": 1.5},
                ),
                BehaviorNode(
                    node_id=str(uuid.uuid4()),
                    node_type=NodeType.ACTION,
                    name="scan_environment",
                    action_fn="scan",
                    parameters={"range_multiplier": 1.2},
                ),
            ],
        )

        # Scout FSM
        fsm = FiniteStateMachine(
            fsm_id=str(uuid.uuid4()),
            current_state=StateType.PATROL,
            transitions=[
                FSMTransition(
                    StateType.PATROL, StateType.ANALYZE, "detect_threat", 0.9
                ),
                FSMTransition(
                    StateType.ANALYZE, StateType.PATROL, "threat_cleared", 0.7
                ),
                FSMTransition(
                    StateType.PATROL, StateType.IDLE, "low_energy", 0.8
                ),
            ],
            state_actions={
                StateType.PATROL: "patrol",
                StateType.ANALYZE: "analyze_threat",
                StateType.IDLE: "rest",
            },
        )

        return BehaviorStrategy(
            strategy_id=strategy_id,
            role=AgentRole.SCOUT,
            behavior_tree=root,
            fsm=fsm,
            speed_modifier=1.5,
            energy_efficiency=0.8,
            detection_range=1.5,
            communication_range=1.2,
            defensive_strength=0.6,
            offensive_strength=0.5,
        )

    @staticmethod
    def create_harvester_strategy() -> BehaviorStrategy:
        """Create a harvester strategy optimized for resource gathering"""
        strategy_id = str(uuid.uuid4())

        root = BehaviorNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.SEQUENCE,
            name="harvester_behavior",
            children=[
                BehaviorNode(
                    node_id=str(uuid.uuid4()),
                    node_type=NodeType.CONDITION,
                    name="resource_available",
                    condition_fn="check_resources",
                    parameters={"threshold": 0.1},
                ),
                BehaviorNode(
                    node_id=str(uuid.uuid4()),
                    node_type=NodeType.ACTION,
                    name="gather_resource",
                    action_fn="gather",
                    parameters={"efficiency": 1.3},
                ),
            ],
        )

        fsm = FiniteStateMachine(
            fsm_id=str(uuid.uuid4()),
            current_state=StateType.GATHER,
            transitions=[
                FSMTransition(
                    StateType.GATHER, StateType.IDLE, "inventory_full", 0.9
                ),
                FSMTransition(
                    StateType.IDLE, StateType.GATHER, "inventory_empty", 0.8
                ),
            ],
            state_actions={
                StateType.GATHER: "harvest_resources",
                StateType.IDLE: "process_resources",
            },
        )

        return BehaviorStrategy(
            strategy_id=strategy_id,
            role=AgentRole.HARVESTER,
            behavior_tree=root,
            fsm=fsm,
            speed_modifier=0.9,
            energy_efficiency=1.3,
            detection_range=0.8,
            communication_range=0.9,
            defensive_strength=0.5,
            offensive_strength=0.3,
        )

    @staticmethod
    def create_guardian_strategy() -> BehaviorStrategy:
        """Create a guardian strategy optimized for defense"""
        strategy_id = str(uuid.uuid4())

        root = BehaviorNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.SELECTOR,
            name="guardian_behavior",
            children=[
                BehaviorNode(
                    node_id=str(uuid.uuid4()),
                    node_type=NodeType.SEQUENCE,
                    name="defend_sequence",
                    children=[
                        BehaviorNode(
                            node_id=str(uuid.uuid4()),
                            node_type=NodeType.CONDITION,
                            name="threat_detected",
                            condition_fn="detect_threat",
                            parameters={"sensitivity": 0.7},
                        ),
                        BehaviorNode(
                            node_id=str(uuid.uuid4()),
                            node_type=NodeType.ACTION,
                            name="engage_threat",
                            action_fn="defend",
                            parameters={"strength": 1.5},
                        ),
                    ],
                ),
                BehaviorNode(
                    node_id=str(uuid.uuid4()),
                    node_type=NodeType.ACTION,
                    name="patrol_area",
                    action_fn="patrol",
                    parameters={"vigilance": 1.2},
                ),
            ],
        )

        fsm = FiniteStateMachine(
            fsm_id=str(uuid.uuid4()),
            current_state=StateType.PATROL,
            transitions=[
                FSMTransition(
                    StateType.PATROL, StateType.DEFEND, "threat_detected", 0.95
                ),
                FSMTransition(
                    StateType.DEFEND,
                    StateType.PATROL,
                    "threat_neutralized",
                    0.9,
                ),
                FSMTransition(
                    StateType.DEFEND, StateType.RETREAT, "overwhelmed", 0.85
                ),
            ],
            state_actions={
                StateType.PATROL: "patrol_perimeter",
                StateType.DEFEND: "engage_threat",
                StateType.RETREAT: "tactical_retreat",
            },
        )

        return BehaviorStrategy(
            strategy_id=strategy_id,
            role=AgentRole.GUARDIAN,
            behavior_tree=root,
            fsm=fsm,
            speed_modifier=1.0,
            energy_efficiency=0.9,
            detection_range=1.3,
            communication_range=1.0,
            defensive_strength=1.5,
            offensive_strength=1.4,
        )

    @staticmethod
    def create_generalist_strategy() -> BehaviorStrategy:
        """Create a balanced generalist strategy"""
        strategy_id = str(uuid.uuid4())

        return BehaviorStrategy(
            strategy_id=strategy_id,
            role=AgentRole.GENERALIST,
            behavior_tree=None,  # Will use default behaviors
            fsm=None,
            speed_modifier=1.0,
            energy_efficiency=1.0,
            detection_range=1.0,
            communication_range=1.0,
            defensive_strength=1.0,
            offensive_strength=1.0,
        )
