"""
Swarm Robotics Plugin - Physical robot coordination and control

This plugin provides swarm coordination capabilities for real autonomous systems:
- Formation control and maintenance
- Collective movement with flocking behaviors
- Energy-aware task allocation
- Self-healing formations
- Obstacle avoidance and path planning
"""

import logging
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.plugin_base import Plugin

logger = logging.getLogger('swarm_robotics')


class FormationType(str, Enum):
    """Swarm formation types"""
    LINE = "line"
    CIRCLE = "circle"
    WEDGE = "wedge"
    GRID = "grid"
    CUSTOM = "custom"


class SwarmBehavior(str, Enum):
    """Swarm behavior modes"""
    FLOCKING = "flocking"  # Cohesion, separation, alignment
    FORAGING = "foraging"  # Distributed resource collection
    PERIMETER_DEFENSE = "perimeter_defense"  # Coordinate to protect area
    EMERGENCY_REGROUP = "emergency_regroup"  # Respond to attacks


@dataclass
class SwarmTask:
    """Task to be allocated within swarm"""
    task_id: str
    task_type: str
    energy_requirement: float
    priority: float
    position: Tuple[float, float]
    assigned_agent: Optional[int] = None
    completed: bool = False


class SwarmRoboticsPlugin(Plugin):
    """
    Plugin for swarm robotics coordination.
    
    Implements distributed coordination algorithms for physical robots
    with realistic constraints (battery, communication range, movement costs).
    """
    
    def __init__(
        self,
        formation_type: FormationType = FormationType.CIRCLE,
        behavior_mode: SwarmBehavior = SwarmBehavior.FLOCKING,
        perception_radius: float = 3.0,
        obstacle_avoidance_range: float = 1.5
    ):
        """
        Initialize swarm robotics plugin.
        
        Args:
            formation_type: Desired formation geometry
            behavior_mode: Active swarm behavior
            perception_radius: Range for sensing neighbors
            obstacle_avoidance_range: Range for obstacle detection
        """
        super().__init__()
        self.formation_type = formation_type
        self.behavior_mode = behavior_mode
        self.perception_radius = perception_radius
        self.obstacle_avoidance_range = obstacle_avoidance_range
        
        # Swarm coordination
        self.swarm_center = np.array([0.0, 0.0])
        self.desired_spacing = 2.0  # Desired distance between agents
        
        # Task management
        self.tasks: List[SwarmTask] = []
        self.task_allocations: Dict[int, List[str]] = {}
        
        # Hardware monitoring
        self.hardware_failures: Dict[int, str] = {}
        self.low_battery_threshold = 0.2  # 20% energy
        
        # Flocking parameters
        self.cohesion_weight = 0.3
        self.separation_weight = 0.5
        self.alignment_weight = 0.2
        self.obstacle_avoidance_weight = 0.8
        
        # Performance metrics
        self.formation_coherence = 0.0
        self.task_completion_rate = 0.0
        self.energy_efficiency = 0.0
        
        logger.info(f"SwarmRoboticsPlugin initialized: "
                   f"formation={formation_type.value}, "
                   f"behavior={behavior_mode.value}")
    
    def initialize(self, node: Any) -> None:
        """Initialize plugin for a node"""
        logger.debug(f"SwarmRoboticsPlugin initialized for node {node.node_id}")
    
    def update(self, node: Any, delta_time: float) -> None:
        """
        Update swarm coordination for this node.
        
        Args:
            node: AliveLoopNode instance
            delta_time: Time step
        """
        # Check for hardware failures
        self._check_hardware_status(node)
        
        # Update behavior based on mode
        if self.behavior_mode == SwarmBehavior.FLOCKING:
            self._update_flocking(node, delta_time)
        elif self.behavior_mode == SwarmBehavior.FORAGING:
            self._update_foraging(node, delta_time)
        elif self.behavior_mode == SwarmBehavior.PERIMETER_DEFENSE:
            self._update_perimeter_defense(node, delta_time)
        elif self.behavior_mode == SwarmBehavior.EMERGENCY_REGROUP:
            self._update_emergency_regroup(node, delta_time)
        
        # Update formation control
        self._maintain_formation(node, delta_time)
    
    def _check_hardware_status(self, node: Any) -> None:
        """
        Monitor hardware health via energy levels.
        
        Low energy indicates potential battery failure or hardware issues.
        
        Args:
            node: AliveLoopNode instance
        """
        # Check for low battery
        if node.energy < self.low_battery_threshold * getattr(node, 'max_energy', 10.0):
            if node.node_id not in self.hardware_failures:
                self.hardware_failures[node.node_id] = "low_battery"
                logger.warning(f"Hardware failure detected: Node {node.node_id} - low battery")
        else:
            # Recovery from low battery
            if node.node_id in self.hardware_failures:
                if self.hardware_failures[node.node_id] == "low_battery":
                    del self.hardware_failures[node.node_id]
                    logger.info(f"Node {node.node_id} recovered from low battery")
        
        # Check for communication failures (no recent signals)
        if hasattr(node, 'signal_history') and len(node.signal_history) == 0:
            if node.node_id not in self.hardware_failures:
                self.hardware_failures[node.node_id] = "communication_failure"
                logger.warning(f"Hardware failure detected: Node {node.node_id} - communication failure")
    
    def _update_flocking(self, node: Any, delta_time: float) -> None:
        """
        Update flocking behavior: cohesion, separation, alignment.
        
        Classic boids algorithm adapted for energy-constrained agents.
        
        Args:
            node: AliveLoopNode instance
            delta_time: Time step
        """
        # Skip if hardware failure
        if node.node_id in self.hardware_failures:
            return
        
        # Get nearby agents (simulated - in real system would use sensors)
        neighbors = self._get_neighbors(node)
        
        if not neighbors:
            return
        
        # Calculate flocking forces
        cohesion = self._calculate_cohesion(node, neighbors)
        separation = self._calculate_separation(node, neighbors)
        alignment = self._calculate_alignment(node, neighbors)
        
        # Combine forces with weights
        desired_velocity = (
            self.cohesion_weight * cohesion +
            self.separation_weight * separation +
            self.alignment_weight * alignment
        )
        
        # Limit velocity based on energy
        max_velocity = 1.0 if node.energy > 0.5 else 0.3
        velocity_magnitude = np.linalg.norm(desired_velocity)
        if velocity_magnitude > max_velocity:
            desired_velocity = (desired_velocity / velocity_magnitude) * max_velocity
        
        # Update velocity (smoothly transition)
        node.velocity = 0.7 * node.velocity + 0.3 * desired_velocity
    
    def _calculate_cohesion(self, node: Any, neighbors: List[Any]) -> np.ndarray:
        """
        Calculate cohesion force - move toward center of neighbors.
        
        Args:
            node: AliveLoopNode instance
            neighbors: List of nearby nodes
            
        Returns:
            Cohesion force vector
        """
        if not neighbors:
            return np.zeros_like(node.position)
        
        # Calculate center of mass
        center = np.mean([n.position for n in neighbors], axis=0)
        
        # Direction toward center
        direction = center - node.position
        
        return direction
    
    def _calculate_separation(self, node: Any, neighbors: List[Any]) -> np.ndarray:
        """
        Calculate separation force - avoid crowding neighbors.
        
        Args:
            node: AliveLoopNode instance
            neighbors: List of nearby nodes
            
        Returns:
            Separation force vector
        """
        if not neighbors:
            return np.zeros_like(node.position)
        
        separation = np.zeros_like(node.position)
        
        for neighbor in neighbors:
            distance = np.linalg.norm(node.position - neighbor.position)
            if distance < self.desired_spacing and distance > 0:
                # Push away, stronger when closer
                direction = (node.position - neighbor.position) / distance
                strength = (self.desired_spacing - distance) / self.desired_spacing
                separation += direction * strength
        
        return separation
    
    def _calculate_alignment(self, node: Any, neighbors: List[Any]) -> np.ndarray:
        """
        Calculate alignment force - match velocity with neighbors.
        
        Args:
            node: AliveLoopNode instance
            neighbors: List of nearby nodes
            
        Returns:
            Alignment force vector
        """
        if not neighbors:
            return np.zeros_like(node.velocity)
        
        # Average velocity of neighbors
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        
        # Align with average
        alignment = avg_velocity - node.velocity
        
        return alignment
    
    def _update_foraging(self, node: Any, delta_time: float) -> None:
        """
        Update foraging behavior - distributed resource collection.
        
        Args:
            node: AliveLoopNode instance
            delta_time: Time step
        """
        # Find uncompleted tasks
        available_tasks = [t for t in self.tasks if not t.completed and t.assigned_agent is None]
        
        if not available_tasks:
            # No tasks, return to center
            direction = self.swarm_center - node.position
            distance = np.linalg.norm(direction)
            if distance > 0.5:
                node.velocity = (direction / distance) * 0.5
            return
        
        # Find nearest task within perception range
        nearest_task = None
        min_distance = float('inf')
        
        for task in available_tasks:
            task_pos = np.array(task.position)
            distance = np.linalg.norm(node.position - task_pos)
            if distance < self.perception_radius and distance < min_distance:
                # Check if we have enough energy
                if node.energy > task.energy_requirement:
                    min_distance = distance
                    nearest_task = task
        
        if nearest_task:
            # Move toward task
            task_pos = np.array(nearest_task.position)
            direction = task_pos - node.position
            distance = np.linalg.norm(direction)
            
            if distance < 0.5:
                # Close enough to complete task
                nearest_task.assigned_agent = node.node_id
                nearest_task.completed = True
                node.energy -= nearest_task.energy_requirement
                logger.info(f"Node {node.node_id} completed task {nearest_task.task_id}")
            else:
                # Move toward task
                node.velocity = (direction / distance) * 0.7
    
    def _update_perimeter_defense(self, node: Any, delta_time: float) -> None:
        """
        Update perimeter defense - coordinate to protect area.
        
        Agents distribute evenly around protected area.
        
        Args:
            node: AliveLoopNode instance
            delta_time: Time step
        """
        # Get neighbors for coordination
        neighbors = self._get_neighbors(node)
        
        # Calculate desired position on perimeter
        # Use node ID to distribute evenly
        angle = (node.node_id * 2 * math.pi) / max(1, len(neighbors) + 1)
        radius = 5.0  # Perimeter radius
        
        desired_pos = self.swarm_center + np.array([
            radius * math.cos(angle),
            radius * math.sin(angle)
        ])
        
        # Move toward desired position
        direction = desired_pos - node.position
        distance = np.linalg.norm(direction)
        
        if distance > 0.5:
            node.velocity = (direction / distance) * 0.6
        else:
            # At position, patrol along perimeter
            tangent = np.array([-math.sin(angle), math.cos(angle)])
            node.velocity = tangent * 0.3
    
    def _update_emergency_regroup(self, node: Any, delta_time: float) -> None:
        """
        Update emergency regroup - respond to attacks collectively.
        
        All agents move toward swarm center for mutual protection.
        
        Args:
            node: AliveLoopNode instance
            delta_time: Time step
        """
        # Move urgently toward swarm center
        direction = self.swarm_center - node.position
        distance = np.linalg.norm(direction)
        
        if distance > 1.0:
            # Move quickly
            node.velocity = (direction / distance) * 1.0
        else:
            # At center, form defensive circle
            neighbors = self._get_neighbors(node)
            if neighbors:
                separation = self._calculate_separation(node, neighbors)
                node.velocity = separation * 0.3
    
    def _maintain_formation(self, node: Any, delta_time: float) -> None:
        """
        Maintain desired swarm formation.
        
        Args:
            node: AliveLoopNode instance
            delta_time: Time step
        """
        if self.formation_type == FormationType.CIRCLE:
            self._maintain_circle_formation(node)
        elif self.formation_type == FormationType.LINE:
            self._maintain_line_formation(node)
        elif self.formation_type == FormationType.GRID:
            self._maintain_grid_formation(node)
        # WEDGE and CUSTOM formations can be added similarly
    
    def _maintain_circle_formation(self, node: Any) -> None:
        """Maintain circular formation"""
        # Calculate position on circle
        neighbors = self._get_neighbors(node)
        total_agents = len(neighbors) + 1
        
        angle = (node.node_id * 2 * math.pi) / total_agents
        radius = 3.0
        
        desired_pos = self.swarm_center + np.array([
            radius * math.cos(angle),
            radius * math.sin(angle)
        ])
        
        # Adjust velocity toward desired position
        correction = (desired_pos - node.position) * 0.1
        node.velocity += correction
    
    def _maintain_line_formation(self, node: Any) -> None:
        """Maintain line formation"""
        # Arrange agents in a line
        neighbors = self._get_neighbors(node)
        total_agents = len(neighbors) + 1
        
        # Position along x-axis
        spacing = 2.0
        desired_x = (node.node_id - total_agents / 2) * spacing
        desired_pos = self.swarm_center + np.array([desired_x, 0.0])
        
        # Adjust velocity
        correction = (desired_pos - node.position) * 0.1
        node.velocity += correction
    
    def _maintain_grid_formation(self, node: Any) -> None:
        """Maintain grid formation"""
        # Arrange agents in grid
        neighbors = self._get_neighbors(node)
        total_agents = len(neighbors) + 1
        
        grid_size = int(math.sqrt(total_agents)) + 1
        spacing = 2.0
        
        row = node.node_id // grid_size
        col = node.node_id % grid_size
        
        desired_pos = self.swarm_center + np.array([
            (col - grid_size / 2) * spacing,
            (row - grid_size / 2) * spacing
        ])
        
        # Adjust velocity
        correction = (desired_pos - node.position) * 0.1
        node.velocity += correction
    
    def _get_neighbors(self, node: Any) -> List[Any]:
        """
        Get neighboring agents within perception radius.
        
        In real system, this would use sensors. Here we simulate it.
        
        Args:
            node: AliveLoopNode instance
            
        Returns:
            List of nearby nodes
        """
        # This would be populated by the simulation environment
        # For now, return empty list (will be filled by external coordinator)
        return []
    
    def set_neighbors(self, node: Any, neighbors: List[Any]) -> None:
        """
        Set neighbors for a node (called by external coordinator).
        
        Args:
            node: AliveLoopNode instance
            neighbors: List of nearby nodes
        """
        # Store neighbors for use in next update
        if not hasattr(node, '_swarm_neighbors'):
            node._swarm_neighbors = []
        node._swarm_neighbors = neighbors
    
    def allocate_tasks(self, nodes: List[Any]) -> Dict[int, List[str]]:
        """
        Allocate tasks to agents based on energy levels and proximity.
        
        Args:
            nodes: List of AliveLoopNode instances
            
        Returns:
            Dictionary of node_id -> list of task_ids
        """
        allocations: Dict[int, List[str]] = {n.node_id: [] for n in nodes}
        
        # Sort tasks by priority
        available_tasks = sorted(
            [t for t in self.tasks if not t.completed and t.assigned_agent is None],
            key=lambda t: t.priority,
            reverse=True
        )
        
        for task in available_tasks:
            # Find best agent for task
            best_node = None
            best_score = -float('inf')
            
            task_pos = np.array(task.position)
            
            for node in nodes:
                # Skip if hardware failure or low energy
                if node.node_id in self.hardware_failures:
                    continue
                if node.energy < task.energy_requirement:
                    continue
                
                # Calculate score based on distance and energy
                distance = np.linalg.norm(node.position - task_pos)
                energy_ratio = node.energy / getattr(node, 'max_energy', 10.0)
                
                # Prefer closer agents with more energy
                score = energy_ratio * 2.0 - distance * 0.5
                
                if score > best_score:
                    best_score = score
                    best_node = node
            
            if best_node:
                task.assigned_agent = best_node.node_id
                allocations[best_node.node_id].append(task.task_id)
        
        self.task_allocations = allocations
        return allocations
    
    def add_task(self, task: SwarmTask) -> None:
        """Add a task to the swarm"""
        self.tasks.append(task)
        logger.info(f"Added task {task.task_id}: type={task.task_type}, "
                   f"energy={task.energy_requirement}")
    
    def set_behavior_mode(self, mode: SwarmBehavior) -> None:
        """Change swarm behavior mode"""
        self.behavior_mode = mode
        logger.info(f"Swarm behavior changed to {mode.value}")
    
    def set_formation_type(self, formation: FormationType) -> None:
        """Change formation type"""
        self.formation_type = formation
        logger.info(f"Formation type changed to {formation.value}")
    
    def set_swarm_center(self, center: np.ndarray) -> None:
        """Update swarm center position"""
        self.swarm_center = center
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm performance statistics"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for t in self.tasks if t.completed)
        
        self.task_completion_rate = (
            completed_tasks / total_tasks if total_tasks > 0 else 0.0
        )
        
        return {
            'formation_type': self.formation_type.value,
            'behavior_mode': self.behavior_mode.value,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'completion_rate': self.task_completion_rate,
            'hardware_failures': len(self.hardware_failures),
            'failed_nodes': list(self.hardware_failures.keys())
        }
