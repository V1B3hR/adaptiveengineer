#!/usr/bin/env python3
"""
Swarm Robotics Demonstration - Coordinated Multi-Agent Control

Demonstrates:
- Formation control and maintenance
- Flocking behaviors (cohesion, separation, alignment)
- Energy-aware task allocation
- Self-healing formations when nodes fail
- Collective movement coordination
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptiveengineer import AliveLoopNode
from plugins.swarm_robotics import (
    SwarmRoboticsPlugin,
    FormationType,
    SwarmBehavior,
    SwarmTask
)


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def visualize_swarm(nodes, title="Swarm State"):
    """Simple text-based visualization of swarm"""
    print(f"\n{title}:")
    print("Position Map (X: 0-20, Y: 0-20):")
    
    # Create simple ASCII grid
    grid_size = 20
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    for node_id, node in nodes.items():
        x = int(node.position[0]) % grid_size
        y = int(node.position[1]) % grid_size
        
        # Mark node position
        if 0 <= x < grid_size and 0 <= y < grid_size:
            if node.energy < 0.3:
                grid[y][x] = '!'  # Low energy
            elif node.energy < 0.6:
                grid[y][x] = 'o'  # Medium energy
            else:
                grid[y][x] = 'O'  # Full energy
    
    # Print grid
    print("  +" + "-" * grid_size + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * grid_size + "+")
    print("  Legend: O=High Energy, o=Medium, !=Low, ' '=Empty\n")


def main():
    print("\n🤖 SWARM ROBOTICS DEMONSTRATION 🤖")
    print("Distributed Coordination and Control\n")
    
    # Initialize swarm agents
    print_section("Initializing Swarm Agents")
    
    num_agents = 8
    agents = {}
    
    for i in range(num_agents):
        # Distribute agents randomly
        node = AliveLoopNode(
            position=(
                np.random.uniform(5, 15),
                np.random.uniform(5, 15)
            ),
            velocity=(0.0, 0.0),
            initial_energy=10.0,
            node_id=i
        )
        node.max_energy = 10.0
        agents[i] = node
        print(f"✓ Agent {i}: Position ({node.position[0]:.1f}, {node.position[1]:.1f})")
    
    # Initialize swarm robotics plugin
    print_section("Configuring Swarm Behaviors")
    
    swarm_plugin = SwarmRoboticsPlugin(
        formation_type=FormationType.CIRCLE,
        behavior_mode=SwarmBehavior.FLOCKING,
        perception_radius=5.0
    )
    
    print(f"✓ Formation: {swarm_plugin.formation_type.value}")
    print(f"✓ Behavior: {swarm_plugin.behavior_mode.value}")
    print(f"✓ Perception Radius: {swarm_plugin.perception_radius}")
    
    # Initialize plugin for each agent
    for node in agents.values():
        swarm_plugin.initialize(node)
    
    # Set swarm center
    swarm_plugin.set_swarm_center(np.array([10.0, 10.0]))
    print(f"✓ Swarm Center: (10.0, 10.0)")
    
    # Phase 1: Formation Control
    print_section("Phase 1: Formation Control")
    print("Agents forming circular formation...\n")
    
    for step in range(30):
        # Update each agent with neighbors
        for node_id, node in agents.items():
            # Find neighbors within perception radius
            neighbors = []
            for other_id, other in agents.items():
                if other_id != node_id:
                    distance = np.linalg.norm(node.position - other.position)
                    if distance < swarm_plugin.perception_radius:
                        neighbors.append(other)
            
            # Store neighbors for plugin
            node._swarm_neighbors = neighbors
            
            # Update swarm behavior
            swarm_plugin.update(node, delta_time=0.1)
            
            # Move agent
            node.position = node.position + node.velocity * 0.1
            
            # Energy cost for movement
            movement_cost = np.linalg.norm(node.velocity) * 0.01
            node.energy = max(0.0, node.energy - movement_cost)
        
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}:")
            for node_id, node in agents.items():
                print(f"  Agent {node_id}: Pos ({node.position[0]:.1f}, {node.position[1]:.1f}), "
                      f"Energy {node.energy:.2f}")
    
    visualize_swarm(agents, "Formation After 30 Steps")
    
    # Phase 2: Task Allocation
    print_section("Phase 2: Energy-Aware Task Allocation")
    
    # Change to foraging mode
    swarm_plugin.set_behavior_mode(SwarmBehavior.FORAGING)
    print("✓ Switched to FORAGING mode")
    
    # Add tasks
    tasks = [
        SwarmTask(
            task_id="task_1",
            task_type="collect",
            energy_requirement=0.5,
            priority=0.8,
            position=(12.0, 8.0)
        ),
        SwarmTask(
            task_id="task_2",
            task_type="collect",
            energy_requirement=0.7,
            priority=0.6,
            position=(8.0, 12.0)
        ),
        SwarmTask(
            task_id="task_3",
            task_type="collect",
            energy_requirement=0.3,
            priority=0.9,
            position=(15.0, 15.0)
        )
    ]
    
    for task in tasks:
        swarm_plugin.add_task(task)
        print(f"✓ Added {task.task_id}: Priority {task.priority}, "
              f"Energy Req {task.energy_requirement}")
    
    # Allocate tasks
    print("\nAllocating tasks...")
    allocations = swarm_plugin.allocate_tasks(list(agents.values()))
    
    for agent_id, task_ids in allocations.items():
        if task_ids:
            print(f"  Agent {agent_id} assigned: {', '.join(task_ids)}")
    
    # Execute foraging
    print("\nExecuting foraging behavior...")
    for step in range(40):
        for node_id, node in agents.items():
            # Update neighbors
            neighbors = []
            for other_id, other in agents.items():
                if other_id != node_id:
                    distance = np.linalg.norm(node.position - other.position)
                    if distance < swarm_plugin.perception_radius:
                        neighbors.append(other)
            
            node._swarm_neighbors = neighbors
            
            # Update
            swarm_plugin.update(node, delta_time=0.1)
            
            # Move
            node.position = node.position + node.velocity * 0.1
            movement_cost = np.linalg.norm(node.velocity) * 0.01
            node.energy = max(0.0, node.energy - movement_cost)
        
        # Check task completion
        completed = sum(1 for t in swarm_plugin.tasks if t.completed)
        if completed == len(tasks):
            print(f"✓ All tasks completed at step {step + 1}!")
            break
    
    # Phase 3: Self-Healing Formation
    print_section("Phase 3: Self-Healing Formation")
    
    # Simulate node failure
    failed_node_id = 2
    print(f"⚠️  Simulating failure of Agent {failed_node_id}...")
    agents[failed_node_id].energy = 0.0  # Simulate battery death
    
    # Check hardware status
    for node in agents.values():
        swarm_plugin._check_hardware_status(node)
    
    print(f"✓ Detected {len(swarm_plugin.hardware_failures)} hardware failures")
    
    # Switch to emergency regroup
    swarm_plugin.set_behavior_mode(SwarmBehavior.EMERGENCY_REGROUP)
    print("✓ Activated EMERGENCY_REGROUP behavior")
    
    # Regroup
    print("\nRegrouping swarm...")
    for step in range(20):
        for node_id, node in agents.items():
            if node_id == failed_node_id:
                continue  # Skip failed node
            
            # Update neighbors
            neighbors = []
            for other_id, other in agents.items():
                if other_id != node_id and other_id != failed_node_id:
                    distance = np.linalg.norm(node.position - other.position)
                    if distance < swarm_plugin.perception_radius:
                        neighbors.append(other)
            
            node._swarm_neighbors = neighbors
            
            # Update
            swarm_plugin.update(node, delta_time=0.1)
            
            # Move
            node.position = node.position + node.velocity * 0.1
            movement_cost = np.linalg.norm(node.velocity) * 0.01
            node.energy = max(0.0, node.energy - movement_cost)
    
    visualize_swarm(agents, "Formation After Self-Healing")
    
    # Final statistics
    print_section("Final Statistics")
    
    stats = swarm_plugin.get_statistics()
    
    print(f"📊 Swarm Performance:")
    print(f"   Formation Type: {stats['formation_type']}")
    print(f"   Behavior Mode: {stats['behavior_mode']}")
    print(f"   Total Tasks: {stats['total_tasks']}")
    print(f"   Completed Tasks: {stats['completed_tasks']}")
    print(f"   Completion Rate: {stats['completion_rate']*100:.1f}%")
    print(f"   Hardware Failures: {stats['hardware_failures']}")
    
    print(f"\n🤖 Agent Status:")
    for agent_id, node in agents.items():
        status = "FAILED" if agent_id in swarm_plugin.hardware_failures else "ACTIVE"
        print(f"   Agent {agent_id}: Energy={node.energy:.2f}, "
              f"Position=({node.position[0]:.1f}, {node.position[1]:.1f}), "
              f"Status={status}")
    
    print("\n✅ Swarm Robotics Demonstration Complete!\n")
    print("Key Observations:")
    print("1. Agents autonomously formed circular formation")
    print("2. Tasks allocated based on energy and proximity")
    print("3. Swarm adapted to node failure and regrouped")
    print("4. Distributed coordination without central controller\n")


if __name__ == "__main__":
    main()
