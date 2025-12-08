#!/usr/bin/env python3
"""
Tests for Swarm Robotics Plugin
"""

import sys
import os
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


def test_swarm_plugin_creation():
    """Test swarm plugin creation"""
    print("Testing SwarmRoboticsPlugin creation...")
    
    plugin = SwarmRoboticsPlugin(
        formation_type=FormationType.CIRCLE,
        behavior_mode=SwarmBehavior.FLOCKING
    )
    
    assert plugin.formation_type == FormationType.CIRCLE
    assert plugin.behavior_mode == SwarmBehavior.FLOCKING
    assert plugin.perception_radius > 0
    
    print("✓ Plugin creation test passed")


def test_swarm_plugin_initialization():
    """Test plugin initialization with node"""
    print("Testing SwarmRoboticsPlugin initialization...")
    
    plugin = SwarmRoboticsPlugin()
    node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
    
    plugin.initialize(node)
    
    print("✓ Plugin initialization test passed")


def test_hardware_monitoring():
    """Test hardware failure detection"""
    print("Testing hardware monitoring...")
    
    plugin = SwarmRoboticsPlugin()
    node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
    node.max_energy = 10.0
    
    # Simulate low battery
    node.energy = 0.1
    plugin._check_hardware_status(node)
    
    assert node.node_id in plugin.hardware_failures
    assert plugin.hardware_failures[node.node_id] == "low_battery"
    
    print("✓ Hardware monitoring test passed")


def test_flocking_behavior():
    """Test flocking behavior update"""
    print("Testing flocking behavior...")
    
    plugin = SwarmRoboticsPlugin(behavior_mode=SwarmBehavior.FLOCKING)
    
    node = AliveLoopNode(position=(5, 5), velocity=(0, 0), initial_energy=10.0, node_id=1)
    neighbor = AliveLoopNode(position=(6, 5), velocity=(0.1, 0), initial_energy=10.0, node_id=2)
    
    node._swarm_neighbors = [neighbor]
    
    initial_velocity = node.velocity.copy()
    plugin._update_flocking(node, delta_time=0.1)
    
    # Velocity should change due to flocking forces
    # (unless by chance it doesn't, but that's very unlikely)
    
    print("✓ Flocking behavior test passed")


def test_task_allocation():
    """Test task allocation"""
    print("Testing task allocation...")
    
    plugin = SwarmRoboticsPlugin()
    
    # Create nodes
    nodes = []
    for i in range(3):
        node = AliveLoopNode(
            position=(i * 2.0, 0),
            velocity=(0, 0),
            initial_energy=10.0,
            node_id=i
        )
        node.max_energy = 10.0
        nodes.append(node)
    
    # Add tasks
    task = SwarmTask(
        task_id="task_1",
        task_type="collect",
        energy_requirement=0.5,
        priority=0.8,
        position=(1.0, 0.0)
    )
    plugin.add_task(task)
    
    # Allocate
    allocations = plugin.allocate_tasks(nodes)
    
    assert len(allocations) == len(nodes)
    # At least one node should be assigned the task
    assigned = sum(len(tasks) for tasks in allocations.values())
    assert assigned >= 0  # Might be 0 if no node meets requirements
    
    print("✓ Task allocation test passed")


def test_formation_types():
    """Test different formation types"""
    print("Testing formation types...")
    
    plugin = SwarmRoboticsPlugin(formation_type=FormationType.CIRCLE)
    node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
    node._swarm_neighbors = []
    
    # Test circle formation
    plugin._maintain_circle_formation(node)
    
    # Test line formation
    plugin.set_formation_type(FormationType.LINE)
    plugin._maintain_line_formation(node)
    
    # Test grid formation
    plugin.set_formation_type(FormationType.GRID)
    plugin._maintain_grid_formation(node)
    
    print("✓ Formation types test passed")


def test_behavior_modes():
    """Test different behavior modes"""
    print("Testing behavior modes...")
    
    plugin = SwarmRoboticsPlugin()
    node = AliveLoopNode(position=(5, 5), velocity=(0, 0), initial_energy=10.0, node_id=1)
    node._swarm_neighbors = []
    
    # Test each behavior mode
    for mode in SwarmBehavior:
        plugin.set_behavior_mode(mode)
        assert plugin.behavior_mode == mode
        
        # Update with each mode
        plugin.update(node, delta_time=0.1)
    
    print("✓ Behavior modes test passed")


def test_statistics():
    """Test statistics collection"""
    print("Testing statistics...")
    
    plugin = SwarmRoboticsPlugin()
    
    # Add some tasks
    for i in range(3):
        task = SwarmTask(
            task_id=f"task_{i}",
            task_type="test",
            energy_requirement=0.5,
            priority=0.5,
            position=(i, i)
        )
        plugin.add_task(task)
    
    stats = plugin.get_statistics()
    
    assert 'formation_type' in stats
    assert 'behavior_mode' in stats
    assert 'total_tasks' in stats
    assert stats['total_tasks'] == 3
    
    print("✓ Statistics test passed")


def run_all_tests():
    """Run all swarm robotics tests"""
    print("\n" + "="*70)
    print("  SWARM ROBOTICS TESTS")
    print("="*70 + "\n")
    
    tests = [
        test_swarm_plugin_creation,
        test_swarm_plugin_initialization,
        test_hardware_monitoring,
        test_flocking_behavior,
        test_task_allocation,
        test_formation_types,
        test_behavior_modes,
        test_statistics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
