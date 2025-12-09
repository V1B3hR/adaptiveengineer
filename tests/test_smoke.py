#!/usr/bin/env python3
"""
Smoke test for Adaptive Engineer simulation.
Verifies that the main simulation path executes without errors.
"""

import os
import sys
import yaml
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptiveengineer import AliveLoopNode
from core.plugin_manager import PluginManager
from plugins.it_operations import ITOperationsPlugin
from plugins.security import SecurityPlugin
from plugins.artificial_life import ArtificialLifePlugin


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_basic_simulation():
    """Test basic simulation with sample configuration."""
    # Load sample config
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'configs', 
        'sample_run.yaml'
    )
    config = load_config(config_path)
    
    # Extract configuration
    num_nodes = config['simulation']['num_nodes']
    initial_energy = config['nodes']['initial_energy']
    timesteps = config['simulation']['timesteps']
    spatial_dims = config['simulation']['spatial_dims']
    
    # Create nodes
    nodes = []
    for i in range(num_nodes):
        node = AliveLoopNode(
            position=[0.0] * spatial_dims,
            velocity=[0.0] * spatial_dims,
            initial_energy=initial_energy,
            node_id=i,
            spatial_dims=spatial_dims
        )
        nodes.append(node)
    
    # Setup plugins for each node
    # Note: This is a minimal smoke test that verifies basic simulation execution.
    # Plugin functionality is tested in dedicated plugin tests.
    for node in nodes:
        manager = PluginManager()
        if config['plugins'].get('it_operations', False):
            manager.register_plugin(ITOperationsPlugin())
        if config['plugins'].get('security', False):
            manager.register_plugin(SecurityPlugin())
        if config['plugins'].get('artificial_life', False):
            manager.register_plugin(ArtificialLifePlugin())
        manager.initialize_all(node)
    
    # Run simulation for specified timesteps
    for t in range(timesteps):
        for node in nodes:
            node.step_phase(t)
            node.move()
    
    # Basic assertions - simulation completed without errors
    assert len(nodes) == num_nodes
    assert all(hasattr(node, 'energy') for node in nodes)
    assert all(hasattr(node, 'position') for node in nodes)
    
    print(f"✓ Smoke test passed: Simulated {num_nodes} nodes for {timesteps} timesteps")


def test_config_file_exists():
    """Test that the sample configuration file exists and is valid."""
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'configs', 
        'sample_run.yaml'
    )
    assert os.path.exists(config_path), "Sample configuration file not found"
    
    # Load and validate config structure
    config = load_config(config_path)
    assert 'simulation' in config
    assert 'nodes' in config
    assert 'plugins' in config
    assert 'outputs' in config
    
    print("✓ Configuration file validation passed")


def test_output_directories():
    """Test that output directories can be created."""
    # Create temporary output directories
    temp_dir = tempfile.mkdtemp()
    try:
        output_dir = os.path.join(temp_dir, 'outputs')
        log_dir = os.path.join(temp_dir, 'logs')
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        assert os.path.exists(output_dir)
        assert os.path.exists(log_dir)
        
        print("✓ Output directory creation test passed")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    print("Running smoke tests for Adaptive Engineer...")
    print("-" * 60)
    
    try:
        test_config_file_exists()
        test_output_directories()
        test_basic_simulation()
        
        print("-" * 60)
        print("All smoke tests passed! ✓")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
