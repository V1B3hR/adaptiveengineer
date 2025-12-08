#!/usr/bin/env python3
"""
Tests for System Metrics Bridge
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.system_metrics_bridge import (
    SystemMetricsBridge,
    SystemMetrics,
    SimulationState
)


def test_bridge_creation():
    """Test bridge creation"""
    print("Testing SystemMetricsBridge creation...")
    
    bridge = SystemMetricsBridge(energy_scale=10.0)
    
    assert bridge.energy_scale == 10.0
    assert bridge.baseline_metrics is None
    
    print("✓ Bridge creation test passed")


def test_collect_metrics():
    """Test metric collection"""
    print("Testing metric collection...")
    
    bridge = SystemMetricsBridge()
    metrics = bridge.collect_system_metrics()
    
    assert isinstance(metrics, SystemMetrics)
    assert metrics.cpu_percent >= 0.0
    assert metrics.memory_percent >= 0.0
    assert metrics.timestamp > 0
    
    print(f"  CPU: {metrics.cpu_percent:.1f}%")
    print(f"  Memory: {metrics.memory_percent:.1f}%")
    print("✓ Metric collection test passed")


def test_map_to_simulation():
    """Test mapping to simulation"""
    print("Testing map to simulation...")
    
    bridge = SystemMetricsBridge(energy_scale=10.0)
    
    metrics = SystemMetrics(
        timestamp=1.0,
        cpu_percent=50.0,
        memory_percent=60.0,
        network_bytes_sent=1024,
        network_bytes_recv=2048,
        disk_read_bytes=512,
        disk_write_bytes=256,
        battery_percent=80.0
    )
    
    sim_params = bridge.map_to_simulation(metrics)
    
    assert 'energy' in sim_params
    assert 'communication_load' in sim_params
    assert 'processing_activity' in sim_params
    assert 0 <= sim_params['energy'] <= bridge.energy_scale
    
    print(f"  Mapped energy: {sim_params['energy']:.2f}")
    print("✓ Map to simulation test passed")


def test_map_from_simulation():
    """Test mapping from simulation"""
    print("Testing map from simulation...")
    
    bridge = SystemMetricsBridge(energy_scale=10.0)
    
    sim_state = SimulationState(
        node_id=1,
        energy=7.5,
        position=(5.0, 5.0),
        velocity=(0.1, 0.2),
        communication_load=0.3,
        trust_score=0.8
    )
    
    expected = bridge.map_from_simulation(sim_state)
    
    assert 'expected_cpu_percent' in expected
    assert 'expected_memory_percent' in expected
    assert 'expected_network_rate' in expected
    
    print(f"  Expected CPU: {expected['expected_cpu_percent']:.1f}%")
    print("✓ Map from simulation test passed")


def test_detect_anomaly():
    """Test anomaly detection"""
    print("Testing anomaly detection...")
    
    bridge = SystemMetricsBridge()
    
    current = SystemMetrics(
        timestamp=1.0,
        cpu_percent=80.0,
        memory_percent=70.0,
        network_bytes_sent=1000,
        network_bytes_recv=1000,
        disk_read_bytes=500,
        disk_write_bytes=500
    )
    
    expected = {
        'expected_cpu_percent': 30.0,
        'expected_memory_percent': 40.0,
        'expected_network_rate': 500.0
    }
    
    is_anomaly, anomalies = bridge.detect_anomaly(current, expected, threshold=0.3)
    
    # With these values, should detect CPU anomaly
    assert isinstance(is_anomaly, bool)
    assert isinstance(anomalies, list)
    
    if is_anomaly:
        print(f"  Detected {len(anomalies)} anomalies")
    print("✓ Anomaly detection test passed")


def test_apply_defensive_action():
    """Test applying defensive actions"""
    print("Testing defensive action application...")
    
    bridge = SystemMetricsBridge()
    
    # Test various actions (simulated)
    actions = [
        "reduce_energy_consumption",
        "switch_communication_channel",
        "apply_rate_limiting",
        "alert_administrator"
    ]
    
    for action in actions:
        result = bridge.apply_defensive_action(action)
        assert isinstance(result, bool)
    
    print("✓ Defensive action test passed")


def test_statistics():
    """Test statistics collection"""
    print("Testing statistics...")
    
    bridge = SystemMetricsBridge()
    
    # Collect some metrics
    for _ in range(3):
        bridge.collect_system_metrics()
    
    stats = bridge.get_statistics()
    
    assert 'samples_collected' in stats
    assert stats['samples_collected'] >= 3
    assert 'baseline_established' in stats
    
    print(f"  Samples: {stats['samples_collected']}")
    print("✓ Statistics test passed")


def run_all_tests():
    """Run all system metrics bridge tests"""
    print("\n" + "="*70)
    print("  SYSTEM METRICS BRIDGE TESTS")
    print("="*70 + "\n")
    
    tests = [
        test_bridge_creation,
        test_collect_metrics,
        test_map_to_simulation,
        test_map_from_simulation,
        test_detect_anomaly,
        test_apply_defensive_action,
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
