#!/usr/bin/env python3
"""
Basic tests for Frequency Intelligence System

Tests core functionality of frequency analyzers to ensure they work correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plugins.rf_spectrum_analyzer import RFSpectrumAnalyzer, SignalType
from plugins.acoustic_frequency import AcousticFrequencyAnalyzer, AcousticSignature
from plugins.vibration_frequency import VibrationFrequencyAnalyzer, BearingGeometry
from core.network_frequency_analyzer import NetworkFrequencyAnalyzer, NetworkEvent
from core.behavioral_frequency import BehavioralFrequencyAnalyzer, MalwareType
from core.keystroke_frequency import KeystrokeFrequencyAnalyzer, KeystrokeEvent, AuthenticationStatus
from core.unified_frequency_intelligence import UnifiedFrequencyIntelligence
import time


def test_rf_spectrum_analyzer():
    """Test RF Spectrum Analyzer basic functionality."""
    print("Testing RF Spectrum Analyzer...")
    
    analyzer = RFSpectrumAnalyzer(scan_range=(20e6, 6e9))
    
    # Test spectrum scan
    signals = analyzer.scan_spectrum(duration_seconds=1.0, battery_level=1.0)
    assert isinstance(signals, list), "scan_spectrum should return a list"
    
    # Test adding authorized transmitter
    analyzer.add_authorized_transmitter(2.437e9, 100e6)
    assert len(analyzer.authorized_transmitters) > 0, "Should have authorized transmitters"
    
    # Test analysis
    results = analyzer.analyze_spectrum(duration_seconds=1.0, battery_level=1.0)
    assert 'signals_detected' in results, "Results should include signals_detected"
    assert 'threats' in results, "Results should include threats"
    
    # Test statistics
    stats = analyzer.get_statistics()
    assert 'scan_range_ghz' in stats, "Stats should include scan_range_ghz"
    
    print("✓ RF Spectrum Analyzer tests passed")
    return True


def test_acoustic_frequency_analyzer():
    """Test Acoustic Frequency Analyzer basic functionality."""
    print("Testing Acoustic Frequency Analyzer...")
    
    analyzer = AcousticFrequencyAnalyzer(sample_rate_hz=44100)
    
    # Test audio analysis
    events = analyzer.analyze_audio_sample(duration_seconds=1.0)
    assert isinstance(events, list), "analyze_audio_sample should return a list"
    
    # Test machinery diagnostics
    diagnostic = analyzer.diagnose_machinery(
        machine_id="test_machine",
        duration_seconds=2.0
    )
    assert diagnostic.machine_id == "test_machine", "Should match machine ID"
    assert 0 <= diagnostic.health_score <= 1.0, "Health score should be 0-1"
    
    # Test frequency range analysis
    freq_analysis = analyzer.get_frequency_range_analysis()
    assert 'infrasound' in freq_analysis, "Should include infrasound range"
    assert 'ultrasound' in freq_analysis, "Should include ultrasound range"
    
    # Test statistics
    stats = analyzer.get_statistics()
    assert 'sample_rate_hz' in stats, "Stats should include sample_rate_hz"
    
    print("✓ Acoustic Frequency Analyzer tests passed")
    return True


def test_vibration_frequency_analyzer():
    """Test Vibration Frequency Analyzer basic functionality."""
    print("Testing Vibration Frequency Analyzer...")
    
    analyzer = VibrationFrequencyAnalyzer(sample_rate_hz=10000)
    
    # Test bearing registration
    bearing = BearingGeometry(
        pitch_diameter_mm=100,
        ball_diameter_mm=12,
        num_balls=9,
        contact_angle_deg=15,
        shaft_speed_rpm=1800
    )
    analyzer.register_bearing("test_bearing", bearing)
    assert "test_bearing" in analyzer.bearing_geometries, "Bearing should be registered"
    
    # Test bearing frequency calculation
    frequencies = analyzer.calculate_bearing_frequencies(bearing)
    assert 'bpfo' in frequencies, "Should include BPFO frequency"
    assert 'bpfi' in frequencies, "Should include BPFI frequency"
    assert frequencies['shaft_freq'] > 0, "Shaft frequency should be positive"
    
    # Test vibration analysis
    events = analyzer.analyze_vibration(bearing_id="test_bearing", duration_seconds=1.0)
    assert isinstance(events, list), "analyze_vibration should return a list"
    
    # Test structural health monitoring
    health = analyzer.monitor_structural_health("test_structure", duration_seconds=2.0)
    assert 0 <= health.health_score <= 1.0, "Health score should be 0-1"
    
    # Test statistics
    stats = analyzer.get_statistics()
    assert 'bearings_monitored' in stats, "Stats should include bearings_monitored"
    
    print("✓ Vibration Frequency Analyzer tests passed")
    return True


def test_network_frequency_analyzer():
    """Test Network Frequency Analyzer basic functionality."""
    print("Testing Network Frequency Analyzer...")
    
    analyzer = NetworkFrequencyAnalyzer()
    
    # Test adding network events
    for i in range(10):
        event = NetworkEvent(
            timestamp=time.time() + i * 0.1,
            source_ip="192.168.1.100",
            dest_ip="8.8.8.8",
            protocol="TCP",
            size_bytes=1500,
            port=443
        )
        analyzer.add_network_event(event)
    
    assert len(analyzer.event_buffer) > 0, "Should have buffered events"
    
    # Test C2 detection (simulate beacon pattern)
    for i in range(5):
        event = NetworkEvent(
            timestamp=time.time() + i * 60,
            source_ip="192.168.1.50",
            dest_ip="malicious.com",
            protocol="HTTPS",
            size_bytes=256,
            port=443
        )
        analyzer.add_network_event(event)
    
    c2_threats = analyzer.detect_c2_beaconing()
    assert isinstance(c2_threats, list), "detect_c2_beaconing should return a list"
    
    # Test statistics
    stats = analyzer.get_statistics()
    assert 'events_buffered' in stats, "Stats should include events_buffered"
    
    print("✓ Network Frequency Analyzer tests passed")
    return True


def test_behavioral_frequency_analyzer():
    """Test Behavioral Frequency Analyzer basic functionality."""
    print("Testing Behavioral Frequency Analyzer...")
    
    analyzer = BehavioralFrequencyAnalyzer()
    
    # Test simulated process behavior
    behaviors = analyzer.simulate_process_behavior(
        process_id=1000,
        process_name="test.exe",
        behavior_type="normal",
        duration_seconds=30.0
    )
    assert len(behaviors) > 0, "Should generate behavior samples"
    
    # Test process analysis
    threats = analyzer.analyze_process(1000)
    assert isinstance(threats, list), "analyze_process should return a list"
    
    # Test cryptominer simulation and detection
    miner_behaviors = analyzer.simulate_process_behavior(
        process_id=2000,
        process_name="miner.exe",
        behavior_type="cryptominer",
        duration_seconds=60.0
    )
    
    miner_threats = analyzer.analyze_process(2000)
    # Should detect cryptominer with high probability
    if len(miner_threats) > 0:
        assert any(t.malware_type == MalwareType.CRYPTOMINER for t in miner_threats), \
            "Should detect cryptominer"
    
    # Test statistics
    stats = analyzer.get_statistics()
    assert 'processes_tracked' in stats, "Stats should include processes_tracked"
    
    print("✓ Behavioral Frequency Analyzer tests passed")
    return True


def test_keystroke_frequency_analyzer():
    """Test Keystroke Frequency Analyzer basic functionality."""
    print("Testing Keystroke Frequency Analyzer...")
    
    analyzer = KeystrokeFrequencyAnalyzer(min_samples_for_baseline=100)
    
    # Test simulated typing
    keystrokes = analyzer.simulate_typing(
        user_id="test_user",
        text="hello world",
        num_repetitions=10,
        typing_speed_wpm=60.0
    )
    assert len(keystrokes) > 0, "Should generate keystrokes"
    
    # Test baseline establishment
    signature = analyzer.establish_baseline("test_user")
    assert signature is not None, "Should establish baseline"
    assert signature.user_id == "test_user", "Signature should match user"
    assert signature.mean_dwell_time > 0, "Dwell time should be positive"
    
    # Test authentication
    analyzer.simulate_typing(
        user_id="test_user",
        text="authenticate",
        num_repetitions=2,
        typing_speed_wpm=60.0
    )
    
    result = analyzer.authenticate("test_user")
    assert result.user_id == "test_user", "Result should match user"
    assert result.status in AuthenticationStatus, "Status should be valid"
    
    # Test statistics
    stats = analyzer.get_statistics()
    assert 'users_tracked' in stats, "Stats should include users_tracked"
    
    print("✓ Keystroke Frequency Analyzer tests passed")
    return True


def test_unified_frequency_intelligence():
    """Test Unified Frequency Intelligence basic functionality."""
    print("Testing Unified Frequency Intelligence...")
    
    unified = UnifiedFrequencyIntelligence(
        enable_rf=True,
        enable_acoustic=True,
        enable_vibration=True,
        enable_network=True,
        enable_behavioral=True,
        enable_keystroke=True
    )
    
    # Test comprehensive analysis
    environment = unified.analyze_all_frequencies(
        battery_level=1.0,
        duration_seconds=1.0
    )
    
    assert hasattr(environment, 'overall_threat_score'), "Should have threat score"
    assert 0 <= environment.overall_threat_score <= 1.0, "Threat score should be 0-1"
    assert hasattr(environment, 'rf_signals'), "Should have RF signals count"
    
    # Test threat report
    report = unified.get_threat_report()
    assert 'current_environment' in report, "Report should include current_environment"
    assert 'recommendations' in report, "Report should include recommendations"
    
    # Test statistics
    stats = unified.get_statistics()
    assert 'environment_snapshots' in stats, "Stats should include snapshots"
    
    print("✓ Unified Frequency Intelligence tests passed")
    return True


def test_node_integration():
    """Test AliveLoopNode integration."""
    print("Testing AliveLoopNode Integration...")
    
    from adaptiveengineer import AliveLoopNode
    
    node = AliveLoopNode(
        position=(0, 0),
        velocity=(0, 0),
        initial_energy=10.0,
        node_id=1
    )
    
    # Test enabling frequency intelligence
    success = node.enable_frequency_intelligence(
        enable_rf=True,
        enable_acoustic=True,
        enable_network=True
    )
    assert success, "Should successfully enable frequency intelligence"
    assert node.frequency_sensors is not None, "Should have frequency sensors"
    
    # Test environment analysis
    results = node.analyze_frequency_environment(duration_seconds=1.0)
    assert 'overall_threat_score' in results, "Results should include threat score"
    
    # Test threat adaptation
    actions = node.adapt_to_frequency_threats()
    assert isinstance(actions, list), "adapt_to_frequency_threats should return a list"
    
    print("✓ AliveLoopNode Integration tests passed")
    return True


def run_all_tests():
    """Run all frequency analyzer tests."""
    print("\n" + "="*80)
    print("FREQUENCY INTELLIGENCE SYSTEM TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_rf_spectrum_analyzer,
        test_acoustic_frequency_analyzer,
        test_vibration_frequency_analyzer,
        test_network_frequency_analyzer,
        test_behavioral_frequency_analyzer,
        test_keystroke_frequency_analyzer,
        test_unified_frequency_intelligence,
        test_node_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
