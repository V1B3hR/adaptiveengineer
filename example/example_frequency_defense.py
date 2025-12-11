#!/usr/bin/env python3
"""
Example: Comprehensive Frequency Defense System

Demonstrates the complete Frequency Intelligence System with cross-domain
threat detection and correlation.

This example showcases:
1. RF Spectrum Analysis - Detecting jamming and rogue transmitters
2. Acoustic Frequency Analysis - Gunshot detection and ultrasonic channels
3. Vibration Analysis - Machinery diagnostics and bearing defects
4. Network Frequency Analysis - C2 beaconing and port scanning
5. Behavioral Analysis - Cryptominer and ransomware detection
6. Keystroke Analysis - Behavioral biometrics and impersonation
7. Unified Intelligence - Cross-domain threat correlation

Usage:
    python3 example/example_frequency_defense.py
"""

import sys
import os
import logging
import time
import numpy as np

# Add parent directory to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Import frequency analyzers
from plugins.rf_spectrum_analyzer import RFSpectrumAnalyzer
from plugins.acoustic_frequency import AcousticFrequencyAnalyzer
from plugins.vibration_frequency import (
    VibrationFrequencyAnalyzer,
    BearingGeometry,
)
from core.network_frequency_analyzer import (
    NetworkFrequencyAnalyzer,
    NetworkEvent,
)
from core.behavioral_frequency import BehavioralFrequencyAnalyzer
from core.keystroke_frequency import KeystrokeFrequencyAnalyzer
from core.unified_frequency_intelligence import UnifiedFrequencyIntelligence
from adaptiveengineer import AliveLoopNode

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demonstrate_rf_analysis():
    """Demonstrate RF spectrum analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("1. RF SPECTRUM ANALYSIS")
    logger.info("=" * 80)

    analyzer = RFSpectrumAnalyzer(
        scan_range=(20e6, 6e9), sensitivity_dbm=-120.0  # 20 MHz to 6 GHz
    )

    # Add authorized transmitters
    analyzer.add_authorized_transmitter(2.437e9, 100e6)  # WiFi
    analyzer.add_authorized_transmitter(1575.42e6, 10e6)  # GPS

    # Perform spectrum analysis
    logger.info("   Scanning RF spectrum...")
    results = analyzer.analyze_spectrum(
        duration_seconds=2.0, battery_level=1.0
    )

    logger.info(f"   ✓ Detected {results['signals_detected']} RF signals")
    logger.info(f"   ✓ Signal breakdown: {results['signal_breakdown']}")
    logger.info(
        f"   ✓ Spectrum utilization: {results['spectrum_utilization']:.2f}%"
    )
    logger.info(
        f"   ✓ Threats detected: {results['threats']['total_threats']}"
    )

    # Get statistics
    stats = analyzer.get_statistics()
    logger.info(
        f"   ✓ Total signals tracked: {stats['total_signals_detected']}"
    )
    logger.info(f"   ✓ Total threats: {stats['total_threats_detected']}")

    return analyzer


def demonstrate_acoustic_analysis():
    """Demonstrate acoustic frequency analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("2. ACOUSTIC FREQUENCY ANALYSIS")
    logger.info("=" * 80)

    analyzer = AcousticFrequencyAnalyzer(
        sample_rate_hz=44100, fft_size=2048  # CD quality
    )

    # Analyze audio sample
    logger.info("   Analyzing acoustic environment...")
    events = analyzer.analyze_audio_sample(duration_seconds=2.0)

    logger.info(f"   ✓ Detected {len(events)} acoustic events")
    for event in events:
        logger.info(
            f"      - {event.signature.value} at {event.frequency_hz:.1f} Hz "
            f"(confidence: {event.confidence:.2f})"
        )

    # Machinery diagnostics
    logger.info("\n   Performing machinery diagnostics...")
    diagnostic = analyzer.diagnose_machinery(
        machine_id="motor_001", duration_seconds=5.0
    )

    logger.info(f"   ✓ Machine health score: {diagnostic.health_score:.2f}")
    logger.info(
        f"   ✓ Fundamental frequency: {diagnostic.fundamental_freq_hz:.1f} Hz"
    )
    logger.info(f"   ✓ Harmonics detected: {len(diagnostic.harmonics)}")
    if diagnostic.anomalies:
        logger.info(f"   ✓ Anomalies: {', '.join(diagnostic.anomalies)}")

    # Frequency range analysis
    freq_analysis = analyzer.get_frequency_range_analysis()
    logger.info("\n   Energy distribution by frequency range:")
    for range_name, energy in freq_analysis.items():
        logger.info(f"      - {range_name}: {energy*100:.1f}%")

    return analyzer


def demonstrate_vibration_analysis():
    """Demonstrate vibration frequency analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("3. VIBRATION FREQUENCY ANALYSIS")
    logger.info("=" * 80)

    analyzer = VibrationFrequencyAnalyzer(
        sample_rate_hz=10000, sensitivity_g=0.001  # 10 kHz
    )

    # Register bearing for monitoring
    bearing = BearingGeometry(
        pitch_diameter_mm=100,
        ball_diameter_mm=12,
        num_balls=9,
        contact_angle_deg=15,
        shaft_speed_rpm=1800,
    )

    analyzer.register_bearing("bearing_001", bearing)
    logger.info("   ✓ Registered bearing for monitoring")

    # Calculate bearing frequencies
    frequencies = analyzer.calculate_bearing_frequencies(bearing)
    logger.info(f"   ✓ Shaft frequency: {frequencies['shaft_freq']:.2f} Hz")
    logger.info(f"   ✓ BPFO (outer race): {frequencies['bpfo']:.2f} Hz")
    logger.info(f"   ✓ BPFI (inner race): {frequencies['bpfi']:.2f} Hz")
    logger.info(f"   ✓ BSF (ball spin): {frequencies['bsf']:.2f} Hz")

    # Analyze vibration
    logger.info("\n   Analyzing vibration patterns...")
    events = analyzer.analyze_vibration(
        bearing_id="bearing_001", duration_seconds=2.0
    )

    logger.info(f"   ✓ Detected {len(events)} vibration events")
    for event in events:
        if event.defect_type.value != "none":
            logger.info(
                f"      - {event.defect_type.value} at {event.frequency_hz:.2f} Hz "
                f"(severity: {event.severity.value})"
            )

    # Structural health monitoring
    logger.info("\n   Monitoring structural health...")
    health = analyzer.monitor_structural_health(
        structure_id="bridge_001", duration_seconds=10.0
    )

    logger.info(f"   ✓ Structural health score: {health.health_score:.2f}")
    logger.info(
        f"   ✓ Resonance frequencies: {[f'{f:.2f}Hz' for f in health.resonance_frequencies[:3]]}"
    )
    if health.anomalies:
        logger.info(f"   ✓ Anomalies: {', '.join(health.anomalies)}")

    return analyzer


def demonstrate_network_analysis():
    """Demonstrate network frequency analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("4. NETWORK FREQUENCY ANALYSIS")
    logger.info("=" * 80)

    analyzer = NetworkFrequencyAnalyzer(
        window_size=1000, min_period=0.1, max_period=300.0
    )

    # Simulate network traffic
    logger.info("   Simulating network traffic...")

    # Normal traffic
    for i in range(50):
        event = NetworkEvent(
            timestamp=time.time() + i * 0.1,
            source_ip="192.168.1.100",
            dest_ip="8.8.8.8",
            protocol="TCP",
            size_bytes=1500,
            port=443,
        )
        analyzer.add_network_event(event)

    # Simulated C2 beacon
    beacon_period = 60  # seconds
    for i in range(10):
        event = NetworkEvent(
            timestamp=time.time() + i * beacon_period,
            source_ip="192.168.1.50",
            dest_ip="malicious.example.com",
            protocol="HTTPS",
            size_bytes=256,
            port=443,
        )
        analyzer.add_network_event(event)

    logger.info("   ✓ Added network events")

    # Detect C2 beaconing
    logger.info("\n   Detecting C2 beaconing...")
    c2_threats = analyzer.detect_c2_beaconing()
    logger.info(f"   ✓ Detected {len(c2_threats)} C2 beacon patterns")

    for threat in c2_threats:
        logger.info(f"      - {threat.description}")
        logger.info(
            f"        Period: {threat.period_seconds:.1f}s, Confidence: {threat.confidence:.2f}"
        )

    # Port scanning detection
    logger.info("\n   Checking for port scanning...")
    scan_threats = analyzer.detect_port_scanning()
    logger.info(f"   ✓ Detected {len(scan_threats)} port scan patterns")

    return analyzer


def demonstrate_behavioral_analysis():
    """Demonstrate behavioral frequency analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("5. BEHAVIORAL FREQUENCY ANALYSIS")
    logger.info("=" * 80)

    analyzer = BehavioralFrequencyAnalyzer(
        window_size=300, sample_interval=1.0
    )

    # Simulate normal process
    logger.info("   Simulating normal process behavior...")
    normal_behaviors = analyzer.simulate_process_behavior(
        process_id=1000,
        process_name="normal_app.exe",
        behavior_type="normal",
        duration_seconds=30.0,
    )
    logger.info(
        f"   ✓ Simulated {len(normal_behaviors)} normal behavior samples"
    )

    # Analyze normal process
    threats = analyzer.analyze_process(1000)
    logger.info(f"   ✓ Normal process threats: {len(threats)}")

    # Simulate cryptominer
    logger.info("\n   Simulating cryptominer behavior...")
    miner_behaviors = analyzer.simulate_process_behavior(
        process_id=2000,
        process_name="suspicious.exe",
        behavior_type="cryptominer",
        duration_seconds=60.0,
    )
    logger.info(f"   ✓ Simulated {len(miner_behaviors)} cryptominer samples")

    # Detect cryptominer
    threats = analyzer.analyze_process(2000)
    logger.info(f"   ✓ Detected {len(threats)} behavioral threats")

    for threat in threats:
        logger.info(
            f"      - {threat.malware_type.value}: {threat.description}"
        )
        logger.info(
            f"        Severity: {threat.severity.value}, Confidence: {threat.confidence:.2f}"
        )
        logger.info(f"        Indicators: {', '.join(threat.indicators[:2])}")

    return analyzer


def demonstrate_keystroke_analysis():
    """Demonstrate keystroke frequency analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("6. KEYSTROKE FREQUENCY ANALYSIS")
    logger.info("=" * 80)

    analyzer = KeystrokeFrequencyAnalyzer(
        min_samples_for_baseline=100, authentication_threshold=0.3
    )

    # Simulate user typing to establish baseline
    logger.info("   Establishing user baseline...")
    user_keystrokes = analyzer.simulate_typing(
        user_id="alice",
        text="the quick brown fox jumps over the lazy dog",
        num_repetitions=5,
        typing_speed_wpm=60.0,
    )

    logger.info(
        f"   ✓ Captured {len(user_keystrokes)} keystrokes for baseline"
    )

    # Establish baseline
    signature = analyzer.establish_baseline("alice")

    if signature:
        logger.info(f"   ✓ Baseline established for alice")
        logger.info(
            f"      - Mean dwell time: {signature.mean_dwell_time*1000:.1f}ms"
        )
        logger.info(
            f"      - Mean flight time: {signature.mean_flight_time*1000:.1f}ms"
        )

    # Authenticate legitimate user
    logger.info("\n   Testing authentication with legitimate user...")
    auth_keystrokes = analyzer.simulate_typing(
        user_id="alice",
        text="hello world",
        num_repetitions=2,
        typing_speed_wpm=60.0,
    )

    result = analyzer.authenticate("alice")
    logger.info(f"   ✓ Authentication result: {result.status.value}")
    logger.info(f"      - Confidence: {result.confidence:.2f}")
    logger.info(
        f"      - Distance from baseline: {result.distance_from_baseline:.3f}"
    )

    # Simulate impersonator (different typing speed)
    logger.info("\n   Testing with impersonator (different typing pattern)...")
    impostor_keystrokes = analyzer.simulate_typing(
        user_id="alice",  # Claims to be alice
        text="hello world",
        num_repetitions=2,
        typing_speed_wpm=90.0,  # Different speed
    )

    result = analyzer.authenticate("alice")
    logger.info(f"   ✓ Authentication result: {result.status.value}")
    logger.info(f"      - Confidence: {result.confidence:.2f}")
    logger.info(
        f"      - Distance from baseline: {result.distance_from_baseline:.3f}"
    )

    return analyzer


def demonstrate_unified_intelligence():
    """Demonstrate unified frequency intelligence with cross-domain correlation."""
    logger.info("\n" + "=" * 80)
    logger.info("7. UNIFIED FREQUENCY INTELLIGENCE")
    logger.info("=" * 80)

    # Create unified intelligence system
    unified = UnifiedFrequencyIntelligence(
        enable_rf=True,
        enable_acoustic=True,
        enable_vibration=True,
        enable_network=True,
        enable_behavioral=True,
        enable_keystroke=True,
    )

    logger.info("   ✓ Unified Intelligence System initialized")

    # Analyze all frequencies
    logger.info("\n   Performing comprehensive frequency analysis...")
    environment = unified.analyze_all_frequencies(
        battery_level=1.0, duration_seconds=2.0
    )

    logger.info(
        f"   ✓ Overall threat score: {environment.overall_threat_score:.2f}"
    )
    logger.info(
        f"   ✓ RF signals: {environment.rf_signals}, threats: {environment.rf_threats}"
    )
    logger.info(f"   ✓ Acoustic events: {environment.acoustic_events}")
    logger.info(f"   ✓ Vibration events: {environment.vibration_events}")
    logger.info(f"   ✓ Network threats: {environment.network_threats}")
    logger.info(f"   ✓ Behavioral threats: {environment.behavior_threats}")
    logger.info(f"   ✓ Authentication: {environment.authentication_status}")

    # Check for correlations
    if environment.active_correlations:
        logger.info(f"\n   ⚠️  CORRELATED THREATS DETECTED:")
        for correlation in environment.active_correlations:
            logger.info(f"      - {correlation.value}")

    # Get threat report
    logger.info("\n   Generating comprehensive threat report...")
    report = unified.get_threat_report()

    logger.info(
        f"   ✓ Active correlations: {len(report['active_correlations'])}"
    )
    logger.info(
        f"   ✓ Recent correlated threats: {report['recent_correlated_threats']}"
    )

    if report["recommendations"]:
        logger.info("\n   Recommendations:")
        for rec in report["recommendations"][:5]:
            logger.info(f"      - {rec}")

    # Get statistics
    stats = unified.get_statistics()
    logger.info(
        f"\n   ✓ Environment snapshots: {stats['environment_snapshots']}"
    )
    logger.info(f"   ✓ Correlated threats: {stats['correlated_threats']}")
    logger.info(f"   ✓ Baseline established: {stats['baseline_established']}")

    return unified


def demonstrate_node_integration():
    """Demonstrate AliveLoopNode integration with Frequency Intelligence."""
    logger.info("\n" + "=" * 80)
    logger.info("8. ALIVELOOPNODE INTEGRATION")
    logger.info("=" * 80)

    # Create node
    node = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1
    )

    logger.info("   ✓ Created AliveLoopNode")

    # Enable frequency intelligence
    logger.info("\n   Enabling Frequency Intelligence...")
    success = node.enable_frequency_intelligence(
        enable_rf=True, enable_acoustic=True, enable_network=True
    )

    if success:
        logger.info("   ✓ Frequency Intelligence enabled for node")

        # Analyze frequency environment
        logger.info("\n   Analyzing frequency environment through node...")
        results = node.analyze_frequency_environment(duration_seconds=1.0)

        logger.info(
            f"   ✓ Threat score: {results.get('overall_threat_score', 0):.2f}"
        )
        logger.info(
            f"   ✓ Active correlations: {results.get('active_correlations', [])}"
        )

        # Adapt to threats
        logger.info("\n   Adapting to detected threats...")
        actions = node.adapt_to_frequency_threats()

        if actions:
            logger.info(f"   ✓ Took {len(actions)} adaptive actions:")
            for action in actions[:5]:
                logger.info(f"      - {action}")
        else:
            logger.info("   ✓ No immediate threats - normal operations")

        logger.info(
            f"\n   ✓ Node threat assessment level: {node.threat_assessment_level}"
        )
        logger.info(f"   ✓ Node anxiety level: {node.anxiety:.2f}")

    return node


def main():
    """Run comprehensive frequency defense demonstration."""
    logger.info("\n" + "#" * 80)
    logger.info("# COMPREHENSIVE FREQUENCY DEFENSE SYSTEM DEMONSTRATION")
    logger.info("#" * 80)

    start_time = time.time()

    try:
        # Demonstrate each analyzer
        rf_analyzer = demonstrate_rf_analysis()
        acoustic_analyzer = demonstrate_acoustic_analysis()
        vibration_analyzer = demonstrate_vibration_analysis()
        network_analyzer = demonstrate_network_analysis()
        behavioral_analyzer = demonstrate_behavioral_analysis()
        keystroke_analyzer = demonstrate_keystroke_analysis()

        # Demonstrate unified intelligence
        unified = demonstrate_unified_intelligence()

        # Demonstrate node integration
        node = demonstrate_node_integration()

        # Summary
        elapsed = time.time() - start_time

        logger.info("\n" + "=" * 80)
        logger.info("DEMONSTRATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"   ✓ All frequency analyzers functional")
        logger.info(f"   ✓ Cross-domain correlation working")
        logger.info(f"   ✓ Node integration successful")
        logger.info(f"   ✓ Total elapsed time: {elapsed:.2f} seconds")
        logger.info("\n" + "#" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
