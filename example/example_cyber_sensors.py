#!/usr/bin/env python3
"""
Example: Cyber Defense with Frequency Sensors

Focused demonstration of cyber-defense capabilities using frequency analysis.
Shows network, behavioral, and keystroke analysis for security operations.

Usage:
    python3 example/example_cyber_sensors.py
"""

import sys
import os
import logging
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.network_frequency_analyzer import (
    NetworkFrequencyAnalyzer,
    NetworkEvent,
)
from core.behavioral_frequency import BehavioralFrequencyAnalyzer
from core.keystroke_frequency import KeystrokeFrequencyAnalyzer
from adaptiveengineer import AliveLoopNode

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def simulate_security_operations():
    """Simulate security operations center monitoring."""
    logger.info("\n" + "=" * 80)
    logger.info("CYBER DEFENSE OPERATIONS CENTER")
    logger.info("=" * 80)

    # Initialize analyzers
    logger.info("\n1. Initializing Security Sensors...")

    network_analyzer = NetworkFrequencyAnalyzer()
    behavioral_analyzer = BehavioralFrequencyAnalyzer()
    keystroke_analyzer = KeystrokeFrequencyAnalyzer()

    logger.info("   ✓ Network traffic analyzer online")
    logger.info("   ✓ Behavioral analyzer online")
    logger.info("   ✓ Keystroke biometrics online")

    # Network monitoring
    logger.info("\n2. Network Traffic Monitoring...")

    # Simulate various traffic patterns
    logger.info("   Injecting test traffic patterns...")

    # Normal traffic
    for i in range(30):
        event = NetworkEvent(
            timestamp=time.time() + i * 0.5,
            source_ip="10.0.0.100",
            dest_ip="8.8.8.8",
            protocol="HTTPS",
            size_bytes=1500,
            port=443,
        )
        network_analyzer.add_network_event(event)

    # C2 beacon pattern
    for i in range(5):
        event = NetworkEvent(
            timestamp=time.time() + i * 60,
            source_ip="10.0.0.50",
            dest_ip="badactor.com",
            protocol="HTTPS",
            size_bytes=200,
            port=443,
        )
        network_analyzer.add_network_event(event)

    logger.info("   ✓ Analyzing traffic patterns...")

    # Detect threats
    c2_threats = network_analyzer.detect_c2_beaconing()
    port_scans = network_analyzer.detect_port_scanning()
    covert_channels = network_analyzer.detect_covert_timing_channel()

    logger.info(f"   📊 C2 beacons detected: {len(c2_threats)}")
    logger.info(f"   📊 Port scans detected: {len(port_scans)}")
    logger.info(f"   📊 Covert channels detected: {len(covert_channels)}")

    if c2_threats:
        for threat in c2_threats:
            logger.warning(f"      ⚠️  {threat.description}")

    # Behavioral monitoring
    logger.info("\n3. Process Behavioral Analysis...")

    # Simulate normal and malicious processes
    logger.info("   Monitoring system processes...")

    behavioral_analyzer.simulate_process_behavior(
        process_id=1000,
        process_name="chrome.exe",
        behavior_type="normal",
        duration_seconds=30.0,
    )

    behavioral_analyzer.simulate_process_behavior(
        process_id=2000,
        process_name="miner.exe",
        behavior_type="cryptominer",
        duration_seconds=60.0,
    )

    behavioral_analyzer.simulate_process_behavior(
        process_id=3000,
        process_name="ransomlock.exe",
        behavior_type="ransomware",
        duration_seconds=30.0,
    )

    logger.info("   ✓ Analyzing process behaviors...")

    # Analyze all processes
    all_threats = behavioral_analyzer.analyze_all_processes()

    logger.info(f"   📊 Total processes monitored: {len(all_threats)}")
    logger.info(
        f"   📊 Malicious processes detected: {sum(len(t) for t in all_threats.values())}"
    )

    for pid, threats in all_threats.items():
        for threat in threats:
            if threat.severity.value in ["malicious", "critical"]:
                logger.warning(
                    f"      ⚠️  PID {pid}: {threat.malware_type.value} "
                    f"(severity: {threat.severity.value})"
                )

    # User authentication monitoring
    logger.info("\n4. Keystroke Biometric Authentication...")

    # Establish baseline for legitimate user
    logger.info("   Establishing baseline for user 'admin'...")

    keystroke_analyzer.simulate_typing(
        user_id="admin",
        text="the quick brown fox jumps over the lazy dog",
        num_repetitions=5,
        typing_speed_wpm=60.0,
    )

    baseline = keystroke_analyzer.establish_baseline("admin")
    if baseline:
        logger.info(
            f"   ✓ Baseline established (dwell: {baseline.mean_dwell_time*1000:.1f}ms)"
        )

    # Test legitimate user
    logger.info("\n   Testing legitimate user authentication...")
    keystroke_analyzer.simulate_typing(
        user_id="admin",
        text="login successful",
        num_repetitions=2,
        typing_speed_wpm=60.0,
    )

    result = keystroke_analyzer.authenticate("admin")
    logger.info(
        f"   📊 Auth result: {result.status.value} (confidence: {result.confidence:.2f})"
    )

    # Test impersonator
    logger.info("\n   Testing impersonator detection...")
    keystroke_analyzer.simulate_typing(
        user_id="admin",  # Claims to be admin
        text="malicious command",
        num_repetitions=2,
        typing_speed_wpm=100.0,  # Different typing pattern
    )

    result = keystroke_analyzer.authenticate("admin")
    logger.info(
        f"   📊 Auth result: {result.status.value} (confidence: {result.confidence:.2f})"
    )

    if result.status.value in ["rejected", "suspicious"]:
        logger.warning(f"      ⚠️  Potential impersonation detected!")

    # Integration with AliveLoopNode
    logger.info("\n5. Security Node Integration...")

    security_node = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=999
    )

    # Enable frequency intelligence
    security_node.enable_frequency_intelligence(
        enable_rf=False,
        enable_acoustic=False,
        enable_vibration=False,
        enable_network=True,
        enable_behavioral=True,
        enable_keystroke=True,
    )

    logger.info("   ✓ Security node initialized with cyber sensors")

    # Analyze and adapt
    results = security_node.analyze_frequency_environment()
    actions = security_node.adapt_to_frequency_threats()

    logger.info(
        f"   📊 Threat score: {results.get('overall_threat_score', 0):.2f}"
    )
    logger.info(f"   📊 Adaptive actions taken: {len(actions)}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SECURITY OPERATIONS SUMMARY")
    logger.info("=" * 80)

    total_network_threats = (
        len(c2_threats) + len(port_scans) + len(covert_channels)
    )
    total_behavioral_threats = sum(len(t) for t in all_threats.values())

    logger.info(f"   Network threats: {total_network_threats}")
    logger.info(f"   Behavioral threats: {total_behavioral_threats}")
    logger.info(
        f"   Authentication events: {len(keystroke_analyzer.authentication_results)}"
    )
    logger.info(
        f"   Security posture: {'CRITICAL' if total_network_threats + total_behavioral_threats > 3 else 'ELEVATED'}"
    )

    logger.info("\n   Recommended actions:")
    if c2_threats:
        logger.info("      - Isolate systems with C2 beaconing")
    if total_behavioral_threats > 0:
        logger.info("      - Terminate malicious processes immediately")
    if result.status.value == "rejected":
        logger.info("      - Lock suspicious user accounts")

    logger.info("\n" + "=" * 80)
    logger.info("CYBER DEFENSE DEMONSTRATION COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(simulate_security_operations())
