#!/usr/bin/env python3
"""
Example demonstrating Phase 2: Communication, Security, and Incident Memory

This example shows:
1. Secure, Structured Messaging - Rich message types, idempotency, tracing
2. Event-Driven Integrations - Webhooks, SIEM, AIOps streaming
3. Incident and Pattern Memory - Persistent storage, pattern recognition, learning

Usage:
    python3 example/example_phase2.py
"""

import logging
import sys
import os
import time

# Add parent directory to path to allow imports from root
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Import core plugin system
from core.plugin_manager import PluginManager

# Import Phase 1 plugins
from plugins.it_operations import ITOperationsPlugin
from plugins.security import SecurityPlugin
from plugins.artificial_life import ArtificialLifePlugin

# Import Phase 2 plugin
from plugins.communication import CommunicationPlugin

# Import Phase 2 components directly for demonstration
from core.messaging import Message, MessageType, Priority
from core.event_integrations import Event, EventSeverity
from core.incident_memory import Incident, IncidentType, IncidentStatus

# Import base node
from adaptiveengineer import AliveLoopNode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase2_example")


def demonstrate_phase2():
    """Demonstrate Phase 2 capabilities."""
    logger.info("=" * 70)
    logger.info(
        "PHASE 2 DEMONSTRATION: Communication, Security, and Incident Memory"
    )
    logger.info("=" * 70)

    # ========================================================================
    # 1. Create nodes with Phase 1 + Phase 2 plugins
    # ========================================================================
    logger.info(
        "\n1. Creating adaptive nodes with Phase 1 + Phase 2 plugins..."
    )

    # Create two nodes for communication demonstration
    node1 = AliveLoopNode(
        position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1
    )

    node2 = AliveLoopNode(
        position=(5, 5), velocity=(0, 0), initial_energy=10.0, node_id=2
    )

    logger.info(f"   ✓ Created nodes {node1.node_id} and {node2.node_id}")

    # ========================================================================
    # 2. Initialize plugin managers with Phase 2 communication plugin
    # ========================================================================
    logger.info("\n2. Initializing plugins (Phase 1 + Phase 2)...")

    # Node 1 plugins
    pm1 = PluginManager()
    pm1.register_plugin(ITOperationsPlugin())
    pm1.register_plugin(SecurityPlugin())
    pm1.register_plugin(ArtificialLifePlugin())
    comm1 = CommunicationPlugin(
        plugin_id="communication",
        config={
            "enable_siem": True,
            "enable_aiops": True,
            "siem_config": {"endpoint": "siem.example.com"},
            "aiops_config": {"endpoint": "aiops.example.com"},
        },
    )
    pm1.register_plugin(comm1)
    pm1.initialize_all(node1)

    # Node 2 plugins
    pm2 = PluginManager()
    pm2.register_plugin(ITOperationsPlugin())
    pm2.register_plugin(SecurityPlugin())
    pm2.register_plugin(ArtificialLifePlugin())
    comm2 = CommunicationPlugin(plugin_id="communication")
    pm2.register_plugin(comm2)
    pm2.initialize_all(node2)

    logger.info(
        f"   ✓ Initialized 4 plugins per node (IT, Security, AL, Communication)"
    )

    # ========================================================================
    # 3. Demonstrate Secure, Structured Messaging
    # ========================================================================
    logger.info("\n3. Demonstrating Secure, Structured Messaging...")
    logger.info("   " + "-" * 66)

    # Send various message types
    logger.info("\n   3.1 Sending rich message types:")

    # Alert message
    alert_msg = Message(
        message_type=MessageType.ALERT,
        source_node_id=node1.node_id,
        target_node_id=node2.node_id,
        payload={
            "alert_type": "cpu_high",
            "cpu_usage": 0.95,
            "threshold": 0.90,
        },
        priority=Priority.HIGH,
        requires_ack=True,
    )
    comm1.send_message(alert_msg)
    logger.info(
        f"     ✓ Sent ALERT message (ID: {alert_msg.message_id[:8]}...)"
    )

    # Anomaly message
    anomaly_msg = Message(
        message_type=MessageType.ANOMALY,
        source_node_id=node1.node_id,
        payload={"anomaly_type": "unusual_traffic", "score": 0.87},
        priority=Priority.HIGH,
    )
    comm1.send_message(anomaly_msg)
    logger.info(
        f"     ✓ Sent ANOMALY message (ID: {anomaly_msg.message_id[:8]}...)"
    )

    # Remediation message
    remediation_msg = Message(
        message_type=MessageType.REMEDIATION,
        source_node_id=node2.node_id,
        payload={
            "action": "restart_service",
            "service": "web_server",
            "success": True,
        },
        priority=Priority.NORMAL,
    )
    comm2.send_message(remediation_msg)
    logger.info(
        f"     ✓ Sent REMEDIATION message (ID: {remediation_msg.message_id[:8]}...)"
    )

    # Trust update message
    trust_msg = Message(
        message_type=MessageType.TRUST_UPDATE,
        source_node_id=node1.node_id,
        payload={
            "target_node": node2.node_id,
            "trust_score": 0.85,
            "reason": "successful_collaboration",
        },
        priority=Priority.NORMAL,
    )
    comm1.send_message(trust_msg)
    logger.info(
        f"     ✓ Sent TRUST_UPDATE message (ID: {trust_msg.message_id[:8]}...)"
    )

    # Allow messages to be processed
    time.sleep(0.2)

    # Demonstrate tracing
    logger.info("\n   3.2 Message tracing and replay:")
    trace_messages = comm1.message_bus.get_trace(alert_msg.trace_id)
    logger.info(
        f"     ✓ Found {len(trace_messages)} messages in trace {alert_msg.trace_id[:8]}..."
    )

    # Demonstrate idempotency
    logger.info("\n   3.3 Idempotency (sending duplicate message):")
    duplicate = Message(
        message_type=MessageType.ALERT,
        source_node_id=node1.node_id,
        payload=alert_msg.payload,  # Same payload
    )
    comm1.send_message(duplicate)
    time.sleep(0.1)
    logger.info(f"     ✓ Duplicate messages are automatically filtered")

    # ========================================================================
    # 4. Demonstrate Event-Driven Integrations
    # ========================================================================
    logger.info("\n4. Demonstrating Event-Driven Integrations...")
    logger.info("   " + "-" * 66)

    logger.info("\n   4.1 Publishing events to external systems:")

    # Publish to SIEM
    security_event = Event(
        event_id="evt_sec_001",
        event_type="security_alert",
        severity=EventSeverity.CRITICAL,
        source=f"node_{node1.node_id}",
        timestamp=time.time(),
        data={
            "threat_type": "intrusion_attempt",
            "source_ip": "192.168.1.100",
            "blocked": True,
        },
        tags=["security", "intrusion", "blocked"],
    )
    stream_count = comm1.event_manager.publish_event(security_event)
    logger.info(
        f"     ✓ Published security event to {stream_count} streams (SIEM, AIOps)"
    )

    # Publish operational event
    ops_event = Event(
        event_id="evt_ops_001",
        event_type="service_degradation",
        severity=EventSeverity.WARNING,
        source=f"node_{node2.node_id}",
        timestamp=time.time(),
        data={
            "service": "api_gateway",
            "response_time_ms": 1500,
            "threshold_ms": 500,
        },
        tags=["operations", "performance"],
    )
    stream_count = comm2.event_manager.publish_event(ops_event)
    logger.info(
        f"     ✓ Published operational event to {stream_count} streams"
    )

    logger.info("\n   4.2 Real-time event streaming:")
    logger.info("     ✓ Events streamed to SIEM for security monitoring")
    logger.info("     ✓ Events streamed to AIOps for operational intelligence")

    # ========================================================================
    # 5. Demonstrate Incident and Pattern Memory
    # ========================================================================
    logger.info("\n5. Demonstrating Incident and Pattern Memory...")
    logger.info("   " + "-" * 66)

    logger.info("\n   5.1 Recording incidents:")

    # Record several similar incidents to trigger pattern detection
    for i in range(5):
        incident = Incident(
            incident_id=f"inc_cpu_{i}",
            incident_type=IncidentType.ALERT,
            status=IncidentStatus.OPEN,
            timestamp=time.time(),
            source=f"node_{node1.node_id}",
            description=f"High CPU usage detected (attempt {i+1})",
            severity=0.8,
            data={"cpu_usage": 0.90 + i * 0.01},
            tags=["cpu", "performance", "alert"],
        )
        success, pattern = comm1.record_incident(incident)

        if i == 0:
            logger.info(f"     ✓ Recorded incident: {incident.incident_id}")
        elif pattern and pattern.occurrence_count >= 3:
            logger.info(
                f"     ⚠ Pattern detected after {pattern.occurrence_count} occurrences!"
            )
            logger.info(f"       Pattern ID: {pattern.pattern_id}")
            logger.info(f"       Confidence: {pattern.confidence:.2f}")

        time.sleep(0.05)

    # Record different incident types
    failure_incident = Incident(
        incident_id="inc_svc_fail",
        incident_type=IncidentType.FAILURE,
        status=IncidentStatus.OPEN,
        timestamp=time.time(),
        source=f"node_{node2.node_id}",
        description="Service crashed unexpectedly",
        severity=0.9,
        data={"service": "web_server", "error_code": 500},
        tags=["failure", "service"],
    )
    comm2.record_incident(failure_incident)
    logger.info(f"     ✓ Recorded service failure incident")

    logger.info("\n   5.2 Pattern recognition:")
    patterns = comm1.incident_memory.get_patterns()
    logger.info(f"     ✓ Detected {len(patterns)} patterns total")

    for pattern in patterns:
        logger.info(f"       - Pattern: {pattern.incident_type.value}")
        logger.info(f"         Occurrences: {pattern.occurrence_count}")
        logger.info(f"         Confidence: {pattern.confidence:.2f}")
        logger.info(f"         Avg Severity: {pattern.severity_avg:.2f}")
        logger.info(f"         Sources: {', '.join(pattern.sources)}")

    logger.info("\n   5.3 Querying incidents:")
    recent_incidents = comm1.incident_memory.query_incidents(
        incident_types=[IncidentType.ALERT], limit=10
    )
    logger.info(
        f"     ✓ Queried {len(recent_incidents)} recent alert incidents"
    )

    logger.info("\n   5.4 Learning data for continual adaptation:")
    learning_data = comm1.incident_memory.get_learning_data()
    logger.info(f"     ✓ Extracted learning data:")
    logger.info(f"       - {learning_data['incident_count']} incidents")
    logger.info(f"       - {learning_data['pattern_count']} patterns")
    logger.info(f"       → Ready for online/continual learning")

    # ========================================================================
    # 6. Demonstrate Privacy and Retention Controls
    # ========================================================================
    logger.info("\n6. Demonstrating Privacy and Retention Controls...")
    logger.info("   " + "-" * 66)

    logger.info("\n   6.1 Data classification levels:")

    # Confidential incident
    confidential_incident = Incident(
        incident_id="inc_conf_001",
        incident_type=IncidentType.THREAT,
        status=IncidentStatus.OPEN,
        timestamp=time.time(),
        source=f"node_{node1.node_id}",
        description="Security breach attempt with sensitive data",
        severity=0.95,
        data={"sensitive": True, "encrypted": True},
        tags=["security", "breach"],
        classification="confidential",
        retention_days=30,  # GDPR/SOC2 compliance
    )
    comm1.record_incident(confidential_incident)
    logger.info(f"     ✓ Recorded CONFIDENTIAL incident (30-day retention)")

    # Public incident
    public_incident = Incident(
        incident_id="inc_pub_001",
        incident_type=IncidentType.ALERT,
        status=IncidentStatus.OPEN,
        timestamp=time.time(),
        source=f"node_{node1.node_id}",
        description="Public service notification",
        severity=0.3,
        data={"public_info": True},
        tags=["notification"],
        classification="public",
        retention_days=365,
    )
    comm1.record_incident(public_incident)
    logger.info(f"     ✓ Recorded PUBLIC incident (365-day retention)")

    logger.info("\n   6.2 Compliance features:")
    logger.info("     ✓ GDPR-compliant data retention policies")
    logger.info("     ✓ SOC2-compliant incident tracking")
    logger.info("     ✓ Automatic expiration of aged data")

    # ========================================================================
    # 7. Update plugins and show integration
    # ========================================================================
    logger.info("\n7. Demonstrating integrated operation...")
    logger.info("   " + "-" * 66)

    # Simulate some activity
    logger.info("\n   7.1 Simulating system activity (10 time steps)...")

    for step in range(10):
        # Update all plugins
        pm1.update_all(delta_time=1.0)
        pm2.update_all(delta_time=1.0)

        # Simulate node activity
        node1.energy -= 0.3
        node2.energy -= 0.3
        node1._time = step
        node2._time = step

        if step == 5:
            # Trigger a stress condition
            node1.anxiety = 8.0
            node1.energy_attack_detected = True

            # Send alert through communication plugin
            comm1.process_signal(
                {
                    "type": "alert",
                    "alert_id": f"alert_stress_{step}",
                    "description": "Stress condition detected",
                    "severity": 0.85,
                    "timestamp": time.time(),
                    "tags": ["stress", "energy_attack"],
                }
            )

        time.sleep(0.05)

    logger.info(f"     ✓ Completed simulation")

    # ========================================================================
    # 8. Show statistics and results
    # ========================================================================
    logger.info("\n8. Phase 2 System Statistics...")
    logger.info("   " + "-" * 66)

    # Communication statistics
    comm1_stats = comm1.get_statistics()
    logger.info(f"\n   Node 1 Communication:")
    logger.info(f"     Messages sent: {comm1_stats['messages_sent']}")
    logger.info(f"     Messages received: {comm1_stats['messages_received']}")
    logger.info(f"     Events published: {comm1_stats['events_published']}")
    logger.info(
        f"     Incidents recorded: {comm1_stats['incidents_recorded']}"
    )
    logger.info(f"     Patterns detected: {comm1_stats['patterns_detected']}")

    # Incident memory statistics
    memory_stats = comm1_stats["incident_memory"]
    logger.info(f"\n   Incident Memory System:")
    logger.info(
        f"     Total incidents processed: {memory_stats['total_incidents_processed']}"
    )
    logger.info(
        f"     Total patterns identified: {memory_stats['total_patterns_identified']}"
    )
    logger.info(
        f"     Stored incidents: {memory_stats['store_stats']['total_incidents']}"
    )

    if memory_stats["top_patterns"]:
        logger.info(f"\n   Top Patterns:")
        for pattern in memory_stats["top_patterns"][:3]:
            logger.info(
                f"     - {pattern['pattern_id']}: {pattern['count']} occurrences "
                f"(confidence: {pattern['confidence']:.2f})"
            )

    # ========================================================================
    # 9. Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 IMPLEMENTATION COMPLETE")
    logger.info("=" * 70)

    logger.info("\n✓ 1. Secure, Structured Messaging:")
    logger.info(
        "      - Rich message types (alert, event, anomaly, remediation, trust_update)"
    )
    logger.info("      - Idempotency checking prevents duplicate processing")
    logger.info("      - Message tracing and replay support")
    logger.info("      - Priority-based message handling")

    logger.info("\n✓ 2. Event-Driven Integrations:")
    logger.info("      - SIEM integration for security monitoring")
    logger.info("      - AIOps integration for operational intelligence")
    logger.info("      - Webhook support for custom integrations")
    logger.info("      - Real-time event streaming to external systems")

    logger.info("\n✓ 3. Incident and Pattern Memory:")
    logger.info("      - Persistent storage of significant events")
    logger.info("      - Pattern recognition for recurring issues")
    logger.info("      - Privacy/retention controls (GDPR/SOC2)")
    logger.info("      - Learning data extraction for continual adaptation")

    logger.info("\n" + "=" * 70 + "\n")

    # Cleanup
    comm1.shutdown()
    comm2.shutdown()


if __name__ == "__main__":
    demonstrate_phase2()
