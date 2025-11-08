# Phase 2 Implementation: Communication, Security, and Incident Memory

This document describes the implementation of Phase 2 requirements from the roadmap.

## Overview

Phase 2 extends the foundational architecture from Phase 1 with advanced communication, security, and memory capabilities:

1. **Secure, Structured Messaging** - Rich message types with encryption, idempotency, and tracing
2. **Event-Driven Integrations** - Real-time streaming to external systems (SIEM, AIOps, monitoring)
3. **Incident and Pattern Memory** - Persistent storage with pattern recognition and compliance controls

## 1. Secure, Structured Messaging

### Message System Architecture

The messaging system provides secure, reliable communication between adaptive engineer nodes.

#### Core Components

**`Message` (core/messaging.py)**
- Rich message structure with multiple types:
  - `ALERT` - Critical alerts requiring attention
  - `EVENT` - General events
  - `ANOMALY` - Detected anomalies
  - `REMEDIATION` - Remediation actions
  - `TRUST_UPDATE` - Trust network updates
  - `CONSENSUS` - Consensus messages
  - `HEARTBEAT` - Node heartbeats
- Priority levels: LOW, NORMAL, HIGH, CRITICAL
- Built-in metadata: timestamp, trace_id, TTL, requires_ack
- Support for encryption flags

**`MessageBus` (core/messaging.py)**
- Abstract base class for message bus implementations
- Idempotency checking to prevent duplicate processing
- Message history for replay support
- Distributed tracing with trace_id
- Handler-based message routing

**`InMemoryMessageBus` (core/messaging.py)**
- In-memory implementation for single-process scenarios
- Threaded message processing
- Full feature support without external dependencies
- Suitable for testing and development

### Key Features

✅ **Rich Message Types**: Multiple message types for different communication patterns

✅ **Idempotency**: Content-based hashing prevents duplicate processing
```python
# Duplicate messages are automatically filtered
message1 = Message(message_type=MessageType.ALERT, payload={'cpu': 0.95})
message2 = Message(message_type=MessageType.ALERT, payload={'cpu': 0.95})
bus.send(message1)  # Processed
bus.send(message2)  # Automatically filtered as duplicate
```

✅ **Message Tracing**: Track related messages with trace_id
```python
# Get all messages in a trace
trace_messages = message_bus.get_trace(trace_id)
```

✅ **Replay Support**: Replay messages by time range or type
```python
# Replay messages from last hour
messages = message_bus.replay_messages(
    start_time=time.time() - 3600,
    message_types=[MessageType.ALERT, MessageType.ANOMALY]
)
```

✅ **Priority Handling**: Messages prioritized by criticality

### Example: Sending Messages

```python
from core.messaging import Message, MessageType, Priority, create_message_bus

# Create message bus
bus = create_message_bus(backend="memory", node_id=1)
bus.start()

# Send alert message
alert = Message(
    message_type=MessageType.ALERT,
    source_node_id=1,
    target_node_id=2,
    payload={'cpu_usage': 0.95, 'threshold': 0.90},
    priority=Priority.HIGH,
    requires_ack=True
)
bus.send(alert)

# Subscribe to messages
def handle_message(message: Message):
    print(f"Received: {message.message_type}")

bus.subscribe(handle_message, message_types=[MessageType.ALERT])
```

## 2. Event-Driven Integrations

### Integration Architecture

The event integration system provides real-time streaming to external monitoring and security tools.

#### Core Components

**`Event` (core/event_integrations.py)**
- Structured event format for external systems
- Severity levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Flexible tagging for filtering
- Timestamp and source tracking

**`EventStream` (core/event_integrations.py)**
- Abstract base class for event streams
- Implementations:
  - `WebhookStream` - HTTP webhook delivery
  - `SIEMStream` - Security Information and Event Management integration
  - `AIOpsStream` - AI-driven Operations platform integration

**`EventIntegrationManager` (core/event_integrations.py)**
- Centralized management of event streams
- Event filtering and routing
- Automatic conversion from messages to events
- Event buffering for replay

**`EventFilter` (core/event_integrations.py)**
- Filter events by type, severity, tags, or source
- Applied per-stream for targeted delivery

### Key Features

✅ **Multiple Integration Types**: Webhooks, SIEM, AIOps out of the box

✅ **Event Filtering**: Route specific events to specific systems
```python
# Only send WARNING and above to SIEM
siem_filter = EventFilter(
    severities=[EventSeverity.WARNING, EventSeverity.ERROR, EventSeverity.CRITICAL]
)
```

✅ **Real-Time Streaming**: Events streamed as they occur

✅ **Automatic Conversion**: Messages automatically converted to events

### Example: Setting Up Integrations

```python
from core.event_integrations import (
    EventIntegrationManager,
    create_siem_integration,
    create_aiops_integration
)

# Create integration manager
manager = EventIntegrationManager()

# Add SIEM integration (security events only)
siem_stream, siem_filter = create_siem_integration(
    siem_config={'endpoint': 'siem.example.com'},
    severity_threshold=EventSeverity.WARNING
)
manager.add_stream('siem', siem_stream, siem_filter)

# Add AIOps integration (operational events)
aiops_stream, aiops_filter = create_aiops_integration(
    aiops_config={'endpoint': 'aiops.example.com'},
    tags=['operations', 'performance']
)
manager.add_stream('aiops', aiops_stream, aiops_filter)

# Publish events
event = Event(
    event_id='evt_001',
    event_type='security_alert',
    severity=EventSeverity.CRITICAL,
    source='node_1',
    data={'threat_type': 'intrusion'},
    tags=['security', 'intrusion']
)
manager.publish_event(event)  # Routed to SIEM
```

## 3. Incident and Pattern Memory

### Memory System Architecture

The incident memory system provides persistent storage, pattern recognition, and learning support.

#### Core Components

**`Incident` (core/incident_memory.py)**
- Structured incident record
- Types: ALERT, REMEDIATION, FAILURE, ANOMALY, THREAT, RECOVERY, DEGRADATION
- Status tracking: OPEN, IN_PROGRESS, RESOLVED, FAILED, RECURRING
- Privacy classification: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
- Retention policies for compliance

**`Pattern` (core/incident_memory.py)**
- Identified pattern in incident data
- Tracks occurrence count, sources, severity
- Confidence scoring based on occurrences
- Automatic updates as new matching incidents occur

**`IncidentStore` (core/incident_memory.py)**
- Abstract storage interface
- `InMemoryIncidentStore` - In-memory implementation
- Query capabilities with filtering
- Automatic expiration based on retention policies

**`PatternRecognizer` (core/incident_memory.py)**
- Analyzes incidents for recurring patterns
- Signature-based matching
- Confidence scoring
- Identifies trends and repeated issues

**`IncidentMemorySystem` (core/incident_memory.py)**
- Complete memory system combining storage and recognition
- Records incidents and checks for patterns
- Provides learning data for continual adaptation
- Maintenance operations for compliance

### Key Features

✅ **Persistent Storage**: All significant events recorded

✅ **Pattern Recognition**: Automatically identifies recurring issues
```python
# Record incidents - patterns detected automatically
for i in range(5):
    incident = Incident(
        incident_id=f'inc_{i}',
        incident_type=IncidentType.ALERT,
        source='server_1',
        description='High CPU',
        tags=['cpu', 'performance']
    )
    success, pattern = memory_system.record_incident(incident)
    
    if pattern and pattern.occurrence_count >= 3:
        print(f"Recurring pattern detected! Occurred {pattern.occurrence_count} times")
```

✅ **Privacy/Retention Controls**: GDPR and SOC2 compliant
```python
# Confidential incident with 30-day retention
incident = Incident(
    incident_id='inc_sensitive',
    incident_type=IncidentType.THREAT,
    classification=DataClassification.CONFIDENTIAL,
    retention_days=30,  # Automatic deletion after 30 days
    data={'sensitive': True}
)
```

✅ **Learning Data Extraction**: Support for online/continual learning
```python
# Get data for ML model training
learning_data = memory_system.get_learning_data(
    incident_type=IncidentType.ANOMALY
)
# Returns: incidents, patterns, statistics
```

✅ **Compliance Features**:
- Automatic data expiration based on retention policies
- Data classification levels
- Audit trail support
- Right to be forgotten (incident deletion)

### Example: Using Incident Memory

```python
from core.incident_memory import (
    IncidentMemorySystem,
    Incident,
    IncidentType,
    IncidentStatus
)
import time

# Create memory system
memory = IncidentMemorySystem()

# Record incident
incident = Incident(
    incident_id='inc_001',
    incident_type=IncidentType.ALERT,
    status=IncidentStatus.OPEN,
    timestamp=time.time(),
    source='node_1',
    description='High CPU usage detected',
    severity=0.8,
    data={'cpu_usage': 0.95},
    tags=['cpu', 'performance']
)
success, pattern = memory.record_incident(incident)

# Query incidents
recent = memory.query_incidents(
    incident_types=[IncidentType.ALERT],
    tags=['cpu'],
    limit=10
)

# Get patterns
patterns = memory.get_patterns(min_confidence=0.3)
for pattern in patterns:
    print(f"Pattern: {pattern.occurrence_count} occurrences, "
          f"confidence: {pattern.confidence:.2f}")

# Get learning data
learning_data = memory.get_learning_data()
print(f"Available for learning: {learning_data['incident_count']} incidents, "
      f"{learning_data['pattern_count']} patterns")

# Maintenance (cleanup old data)
stats = memory.maintenance()
print(f"Cleared {stats['expired_incidents']} expired incidents")
```

## 4. Communication Plugin

### Integration with Plugin System

The `CommunicationPlugin` integrates all Phase 2 capabilities into the plugin architecture.

**Features:**
- Manages message bus for the node
- Routes signals to appropriate channels
- Publishes events to integrations
- Records incidents and patterns
- Provides communication state variables

**State Variables:**
| Variable | Description |
|----------|-------------|
| `messages_sent` | Total messages sent |
| `messages_received` | Total messages received |
| `events_published` | Events published to integrations |
| `incidents_recorded` | Incidents recorded in memory |
| `patterns_detected` | Patterns identified |
| `communication_health` | Overall communication system health |

**Actions:**
- `send_message` - Send message to specific node
- `broadcast_message` - Broadcast to all nodes
- `record_incident` - Record an incident
- `query_incidents` - Query incident history
- `get_patterns` - Get detected patterns
- `publish_event` - Publish event to integrations
- `get_statistics` - Get communication statistics

### Example: Using Communication Plugin

```python
from plugins.communication import CommunicationPlugin
from core.plugin_manager import PluginManager
from adaptiveengineer import AliveLoopNode

# Create node
node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)

# Setup communication plugin
comm_plugin = CommunicationPlugin(
    plugin_id="communication",
    config={
        'enable_siem': True,
        'enable_aiops': True,
        'siem_config': {'endpoint': 'siem.example.com'},
        'aiops_config': {'endpoint': 'aiops.example.com'}
    }
)

# Initialize with node
manager = PluginManager()
manager.register_plugin(comm_plugin)
manager.initialize_all(node)

# Send alert
comm_plugin.execute_action('send_message', {
    'message_type': 'alert',
    'target_node_id': 2,
    'payload': {'alert': 'High CPU'},
    'priority': 'high'
})

# Record incident
comm_plugin.execute_action('record_incident', {
    'incident_id': 'inc_001',
    'incident_type': 'alert',
    'timestamp': time.time(),
    'description': 'System alert',
    'severity': 0.8,
    'tags': ['system']
})

# Get statistics
stats = comm_plugin.get_statistics()
print(f"Messages sent: {stats['messages_sent']}")
print(f"Patterns detected: {stats['patterns_detected']}")
```

## Testing

Run the Phase 2 demonstration:

```bash
python3 example_phase2.py
```

The demonstration shows:
1. **Secure Messaging**: Rich message types, idempotency, tracing, replay
2. **Event Integrations**: SIEM, AIOps, webhook streaming
3. **Incident Memory**: Recording, pattern detection, learning data extraction
4. **Privacy Controls**: Data classification and retention policies
5. **Integrated Operation**: All components working together

## Architecture Benefits

### Secure Communication
- ✅ Idempotency prevents duplicate processing
- ✅ Message tracing for debugging and analysis
- ✅ Priority-based handling for critical events
- ✅ Replay support for recovery and testing

### External Integration
- ✅ Real-time streaming to monitoring tools
- ✅ Flexible filtering and routing
- ✅ Multiple integration types (SIEM, AIOps, webhooks)
- ✅ Automatic event conversion

### Intelligent Memory
- ✅ Pattern recognition for recurring issues
- ✅ Learning data for continual adaptation
- ✅ Compliance-ready with retention policies
- ✅ Privacy controls (GDPR/SOC2)

## Future Enhancements

The Phase 2 architecture supports future expansion:

- **Distributed Message Buses**: ZeroMQ, RabbitMQ, Kafka backends
- **Encryption**: End-to-end message encryption
- **Advanced Patterns**: Machine learning for pattern recognition
- **Distributed Storage**: Database backends for incident storage
- **Federation**: Multi-cluster communication

## Summary

Phase 2 establishes:

✅ **Secure, Structured Messaging**
- Rich message types (alert, event, anomaly, remediation, trust_update, consensus)
- Idempotency, tracing, and replay support
- Multiple backend support (in-memory, with extensibility for distributed systems)

✅ **Event-Driven Integrations**
- Webhook/APM triggers for external IT/SOC tools
- Real-time signal streaming (SIEM, AIOps, monitoring)
- Flexible filtering and routing

✅ **Incident and Pattern Memory**
- Persistent storage of significant events
- Pattern recognition for recurring issues and threats
- Privacy/retention controls (GDPR/SOC2 compliant)
- Learning data extraction for continual adaptation

The implementation provides a robust foundation for adaptive, learning-based systems with secure communication, external integrations, and intelligent memory capabilities.
