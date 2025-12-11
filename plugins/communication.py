"""
Communication Plugin for Phase 2.

Integrates:
- Secure messaging
- Event-driven integrations
- Incident and pattern memory
"""

import logging
from typing import Any, Dict, List, Optional

from core.plugin_base import PluginBase, StateVariable
from core.messaging import (
    Message,
    MessageType,
    Priority,
    create_message_bus,
    InMemoryMessageBus,
)
from core.event_integrations import (
    EventIntegrationManager,
    Event,
    EventSeverity,
    create_siem_integration,
    create_aiops_integration,
    create_webhook_integration,
)
from core.incident_memory import (
    IncidentMemorySystem,
    Incident,
    IncidentType,
    IncidentStatus,
    create_incident_from_alert,
    create_incident_from_failure,
)


logger = logging.getLogger("communication_plugin")


class CommunicationPlugin(PluginBase):
    """
    Communication plugin providing Phase 2 capabilities.

    Features:
    - Secure, structured messaging
    - Event-driven integrations (webhooks, SIEM, AIOps)
    - Incident and pattern memory
    """

    def __init__(
        self,
        plugin_id: str = "communication",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize communication plugin.

        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration
        """
        super().__init__(plugin_id, config)

        # Initialize components
        self.message_bus: Optional[InMemoryMessageBus] = None
        self.event_manager = EventIntegrationManager()
        self.incident_memory = IncidentMemorySystem()

        self.node = None

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.events_published = 0
        self.incidents_recorded = 0
        self.patterns_detected = 0

        logger.info(f"Communication plugin '{plugin_id}' created")

    def get_plugin_type(self) -> str:
        """Return plugin type."""
        return "Communication"

    def initialize(self, node: Any) -> None:
        """
        Initialize plugin with host node.

        Args:
            node: Host AliveLoopNode
        """
        self.node = node

        # Initialize message bus
        self.message_bus = create_message_bus(
            backend="memory", node_id=node.node_id
        )

        # Subscribe to messages
        self.message_bus.subscribe(self._handle_incoming_message)

        # Start message bus
        self.message_bus.start()

        # Setup event integrations if configured
        self._setup_integrations()

        # Initialize state variables
        schema = self.get_state_schema()
        for name, var in schema.items():
            self.state_variables[name] = var

        logger.info(
            f"Communication plugin initialized for node {node.node_id}"
        )

    def _setup_integrations(self) -> None:
        """Setup event integrations based on configuration."""
        config = self.config

        # Setup SIEM integration if configured
        if config.get("enable_siem", True):
            siem_config = config.get("siem_config", {})
            stream, event_filter = create_siem_integration(siem_config)
            self.event_manager.add_stream("siem", stream, event_filter)
            logger.info("SIEM integration enabled")

        # Setup AIOps integration if configured
        if config.get("enable_aiops", True):
            aiops_config = config.get("aiops_config", {})
            stream, event_filter = create_aiops_integration(aiops_config)
            self.event_manager.add_stream("aiops", stream, event_filter)
            logger.info("AIOps integration enabled")

        # Setup webhook integration if configured
        webhook_url = config.get("webhook_url")
        if webhook_url:
            stream, event_filter = create_webhook_integration(webhook_url)
            self.event_manager.add_stream("webhook", stream, event_filter)
            logger.info(f"Webhook integration enabled: {webhook_url}")

    def get_state_schema(self) -> Dict[str, StateVariable]:
        """Define communication plugin state variables."""
        return {
            "messages_sent": StateVariable(
                name="messages_sent",
                value=0.0,
                min_value=0.0,
                max_value=10000.0,
                metadata={
                    "unit": "count",
                    "description": "Total messages sent",
                },
            ),
            "messages_received": StateVariable(
                name="messages_received",
                value=0.0,
                min_value=0.0,
                max_value=10000.0,
                metadata={
                    "unit": "count",
                    "description": "Total messages received",
                },
            ),
            "events_published": StateVariable(
                name="events_published",
                value=0.0,
                min_value=0.0,
                max_value=10000.0,
                metadata={
                    "unit": "count",
                    "description": "Events published to integrations",
                },
            ),
            "incidents_recorded": StateVariable(
                name="incidents_recorded",
                value=0.0,
                min_value=0.0,
                max_value=10000.0,
                metadata={
                    "unit": "count",
                    "description": "Incidents recorded in memory",
                },
            ),
            "patterns_detected": StateVariable(
                name="patterns_detected",
                value=0.0,
                min_value=0.0,
                max_value=1000.0,
                metadata={"unit": "count", "description": "Patterns detected"},
            ),
            "communication_health": StateVariable(
                name="communication_health",
                value=1.0,
                min_value=0.0,
                max_value=1.0,
                metadata={
                    "unit": "ratio",
                    "description": "Communication system health",
                },
            ),
        }

    def update_state(self, delta_time: float) -> None:
        """
        Update plugin state.

        Args:
            delta_time: Time elapsed since last update
        """
        if not self.node:
            return

        # Update state variables
        self.state_variables["messages_sent"].update(float(self.messages_sent))
        self.state_variables["messages_received"].update(
            float(self.messages_received)
        )
        self.state_variables["events_published"].update(
            float(self.events_published)
        )
        self.state_variables["incidents_recorded"].update(
            float(self.incidents_recorded)
        )

        # Count current patterns
        patterns = self.incident_memory.get_patterns()
        self.patterns_detected = len(patterns)
        self.state_variables["patterns_detected"].update(
            float(self.patterns_detected)
        )

        # Update communication health (simple metric based on message flow)
        health = 1.0
        if self.messages_sent == 0 and delta_time > 10:
            health = 0.5  # No messages sent for a while
        self.state_variables["communication_health"].update(health)

        # Periodic maintenance
        if int(self.node._time) % 100 == 0:  # Every 100 time steps
            self.incident_memory.maintenance()

    def process_signal(self, signal: Dict[str, Any]) -> None:
        """
        Process incoming signals.

        Args:
            signal: Signal dictionary
        """
        signal_type = signal.get("type")

        if signal_type == "alert":
            self._handle_alert(signal)
        elif signal_type == "anomaly":
            self._handle_anomaly(signal)
        elif signal_type == "failure":
            self._handle_failure(signal)
        elif signal_type == "remediation":
            self._handle_remediation(signal)

    def _handle_alert(self, signal: Dict[str, Any]) -> None:
        """Handle alert signal."""
        # Send alert message
        message = Message(
            message_type=MessageType.ALERT,
            source_node_id=self.node.node_id if self.node else None,
            payload=signal,
            priority=Priority.HIGH,
            requires_ack=True,
        )
        self.send_message(message)

        # Record as incident
        incident = create_incident_from_alert(
            signal, f"node_{self.node.node_id}"
        )
        self.record_incident(incident)

    def _handle_anomaly(self, signal: Dict[str, Any]) -> None:
        """Handle anomaly signal."""
        # Send anomaly message
        message = Message(
            message_type=MessageType.ANOMALY,
            source_node_id=self.node.node_id if self.node else None,
            payload=signal,
            priority=Priority.HIGH,
        )
        self.send_message(message)

        # Record as incident
        incident = Incident(
            incident_id=f"anomaly_{self.node.node_id}_{int(signal.get('timestamp', 0))}",
            incident_type=IncidentType.ANOMALY,
            status=IncidentStatus.OPEN,
            timestamp=signal.get("timestamp", 0),
            source=f"node_{self.node.node_id}",
            description=signal.get("description", "Anomaly detected"),
            severity=signal.get("severity", 0.7),
            data=signal,
            tags=["anomaly"],
        )
        self.record_incident(incident)

    def _handle_failure(self, signal: Dict[str, Any]) -> None:
        """Handle failure signal."""
        # Send failure message
        message = Message(
            message_type=MessageType.ALERT,
            source_node_id=self.node.node_id if self.node else None,
            payload=signal,
            priority=Priority.CRITICAL,
            requires_ack=True,
        )
        self.send_message(message)

        # Record as incident
        incident = create_incident_from_failure(
            signal, f"node_{self.node.node_id}"
        )
        self.record_incident(incident)

    def _handle_remediation(self, signal: Dict[str, Any]) -> None:
        """Handle remediation signal."""
        # Send remediation message
        message = Message(
            message_type=MessageType.REMEDIATION,
            source_node_id=self.node.node_id if self.node else None,
            payload=signal,
            priority=Priority.NORMAL,
        )
        self.send_message(message)

    def _handle_incoming_message(self, message: Message) -> None:
        """Handle incoming message from message bus."""
        self.messages_received += 1
        logger.debug(
            f"Node {self.node.node_id} received message: {message.message_type.value}"
        )

        # Publish to event integrations
        self.event_manager.publish_from_message(message)
        self.events_published += 1

    def get_actions(self) -> List[str]:
        """Return list of available actions."""
        return [
            "send_message",
            "broadcast_message",
            "record_incident",
            "query_incidents",
            "get_patterns",
            "publish_event",
            "get_statistics",
        ]

    def execute_action(self, action_type: str, params: Dict[str, Any]) -> Any:
        """
        Execute a plugin action.

        Args:
            action_type: Type of action to execute
            params: Action parameters

        Returns:
            Action result
        """
        if action_type == "send_message":
            return self.send_message_action(params)
        elif action_type == "broadcast_message":
            return self.broadcast_message(params)
        elif action_type == "record_incident":
            return self.record_incident_action(params)
        elif action_type == "query_incidents":
            return self.query_incidents(params)
        elif action_type == "get_patterns":
            return self.get_patterns()
        elif action_type == "publish_event":
            return self.publish_event(params)
        elif action_type == "get_statistics":
            return self.get_statistics()
        else:
            logger.warning(f"Unknown action: {action_type}")
            return False

    def send_message_action(self, params: Dict[str, Any]) -> bool:
        """Send a message to specific node."""
        message = Message(
            message_type=MessageType(params.get("message_type", "event")),
            source_node_id=self.node.node_id if self.node else None,
            target_node_id=params.get("target_node_id"),
            payload=params.get("payload", {}),
            priority=Priority(params.get("priority", "normal")),
        )
        return self.send_message(message)

    def send_message(self, message: Message) -> bool:
        """Send a message through the message bus."""
        if not self.message_bus:
            return False

        success = self.message_bus.send(message)
        if success:
            self.messages_sent += 1
            logger.debug(
                f"Node {self.node.node_id} sent message: {message.message_type.value}"
            )
        return success

    def broadcast_message(self, params: Dict[str, Any]) -> bool:
        """Broadcast a message to all nodes."""
        message = Message(
            message_type=MessageType(params.get("message_type", "event")),
            source_node_id=self.node.node_id if self.node else None,
            target_node_id=None,  # Broadcast
            payload=params.get("payload", {}),
            priority=Priority(params.get("priority", "normal")),
        )
        return self.send_message(message)

    def record_incident_action(self, params: Dict[str, Any]) -> bool:
        """Record an incident."""
        incident = Incident(
            incident_id=params.get(
                "incident_id", f"incident_{int(params.get('timestamp', 0))}"
            ),
            incident_type=IncidentType(params.get("incident_type", "event")),
            status=IncidentStatus(params.get("status", "open")),
            timestamp=params.get("timestamp", 0),
            source=params.get("source", f"node_{self.node.node_id}"),
            description=params.get("description", ""),
            severity=params.get("severity", 0.5),
            data=params.get("data", {}),
            tags=params.get("tags", []),
        )
        return self.record_incident(incident)[0]

    def record_incident(self, incident: Incident) -> tuple:
        """Record an incident and check for patterns."""
        success, pattern = self.incident_memory.record_incident(incident)
        if success:
            self.incidents_recorded += 1
            if pattern:
                logger.info(f"Pattern detected: {pattern.pattern_id}")
        return success, pattern

    def query_incidents(self, params: Dict[str, Any]) -> List[Incident]:
        """Query incidents from memory."""
        return self.incident_memory.query_incidents(**params)

    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get detected patterns."""
        patterns = self.incident_memory.get_patterns()
        return [p.to_dict() for p in patterns]

    def publish_event(self, params: Dict[str, Any]) -> int:
        """Publish an event to integrations."""
        event = Event(
            event_id=params.get(
                "event_id", f"event_{int(params.get('timestamp', 0))}"
            ),
            event_type=params.get("event_type", "custom"),
            severity=EventSeverity(params.get("severity", "info")),
            source=params.get("source", f"node_{self.node.node_id}"),
            timestamp=params.get("timestamp", 0),
            data=params.get("data", {}),
            tags=params.get("tags", []),
        )
        count = self.event_manager.publish_event(event)
        self.events_published += count
        return count

    def get_state(self) -> Dict[str, float]:
        """Get current state values."""
        return {name: var.value for name, var in self.state_variables.items()}

    def get_state_variable(self, name: str) -> Optional[StateVariable]:
        """Get a specific state variable."""
        return self.state_variables.get(name)

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "events_published": self.events_published,
            "incidents_recorded": self.incidents_recorded,
            "patterns_detected": self.patterns_detected,
            "incident_memory": self.incident_memory.get_statistics(),
        }

    def shutdown(self) -> None:
        """Shutdown communication plugin."""
        if self.message_bus:
            self.message_bus.stop()

        self.event_manager.close_all()

        logger.info(
            f"Communication plugin shut down (node {self.node.node_id if self.node else 'unknown'})"
        )
