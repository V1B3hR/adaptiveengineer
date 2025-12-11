"""
Security plugin for threat detection, anomaly tracking, and incident response.
"""

import logging
from typing import Any, Dict, List, Optional
from collections import deque

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.plugin_base import PluginBase, StateVariable


logger = logging.getLogger("security_plugin")


class SecurityPlugin(PluginBase):
    """
    Plugin for cybersecurity monitoring and response.

    Tracks:
    - Threat scores and anomaly detection
    - Security incident tracking
    - Attack patterns and mitigation
    - Trust levels and authentication
    """

    def __init__(
        self,
        plugin_id: str = "security",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Security plugin."""
        super().__init__(plugin_id, config)
        self.node = None
        self.threat_history = deque(maxlen=100)
        self.incident_log = deque(maxlen=200)
        self.blocked_sources = set()

    def get_plugin_type(self) -> str:
        """Return plugin type."""
        return "Security"

    def initialize(self, node: Any) -> None:
        """
        Initialize plugin with host node.

        Args:
            node: Host AliveLoopNode instance
        """
        self.node = node

        # Initialize state variables based on schema
        self.state_variables = self.get_state_schema()

        logger.info(
            f"Security plugin initialized for node {getattr(node, 'node_id', 'unknown')}"
        )

    def get_state_schema(self) -> Dict[str, StateVariable]:
        """
        Define Security state variables.

        Returns:
            Dictionary of state variable definitions
        """
        return {
            # Threat assessment
            "threat_score": StateVariable(
                name="threat_score",
                value=0.0,  # Start with no threats
                min_value=0.0,
                max_value=1.0,
                metadata={
                    "unit": "score",
                    "description": "Overall threat level (0=safe, 1=critical)",
                },
            ),
            "anomaly_score": StateVariable(
                name="anomaly_score",
                value=0.0,  # Start with no anomalies
                min_value=0.0,
                max_value=1.0,
                metadata={
                    "unit": "score",
                    "description": "Anomaly detection score",
                },
            ),
            # Incident tracking
            "active_incidents": StateVariable(
                name="active_incidents",
                value=0.0,
                min_value=0.0,
                max_value=50.0,
                metadata={
                    "unit": "count",
                    "description": "Number of active security incidents",
                },
            ),
            "incidents_mitigated": StateVariable(
                name="incidents_mitigated",
                value=0.0,
                min_value=0.0,
                max_value=1000.0,
                metadata={
                    "unit": "count",
                    "description": "Total incidents successfully mitigated",
                },
            ),
            # Attack detection
            "intrusion_attempts": StateVariable(
                name="intrusion_attempts",
                value=0.0,
                min_value=0.0,
                max_value=100.0,
                metadata={
                    "unit": "count",
                    "description": "Detected intrusion attempts",
                },
            ),
            "ddos_level": StateVariable(
                name="ddos_level",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                metadata={
                    "unit": "score",
                    "description": "DDoS attack intensity level",
                },
            ),
            # Defense status
            "defense_posture": StateVariable(
                name="defense_posture",
                value=0.5,  # Start at medium defense
                min_value=0.0,  # Low/relaxed
                max_value=1.0,  # High/strict
                metadata={
                    "unit": "level",
                    "description": "Current defense posture level",
                },
            ),
            "mitigation_effectiveness": StateVariable(
                name="mitigation_effectiveness",
                value=0.8,  # Start with 80% effectiveness
                min_value=0.0,
                max_value=1.0,
                metadata={
                    "unit": "percentage",
                    "description": "Effectiveness of security mitigations",
                },
            ),
        }

    def update_state(self, delta_time: float) -> None:
        """
        Update Security state based on node conditions and threats.

        Args:
            delta_time: Time elapsed since last update
        """
        if not self.node:
            return

        # Detect energy attacks as security threats
        energy_attack = getattr(self.node, "energy_attack_detected", False)
        if energy_attack:
            self.update_state_variable("threat_score", 0.8)
            self.update_state_variable("anomaly_score", 0.9)
            self._record_threat("energy_attack", severity=0.8)

        # Monitor for suspicious activity based on node behavior
        anxiety = getattr(self.node, "anxiety", 0.0)
        if anxiety > 8.0:
            # High anxiety might indicate detected threat
            current_threat = self.state_variables["threat_score"].value
            self.update_state_variable(
                "threat_score", min(1.0, current_threat + 0.1)
            )

        # Monitor trust network for suspicious nodes
        trust_network = getattr(self.node, "trust_network", {})
        suspicious_count = sum(
            1 for trust in trust_network.values() if trust < 0.3
        )
        if suspicious_count > 0:
            anomaly = min(1.0, suspicious_count / 10.0)
            self.update_state_variable("anomaly_score", anomaly)

        # Check for DDoS patterns (high communication load)
        comm_queue_size = len(getattr(self.node, "communication_queue", []))
        max_queue = (
            getattr(self.node, "communication_queue", deque(maxlen=20)).maxlen
            or 20
        )
        if comm_queue_size > max_queue * 0.9:
            # Queue near capacity, possible DDoS
            self.update_state_variable("ddos_level", 0.7)
            self._record_threat("possible_ddos", severity=0.7)
        else:
            # Gradually decrease DDoS level
            current_ddos = self.state_variables["ddos_level"].value
            self.update_state_variable(
                "ddos_level", max(0.0, current_ddos - 0.1)
            )

        # Adjust defense posture based on threat level
        threat_score = self.state_variables["threat_score"].value
        if threat_score > 0.7:
            # High threat, increase defense
            self.update_state_variable("defense_posture", 1.0)
        elif threat_score > 0.4:
            # Medium threat, medium defense
            self.update_state_variable("defense_posture", 0.7)
        elif threat_score < 0.2:
            # Low threat, can relax defenses
            current_posture = self.state_variables["defense_posture"].value
            self.update_state_variable(
                "defense_posture", max(0.3, current_posture - 0.1)
            )

        # Gradually decay threat score if no new threats
        current_threat = self.state_variables["threat_score"].value
        self.update_state_variable(
            "threat_score", max(0.0, current_threat - 0.05)
        )

        # Decay anomaly score
        current_anomaly = self.state_variables["anomaly_score"].value
        self.update_state_variable(
            "anomaly_score", max(0.0, current_anomaly - 0.05)
        )

    def process_signal(self, signal: Any) -> Optional[Any]:
        """
        Process Security related signals.

        Args:
            signal: Incoming signal

        Returns:
            Optional response signal
        """
        if not hasattr(signal, "signal_type"):
            return None

        signal_type = signal.signal_type

        if signal_type == "security_threat":
            # Process threat notification
            self._handle_threat_signal(signal)
            return None
        elif signal_type == "security_query":
            # Respond with security status
            return self._create_security_response()
        elif signal_type == "incident_alert":
            # Record security incident
            self._record_incident(signal.content)
            return None
        elif signal_type == "mitigation_request":
            # Attempt to mitigate threat
            return self._attempt_mitigation(signal.content)

        return None

    def get_actions(self) -> List[str]:
        """Return available Security actions."""
        return [
            "block_source",
            "unblock_source",
            "increase_defense",
            "decrease_defense",
            "scan_threats",
            "quarantine",
            "emergency_lockdown",
        ]

    def execute_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """
        Execute a Security action.

        Args:
            action_type: Type of action
            params: Action parameters

        Returns:
            True if action succeeded
        """
        if action_type == "block_source":
            return self._block_source(params.get("source_id"))
        elif action_type == "unblock_source":
            return self._unblock_source(params.get("source_id"))
        elif action_type == "increase_defense":
            return self._adjust_defense(increase=True)
        elif action_type == "decrease_defense":
            return self._adjust_defense(increase=False)
        elif action_type == "scan_threats":
            return self._scan_threats()
        elif action_type == "quarantine":
            return self._quarantine_threat(params.get("threat_id"))
        elif action_type == "emergency_lockdown":
            return self._emergency_lockdown()

        logger.warning(f"Unknown action type: {action_type}")
        return False

    def _record_threat(self, threat_type: str, severity: float) -> None:
        """Record a detected threat."""
        self.threat_history.append(
            {
                "timestamp": getattr(self.node, "_time", 0),
                "type": threat_type,
                "severity": severity,
            }
        )

        logger.warning(
            f"Threat detected: {threat_type} (severity: {severity:.2f}) for node {getattr(self.node, 'node_id', 'unknown')}"
        )

    def _record_incident(self, incident_data: Any) -> None:
        """Record a security incident."""
        self.incident_log.append(
            {
                "timestamp": getattr(self.node, "_time", 0),
                "data": incident_data,
            }
        )

        # Increment active incidents
        current_count = self.state_variables["active_incidents"].value
        self.update_state_variable("active_incidents", current_count + 1)

        logger.warning(
            f"Security incident recorded for node {getattr(self.node, 'node_id', 'unknown')}"
        )

    def _handle_threat_signal(self, signal: Any) -> None:
        """Handle incoming threat notification."""
        threat_data = signal.content
        severity = (
            threat_data.get("severity", 0.5)
            if isinstance(threat_data, dict)
            else 0.5
        )

        # Update threat score
        current_threat = self.state_variables["threat_score"].value
        new_threat = min(1.0, current_threat + severity * 0.3)
        self.update_state_variable("threat_score", new_threat)

        # Record threat
        threat_type = (
            threat_data.get("type", "unknown")
            if isinstance(threat_data, dict)
            else "unknown"
        )
        self._record_threat(threat_type, severity)

    def _create_security_response(self) -> Any:
        """Create a security status response signal."""
        try:
            from adaptiveengineer import SocialSignal

            return SocialSignal(
                content={
                    "threat_score": self.state_variables["threat_score"].value,
                    "anomaly_score": self.state_variables[
                        "anomaly_score"
                    ].value,
                    "defense_posture": self.state_variables[
                        "defense_posture"
                    ].value,
                    "active_incidents": self.state_variables[
                        "active_incidents"
                    ].value,
                },
                signal_type="security_status_response",
                urgency=0.6,
                source_id=getattr(self.node, "node_id", 0),
            )
        except ImportError:
            return None

    def _attempt_mitigation(self, mitigation_request: Any) -> Any:
        """Attempt to mitigate a security threat."""
        effectiveness = self.state_variables["mitigation_effectiveness"].value

        import random

        success = random.random() < effectiveness

        if success:
            # Decrease active incidents
            current_count = self.state_variables["active_incidents"].value
            self.update_state_variable(
                "active_incidents", max(0, current_count - 1)
            )

            # Increment mitigated count
            mitigated = self.state_variables["incidents_mitigated"].value
            self.update_state_variable("incidents_mitigated", mitigated + 1)

            # Reduce threat score
            current_threat = self.state_variables["threat_score"].value
            self.update_state_variable(
                "threat_score", max(0.0, current_threat - 0.2)
            )

            logger.info(
                f"Threat mitigation successful for node {getattr(self.node, 'node_id', 'unknown')}"
            )

        try:
            from adaptiveengineer import SocialSignal

            return SocialSignal(
                content={"success": success, "mitigation": mitigation_request},
                signal_type="mitigation_response",
                urgency=0.7,
                source_id=getattr(self.node, "node_id", 0),
            )
        except ImportError:
            return None

    def _block_source(self, source_id: Optional[int]) -> bool:
        """Block a source from communication."""
        if source_id is None:
            return False

        self.blocked_sources.add(source_id)
        logger.info(
            f"Blocked source {source_id} for node {getattr(self.node, 'node_id', 'unknown')}"
        )

        # Increment intrusion attempts
        current_attempts = self.state_variables["intrusion_attempts"].value
        self.update_state_variable("intrusion_attempts", current_attempts + 1)

        return True

    def _unblock_source(self, source_id: Optional[int]) -> bool:
        """Unblock a previously blocked source."""
        if source_id is None or source_id not in self.blocked_sources:
            return False

        self.blocked_sources.remove(source_id)
        logger.info(
            f"Unblocked source {source_id} for node {getattr(self.node, 'node_id', 'unknown')}"
        )

        return True

    def _adjust_defense(self, increase: bool) -> bool:
        """Adjust defense posture."""
        current = self.state_variables["defense_posture"].value

        if increase:
            new_posture = min(1.0, current + 0.2)
            logger.info(f"Increasing defense posture to {new_posture:.2f}")
        else:
            new_posture = max(0.0, current - 0.2)
            logger.info(f"Decreasing defense posture to {new_posture:.2f}")

        self.update_state_variable("defense_posture", new_posture)
        return True

    def _scan_threats(self) -> bool:
        """Perform threat scan."""
        logger.info(
            f"Scanning for threats on node {getattr(self.node, 'node_id', 'unknown')}"
        )

        # Check for various threat indicators
        threats_found = []

        # Check energy attack
        if getattr(self.node, "energy_attack_detected", False):
            threats_found.append("energy_attack")

        # Check suspicious nodes in trust network
        trust_network = getattr(self.node, "trust_network", {})
        suspicious = [
            nid for nid, trust in trust_network.items() if trust < 0.2
        ]
        if suspicious:
            threats_found.append(f"suspicious_nodes({len(suspicious)})")

        # Check DDoS indicators
        if self.state_variables["ddos_level"].value > 0.5:
            threats_found.append("ddos_pattern")

        logger.info(
            f"Threat scan found: {threats_found if threats_found else 'no threats'}"
        )

        # Update anomaly score based on findings
        if threats_found:
            self.update_state_variable(
                "anomaly_score", min(1.0, len(threats_found) * 0.3)
            )

        return True

    def _quarantine_threat(self, threat_id: Optional[str]) -> bool:
        """Quarantine a detected threat."""
        if not threat_id:
            return False

        logger.warning(
            f"Quarantining threat {threat_id} for node {getattr(self.node, 'node_id', 'unknown')}"
        )

        # Reduce threat score
        current_threat = self.state_variables["threat_score"].value
        self.update_state_variable(
            "threat_score", max(0.0, current_threat - 0.3)
        )

        return True

    def _emergency_lockdown(self) -> bool:
        """Activate emergency security lockdown."""
        logger.critical(
            f"Emergency lockdown activated for node {getattr(self.node, 'node_id', 'unknown')}"
        )

        # Maximum defense posture
        self.update_state_variable("defense_posture", 1.0)

        # If node has emergency mode, activate it
        if hasattr(self.node, "activate_emergency_energy_conservation"):
            self.node.activate_emergency_energy_conservation()

        return True

    def get_security_summary(self) -> Dict[str, Any]:
        """Get a summary of security metrics."""
        return {
            "threat_score": self.state_variables["threat_score"].value,
            "anomaly_score": self.state_variables["anomaly_score"].value,
            "active_incidents": self.state_variables["active_incidents"].value,
            "incidents_mitigated": self.state_variables[
                "incidents_mitigated"
            ].value,
            "intrusion_attempts": self.state_variables[
                "intrusion_attempts"
            ].value,
            "ddos_level": self.state_variables["ddos_level"].value,
            "defense_posture": self.state_variables["defense_posture"].value,
            "mitigation_effectiveness": self.state_variables[
                "mitigation_effectiveness"
            ].value,
            "blocked_sources": len(self.blocked_sources),
            "recent_threats": len(self.threat_history),
        }
