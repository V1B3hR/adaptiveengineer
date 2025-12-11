"""
Adaptive Defense System - Self-healing cyber defenses with automated threat response

This module provides automated, adaptive cybersecurity defenses:
- Automated threat response (block IPs, processes, restart services)
- Self-healing actions (rollback, quarantine, configuration adjustment)
- Full auditability for compliance

Enhancements in this version:
- Thread-safe state updates with RLock
- Per-threat response thresholds with adaptive learning
- Cooldown and deduplication to prevent over-response to repeated sources
- Dry-run (simulate) mode for safe testing
- Pluggable action handlers (extend or override built-in actions)
- Structured audit sink (logger and/or file), JSON audit events
- State persistence helpers (save/load)
- Prometheus-style metrics export
- Input validation and serialization helpers (to_dict)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
from uuid import uuid4

logger = logging.getLogger("adaptive_defense")


class ThreatType(str, Enum):
    """Types of threats"""

    MALWARE = "malware"
    DDOS = "ddos"
    INTRUSION = "intrusion"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    BRUTE_FORCE = "brute_force"
    ANOMALY = "anomaly"


class DefenseAction(str, Enum):
    """Automated defense actions"""

    BLOCK_IP = "block_ip"
    BLOCK_PROCESS = "block_process"
    RESTART_SERVICE = "restart_service"
    ROLLBACK_CONFIG = "rollback_config"
    QUARANTINE = "quarantine"
    RATE_LIMIT = "rate_limit"
    ISOLATE_SEGMENT = "isolate_segment"
    APPLY_PATCH = "apply_patch"


class HealingAction(str, Enum):
    """Self-healing actions"""

    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"
    CONFIG_ADJUST = "config_adjust"
    FAILOVER = "failover"
    RESTORE_BACKUP = "restore_backup"


def _ts() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts or _ts()))


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass
class ThreatEvent:
    """Record of a detected threat"""

    event_id: str
    timestamp: float
    threat_type: ThreatType
    severity: float  # 0.0 to 1.0
    source: str  # IP, process, user, etc.
    confidence: float  # 0.0 to 1.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    responded: bool = False
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_iso"] = _iso(self.timestamp)
        d["threat_type"] = str(self.threat_type)
        return d


@dataclass
class DefenseResponse:
    """Record of a defense action taken"""

    response_id: str
    timestamp: float
    threat_event_id: str
    action: DefenseAction
    target: str
    success: bool
    confidence: float
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[str] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_iso"] = _iso(self.timestamp)
        d["action"] = str(self.action)
        return d


@dataclass(frozen=True)
class HealingResponse:
    response_id: str
    timestamp: float
    healing_action: HealingAction
    target: str
    success: bool
    recovery_time: float  # seconds
    audit_trail: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    failure_reason: Optional[str] = None
    error_code: Optional[str] = None
    verification_passed: Optional[bool] = None
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_iso"] = _iso(self.timestamp)
        d["healing_action"] = str(self.healing_action)
        return d


def make_heal_id(node_id: int) -> str:
    return f"heal_{node_id}_{uuid4()}"


ActionHandler = Callable[
    [DefenseAction, str, ThreatEvent], Tuple[bool, List[str]]
]


class AdaptiveDefenseSystem:
    """
    Automated, adaptive cybersecurity defense system.

    Capabilities:
    - Automated threat detection and response
    - Self-healing and recovery actions
    - Adaptive learning from incidents
    - Full auditability
    """

    def __init__(
        self,
        node_id: int,
        auto_response_enabled: bool = True,
        response_threshold: float = 0.7,
        audit_all: bool = True,
        audit_file: Optional[str] = None,
        dry_run: bool = False,
        default_cooldown_seconds: float = 30.0,
        per_action_cooldowns: Optional[Dict[DefenseAction, float]] = None,
        maintenance_quiet_hours_utc: Optional[List[Tuple[int, int]]] = None,
        per_threat_response_thresholds: Optional[
            Dict[ThreatType, float]
        ] = None,
    ):
        """
        Initialize adaptive defense system.

        Args:
            node_id: Unique identifier for this node
            auto_response_enabled: Whether automatic response is enabled
            response_threshold: Default confidence threshold for auto-response (0.0-1.0)
            audit_all: Whether to audit all actions
            audit_file: Optional file path to append JSON audit entries
            dry_run: If True, do not mutate blocklists/quarantine; only log/audit
            default_cooldown_seconds: Cooldown to deduplicate rapid repeats
            per_action_cooldowns: Per-action override cooldowns
            maintenance_quiet_hours_utc: List of (start_hour, end_hour) tuples to suppress auto response
            per_threat_response_thresholds: Optional per-threat thresholds
        """
        self.node_id = node_id
        self.auto_response_enabled = auto_response_enabled
        self.response_threshold = _clamp01(response_threshold)
        self.audit_all = audit_all
        self.audit_file = audit_file
        self.dry_run = dry_run

        # Thread safety
        self._lock = threading.RLock()

        # Tracking
        self.threat_events: deque[ThreatEvent] = deque(maxlen=2000)
        self.defense_responses: deque[DefenseResponse] = deque(maxlen=2000)
        self.healing_responses: deque[HealingResponse] = deque(maxlen=1000)

        # Block lists
        self.blocked_ips: Set[str] = set()
        self.blocked_processes: Set[str] = set()
        self.quarantined_items: Set[str] = set()

        # Configuration states (for rollback)
        self.config_history: deque[Dict[str, Any]] = deque(maxlen=100)
        self.current_config: Dict[str, Any] = {}

        # Metrics
        self.threats_detected = 0
        self.threats_blocked = 0
        self.false_positives = 0
        self.healings_performed = 0

        # Adaptive thresholds (learned over time) used for detection sensitivity
        self.threat_thresholds: Dict[ThreatType, float] = {
            threat_type: 0.5 for threat_type in ThreatType
        }

        # Per-threat response thresholds (for auto-response gating)
        self.response_thresholds: Dict[ThreatType, float] = {
            threat_type: self.response_threshold for threat_type in ThreatType
        }
        if per_threat_response_thresholds:
            for t, v in per_threat_response_thresholds.items():
                self.response_thresholds[t] = _clamp01(v)

        # Cooldowns
        self.default_cooldown_seconds = max(0.0, default_cooldown_seconds)
        self.per_action_cooldowns: Dict[DefenseAction, float] = (
            per_action_cooldowns or {}
        )
        self._last_action_time_by_key: Dict[
            Tuple[DefenseAction, str], float
        ] = {}

        # Quiet hours for auto response
        self.maintenance_quiet_hours_utc = maintenance_quiet_hours_utc or []

        # Action handlers registry for extensibility
        self._action_handlers: Dict[DefenseAction, ActionHandler] = {}

        logger.info(
            "Adaptive defense system initialized for node %s "
            "(auto_response=%s, default_threshold=%.2f, dry_run=%s)",
            node_id,
            auto_response_enabled,
            self.response_threshold,
            dry_run,
        )

    # ---------------------------
    # Registration and utilities
    # ---------------------------
    def register_action_handler(
        self, action: DefenseAction, handler: ActionHandler
    ):
        """Register or override a handler for a specific defense action."""
        with self._lock:
            self._action_handlers[action] = handler
            logger.info("Registered custom handler for action %s", action)

    def _is_quiet_time(self) -> bool:
        """Returns True if current UTC hour is within any quiet window."""
        if not self.maintenance_quiet_hours_utc:
            return False
        hour = int(time.gmtime().tm_hour)
        for start, end in self.maintenance_quiet_hours_utc:
            # Support wrapping windows like (22, 2)
            if start <= end:
                if start <= hour < end:
                    return True
            else:
                if hour >= start or hour < end:
                    return True
        return False

    def _should_dedup(self, action: DefenseAction, target: str) -> bool:
        """Apply cooldown-based dedup to avoid repeated actions on the same target."""
        now = _ts()
        key = (action, target)
        cooldown = self.per_action_cooldowns.get(
            action, self.default_cooldown_seconds
        )
        last = self._last_action_time_by_key.get(key, 0.0)
        if now - last < cooldown:
            return True
        self._last_action_time_by_key[key] = now
        return False

    def _audit_json(self, payload: Dict[str, Any]):
        """Write structured audit event to logger and/or file."""
        if self.audit_all:
            logger.info(
                "Audit: %s",
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )
        if self.audit_file:
            try:
                Path(os.path.dirname(self.audit_file) or ".").mkdir(
                    parents=True, exist_ok=True
                )
                with open(self.audit_file, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    )
                    f.write("\n")
            except Exception as e:
                logger.error(
                    "Failed to write audit file %s: %s", self.audit_file, e
                )

    # ---------------------------
    # Threats
    # ---------------------------
    def detect_threat(
        self,
        threat_type: ThreatType,
        source: str,
        severity: float,
        confidence: float,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> ThreatEvent:
        """
        Detect and record a threat.

        Args:
            threat_type: Type of threat
            source: Source of threat (IP, process, etc.)
            severity: Severity level (0.0-1.0)
            confidence: Detection confidence (0.0-1.0)
            indicators: Optional threat indicators

        Returns:
            ThreatEvent record
        """
        severity = _clamp01(severity)
        confidence = _clamp01(confidence)

        event_id = f"threat_{self.node_id}_{int(_ts()*1000)}_{uuid4().hex[:8]}"
        event = ThreatEvent(
            event_id=event_id,
            timestamp=_ts(),
            threat_type=threat_type,
            severity=severity,
            source=source,
            confidence=confidence,
            indicators=indicators or {},
        )

        with self._lock:
            self.threats_detected += 1
            self.threat_events.append(event)

        logger.warning(
            "Threat detected: %s from %s (severity=%.2f, confidence=%.2f)",
            threat_type,
            source,
            severity,
            confidence,
        )

        self._audit_json(
            {
                "type": "threat_detected",
                "node_id": self.node_id,
                "event": event.to_dict(),
            }
        )

        # Automatic response if enabled, not in quiet time, and confidence high enough
        threshold = self.response_thresholds.get(
            threat_type, self.response_threshold
        )
        if (
            self.auto_response_enabled
            and not self._is_quiet_time()
            and confidence >= threshold
        ):
            self.respond_to_threat(event)

        return event

    def respond_to_threat(
        self, threat_event: ThreatEvent
    ) -> Optional[DefenseResponse]:
        """
        Automatically respond to a detected threat.

        Args:
            threat_event: The threat to respond to

        Returns:
            DefenseResponse record or None if no action taken
        """
        with self._lock:
            if threat_event.responded:
                logger.debug(
                    "Threat %s already responded to", threat_event.event_id
                )
                return None

        # Determine appropriate defense action
        action = self._select_defense_action(threat_event)
        if not action:
            logger.info(
                "No action selected for threat %s", threat_event.event_id
            )
            return None

        # Dedup via cooldown
        if self._should_dedup(action, threat_event.source):
            logger.info(
                "Suppressed repeated action %s for target %s due to cooldown",
                action,
                threat_event.source,
            )
            return None

        response_id = (
            f"response_{self.node_id}_{int(_ts()*1000)}_{uuid4().hex[:8]}"
        )

        # Record metrics before action
        metrics_before = self._get_current_metrics()

        # Execute defense action
        success, audit_log = self._execute_defense_action(
            action, threat_event.source, threat_event
        )

        # Record metrics after action
        metrics_after = self._get_current_metrics()

        response = DefenseResponse(
            response_id=response_id,
            timestamp=_ts(),
            threat_event_id=threat_event.event_id,
            action=action,
            target=threat_event.source,
            success=success,
            confidence=threat_event.confidence,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            audit_trail=audit_log,
        )

        with self._lock:
            self.defense_responses.append(response)
            threat_event.responded = True
            if success:
                self.threats_blocked += 1

        self._audit_json(
            {
                "type": "defense_response",
                "node_id": self.node_id,
                "response": response.to_dict(),
            }
        )

        if success:
            logger.info(
                "Successfully responded to threat with %s: %s",
                action,
                threat_event.source,
            )
        else:
            logger.warning(
                "Failed to respond to threat with %s: %s",
                action,
                threat_event.source,
            )

        return response

    def _select_defense_action(
        self, threat_event: ThreatEvent
    ) -> Optional[DefenseAction]:
        """Select appropriate defense action for threat type"""
        action_map = {
            ThreatType.MALWARE: DefenseAction.QUARANTINE,
            ThreatType.DDOS: DefenseAction.BLOCK_IP,
            ThreatType.INTRUSION: DefenseAction.BLOCK_IP,
            ThreatType.DATA_EXFILTRATION: DefenseAction.BLOCK_PROCESS,
            ThreatType.PRIVILEGE_ESCALATION: DefenseAction.ISOLATE_SEGMENT,
            ThreatType.BRUTE_FORCE: DefenseAction.RATE_LIMIT,
            ThreatType.ANOMALY: DefenseAction.QUARANTINE,
        }
        return action_map.get(threat_event.threat_type)

    def _execute_defense_action(
        self, action: DefenseAction, target: str, threat_event: ThreatEvent
    ) -> Tuple[bool, List[str]]:
        """
        Execute a defense action.

        Returns:
            Tuple of (success, audit_trail)
        """
        audit_log: List[str] = []
        audit_log.append(f"Action: {action} on target: {target}")
        audit_log.append(
            f"Threat: {threat_event.threat_type}, Severity: {threat_event.severity:.2f}"
        )
        audit_log.append(f"Timestamp: {_ts()}")

        # Use custom handler if registered
        handler = self._action_handlers.get(action)

        try:
            if handler is not None:
                success, trail = handler(action, target, threat_event)
                audit_log.extend(trail)
            else:
                # Built-in behavior
                if action == DefenseAction.BLOCK_IP:
                    if not self.dry_run:
                        with self._lock:
                            self.blocked_ips.add(target)
                    audit_log.append(f"Blocked IP: {target}")
                    success = True

                elif action == DefenseAction.BLOCK_PROCESS:
                    if not self.dry_run:
                        with self._lock:
                            self.blocked_processes.add(target)
                    audit_log.append(f"Blocked process: {target}")
                    success = True

                elif action == DefenseAction.QUARANTINE:
                    if not self.dry_run:
                        with self._lock:
                            self.quarantined_items.add(target)
                    audit_log.append(f"Quarantined: {target}")
                    success = True

                elif action == DefenseAction.RESTART_SERVICE:
                    audit_log.append(f"Restarted service: {target}")
                    success = True  # Simulated

                elif action == DefenseAction.ROLLBACK_CONFIG:
                    success = self._rollback_configuration(target)
                    audit_log.append(
                        f"Rolled back configuration for: {target}"
                    )

                elif action == DefenseAction.RATE_LIMIT:
                    audit_log.append(f"Applied rate limiting to: {target}")
                    success = True  # Simulated

                elif action == DefenseAction.ISOLATE_SEGMENT:
                    audit_log.append(f"Isolated network segment for: {target}")
                    success = True  # Simulated

                elif action == DefenseAction.APPLY_PATCH:
                    audit_log.append(f"Applied security patch for: {target}")
                    success = True  # Simulated

                else:
                    audit_log.append(f"Unknown action: {action}")
                    success = False

            if self.audit_all:
                logger.info("Audit trail: %s", " | ".join(audit_log))

            return success, audit_log

        except Exception as e:
            audit_log.append(f"Error: {str(e)}")
            logger.error("Error executing defense action: %s", e)
            return False, audit_log

    # ---------------------------
    # Metrics and audit
    # ---------------------------
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self._lock:
            metrics = {
                "blocked_ips_count": len(self.blocked_ips),
                "blocked_processes_count": len(self.blocked_processes),
                "quarantined_count": len(self.quarantined_items),
                "threats_detected": self.threats_detected,
                "threats_blocked": self.threats_blocked,
                "false_positives": self.false_positives,
            }
        metrics["timestamp"] = _ts()
        return metrics

    def get_defense_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about defense operations.

        Returns:
            Dictionary of metrics
        """
        now = _ts()
        with self._lock:
            recent_threats = len(
                [t for t in self.threat_events if now - t.timestamp < 300]
            )
            recent_blocks = len(
                [
                    r
                    for r in self.defense_responses
                    if now - r.timestamp < 300 and r.success
                ]
            )
            threats_blocked = self.threats_blocked
            threats_detected = self.threats_detected
            false_positives = self.false_positives
            healings_performed = self.healings_performed
            blocked_ips = len(self.blocked_ips)
            blocked_processes = len(self.blocked_processes)
            quarantined_items = len(self.quarantined_items)

        block_rate = (
            (threats_blocked / threats_detected)
            if threats_detected > 0
            else 0.0
        )

        return {
            "node_id": self.node_id,
            "threats_detected": threats_detected,
            "threats_blocked": threats_blocked,
            "block_rate": block_rate,
            "false_positives": false_positives,
            "healings_performed": healings_performed,
            "blocked_ips": blocked_ips,
            "blocked_processes": blocked_processes,
            "quarantined_items": quarantined_items,
            "recent_threats_5min": recent_threats,
            "recent_blocks_5min": recent_blocks,
            "auto_response_enabled": self.auto_response_enabled,
            "dry_run": self.dry_run,
            "timestamp": now,
        }

    def export_metrics_prometheus(self) -> str:
        """Export key metrics as Prometheus exposition format (text)."""
        m = self.get_defense_metrics()
        lines = [
            "# TYPE adaptive_threats_detected counter",
            f"adaptive_threats_detected{{node_id=\"{self.node_id}\"}} {m['threats_detected']}",
            "# TYPE adaptive_threats_blocked counter",
            f"adaptive_threats_blocked{{node_id=\"{self.node_id}\"}} {m['threats_blocked']}",
            "# TYPE adaptive_false_positives counter",
            f"adaptive_false_positives{{node_id=\"{self.node_id}\"}} {m['false_positives']}",
            "# TYPE adaptive_healings_performed counter",
            f"adaptive_healings_performed{{node_id=\"{self.node_id}\"}} {m['healings_performed']}",
            "# TYPE adaptive_block_rate gauge",
            f"adaptive_block_rate{{node_id=\"{self.node_id}\"}} {m['block_rate']}",
            "# TYPE adaptive_blocked_ips gauge",
            f"adaptive_blocked_ips{{node_id=\"{self.node_id}\"}} {m['blocked_ips']}",
            "# TYPE adaptive_blocked_processes gauge",
            f"adaptive_blocked_processes{{node_id=\"{self.node_id}\"}} {m['blocked_processes']}",
            "# TYPE adaptive_quarantined_items gauge",
            f"adaptive_quarantined_items{{node_id=\"{self.node_id}\"}} {m['quarantined_items']}",
            "# TYPE adaptive_recent_threats_5min gauge",
            f"adaptive_recent_threats_5min{{node_id=\"{self.node_id}\"}} {m['recent_threats_5min']}",
            "# TYPE adaptive_recent_blocks_5min gauge",
            f"adaptive_recent_blocks_5min{{node_id=\"{self.node_id}\"}} {m['recent_blocks_5min']}",
        ]
        return "\n".join(lines) + "\n"

    def get_audit_log(
        self, limit: int = 50, action_type: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log of defense and healing actions.

        Args:
            limit: Maximum number of entries to return
            action_type: Optional filter by action type (DefenseAction, HealingAction, or string)

        Returns:
            List of audit entries (newest first)
        """

        def _match_action(a: Any, action_type: Any) -> bool:
            if action_type is None:
                return True
            if isinstance(action_type, (DefenseAction, HealingAction)):
                return a == action_type
            return str(a) == str(action_type)

        entries: List[Dict[str, Any]] = []

        with self._lock:
            for response in list(self.defense_responses)[-limit:]:
                if not _match_action(response.action, action_type):
                    continue
                entries.append(
                    {
                        "type": "defense_response",
                        "response_id": response.response_id,
                        "timestamp": response.timestamp,
                        "action": response.action,
                        "target": response.target,
                        "success": response.success,
                        "audit_trail": response.audit_trail,
                    }
                )

            for response in list(self.healing_responses)[-limit:]:
                if not _match_action(response.healing_action, action_type):
                    continue
                entries.append(
                    {
                        "type": "healing_response",
                        "response_id": response.response_id,
                        "timestamp": response.timestamp,
                        "action": response.healing_action,
                        "target": response.target,
                        "success": response.success,
                        "recovery_time": response.recovery_time,
                        "audit_trail": response.audit_trail,
                    }
                )

        # Sort by timestamp desc
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return entries[:limit]

    # ---------------------------
    # Healing
    # ---------------------------
    def heal_system(
        self, healing_action: HealingAction, target: str, reason: str
    ) -> HealingResponse:
        """
        Perform self-healing action.

        Args:
            healing_action: Type of healing action
            target: Target to heal
            reason: Reason for healing

        Returns:
            HealingResponse record
        """
        response_id = (
            f"heal_{self.node_id}_{int(_ts()*1000)}_{uuid4().hex[:8]}"
        )
        start_time = _ts()

        audit_log: List[str] = []
        audit_log.append(f"Healing action: {healing_action} on {target}")
        audit_log.append(f"Reason: {reason}")
        audit_log.append(f"Started at: {start_time}")

        success = False
        failure_reason: Optional[str] = None

        try:
            if healing_action == HealingAction.ROLLBACK:
                success = self._rollback_configuration(target)
                audit_log.append(f"Configuration rolled back for: {target}")

            elif healing_action == HealingAction.QUARANTINE:
                if not self.dry_run:
                    with self._lock:
                        self.quarantined_items.add(target)
                success = True
                audit_log.append(f"Quarantined: {target}")

            elif healing_action == HealingAction.CONFIG_ADJUST:
                success = self._adjust_configuration(target, reason)
                audit_log.append(f"Configuration adjusted for: {target}")

            elif healing_action == HealingAction.FAILOVER:
                audit_log.append(f"Failover executed for: {target}")
                success = True  # Simulated

            elif healing_action == HealingAction.RESTORE_BACKUP:
                audit_log.append(f"Backup restored for: {target}")
                success = True  # Simulated

            else:
                failure_reason = "Unknown healing action"
                audit_log.append(f"Unknown healing action: {healing_action}")
                success = False

        except Exception as e:
            failure_reason = str(e)
            audit_log.append(f"Error: {failure_reason}")
            logger.error("Error during healing: %s", e)

        recovery_time = _ts() - start_time
        audit_log.append(f"Completed in: {recovery_time:.3f}s")

        response = HealingResponse(
            response_id=response_id,
            timestamp=_ts(),
            healing_action=healing_action,
            target=target,
            success=success,
            recovery_time=recovery_time,
            audit_trail=audit_log,
            reason=reason,
            failure_reason=failure_reason,
        )

        with self._lock:
            self.healing_responses.append(response)
            if success:
                self.healings_performed += 1

        if self.audit_all:
            logger.info("Healing audit: %s", " | ".join(audit_log))

        self._audit_json(
            {
                "type": "healing_response",
                "node_id": self.node_id,
                "response": response.to_dict(),
            }
        )

        if success:
            logger.info("Healing successful: %s on %s", healing_action, target)
        else:
            logger.warning("Healing failed: %s on %s", healing_action, target)

        return response

    # ---------------------------
    # Config ops
    # ---------------------------
    def _rollback_configuration(self, target: str) -> bool:
        """Rollback configuration to previous state"""
        with self._lock:
            if len(self.config_history) == 0:
                logger.warning(
                    "No configuration history available for %s", target
                )
                return False
            # Get previous config
            previous_config = self.config_history.pop()
            # Simulate applying previous config
            self.current_config.update(previous_config)
        logger.info("Rolled back configuration for %s", target)
        return True

    def _adjust_configuration(self, target: str, reason: str) -> bool:
        """Adjust configuration based on threat/issue"""
        adjustment_map = {
            "rate_limiting": {"max_requests": 100, "window": 60},
            "firewall": {"default_deny": True, "log_all": True},
            "access_control": {"strict_mode": True},
        }
        with self._lock:
            # Save current config
            self.config_history.append(self.current_config.copy())
            if target in adjustment_map:
                self.current_config[target] = adjustment_map[target]
                logger.info(
                    "Adjusted configuration for %s: %s",
                    target,
                    adjustment_map[target],
                )
                return True
        logger.info(
            "No predefined config adjustment for %s (reason: %s)",
            target,
            reason,
        )
        return False

    # ---------------------------
    # Adaptation and controls
    # ---------------------------
    def adapt_thresholds(self, threat_type: ThreatType, feedback: str):
        """
        Adapt detection thresholds based on feedback.

        Args:
            threat_type: Type of threat
            feedback: 'true_positive', 'false_positive', or 'false_negative'
        """
        with self._lock:
            current_detect = self.threat_thresholds[threat_type]
            current_response = self.response_thresholds.get(
                threat_type, self.response_threshold
            )

            if feedback == "false_positive":
                # Increase thresholds to be more conservative
                new_detect = min(1.0, current_detect + 0.05)
                new_response = min(1.0, current_response + 0.05)
                self.threat_thresholds[threat_type] = new_detect
                self.response_thresholds[threat_type] = new_response
                self.false_positives += 1
                logger.info(
                    "Adapted %s thresholds detect: %.2f → %.2f, response: %.2f → %.2f (reduce FP)",
                    threat_type,
                    current_detect,
                    new_detect,
                    current_response,
                    new_response,
                )

            elif feedback == "false_negative":
                # Decrease thresholds to be more sensitive
                new_detect = max(0.1, current_detect - 0.05)
                new_response = max(0.1, current_response - 0.05)
                self.threat_thresholds[threat_type] = new_detect
                self.response_thresholds[threat_type] = new_response
                logger.info(
                    "Adapted %s thresholds detect: %.2f → %.2f, response: %.2f → %.2f (reduce FN)",
                    threat_type,
                    current_detect,
                    new_detect,
                    current_response,
                    new_response,
                )

            # true_positive: hold steady

    def unblock_ip(self, ip: str, reason: str = "Manual unblock"):
        """Manually unblock an IP address"""
        with self._lock:
            if ip in self.blocked_ips:
                self.blocked_ips.remove(ip)
                logger.info("Unblocked IP %s: %s", ip, reason)
                self._audit_json(
                    {
                        "type": "unblock_ip",
                        "node_id": self.node_id,
                        "ip": ip,
                        "reason": reason,
                        "ts": _ts(),
                    }
                )

    def unblock_process(self, process: str, reason: str = "Manual unblock"):
        """Manually unblock a process"""
        with self._lock:
            if process in self.blocked_processes:
                self.blocked_processes.remove(process)
                logger.info("Unblocked process %s: %s", process, reason)
                self._audit_json(
                    {
                        "type": "unblock_process",
                        "node_id": self.node_id,
                        "process": process,
                        "reason": reason,
                        "ts": _ts(),
                    }
                )

    def unquarantine(self, target: str, reason: str = "Manual unquarantine"):
        """Remove an item from quarantine."""
        with self._lock:
            if target in self.quarantined_items:
                self.quarantined_items.remove(target)
                logger.info("Unquarantined %s: %s", target, reason)
                self._audit_json(
                    {
                        "type": "unquarantine",
                        "node_id": self.node_id,
                        "target": target,
                        "reason": reason,
                        "ts": _ts(),
                    }
                )

    def clear_blocklists(self, reason: str = "Manual clear"):
        """Clear all blocklists and quarantine."""
        with self._lock:
            self.blocked_ips.clear()
            self.blocked_processes.clear()
            self.quarantined_items.clear()
        logger.info("Cleared blocklists/quarantine: %s", reason)
        self._audit_json(
            {
                "type": "clear_blocklists",
                "node_id": self.node_id,
                "reason": reason,
                "ts": _ts(),
            }
        )

    # ---------------------------
    # Persistence
    # ---------------------------
    def save_state(self, path: str):
        """Persist essential state to a JSON file."""
        with self._lock:
            state = {
                "node_id": self.node_id,
                "blocked_ips": sorted(self.blocked_ips),
                "blocked_processes": sorted(self.blocked_processes),
                "quarantined_items": sorted(self.quarantined_items),
                "current_config": self.current_config,
                "threat_thresholds": {
                    k.value: v for k, v in self.threat_thresholds.items()
                },
                "response_thresholds": {
                    k.value: v for k, v in self.response_thresholds.items()
                },
                "metrics": {
                    "threats_detected": self.threats_detected,
                    "threats_blocked": self.threats_blocked,
                    "false_positives": self.false_positives,
                    "healings_performed": self.healings_performed,
                },
                "timestamp": _ts(),
            }
        try:
            Path(os.path.dirname(path) or ".").mkdir(
                parents=True, exist_ok=True
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    state, f, indent=2, ensure_ascii=False, sort_keys=True
                )
            logger.info("Saved adaptive defense state to %s", path)
        except Exception as e:
            logger.error("Failed to save state to %s: %s", path, e)

    def load_state(self, path: str):
        """Load essential state from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except FileNotFoundError:
            logger.warning("State file not found: %s", path)
            return
        except Exception as e:
            logger.error("Failed to load state from %s: %s", path, e)
            return

        with self._lock:
            self.blocked_ips = set(state.get("blocked_ips", []))
            self.blocked_processes = set(state.get("blocked_processes", []))
            self.quarantined_items = set(state.get("quarantined_items", []))
            self.current_config = state.get("current_config", {})
            for k, v in state.get("threat_thresholds", {}).items():
                try:
                    self.threat_thresholds[ThreatType(k)] = float(v)
                except Exception:
                    pass
            for k, v in state.get("response_thresholds", {}).items():
                try:
                    self.response_thresholds[ThreatType(k)] = float(v)
                except Exception:
                    pass
            metrics = state.get("metrics", {})
            self.threats_detected = int(
                metrics.get("threats_detected", self.threats_detected)
            )
            self.threats_blocked = int(
                metrics.get("threats_blocked", self.threats_blocked)
            )
            self.false_positives = int(
                metrics.get("false_positives", self.false_positives)
            )
            self.healings_performed = int(
                metrics.get("healings_performed", self.healings_performed)
            )

        logger.info("Loaded adaptive defense state from %s", path)
