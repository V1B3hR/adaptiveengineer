"""
Adaptive Defense System - Self-healing cyber defenses with automated threat response

This module provides automated, adaptive cybersecurity defenses:
- Automated threat response (block IPs, processes, restart services)
- Self-healing actions (rollback, quarantine, configuration adjustment)
- Full auditability for compliance
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger('adaptive_defense')


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

def make_heal_id(node_id: int) -> str:
    return f"heal_{node_id}_{uuid4()}"

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
        audit_all: bool = True
    ):
        """
        Initialize adaptive defense system.
        
        Args:
            node_id: Unique identifier for this node
            auto_response_enabled: Whether automatic response is enabled
            response_threshold: Confidence threshold for auto-response (0.0-1.0)
            audit_all: Whether to audit all actions
        """
        self.node_id = node_id
        self.auto_response_enabled = auto_response_enabled
        self.response_threshold = response_threshold
        self.audit_all = audit_all
        
        # Tracking
        self.threat_events: deque = deque(maxlen=1000)
        self.defense_responses: deque = deque(maxlen=1000)
        self.healing_responses: deque = deque(maxlen=500)
        
        # Block lists
        self.blocked_ips: Set[str] = set()
        self.blocked_processes: Set[str] = set()
        self.quarantined_items: Set[str] = set()
        
        # Configuration states (for rollback)
        self.config_history: deque = deque(maxlen=50)
        self.current_config: Dict[str, Any] = {}
        
        # Metrics
        self.threats_detected = 0
        self.threats_blocked = 0
        self.false_positives = 0
        self.healings_performed = 0
        
        # Adaptive thresholds (learned over time)
        self.threat_thresholds: Dict[ThreatType, float] = {
            threat_type: 0.5 for threat_type in ThreatType
        }
        
        logger.info(f"Adaptive defense system initialized for node {node_id} "
                   f"(auto_response={auto_response_enabled}, threshold={response_threshold})")
    
    def detect_threat(
        self,
        threat_type: ThreatType,
        source: str,
        severity: float,
        confidence: float,
        indicators: Optional[Dict[str, Any]] = None
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
        event_id = f"threat_{self.node_id}_{int(time.time()*1000)}"
        
        event = ThreatEvent(
            event_id=event_id,
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            source=source,
            confidence=confidence,
            indicators=indicators or {}
        )
        
        self.threats_detected += 1
        self.threat_events.append(event)
        
        logger.warning(f"Threat detected: {threat_type} from {source} "
                      f"(severity={severity:.2f}, confidence={confidence:.2f})")
        
        # Automatic response if enabled and confidence is high enough
        if self.auto_response_enabled and confidence >= self.response_threshold:
            self.respond_to_threat(event)
        
        return event
    
    def respond_to_threat(
        self,
        threat_event: ThreatEvent
    ) -> Optional[DefenseResponse]:
        """
        Automatically respond to a detected threat.
        
        Args:
            threat_event: The threat to respond to
        
        Returns:
            DefenseResponse record or None if no action taken
        """
        if threat_event.responded:
            logger.debug(f"Threat {threat_event.event_id} already responded to")
            return None
        
        # Determine appropriate defense action
        action = self._select_defense_action(threat_event)
        if not action:
            return None
        
        response_id = f"response_{self.node_id}_{int(time.time()*1000)}"
        
        # Record metrics before action
        metrics_before = self._get_current_metrics()
        
        # Execute defense action
        success, audit_log = self._execute_defense_action(
            action,
            threat_event.source,
            threat_event
        )
        
        # Record metrics after action
        metrics_after = self._get_current_metrics()
        
        response = DefenseResponse(
            response_id=response_id,
            timestamp=time.time(),
            threat_event_id=threat_event.event_id,
            action=action,
            target=threat_event.source,
            success=success,
            confidence=threat_event.confidence,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            audit_trail=audit_log
        )
        
        self.defense_responses.append(response)
        threat_event.responded = True
        
        if success:
            self.threats_blocked += 1
            logger.info(f"Successfully responded to threat with {action}: {threat_event.source}")
        else:
            logger.warning(f"Failed to respond to threat with {action}: {threat_event.source}")
        
        return response
    
    def _select_defense_action(
        self,
        threat_event: ThreatEvent
    ) -> Optional[DefenseAction]:
        """Select appropriate defense action for threat type"""
        action_map = {
            ThreatType.MALWARE: DefenseAction.QUARANTINE,
            ThreatType.DDOS: DefenseAction.BLOCK_IP,
            ThreatType.INTRUSION: DefenseAction.BLOCK_IP,
            ThreatType.DATA_EXFILTRATION: DefenseAction.BLOCK_PROCESS,
            ThreatType.PRIVILEGE_ESCALATION: DefenseAction.ISOLATE_SEGMENT,
            ThreatType.BRUTE_FORCE: DefenseAction.RATE_LIMIT,
            ThreatType.ANOMALY: DefenseAction.QUARANTINE
        }
        
        return action_map.get(threat_event.threat_type)
    
    def _execute_defense_action(
        self,
        action: DefenseAction,
        target: str,
        threat_event: ThreatEvent
    ) -> tuple[bool, List[str]]:
        """
        Execute a defense action.
        
        Returns:
            Tuple of (success, audit_trail)
        """
        audit_log = []
        audit_log.append(f"Action: {action} on target: {target}")
        audit_log.append(f"Threat: {threat_event.threat_type}, Severity: {threat_event.severity:.2f}")
        audit_log.append(f"Timestamp: {time.time()}")
        
        try:
            if action == DefenseAction.BLOCK_IP:
                self.blocked_ips.add(target)
                audit_log.append(f"Blocked IP: {target}")
                success = True
            
            elif action == DefenseAction.BLOCK_PROCESS:
                self.blocked_processes.add(target)
                audit_log.append(f"Blocked process: {target}")
                success = True
            
            elif action == DefenseAction.QUARANTINE:
                self.quarantined_items.add(target)
                audit_log.append(f"Quarantined: {target}")
                success = True
            
            elif action == DefenseAction.RESTART_SERVICE:
                audit_log.append(f"Restarted service: {target}")
                success = True  # Simulated
            
            elif action == DefenseAction.ROLLBACK_CONFIG:
                success = self._rollback_configuration(target)
                audit_log.append(f"Rolled back configuration for: {target}")
            
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
                logger.info(f"Audit: {' | '.join(audit_log)}")
            
            return success, audit_log
        
        except Exception as e:
            audit_log.append(f"Error: {str(e)}")
            logger.error(f"Error executing defense action: {e}")
            return False, audit_log
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'blocked_ips_count': len(self.blocked_ips),
            'blocked_processes_count': len(self.blocked_processes),
            'quarantined_count': len(self.quarantined_items),
            'threats_detected': self.threats_detected,
            'threats_blocked': self.threats_blocked,
            'timestamp': time.time()
        }
    
    def heal_system(
        self,
        healing_action: HealingAction,
        target: str,
        reason: str
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
        response_id = f"heal_{self.node_id}_{int(time.time()*1000)}"
        start_time = time.time()
        
        audit_log = []
        audit_log.append(f"Healing action: {healing_action} on {target}")
        audit_log.append(f"Reason: {reason}")
        audit_log.append(f"Started at: {start_time}")
        
        success = False
        
        try:
            if healing_action == HealingAction.ROLLBACK:
                success = self._rollback_configuration(target)
                audit_log.append(f"Configuration rolled back for: {target}")
            
            elif healing_action == HealingAction.QUARANTINE:
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
        
        except Exception as e:
            audit_log.append(f"Error: {str(e)}")
            logger.error(f"Error during healing: {e}")
        
        recovery_time = time.time() - start_time
        audit_log.append(f"Completed in: {recovery_time:.3f}s")
        
        response = HealingResponse(
            response_id=response_id,
            timestamp=time.time(),
            healing_action=healing_action,
            target=target,
            success=success,
            recovery_time=recovery_time,
            audit_trail=audit_log
        )
        
        self.healing_responses.append(response)
        
        if success:
            self.healings_performed += 1
            logger.info(f"Healing successful: {healing_action} on {target}")
        else:
            logger.warning(f"Healing failed: {healing_action} on {target}")
        
        if self.audit_all:
            logger.info(f"Healing audit: {' | '.join(audit_log)}")
        
        return response
    
    def _rollback_configuration(self, target: str) -> bool:
        """Rollback configuration to previous state"""
        if len(self.config_history) == 0:
            logger.warning(f"No configuration history available for {target}")
            return False
        
        # Get previous config
        previous_config = self.config_history.pop()
        
        # Simulate applying previous config
        logger.info(f"Rolling back configuration for {target}")
        return True
    
    def _adjust_configuration(self, target: str, reason: str) -> bool:
        """Adjust configuration based on threat/issue"""
        # Save current config
        self.config_history.append(self.current_config.copy())
        
        # Simulate configuration adjustment
        adjustment_map = {
            'rate_limiting': {'max_requests': 100, 'window': 60},
            'firewall': {'default_deny': True, 'log_all': True},
            'access_control': {'strict_mode': True}
        }
        
        if target in adjustment_map:
            self.current_config[target] = adjustment_map[target]
            logger.info(f"Adjusted configuration for {target}: {adjustment_map[target]}")
            return True
        
        return False
    
    def adapt_thresholds(
        self,
        threat_type: ThreatType,
        feedback: str
    ):
        """
        Adapt detection thresholds based on feedback.
        
        Args:
            threat_type: Type of threat
            feedback: 'true_positive', 'false_positive', or 'false_negative'
        """
        current = self.threat_thresholds[threat_type]
        
        if feedback == 'false_positive':
            # Increase threshold to be more conservative
            self.threat_thresholds[threat_type] = min(1.0, current + 0.05)
            self.false_positives += 1
            logger.info(f"Adapted {threat_type} threshold: {current:.2f} → "
                       f"{self.threat_thresholds[threat_type]:.2f} (reduce FP)")
        
        elif feedback == 'false_negative':
            # Decrease threshold to be more sensitive
            self.threat_thresholds[threat_type] = max(0.1, current - 0.05)
            logger.info(f"Adapted {threat_type} threshold: {current:.2f} → "
                       f"{self.threat_thresholds[threat_type]:.2f} (reduce FN)")
    
    def unblock_ip(self, ip: str, reason: str = "Manual unblock"):
        """Manually unblock an IP address"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"Unblocked IP {ip}: {reason}")
    
    def unblock_process(self, process: str, reason: str = "Manual unblock"):
        """Manually unblock a process"""
        if process in self.blocked_processes:
            self.blocked_processes.remove(process)
            logger.info(f"Unblocked process {process}: {reason}")
    
    def get_defense_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about defense operations.
        
        Returns:
            Dictionary of metrics
        """
        recent_threats = len([
            t for t in self.threat_events 
            if time.time() - t.timestamp < 300
        ])
        
        recent_blocks = len([
            r for r in self.defense_responses 
            if time.time() - r.timestamp < 300 and r.success
        ])
        
        block_rate = (
            self.threats_blocked / self.threats_detected 
            if self.threats_detected > 0 else 0.0
        )
        
        return {
            'node_id': self.node_id,
            'threats_detected': self.threats_detected,
            'threats_blocked': self.threats_blocked,
            'block_rate': block_rate,
            'false_positives': self.false_positives,
            'healings_performed': self.healings_performed,
            'blocked_ips': len(self.blocked_ips),
            'blocked_processes': len(self.blocked_processes),
            'quarantined_items': len(self.quarantined_items),
            'recent_threats_5min': recent_threats,
            'recent_blocks_5min': recent_blocks,
            'auto_response_enabled': self.auto_response_enabled
        }
    
    def get_audit_log(
        self,
        limit: int = 50,
        action_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log of defense actions.
        
        Args:
            limit: Maximum number of entries to return
            action_type: Optional filter by action type
        
        Returns:
            List of audit entries
        """
        entries = []
        
        # Add defense responses
        for response in list(self.defense_responses)[-limit:]:
            if action_type and response.action != action_type:
                continue
            
            entries.append({
                'type': 'defense_response',
                'response_id': response.response_id,
                'timestamp': response.timestamp,
                'action': response.action,
                'target': response.target,
                'success': response.success,
                'audit_trail': response.audit_trail
            })
        
        # Add healing responses
        for response in list(self.healing_responses)[-limit:]:
            if action_type and response.healing_action != action_type:
                continue
            
            entries.append({
                'type': 'healing_response',
                'response_id': response.response_id,
                'timestamp': response.timestamp,
                'action': response.healing_action,
                'target': response.target,
                'success': response.success,
                'recovery_time': response.recovery_time,
                'audit_trail': response.audit_trail
            })
        
        # Sort by timestamp
        entries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return entries[:limit]
