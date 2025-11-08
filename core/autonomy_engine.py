"""
Autonomy Engine - Self-repair, collaboration, and independent action (AL Principle #4)

This module provides the core autonomy capabilities for adaptive engineer systems:
- Self-repair and service restoration
- Independent decision making
- Ethics and privacy escalation
- Collaborative and competitive behaviors
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger('autonomy_engine')


class ActionType(str, Enum):
    """Types of autonomous actions"""
    SELF_REPAIR = "self_repair"
    SERVICE_RESTORE = "service_restore"
    COLLABORATE = "collaborate"
    COMPETE = "compete"
    COOPERATE = "cooperate"
    ESCALATE = "escalate"


class EscalationReason(str, Enum):
    """Reasons for escalating to human oversight"""
    ETHICS_RISK = "ethics_risk"
    PRIVACY_VIOLATION = "privacy_violation"
    COMPLIANCE_ISSUE = "compliance_issue"
    CRITICAL_DECISION = "critical_decision"
    UNCERTAIN_OUTCOME = "uncertain_outcome"


class RepairStrategy(str, Enum):
    """Self-repair strategies"""
    RESTART = "restart"
    ROLLBACK = "rollback"
    RECONFIGURE = "reconfigure"
    ISOLATE = "isolate"
    REDUNDANCY_SWITCH = "redundancy_switch"


@dataclass
class RepairAction:
    """Self-repair action record"""
    action_id: str
    timestamp: float
    strategy: RepairStrategy
    target: str  # What is being repaired
    status: str  # success, failure, in_progress
    confidence: float  # 0.0 to 1.0
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    escalated: bool = False
    escalation_reason: Optional[EscalationReason] = None


@dataclass
class EscalationRequest:
    """Request for human oversight"""
    request_id: str
    timestamp: float
    reason: EscalationReason
    context: Dict[str, Any]
    proposed_action: str
    risk_level: float  # 0.0 to 1.0
    requires_approval: bool = True


class AutonomyEngine:
    """
    Engine for autonomous agent behavior.
    
    Capabilities:
    - Self-repair and automatic restoration
    - Independent action without human input
    - Ethics and privacy-aware escalation
    - Collaboration, competition, and cooperation
    """
    
    def __init__(
        self,
        node_id: int,
        ethics_threshold: float = 0.7,
        privacy_threshold: float = 0.8,
        auto_repair_enabled: bool = True
    ):
        """
        Initialize autonomy engine.
        
        Args:
            node_id: Unique identifier for this node
            ethics_threshold: Threshold for ethics escalation (0.0-1.0)
            privacy_threshold: Threshold for privacy escalation (0.0-1.0)
            auto_repair_enabled: Whether automatic repair is enabled
        """
        self.node_id = node_id
        self.ethics_threshold = ethics_threshold
        self.privacy_threshold = privacy_threshold
        self.auto_repair_enabled = auto_repair_enabled
        
        # Tracking
        self.repair_history: List[RepairAction] = []
        self.escalation_history: List[EscalationRequest] = []
        self.collaboration_partners: Dict[int, float] = {}  # node_id -> trust
        self.service_states: Dict[str, Dict[str, Any]] = {}
        
        # Metrics
        self.repairs_attempted = 0
        self.repairs_successful = 0
        self.escalations_made = 0
        
        logger.info(f"Autonomy engine initialized for node {node_id} "
                   f"(ethics_thresh={ethics_threshold}, privacy_thresh={privacy_threshold})")
    
    def assess_repair_need(
        self,
        service_name: str,
        health_score: float,
        metrics: Dict[str, Any]
    ) -> Optional[RepairStrategy]:
        """
        Assess if repair is needed and determine strategy.
        
        Args:
            service_name: Name of the service
            health_score: Current health score (0.0-1.0)
            metrics: Current service metrics
        
        Returns:
            Recommended repair strategy or None if no repair needed
        """
        # Store current state
        self.service_states[service_name] = {
            'health': health_score,
            'metrics': metrics.copy(),
            'timestamp': time.time()
        }
        
        # Assess repair need based on health
        if health_score >= 0.8:
            return None  # Healthy, no repair needed
        
        if health_score < 0.2:
            # Critical failure - needs immediate action
            logger.warning(f"Critical failure detected in {service_name}: health={health_score:.2f}")
            return RepairStrategy.RESTART
        
        if health_score < 0.5:
            # Moderate failure
            error_rate = metrics.get('error_rate', 0.0)
            if error_rate > 0.5:
                return RepairStrategy.ROLLBACK
            else:
                return RepairStrategy.RECONFIGURE
        
        if health_score < 0.8:
            # Degraded performance
            return RepairStrategy.ISOLATE
        
        return None
    
    def attempt_self_repair(
        self,
        service_name: str,
        strategy: RepairStrategy,
        repair_callback: Optional[Callable] = None
    ) -> RepairAction:
        """
        Attempt to repair a service autonomously.
        
        Args:
            service_name: Name of service to repair
            strategy: Repair strategy to use
            repair_callback: Optional callback to execute actual repair
        
        Returns:
            RepairAction record with results
        """
        if not self.auto_repair_enabled:
            logger.info(f"Auto-repair disabled, escalating repair of {service_name}")
            return self._escalate_repair(service_name, strategy)
        
        action_id = f"repair_{self.node_id}_{int(time.time()*1000)}"
        
        # Get metrics before repair
        metrics_before = self.service_states.get(service_name, {}).get('metrics', {})
        
        action = RepairAction(
            action_id=action_id,
            timestamp=time.time(),
            strategy=strategy,
            target=service_name,
            status="in_progress",
            confidence=0.7,  # Default confidence
            metrics_before=metrics_before
        )
        
        self.repairs_attempted += 1
        
        try:
            # Execute repair if callback provided
            if repair_callback:
                success = repair_callback(service_name, strategy)
            else:
                # Simulated repair success based on strategy
                success = self._simulate_repair(strategy)
            
            if success:
                action.status = "success"
                action.confidence = 0.9
                self.repairs_successful += 1
                logger.info(f"Successfully repaired {service_name} using {strategy}")
            else:
                action.status = "failure"
                action.confidence = 0.3
                logger.warning(f"Failed to repair {service_name} using {strategy}")
        
        except Exception as e:
            action.status = "failure"
            action.confidence = 0.0
            logger.error(f"Error during repair of {service_name}: {e}")
        
        # Get metrics after repair
        action.metrics_after = self.service_states.get(service_name, {}).get('metrics', {})
        
        self.repair_history.append(action)
        return action
    
    def _simulate_repair(self, strategy: RepairStrategy) -> bool:
        """Simulate repair execution (for testing/demonstration)"""
        # Different strategies have different success rates
        success_rates = {
            RepairStrategy.RESTART: 0.85,
            RepairStrategy.ROLLBACK: 0.90,
            RepairStrategy.RECONFIGURE: 0.75,
            RepairStrategy.ISOLATE: 0.80,
            RepairStrategy.REDUNDANCY_SWITCH: 0.95
        }
        
        import random
        return random.random() < success_rates.get(strategy, 0.7)
    
    def _escalate_repair(
        self,
        service_name: str,
        strategy: RepairStrategy
    ) -> RepairAction:
        """Escalate repair decision to human oversight"""
        escalation = EscalationRequest(
            request_id=f"escalate_{self.node_id}_{int(time.time()*1000)}",
            timestamp=time.time(),
            reason=EscalationReason.CRITICAL_DECISION,
            context={
                'service': service_name,
                'strategy': strategy.value,
                'auto_repair_disabled': not self.auto_repair_enabled
            },
            proposed_action=f"Repair {service_name} using {strategy}",
            risk_level=0.6
        )
        
        self.escalations_made += 1
        self.escalation_history.append(escalation)
        
        action = RepairAction(
            action_id=escalation.request_id,
            timestamp=time.time(),
            strategy=strategy,
            target=service_name,
            status="escalated",
            confidence=0.0,
            escalated=True,
            escalation_reason=EscalationReason.CRITICAL_DECISION
        )
        
        self.repair_history.append(action)
        logger.info(f"Escalated repair of {service_name} to human oversight")
        return action
    
    def check_ethics_risk(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Assess ethical risk of an action.
        
        Args:
            action: Proposed action
            context: Context and parameters
        
        Returns:
            Risk score (0.0-1.0), higher means more risk
        """
        risk_score = 0.0
        
        # Check for sensitive data access
        if context.get('accesses_user_data', False):
            risk_score += 0.3
        
        # Check for system-wide impact
        if context.get('affects_all_users', False):
            risk_score += 0.3
        
        # Check for irreversible actions
        if context.get('irreversible', False):
            risk_score += 0.2
        
        # Check for potential harm
        if context.get('potential_harm', False):
            risk_score += 0.4
        
        return min(1.0, risk_score)
    
    def check_privacy_risk(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Assess privacy risk of an action.
        
        Args:
            action: Proposed action
            context: Context and parameters
        
        Returns:
            Risk score (0.0-1.0), higher means more risk
        """
        risk_score = 0.0
        
        # Check for PII access
        if context.get('accesses_pii', False):
            risk_score += 0.5
        
        # Check for data export
        if context.get('exports_data', False):
            risk_score += 0.3
        
        # Check for logging sensitive data
        if context.get('logs_sensitive_data', False):
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def should_escalate(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[EscalationReason]]:
        """
        Determine if action should be escalated to human oversight.
        
        Args:
            action: Proposed action
            context: Action context
        
        Returns:
            Tuple of (should_escalate, reason)
        """
        # Check ethics risk
        ethics_risk = self.check_ethics_risk(action, context)
        if ethics_risk >= self.ethics_threshold:
            return True, EscalationReason.ETHICS_RISK
        
        # Check privacy risk
        privacy_risk = self.check_privacy_risk(action, context)
        if privacy_risk >= self.privacy_threshold:
            return True, EscalationReason.PRIVACY_VIOLATION
        
        # Check for critical decisions
        if context.get('is_critical', False):
            return True, EscalationReason.CRITICAL_DECISION
        
        # Check for uncertain outcomes
        confidence = context.get('confidence', 1.0)
        if confidence < 0.5:
            return True, EscalationReason.UNCERTAIN_OUTCOME
        
        return False, None
    
    def escalate_action(
        self,
        action: str,
        reason: EscalationReason,
        context: Dict[str, Any],
        risk_level: float
    ) -> EscalationRequest:
        """
        Escalate action to human oversight.
        
        Args:
            action: Action to escalate
            reason: Reason for escalation
            context: Action context
            risk_level: Risk level (0.0-1.0)
        
        Returns:
            EscalationRequest record
        """
        request = EscalationRequest(
            request_id=f"escalate_{self.node_id}_{int(time.time()*1000)}",
            timestamp=time.time(),
            reason=reason,
            context=context,
            proposed_action=action,
            risk_level=risk_level
        )
        
        self.escalations_made += 1
        self.escalation_history.append(request)
        
        logger.warning(f"Action escalated: {action} (reason: {reason}, risk: {risk_level:.2f})")
        return request
    
    def add_collaboration_partner(
        self,
        partner_id: int,
        trust_score: float
    ):
        """
        Add or update a collaboration partner.
        
        Args:
            partner_id: ID of partner node
            trust_score: Trust level (0.0-1.0)
        """
        self.collaboration_partners[partner_id] = trust_score
        logger.debug(f"Node {self.node_id} added partner {partner_id} (trust={trust_score:.2f})")
    
    def get_collaboration_partners(
        self,
        min_trust: float = 0.5
    ) -> List[int]:
        """
        Get list of trusted collaboration partners.
        
        Args:
            min_trust: Minimum trust threshold
        
        Returns:
            List of partner node IDs
        """
        return [
            partner_id 
            for partner_id, trust in self.collaboration_partners.items()
            if trust >= min_trust
        ]
    
    def get_autonomy_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about autonomous operation.
        
        Returns:
            Dictionary of metrics
        """
        success_rate = (
            self.repairs_successful / self.repairs_attempted 
            if self.repairs_attempted > 0 else 0.0
        )
        
        return {
            'node_id': self.node_id,
            'repairs_attempted': self.repairs_attempted,
            'repairs_successful': self.repairs_successful,
            'success_rate': success_rate,
            'escalations_made': self.escalations_made,
            'auto_repair_enabled': self.auto_repair_enabled,
            'collaboration_partners': len(self.collaboration_partners),
            'recent_repairs': len([r for r in self.repair_history if time.time() - r.timestamp < 300]),
            'recent_escalations': len([e for e in self.escalation_history if time.time() - e.timestamp < 300])
        }
