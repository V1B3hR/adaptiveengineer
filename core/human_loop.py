"""
Human-in-the-Loop System: Ethics, Privacy, Compliance, and Oversight

Provides boundaries, overrides, and transparency for sensitive actions.
Ensures all decisions are explainable, auditable, and subject to human approval
when necessary.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import deque

logger = logging.getLogger('human_loop')


class ApprovalStatus(str, Enum):
    """Status of approval request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


class ActionSensitivity(str, Enum):
    """Sensitivity level of actions requiring approval"""
    LOW = "low"  # Can be auto-approved
    MEDIUM = "medium"  # Requires approval for first time
    HIGH = "high"  # Always requires approval
    CRITICAL = "critical"  # Requires multi-party approval


class ComplianceFramework(str, Enum):
    """Compliance frameworks to check against"""
    GDPR = "gdpr"  # EU General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOC2 = "soc2"  # Service Organization Control 2
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    CCPA = "ccpa"  # California Consumer Privacy Act


@dataclass
class ApprovalRequest:
    """Request for human approval of an action"""
    request_id: str
    action: str
    description: str
    sensitivity: ActionSensitivity
    context: Dict[str, Any]
    requester_node_id: int
    risk_level: float  # 0.0 to 1.0
    expiration_time: float  # Unix timestamp
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: Optional[str] = None
    approval_time: Optional[float] = None
    rejection_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if request has expired"""
        return time.time() > self.expiration_time
    
    def approve(self, approver: str):
        """Approve the request"""
        self.status = ApprovalStatus.APPROVED
        self.approver = approver
        self.approval_time = time.time()
    
    def reject(self, approver: str, reason: str):
        """Reject the request"""
        self.status = ApprovalStatus.REJECTED
        self.approver = approver
        self.approval_time = time.time()
        self.rejection_reason = reason


@dataclass
class AuditEntry:
    """Audit log entry for transparency"""
    entry_id: str
    timestamp: float
    node_id: int
    action: str
    decision: str
    reasoning: List[str]
    approval_required: bool
    approval_status: Optional[ApprovalStatus]
    request_id: Optional[str]
    context: Dict[str, Any]
    compliance_checks: Dict[ComplianceFramework, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp,
            'node_id': self.node_id,
            'action': self.action,
            'decision': self.decision,
            'reasoning': self.reasoning,
            'approval_required': self.approval_required,
            'approval_status': self.approval_status.value if self.approval_status else None,
            'request_id': self.request_id,
            'context': self.context,
            'compliance_checks': {
                k.value: v for k, v in self.compliance_checks.items()
            }
        }


@dataclass
class ExplanationTrace:
    """Detailed explanation of a decision"""
    decision_id: str
    action: str
    final_decision: str
    reasoning_steps: List[str]
    factors_considered: Dict[str, float]
    alternatives_evaluated: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)


class HumanLoopSystem:
    """
    Manages human-in-the-loop oversight, approval, and compliance.
    
    Key capabilities:
    - Approval workflows for sensitive actions
    - Override mechanisms for automated decisions
    - Full audit trail for compliance
    - Explainable AI decisions
    - Privacy boundary enforcement
    - Multi-framework compliance checking
    """
    
    def __init__(
        self,
        node_id: int,
        auto_approve_low_sensitivity: bool = True,
        approval_timeout: float = 300.0,  # 5 minutes default
        max_audit_entries: int = 10000,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None
    ):
        """
        Initialize human-in-the-loop system.
        
        Args:
            node_id: Node identifier
            auto_approve_low_sensitivity: Auto-approve low-sensitivity actions
            approval_timeout: Timeout for approval requests (seconds)
            max_audit_entries: Maximum audit entries to keep
            compliance_frameworks: List of compliance frameworks to enforce
        """
        self.node_id = node_id
        self.auto_approve_low = auto_approve_low_sensitivity
        self.approval_timeout = approval_timeout
        self.max_audit_entries = max_audit_entries
        
        # Set default compliance frameworks if none specified
        self.compliance_frameworks = compliance_frameworks or [
            ComplianceFramework.GDPR,
            ComplianceFramework.SOC2
        ]
        
        # Approval tracking
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.completed_requests: deque = deque(maxlen=1000)
        
        # Audit trail
        self.audit_log: deque = deque(maxlen=max_audit_entries)
        
        # Explanation tracking
        self.explanations: Dict[str, ExplanationTrace] = {}
        
        # Override rules (action -> always allow/deny)
        self.override_rules: Dict[str, bool] = {}
        
        # Privacy boundaries
        self.privacy_boundaries = {
            'access_pii': False,  # Personal Identifiable Information
            'access_phi': False,  # Protected Health Information
            'export_data': False,
            'modify_user_data': False,
            'cross_border_transfer': False
        }
        
        # Metrics
        self.approval_requests_made = 0
        self.approvals_granted = 0
        self.approvals_rejected = 0
        self.approvals_expired = 0
        self.override_blocks = 0
        
        logger.info(
            f"Human-in-the-loop system initialized for node {node_id} "
            f"(compliance={[f.value for f in self.compliance_frameworks]})"
        )
    
    def request_approval(
        self,
        action: str,
        description: str,
        sensitivity: ActionSensitivity,
        context: Dict[str, Any],
        risk_level: float = 0.5
    ) -> ApprovalRequest:
        """
        Request human approval for an action.
        
        Args:
            action: Action identifier
            description: Human-readable description
            sensitivity: Sensitivity level
            context: Additional context
            risk_level: Risk level (0.0-1.0)
        
        Returns:
            ApprovalRequest object
        """
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        request = ApprovalRequest(
            request_id=request_id,
            action=action,
            description=description,
            sensitivity=sensitivity,
            context=context,
            requester_node_id=self.node_id,
            risk_level=max(0.0, min(1.0, risk_level)),
            expiration_time=time.time() + self.approval_timeout
        )
        
        # Check for override rules
        if action in self.override_rules:
            if self.override_rules[action]:
                request.status = ApprovalStatus.AUTO_APPROVED
                request.approver = "override_rule"
                logger.info(f"Action {action} auto-approved by override rule")
            else:
                request.status = ApprovalStatus.REJECTED
                request.rejection_reason = "Blocked by override rule"
                self.override_blocks += 1
                logger.warning(f"Action {action} blocked by override rule")
        
        # Auto-approve low sensitivity if configured
        elif (self.auto_approve_low and 
              sensitivity == ActionSensitivity.LOW and
              risk_level < 0.3):
            request.status = ApprovalStatus.AUTO_APPROVED
            request.approver = "auto_approval"
            logger.debug(f"Action {action} auto-approved (low sensitivity)")
        
        else:
            # Add to pending requests
            self.pending_requests[request_id] = request
            logger.info(
                f"Approval requested: {action} "
                f"(sensitivity={sensitivity.value}, risk={risk_level:.2f})"
            )
        
        self.approval_requests_made += 1
        
        # Audit the request
        self._audit_approval_request(request)
        
        return request
    
    def approve_request(
        self,
        request_id: str,
        approver: str
    ) -> bool:
        """
        Approve a pending request.
        
        Args:
            request_id: Request to approve
            approver: Identifier of approver
        
        Returns:
            True if approved successfully
        """
        if request_id not in self.pending_requests:
            logger.warning(f"Cannot approve unknown request: {request_id}")
            return False
        
        request = self.pending_requests[request_id]
        
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            self.approvals_expired += 1
            logger.warning(f"Request {request_id} expired before approval")
            return False
        
        request.approve(approver)
        self.approvals_granted += 1
        
        # Move to completed
        self.completed_requests.append(request)
        del self.pending_requests[request_id]
        
        logger.info(f"Request {request_id} approved by {approver}")
        
        # Audit the approval
        self._audit_approval_decision(request)
        
        return True
    
    def reject_request(
        self,
        request_id: str,
        approver: str,
        reason: str
    ) -> bool:
        """
        Reject a pending request.
        
        Args:
            request_id: Request to reject
            approver: Identifier of approver
            reason: Rejection reason
        
        Returns:
            True if rejected successfully
        """
        if request_id not in self.pending_requests:
            logger.warning(f"Cannot reject unknown request: {request_id}")
            return False
        
        request = self.pending_requests[request_id]
        request.reject(approver, reason)
        self.approvals_rejected += 1
        
        # Move to completed
        self.completed_requests.append(request)
        del self.pending_requests[request_id]
        
        logger.info(f"Request {request_id} rejected by {approver}: {reason}")
        
        # Audit the rejection
        self._audit_approval_decision(request)
        
        return True
    
    def check_approval_status(self, request_id: str) -> ApprovalStatus:
        """
        Check status of an approval request.
        
        Args:
            request_id: Request to check
        
        Returns:
            Current approval status
        """
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                self.approvals_expired += 1
                # Move to completed
                self.completed_requests.append(request)
                del self.pending_requests[request_id]
            return request.status
        
        # Check completed requests
        for request in self.completed_requests:
            if request.request_id == request_id:
                return request.status
        
        return ApprovalStatus.PENDING
    
    def set_override_rule(self, action: str, allow: bool):
        """
        Set override rule for an action.
        
        Args:
            action: Action identifier
            allow: True to always allow, False to always deny
        """
        self.override_rules[action] = allow
        logger.info(f"Override rule set: {action} -> {'allow' if allow else 'deny'}")
    
    def remove_override_rule(self, action: str):
        """Remove override rule for an action"""
        if action in self.override_rules:
            del self.override_rules[action]
            logger.info(f"Override rule removed: {action}")
    
    def set_privacy_boundary(self, boundary: str, allowed: bool):
        """
        Set privacy boundary.
        
        Args:
            boundary: Boundary identifier
            allowed: Whether to allow crossing this boundary
        """
        self.privacy_boundaries[boundary] = allowed
        logger.info(f"Privacy boundary set: {boundary} -> {allowed}")
    
    def check_privacy_boundary(self, boundary: str) -> bool:
        """Check if privacy boundary allows action"""
        return self.privacy_boundaries.get(boundary, False)
    
    def check_compliance(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> Dict[ComplianceFramework, bool]:
        """
        Check action compliance with configured frameworks.
        
        Args:
            action: Action to check
            context: Action context
        
        Returns:
            Dict mapping framework to compliance status
        """
        results = {}
        
        for framework in self.compliance_frameworks:
            compliant = self._check_framework_compliance(framework, action, context)
            results[framework] = compliant
        
        return results
    
    def _check_framework_compliance(
        self,
        framework: ComplianceFramework,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check compliance with specific framework"""
        if framework == ComplianceFramework.GDPR:
            # GDPR: Check data protection requirements
            if context.get('accesses_pii'):
                # Requires consent and purpose limitation
                return (
                    context.get('has_consent', False) and
                    context.get('has_purpose', False)
                )
            return True
        
        elif framework == ComplianceFramework.HIPAA:
            # HIPAA: Check health information protection
            if context.get('accesses_phi'):
                return (
                    context.get('has_authorization', False) and
                    context.get('is_encrypted', False)
                )
            return True
        
        elif framework == ComplianceFramework.SOC2:
            # SOC2: Check security, availability, confidentiality
            return (
                context.get('is_audited', True) and
                context.get('is_secure', True)
            )
        
        elif framework == ComplianceFramework.PCI_DSS:
            # PCI DSS: Check payment card data protection
            if context.get('accesses_payment_data'):
                return (
                    context.get('is_encrypted', False) and
                    context.get('is_logged', False)
                )
            return True
        
        elif framework == ComplianceFramework.CCPA:
            # CCPA: Check California privacy requirements
            if context.get('accesses_personal_data'):
                return context.get('has_opt_out', True)
            return True
        
        return True  # Default to compliant if unknown
    
    def explain_decision(
        self,
        decision_id: str,
        action: str,
        final_decision: str,
        reasoning_steps: List[str],
        factors_considered: Dict[str, float],
        alternatives_evaluated: List[str],
        confidence: float
    ) -> ExplanationTrace:
        """
        Create detailed explanation for a decision.
        
        Args:
            decision_id: Unique decision identifier
            action: Action being decided
            final_decision: Final decision made
            reasoning_steps: Step-by-step reasoning
            factors_considered: Factors and their weights
            alternatives_evaluated: Alternative options considered
            confidence: Confidence in decision (0.0-1.0)
        
        Returns:
            ExplanationTrace object
        """
        explanation = ExplanationTrace(
            decision_id=decision_id,
            action=action,
            final_decision=final_decision,
            reasoning_steps=reasoning_steps,
            factors_considered=factors_considered,
            alternatives_evaluated=alternatives_evaluated,
            confidence=max(0.0, min(1.0, confidence))
        )
        
        self.explanations[decision_id] = explanation
        
        logger.debug(f"Decision explanation recorded: {decision_id}")
        
        return explanation
    
    def get_explanation(self, decision_id: str) -> Optional[ExplanationTrace]:
        """Get explanation for a decision"""
        return self.explanations.get(decision_id)
    
    def _audit_approval_request(self, request: ApprovalRequest):
        """Add approval request to audit log"""
        entry = AuditEntry(
            entry_id=f"audit_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            node_id=self.node_id,
            action=request.action,
            decision="approval_requested",
            reasoning=[
                f"Sensitivity: {request.sensitivity.value}",
                f"Risk level: {request.risk_level:.2f}"
            ],
            approval_required=True,
            approval_status=request.status,
            request_id=request.request_id,
            context=request.context,
            compliance_checks=self.check_compliance(request.action, request.context)
        )
        
        self.audit_log.append(entry)
    
    def _audit_approval_decision(self, request: ApprovalRequest):
        """Add approval decision to audit log"""
        entry = AuditEntry(
            entry_id=f"audit_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            node_id=self.node_id,
            action=request.action,
            decision=request.status.value,
            reasoning=[
                f"Approver: {request.approver}",
                f"Decision time: {request.approval_time}"
            ] if request.approver else ["System decision"],
            approval_required=True,
            approval_status=request.status,
            request_id=request.request_id,
            context=request.context,
            compliance_checks=self.check_compliance(request.action, request.context)
        )
        
        self.audit_log.append(entry)
    
    def get_audit_log(
        self,
        limit: Optional[int] = None,
        since: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            limit: Maximum entries to return
            since: Only entries after this timestamp
        
        Returns:
            List of audit entries as dictionaries
        """
        entries = list(self.audit_log)
        
        # Filter by timestamp if specified
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        # Reverse to get most recent first
        entries.reverse()
        
        # Apply limit
        if limit:
            entries = entries[:limit]
        
        return [e.to_dict() for e in entries]
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests"""
        # Clean up expired requests
        expired = []
        for req_id, request in self.pending_requests.items():
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                expired.append(req_id)
        
        for req_id in expired:
            self.completed_requests.append(self.pending_requests[req_id])
            del self.pending_requests[req_id]
            self.approvals_expired += 1
        
        return list(self.pending_requests.values())
    
    def get_human_loop_metrics(self) -> Dict[str, Any]:
        """Get metrics about human-in-the-loop operations"""
        approval_rate = (
            self.approvals_granted / self.approval_requests_made
            if self.approval_requests_made > 0 else 0.0
        )
        
        rejection_rate = (
            self.approvals_rejected / self.approval_requests_made
            if self.approval_requests_made > 0 else 0.0
        )
        
        return {
            'node_id': self.node_id,
            'approval_requests_made': self.approval_requests_made,
            'approvals_granted': self.approvals_granted,
            'approvals_rejected': self.approvals_rejected,
            'approvals_expired': self.approvals_expired,
            'approval_rate': approval_rate,
            'rejection_rate': rejection_rate,
            'pending_approvals': len(self.pending_requests),
            'override_blocks': self.override_blocks,
            'override_rules': len(self.override_rules),
            'audit_entries': len(self.audit_log),
            'explanations_recorded': len(self.explanations),
            'compliance_frameworks': [f.value for f in self.compliance_frameworks]
        }
