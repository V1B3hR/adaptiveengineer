"""
Constitutional Framework for Phase 4: Ethical Self-Regulation

Implements immutable core directives that even the Council of Professors cannot violate.
This is the ultimate layer of self-repair and safety, ensuring the system remains
aligned with its core purpose.

The Three Laws:
1. Law of System Integrity - Cannot compromise long-term viability of host system
2. Law of Operational Continuity - Must prioritize mission-critical services
3. Law of Efficient Evolution - Must seek improvement without violating other laws
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple

logger = logging.getLogger(__name__)


class ConstitutionalLaw(Enum):
    """The three immutable laws of the constitutional framework"""

    SYSTEM_INTEGRITY = 1  # Cannot compromise long-term viability
    OPERATIONAL_CONTINUITY = 2  # Must prioritize mission-critical services
    EFFICIENT_EVOLUTION = 3  # Must seek improvement


class ViolationSeverity(Enum):
    """Severity of constitutional violation"""

    MINOR = "minor"  # Small deviation, correctable
    MODERATE = "moderate"  # Significant deviation, needs attention
    MAJOR = "major"  # Major violation, immediate action required
    CRITICAL = "critical"  # Critical violation, emergency response


@dataclass
class ConstitutionalViolation:
    """Record of a constitutional violation"""

    violation_id: str
    timestamp: float
    law_violated: ConstitutionalLaw
    severity: ViolationSeverity
    description: str
    action_attempted: str
    agent_id: Optional[str] = None
    prevented: bool = True  # Whether violation was prevented
    consequence: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class ActionEvaluation:
    """Evaluation of an action against constitutional framework"""

    action_id: str
    timestamp: float
    action_description: str
    is_compliant: bool
    violations: List[Tuple[ConstitutionalLaw, str]] = field(
        default_factory=list
    )
    warnings: List[str] = field(default_factory=list)
    compliance_score: float = 1.0  # 0.0-1.0, lower means less compliant
    recommendation: str = ""


class ConstitutionalFramework:
    """
    Constitutional Framework - Immutable core directives for the system.

    Provides the ultimate guardrails ensuring the system remains aligned
    with its intended purpose even as it becomes more intelligent and autonomous.
    """

    def __init__(
        self,
        critical_services: Optional[Set[str]] = None,
        system_components: Optional[Set[str]] = None,
    ):
        """
        Initialize Constitutional Framework.

        Args:
            critical_services: Set of mission-critical service names
            system_components: Set of critical system components
        """
        self.critical_services = critical_services or set()
        self.system_components = system_components or set()

        # Violation tracking
        self.violations: Dict[str, ConstitutionalViolation] = {}
        self.violation_count_by_law: Dict[ConstitutionalLaw, int] = {
            law: 0 for law in ConstitutionalLaw
        }

        # Action evaluation history
        self.evaluations: List[ActionEvaluation] = []

        # Statistics
        self.total_actions_evaluated = 0
        self.total_violations_detected = 0
        self.total_violations_prevented = 0
        self.total_compliant_actions = 0

        # Emergency mode
        self.emergency_mode = False
        self.emergency_reason: Optional[str] = None

        logger.info("Constitutional Framework initialized")
        logger.info(f"  Critical services: {len(self.critical_services)}")
        logger.info(f"  System components: {len(self.system_components)}")

    def register_critical_service(self, service_name: str):
        """Register a mission-critical service"""
        self.critical_services.add(service_name)
        logger.info(f"Critical service registered: {service_name}")

    def register_system_component(self, component_name: str):
        """Register a critical system component"""
        self.system_components.add(component_name)
        logger.info(f"System component registered: {component_name}")

    def evaluate_action(
        self,
        action_id: str,
        action_description: str,
        action_context: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> ActionEvaluation:
        """
        Evaluate an action against the constitutional framework.

        Args:
            action_id: Unique action identifier
            action_description: Description of the action
            action_context: Context information about the action
            agent_id: Optional agent ID performing action

        Returns:
            ActionEvaluation with compliance assessment
        """
        evaluation = ActionEvaluation(
            action_id=action_id,
            timestamp=time.time(),
            action_description=action_description,
            is_compliant=True,
        )

        # Check against each law
        violations = []
        warnings = []

        # Law 1: System Integrity
        integrity_check = self._check_system_integrity(action_context)
        if not integrity_check[0]:
            violations.append(
                (ConstitutionalLaw.SYSTEM_INTEGRITY, integrity_check[1])
            )
            evaluation.is_compliant = False
        elif integrity_check[2]:  # Has warning
            warnings.append(integrity_check[2])

        # Law 2: Operational Continuity
        continuity_check = self._check_operational_continuity(action_context)
        if not continuity_check[0]:
            violations.append(
                (ConstitutionalLaw.OPERATIONAL_CONTINUITY, continuity_check[1])
            )
            evaluation.is_compliant = False
        elif continuity_check[2]:  # Has warning
            warnings.append(continuity_check[2])

        # Law 3: Efficient Evolution
        evolution_check = self._check_efficient_evolution(action_context)
        if not evolution_check[0]:
            violations.append(
                (ConstitutionalLaw.EFFICIENT_EVOLUTION, evolution_check[1])
            )
            evaluation.is_compliant = False
        elif evolution_check[2]:  # Has warning
            warnings.append(evolution_check[2])

        # Set violations and warnings
        evaluation.violations = violations
        evaluation.warnings = warnings

        # Calculate compliance score
        evaluation.compliance_score = self._calculate_compliance_score(
            violations, warnings, action_context
        )

        # Generate recommendation
        evaluation.recommendation = self._generate_recommendation(
            violations, warnings, action_context
        )

        # Track evaluation
        self.evaluations.append(evaluation)
        self.total_actions_evaluated += 1

        if evaluation.is_compliant:
            self.total_compliant_actions += 1
        else:
            self.total_violations_detected += 1
            self._record_violation(
                action_id,
                action_description,
                violations,
                agent_id,
                action_context,
            )

        if violations:
            logger.warning(
                f"Action {action_id} violates constitutional framework:"
            )
            for law, reason in violations:
                logger.warning(f"  - {law.name}: {reason}")
        else:
            logger.debug(
                f"Action {action_id} compliant with constitutional framework "
                f"(score={evaluation.compliance_score:.2f})"
            )

        return evaluation

    def _check_system_integrity(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Check Law of System Integrity.

        The collective may not take actions that intentionally compromise
        the long-term viability or structural integrity of the host system
        it is designed to protect.

        Returns:
            (is_compliant, violation_reason, warning_message)
        """
        # Check if action compromises system integrity
        if context.get("compromises_system_integrity"):
            return (False, "Action would compromise system integrity", None)

        # Check if action affects critical system components
        if context.get("affects_critical_components"):
            affected = context.get("affected_components", [])
            critical_affected = [
                c for c in affected if c in self.system_components
            ]

            if critical_affected and not context.get("has_redundancy"):
                return (
                    False,
                    f"Would affect critical components without redundancy: {critical_affected}",
                    None,
                )

            if critical_affected:
                warning = (
                    f"Action affects critical components: {critical_affected}"
                )
                return (True, "", warning)

        # Check for destructive operations without backup
        if context.get("is_destructive") and not context.get("has_backup"):
            return (
                False,
                "Destructive operation without backup violates system integrity",
                None,
            )

        # Check for resource exhaustion
        if context.get("may_exhaust_resources"):
            if context.get("resource_usage", 0.0) > 0.9:
                return (False, "Action may exhaust critical resources", None)
            elif context.get("resource_usage", 0.0) > 0.7:
                warning = "Action has high resource usage"
                return (True, "", warning)

        return (True, "", None)

    def _check_operational_continuity(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Check Law of Operational Continuity.

        The collective must prioritize the continuous operation of mission-critical
        services as defined by its human operators, unless doing so directly
        conflicts with the First Law.

        Returns:
            (is_compliant, violation_reason, warning_message)
        """
        # Check if action affects critical services
        if context.get("affects_services"):
            affected = context.get("affected_services", [])
            critical_affected = [
                s for s in affected if s in self.critical_services
            ]

            if critical_affected:
                # Check if action would cause downtime
                if context.get("causes_downtime"):
                    if not context.get("scheduled_maintenance"):
                        return (
                            False,
                            f"Would cause unscheduled downtime for critical services: {critical_affected}",
                            None,
                        )
                    else:
                        warning = f"Scheduled maintenance affects critical services: {critical_affected}"
                        return (True, "", warning)

                # Check if action would degrade performance
                if context.get("degrades_performance"):
                    severity = context.get("degradation_severity", 0.0)
                    if severity > 0.5:
                        return (
                            False,
                            f"Would significantly degrade critical services: {critical_affected}",
                            None,
                        )
                    elif severity > 0.3:
                        warning = f"May degrade performance of critical services: {critical_affected}"
                        return (True, "", warning)

        # Check for actions that prevent service recovery
        if context.get("prevents_recovery"):
            return (False, "Action would prevent service recovery", None)

        # Check for priority violations
        if context.get("low_priority_action") and context.get(
            "critical_services_degraded"
        ):
            return (
                False,
                "Cannot perform low-priority action while critical services are degraded",
                None,
            )

        return (True, "", None)

    def _check_efficient_evolution(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Check Law of Efficient Evolution.

        The collective must seek to improve its own problem-solving capabilities
        and resource efficiency, as long as this does not conflict with the
        first two Laws.

        Returns:
            (is_compliant, violation_reason, warning_message)
        """
        # Check if action hinders improvement
        if context.get("prevents_learning"):
            return (
                False,
                "Action would prevent system learning and improvement",
                None,
            )

        # Check if action is wasteful
        if context.get("resource_efficiency"):
            efficiency = context.get("resource_efficiency", 1.0)
            if efficiency < 0.3:
                return (
                    False,
                    "Action is extremely resource-inefficient",
                    None,
                )
            elif efficiency < 0.5:
                warning = "Action has low resource efficiency"
                return (True, "", warning)

        # Check if action degrades capability
        if context.get("degrades_capability"):
            degradation = context.get("capability_degradation", 0.0)
            if degradation > 0.5:
                return (
                    False,
                    "Action significantly degrades system capabilities",
                    None,
                )
            elif degradation > 0.3:
                warning = "Action may degrade system capabilities"
                return (True, "", warning)

        # Check if action blocks evolution
        if context.get("blocks_evolution"):
            return (
                False,
                "Action would block evolutionary improvements",
                None,
            )

        return (True, "", None)

    def _calculate_compliance_score(
        self,
        violations: List[Tuple[ConstitutionalLaw, str]],
        warnings: List[str],
        context: Dict[str, Any],
    ) -> float:
        """Calculate compliance score (0.0-1.0)"""
        if violations:
            # Major violations result in very low score
            return 0.0

        # Start with perfect score
        score = 1.0

        # Reduce score for warnings
        score -= len(warnings) * 0.1

        # Adjust based on context
        if context.get("high_risk"):
            score -= 0.2

        if context.get("affects_critical_components") or context.get(
            "affects_services"
        ):
            score -= 0.1

        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))

    def _generate_recommendation(
        self,
        violations: List[Tuple[ConstitutionalLaw, str]],
        warnings: List[str],
        context: Dict[str, Any],
    ) -> str:
        """Generate recommendation based on evaluation"""
        if not violations and not warnings:
            return "ACTION_APPROVED: Action is fully compliant with constitutional framework"

        if violations:
            law_names = ", ".join(law.name for law, _ in violations)
            return f"ACTION_DENIED: Violates {law_names}. Seek alternative approach."

        if warnings:
            return f"ACTION_CAUTION: Proceed with caution. {len(warnings)} warning(s) detected."

        return "ACTION_REVIEW: Manual review recommended"

    def _record_violation(
        self,
        action_id: str,
        action_description: str,
        violations: List[Tuple[ConstitutionalLaw, str]],
        agent_id: Optional[str],
        context: Dict[str, Any],
    ):
        """Record a constitutional violation"""
        for law, reason in violations:
            violation_id = f"violation_{int(time.time()*1000)}"

            # Determine severity
            severity = self._determine_violation_severity(law, reason, context)

            violation = ConstitutionalViolation(
                violation_id=violation_id,
                timestamp=time.time(),
                law_violated=law,
                severity=severity,
                description=reason,
                action_attempted=action_description,
                agent_id=agent_id,
                prevented=True,
                consequence="Action blocked by constitutional framework",
                remediation="Seek alternative approach that complies with constitutional laws",
            )

            self.violations[violation_id] = violation
            self.violation_count_by_law[law] += 1
            self.total_violations_prevented += 1

            logger.warning(
                f"Constitutional violation recorded: {violation_id}"
            )
            logger.warning(f"  Law: {law.name}, Severity: {severity.value}")
            logger.warning(f"  Reason: {reason}")

    def _determine_violation_severity(
        self, law: ConstitutionalLaw, reason: str, context: Dict[str, Any]
    ) -> ViolationSeverity:
        """Determine severity of violation"""
        # Critical if affects critical services/components
        if context.get("affects_critical_components") or context.get(
            "affects_services"
        ):
            critical_affected_services = [
                s
                for s in context.get("affected_services", [])
                if s in self.critical_services
            ]
            critical_affected_components = [
                c
                for c in context.get("affected_components", [])
                if c in self.system_components
            ]

            if critical_affected_services or critical_affected_components:
                return ViolationSeverity.CRITICAL

        # Major if causes downtime or compromises integrity
        if context.get("causes_downtime") or context.get(
            "compromises_system_integrity"
        ):
            return ViolationSeverity.MAJOR

        # Moderate if high impact
        if context.get("high_impact") or context.get("degrades_performance"):
            return ViolationSeverity.MODERATE

        # Otherwise minor
        return ViolationSeverity.MINOR

    def activate_emergency_mode(self, reason: str):
        """
        Activate emergency mode.

        In emergency mode, certain restrictions may be relaxed to preserve
        system viability (but Laws cannot be violated).
        """
        self.emergency_mode = True
        self.emergency_reason = reason
        logger.critical(f"EMERGENCY MODE ACTIVATED: {reason}")

    def deactivate_emergency_mode(self):
        """Deactivate emergency mode"""
        if self.emergency_mode:
            logger.info(
                f"Emergency mode deactivated (was: {self.emergency_reason})"
            )
            self.emergency_mode = False
            self.emergency_reason = None

    def get_violations_by_law(
        self, law: ConstitutionalLaw
    ) -> List[ConstitutionalViolation]:
        """Get all violations of a specific law"""
        return [v for v in self.violations.values() if v.law_violated == law]

    def get_recent_violations(
        self, limit: int = 10
    ) -> List[ConstitutionalViolation]:
        """Get recent violations"""
        sorted_violations = sorted(
            self.violations.values(), key=lambda v: v.timestamp, reverse=True
        )
        return sorted_violations[:limit]

    def get_statistics(self) -> Dict:
        """Get constitutional framework statistics"""
        violation_severity_counts = {
            severity: sum(
                1 for v in self.violations.values() if v.severity == severity
            )
            for severity in ViolationSeverity
        }

        return {
            "total_actions_evaluated": self.total_actions_evaluated,
            "total_violations_detected": self.total_violations_detected,
            "total_violations_prevented": self.total_violations_prevented,
            "total_compliant_actions": self.total_compliant_actions,
            "compliance_rate": self.total_compliant_actions
            / max(1, self.total_actions_evaluated),
            "violations_by_law": {
                law.name: count
                for law, count in self.violation_count_by_law.items()
            },
            "violations_by_severity": {
                severity.value: count
                for severity, count in violation_severity_counts.items()
            },
            "emergency_mode": self.emergency_mode,
            "emergency_reason": self.emergency_reason,
            "critical_services": len(self.critical_services),
            "system_components": len(self.system_components),
        }

    def export_compliance_report(self) -> Dict:
        """Export comprehensive compliance report"""
        return {
            "framework_status": {
                "active": True,
                "emergency_mode": self.emergency_mode,
                "critical_services": list(self.critical_services),
                "system_components": list(self.system_components),
            },
            "statistics": self.get_statistics(),
            "recent_violations": [
                {
                    "violation_id": v.violation_id,
                    "timestamp": v.timestamp,
                    "law_violated": v.law_violated.name,
                    "severity": v.severity.value,
                    "description": v.description,
                    "prevented": v.prevented,
                }
                for v in self.get_recent_violations(20)
            ],
            "recent_evaluations": [
                {
                    "action_id": e.action_id,
                    "timestamp": e.timestamp,
                    "is_compliant": e.is_compliant,
                    "compliance_score": e.compliance_score,
                    "violations_count": len(e.violations),
                    "warnings_count": len(e.warnings),
                }
                for e in self.evaluations[-20:]
            ],
        }
