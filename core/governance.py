"""
Council of Professors: The Governance Layer for Phase 4

Implements three specialized governor agents that perform critical oversight:
1. Systemic Pathologist - Analyzes failures and publishes lessons learned
2. Strategic Immunologist - Monitors for chronic vulnerabilities and injects antibodies
3. Evolutionary Biologist - Curates genetic health and manages evolution

These professors are the most powerful agents in the collective, instantiated by
the system (not evolved) to perform oversight functions.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class ProfessorType(Enum):
    """Types of professor agents"""

    PATHOLOGIST = "pathologist"  # Analyzes failures
    IMMUNOLOGIST = "immunologist"  # Monitors vulnerabilities
    BIOLOGIST = "biologist"  # Curates genetic health


class AntibodyType(Enum):
    """Types of antibodies that can be injected"""

    SECURITY_MODEL = "security_model"  # New pre-trained security model
    GENETIC_SEQUENCE = "genetic_sequence"  # Mandatory genetic sequence
    OPERATIONAL_POLICY = "operational_policy"  # Globally enforced policy
    FIREWALL_RULE = "firewall_rule"  # New firewall rule
    THRESHOLD_ADJUSTMENT = (
        "threshold_adjustment"  # Adjust detection thresholds
    )


@dataclass
class FailureAnalysis:
    """Analysis of a failure by the Systemic Pathologist"""

    failure_id: str
    timestamp: float
    agent_id: Optional[str]
    squad_id: Optional[str]
    failure_type: str
    root_cause: str
    contributing_factors: List[str] = field(default_factory=list)
    impact_severity: float = 0.0  # 0.0-1.0
    recommendation: str = ""
    lesson_learned: str = ""
    affected_systems: List[str] = field(default_factory=list)


@dataclass
class LessonLearned:
    """A lesson learned published to the collective"""

    lesson_id: str
    timestamp: float
    title: str
    description: str
    failure_pattern: str
    what_not_to_do: str
    recommended_approach: str
    severity: str  # "low", "medium", "high", "critical"
    related_failures: List[str] = field(default_factory=list)
    published_by: str = "systemic_pathologist"


@dataclass
class Vulnerability:
    """A chronic vulnerability identified by the Strategic Immunologist"""

    vuln_id: str
    timestamp: float
    vulnerability_type: str
    description: str
    frequency: int  # How many times observed
    severity: float  # 0.0-1.0
    affected_components: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    remediation_attempts: int = 0


@dataclass
class Antibody:
    """An antibody injected by the Strategic Immunologist"""

    antibody_id: str
    timestamp: float
    antibody_type: AntibodyType
    target_vulnerability: str
    description: str
    implementation: Dict[str, Any]  # Specific implementation details
    deployed: bool = False
    effectiveness: Optional[float] = None  # Measured after deployment
    deployment_time: Optional[float] = None


@dataclass
class GeneticHealth:
    """Genetic health metrics tracked by the Evolutionary Biologist"""

    timestamp: float
    population_size: int
    genetic_diversity: float  # 0.0-1.0, higher is more diverse
    avg_fitness: float
    stagnation_level: float  # 0.0-1.0, higher means more stagnant
    dangerous_mutations: int
    beneficial_mutations: int
    mutation_rate: float
    recommendation: str = ""


class SystemicPathologist:
    """
    Professor that analyzes failures and publishes lessons learned.

    Retrieves logs and strategic decisions of "bad agents" or failed squads
    to perform root cause analysis, then publishes "Lessons Learned" bulletins
    to train the entire collective on what NOT to do.
    """

    def __init__(self, knowledge_graph: Optional[object] = None):
        """
        Initialize Systemic Pathologist.

        Args:
            knowledge_graph: Optional KnowledgeGraph for integration
        """
        self.professor_id = "prof_pathologist"
        self.knowledge_graph = knowledge_graph

        # Failure tracking
        self.failures_analyzed: Dict[str, FailureAnalysis] = {}
        self.lessons_learned: Dict[str, LessonLearned] = {}

        # Pattern detection
        self.failure_patterns: Dict[str, List[str]] = defaultdict(list)
        self.recurring_failures: Set[str] = set()

        # Statistics
        self.total_failures_analyzed = 0
        self.total_lessons_published = 0
        self.total_root_causes_identified = 0

        logger.info("Systemic Pathologist initialized")

    def analyze_failure(
        self,
        failure_id: str,
        agent_id: Optional[str] = None,
        squad_id: Optional[str] = None,
        failure_type: str = "unknown",
        logs: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
    ) -> FailureAnalysis:
        """
        Perform root cause analysis on a failure.

        Args:
            failure_id: Unique failure identifier
            agent_id: ID of agent that failed (if applicable)
            squad_id: ID of squad that failed (if applicable)
            failure_type: Type of failure
            logs: Log entries related to failure
            context: Additional context about failure

        Returns:
            FailureAnalysis with root cause and recommendations
        """
        logger.info(f"Analyzing failure {failure_id} (type={failure_type})")

        # Perform analysis
        root_cause = self._identify_root_cause(failure_type, logs, context)
        contributing_factors = self._identify_contributing_factors(
            logs, context
        )
        impact_severity = self._assess_impact(failure_type, context)
        recommendation = self._generate_recommendation(
            root_cause, contributing_factors
        )
        lesson = self._extract_lesson(root_cause, recommendation)

        analysis = FailureAnalysis(
            failure_id=failure_id,
            timestamp=time.time(),
            agent_id=agent_id,
            squad_id=squad_id,
            failure_type=failure_type,
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            impact_severity=impact_severity,
            recommendation=recommendation,
            lesson_learned=lesson,
            affected_systems=self._identify_affected_systems(context),
        )

        self.failures_analyzed[failure_id] = analysis
        self.total_failures_analyzed += 1
        self.total_root_causes_identified += 1

        # Track patterns
        pattern_key = f"{failure_type}:{root_cause}"
        self.failure_patterns[pattern_key].append(failure_id)

        # Detect recurring failures
        if len(self.failure_patterns[pattern_key]) >= 3:
            self.recurring_failures.add(pattern_key)
            logger.warning(
                f"Recurring failure pattern detected: {pattern_key}"
            )

        # Integrate with knowledge graph
        if self.knowledge_graph:
            self._update_knowledge_graph(analysis)

        logger.info(f"Failure analysis complete: {root_cause}")

        return analysis

    def publish_lesson_learned(
        self, failure_id: str, title: Optional[str] = None
    ) -> Optional[LessonLearned]:
        """
        Publish a lesson learned bulletin to the collective.

        Args:
            failure_id: ID of failure to publish lesson for
            title: Optional custom title

        Returns:
            LessonLearned bulletin
        """
        analysis = self.failures_analyzed.get(failure_id)
        if not analysis:
            logger.warning(
                f"Cannot publish lesson for unknown failure {failure_id}"
            )
            return None

        lesson_id = f"lesson_{int(time.time()*1000)}"

        # Determine severity
        if analysis.impact_severity >= 0.8:
            severity = "critical"
        elif analysis.impact_severity >= 0.6:
            severity = "high"
        elif analysis.impact_severity >= 0.3:
            severity = "medium"
        else:
            severity = "low"

        # Find related failures
        pattern_key = f"{analysis.failure_type}:{analysis.root_cause}"
        related = self.failure_patterns.get(pattern_key, [])

        lesson = LessonLearned(
            lesson_id=lesson_id,
            timestamp=time.time(),
            title=title
            or f"Lesson: {analysis.failure_type} due to {analysis.root_cause}",
            description=f"Analysis of failure {failure_id}",
            failure_pattern=pattern_key,
            what_not_to_do=self._formulate_what_not_to_do(analysis),
            recommended_approach=analysis.recommendation,
            severity=severity,
            related_failures=related,
        )

        self.lessons_learned[lesson_id] = lesson
        self.total_lessons_published += 1

        logger.info(
            f"Lesson learned published: {lesson.title} (severity={severity})"
        )

        return lesson

    def _identify_root_cause(
        self,
        failure_type: str,
        logs: Optional[List[Dict]],
        context: Optional[Dict],
    ) -> str:
        """Identify root cause of failure"""
        # Simplified root cause analysis
        # In production, this would use more sophisticated techniques

        if failure_type == "task_failure":
            if context and context.get("insufficient_resources"):
                return "insufficient_resources"
            elif context and context.get("capability_mismatch"):
                return "capability_mismatch"
            else:
                return "unknown_task_failure"

        elif failure_type == "security_breach":
            if context and context.get("weak_authentication"):
                return "weak_authentication"
            elif context and context.get("unpatched_vulnerability"):
                return "unpatched_vulnerability"
            else:
                return "unknown_security_failure"

        elif failure_type == "system_degradation":
            if context and context.get("resource_exhaustion"):
                return "resource_exhaustion"
            elif context and context.get("cascading_failure"):
                return "cascading_failure"
            else:
                return "unknown_degradation"

        return f"unknown_root_cause_{failure_type}"

    def _identify_contributing_factors(
        self, logs: Optional[List[Dict]], context: Optional[Dict]
    ) -> List[str]:
        """Identify contributing factors to failure"""
        factors = []

        if context:
            if context.get("high_load"):
                factors.append("high_system_load")
            if context.get("insufficient_monitoring"):
                factors.append("insufficient_monitoring")
            if context.get("lack_of_redundancy"):
                factors.append("lack_of_redundancy")
            if context.get("poor_communication"):
                factors.append("poor_agent_communication")

        return factors

    def _assess_impact(
        self, failure_type: str, context: Optional[Dict]
    ) -> float:
        """Assess impact severity of failure (0.0-1.0)"""
        # Base severity by type
        severity_map = {
            "security_breach": 0.9,
            "system_degradation": 0.7,
            "task_failure": 0.4,
            "communication_failure": 0.3,
        }
        base_severity = severity_map.get(failure_type, 0.5)

        # Adjust based on context
        if context:
            if context.get("critical_service"):
                base_severity = min(1.0, base_severity * 1.2)
            if context.get("widespread_impact"):
                base_severity = min(1.0, base_severity * 1.3)

        return base_severity

    def _generate_recommendation(
        self, root_cause: str, factors: List[str]
    ) -> str:
        """Generate recommendation for addressing root cause"""
        recommendations = {
            "insufficient_resources": "Implement resource monitoring and dynamic allocation",
            "capability_mismatch": "Improve task-to-agent matching and capability assessment",
            "weak_authentication": "Strengthen authentication mechanisms and access controls",
            "unpatched_vulnerability": "Implement automated patching and vulnerability scanning",
            "resource_exhaustion": "Implement predictive resource management and auto-scaling",
            "cascading_failure": "Implement circuit breakers and isolation mechanisms",
        }
        return recommendations.get(
            root_cause, "Conduct deeper investigation and implement monitoring"
        )

    def _extract_lesson(self, root_cause: str, recommendation: str) -> str:
        """Extract lesson learned from root cause and recommendation"""
        return f"When {root_cause} occurs, {recommendation}"

    def _identify_affected_systems(self, context: Optional[Dict]) -> List[str]:
        """Identify systems affected by failure"""
        if not context:
            return []
        return context.get("affected_systems", [])

    def _formulate_what_not_to_do(self, analysis: FailureAnalysis) -> str:
        """Formulate what NOT to do based on failure"""
        # Simplified formulation
        what_not_to_do = {
            "insufficient_resources": "Do not attempt tasks without verifying sufficient resources",
            "capability_mismatch": "Do not bid on tasks that exceed your capabilities",
            "weak_authentication": "Do not bypass authentication checks for convenience",
            "unpatched_vulnerability": "Do not delay security patches for non-critical systems",
        }
        return what_not_to_do.get(
            analysis.root_cause,
            f"Do not repeat actions that led to {analysis.failure_type}",
        )

    def _update_knowledge_graph(self, analysis: FailureAnalysis):
        """Update knowledge graph with failure analysis"""
        # In production, this would integrate with the KnowledgeGraph
        logger.debug(
            f"Would update knowledge graph with analysis {analysis.failure_id}"
        )

    def get_recurring_failures(self) -> List[Tuple[str, int]]:
        """Get recurring failure patterns"""
        return [
            (pattern, len(failures))
            for pattern, failures in self.failure_patterns.items()
            if len(failures) >= 3
        ]

    def get_statistics(self) -> Dict:
        """Get pathologist statistics"""
        return {
            "total_failures_analyzed": self.total_failures_analyzed,
            "total_lessons_published": self.total_lessons_published,
            "total_root_causes_identified": self.total_root_causes_identified,
            "recurring_failure_patterns": len(self.recurring_failures),
            "unique_failure_types": len(self.failure_patterns),
        }


class StrategicImmunologist:
    """
    Professor that monitors for chronic vulnerabilities and injects antibodies.

    Monitors the entire system for chronic, recurring vulnerabilities. When a
    persistent weakness is identified, it has authority to inject an "antibody" -
    a new security model, mandatory genetic sequence, or globally enforced policy.
    """

    def __init__(self):
        """Initialize Strategic Immunologist"""
        self.professor_id = "prof_immunologist"

        # Vulnerability tracking
        self.vulnerabilities: Dict[str, Vulnerability] = {}
        self.chronic_vulnerabilities: Set[str] = set()

        # Antibodies
        self.antibodies: Dict[str, Antibody] = {}
        self.deployed_antibodies: List[str] = []

        # Thresholds
        self.chronic_threshold = 5  # Occurrences to be considered chronic
        self.severity_threshold = (
            0.7  # Severity threshold for antibody injection
        )

        # Statistics
        self.total_vulnerabilities_tracked = 0
        self.total_antibodies_injected = 0
        self.total_chronic_identified = 0

        logger.info("Strategic Immunologist initialized")

    def monitor_vulnerability(
        self,
        vuln_id: str,
        vulnerability_type: str,
        description: str,
        severity: float,
        affected_components: List[str],
    ) -> Vulnerability:
        """
        Monitor a vulnerability in the system.

        Args:
            vuln_id: Unique vulnerability identifier
            vulnerability_type: Type of vulnerability
            description: Description of vulnerability
            severity: Severity (0.0-1.0)
            affected_components: List of affected components

        Returns:
            Vulnerability record
        """
        if vuln_id in self.vulnerabilities:
            # Update existing vulnerability
            vuln = self.vulnerabilities[vuln_id]
            vuln.frequency += 1
            vuln.last_seen = time.time()
            logger.info(
                f"Vulnerability {vuln_id} observed again (frequency={vuln.frequency})"
            )
        else:
            # Create new vulnerability record
            vuln = Vulnerability(
                vuln_id=vuln_id,
                timestamp=time.time(),
                vulnerability_type=vulnerability_type,
                description=description,
                frequency=1,
                severity=severity,
                affected_components=affected_components,
            )
            self.vulnerabilities[vuln_id] = vuln
            self.total_vulnerabilities_tracked += 1
            logger.info(
                f"New vulnerability tracked: {vuln_id} (type={vulnerability_type})"
            )

        # Check if chronic
        if (
            vuln.frequency >= self.chronic_threshold
            and vuln.severity >= self.severity_threshold
        ):
            if vuln_id not in self.chronic_vulnerabilities:
                self.chronic_vulnerabilities.add(vuln_id)
                self.total_chronic_identified += 1
                logger.warning(
                    f"Chronic vulnerability identified: {vuln_id} "
                    f"(frequency={vuln.frequency}, severity={vuln.severity:.2f})"
                )

                # Consider injecting antibody
                self._consider_antibody_injection(vuln)

        return vuln

    def inject_antibody(
        self,
        target_vulnerability: str,
        antibody_type: AntibodyType,
        description: str,
        implementation: Dict[str, Any],
    ) -> Antibody:
        """
        Inject an antibody to address a chronic vulnerability.

        Args:
            target_vulnerability: ID of vulnerability to address
            antibody_type: Type of antibody to inject
            description: Description of antibody
            implementation: Implementation details

        Returns:
            Injected Antibody
        """
        antibody_id = f"antibody_{int(time.time()*1000)}"

        antibody = Antibody(
            antibody_id=antibody_id,
            timestamp=time.time(),
            antibody_type=antibody_type,
            target_vulnerability=target_vulnerability,
            description=description,
            implementation=implementation,
        )

        self.antibodies[antibody_id] = antibody
        self.total_antibodies_injected += 1

        logger.info(
            f"Antibody injected: {antibody_type.value} for {target_vulnerability}"
        )
        logger.info(f"  Description: {description}")

        return antibody

    def deploy_antibody(self, antibody_id: str) -> bool:
        """
        Deploy an antibody to the system.

        Args:
            antibody_id: ID of antibody to deploy

        Returns:
            True if deployed successfully
        """
        antibody = self.antibodies.get(antibody_id)
        if not antibody:
            logger.warning(f"Cannot deploy unknown antibody {antibody_id}")
            return False

        if antibody.deployed:
            logger.warning(f"Antibody {antibody_id} already deployed")
            return False

        # Deploy antibody (implementation would vary by type)
        logger.info(
            f"Deploying antibody {antibody_id} ({antibody.antibody_type.value})"
        )

        antibody.deployed = True
        antibody.deployment_time = time.time()
        self.deployed_antibodies.append(antibody_id)

        # Mark vulnerability as having remediation attempt
        vuln = self.vulnerabilities.get(antibody.target_vulnerability)
        if vuln:
            vuln.remediation_attempts += 1

        logger.info(f"Antibody {antibody_id} deployed successfully")

        return True

    def measure_antibody_effectiveness(
        self, antibody_id: str, effectiveness: float
    ) -> bool:
        """
        Measure effectiveness of deployed antibody.

        Args:
            antibody_id: ID of antibody
            effectiveness: Measured effectiveness (0.0-1.0)

        Returns:
            True if measurement recorded
        """
        antibody = self.antibodies.get(antibody_id)
        if not antibody:
            logger.warning(f"Cannot measure unknown antibody {antibody_id}")
            return False

        if not antibody.deployed:
            logger.warning(f"Antibody {antibody_id} not deployed yet")
            return False

        antibody.effectiveness = effectiveness

        logger.info(
            f"Antibody {antibody_id} effectiveness: {effectiveness:.2%}"
        )

        return True

    def _consider_antibody_injection(self, vuln: Vulnerability):
        """Consider injecting an antibody for a chronic vulnerability"""
        logger.info(f"Considering antibody injection for {vuln.vuln_id}")

        # Determine appropriate antibody type
        antibody_type = self._determine_antibody_type(vuln.vulnerability_type)

        # Create implementation based on type
        implementation = self._create_antibody_implementation(
            vuln.vulnerability_type, antibody_type
        )

        # Inject antibody
        description = (
            f"Antibody for chronic {vuln.vulnerability_type} vulnerability"
        )
        self.inject_antibody(
            target_vulnerability=vuln.vuln_id,
            antibody_type=antibody_type,
            description=description,
            implementation=implementation,
        )

    def _determine_antibody_type(
        self, vulnerability_type: str
    ) -> AntibodyType:
        """Determine appropriate antibody type for vulnerability"""
        antibody_map = {
            "authentication_weakness": AntibodyType.SECURITY_MODEL,
            "unpatched_software": AntibodyType.OPERATIONAL_POLICY,
            "network_exposure": AntibodyType.FIREWALL_RULE,
            "false_negative": AntibodyType.THRESHOLD_ADJUSTMENT,
            "genetic_weakness": AntibodyType.GENETIC_SEQUENCE,
        }
        return antibody_map.get(
            vulnerability_type, AntibodyType.OPERATIONAL_POLICY
        )

    def _create_antibody_implementation(
        self, vulnerability_type: str, antibody_type: AntibodyType
    ) -> Dict[str, Any]:
        """Create antibody implementation details"""
        # Simplified implementation
        # In production, this would contain actual implementation details
        return {
            "vulnerability_type": vulnerability_type,
            "antibody_type": antibody_type.value,
            "action": f"remediate_{vulnerability_type}",
            "parameters": {},
        }

    def get_chronic_vulnerabilities(self) -> List[Vulnerability]:
        """Get all chronic vulnerabilities"""
        return [
            self.vulnerabilities[vuln_id]
            for vuln_id in self.chronic_vulnerabilities
            if vuln_id in self.vulnerabilities
        ]

    def get_statistics(self) -> Dict:
        """Get immunologist statistics"""
        return {
            "total_vulnerabilities_tracked": self.total_vulnerabilities_tracked,
            "total_antibodies_injected": self.total_antibodies_injected,
            "total_chronic_identified": self.total_chronic_identified,
            "deployed_antibodies": len(self.deployed_antibodies),
            "active_vulnerabilities": len(self.vulnerabilities),
        }


class EvolutionaryBiologist:
    """
    Professor that curates genetic health of the collective.

    Monitors the gene pool for signs of stagnation or dangerous mutations,
    and can adjust global parameters like mutation rate. Can create
    "evolutionary sandboxes" to test radical new agent designs.
    """

    def __init__(self, evolution_engine: Optional[object] = None):
        """
        Initialize Evolutionary Biologist.

        Args:
            evolution_engine: Optional EvolutionEngine for integration
        """
        self.professor_id = "prof_biologist"
        self.evolution_engine = evolution_engine

        # Genetic health tracking
        self.health_history: deque = deque(maxlen=100)

        # Mutation tracking
        self.beneficial_mutations: List[Dict] = []
        self.dangerous_mutations: List[Dict] = []

        # Sandboxes
        self.sandboxes: Dict[str, Dict] = {}

        # Parameters
        self.diversity_threshold = 0.3  # Min acceptable diversity
        self.stagnation_threshold = 0.7  # Max acceptable stagnation

        # Statistics
        self.total_assessments = 0
        self.total_interventions = 0
        self.total_sandboxes_created = 0

        logger.info("Evolutionary Biologist initialized")

    def assess_genetic_health(
        self,
        population_size: int,
        genetic_diversity: float,
        avg_fitness: float,
        stagnation_level: float,
        mutation_rate: float,
    ) -> GeneticHealth:
        """
        Assess genetic health of the population.

        Args:
            population_size: Current population size
            genetic_diversity: Genetic diversity (0.0-1.0)
            avg_fitness: Average fitness of population
            stagnation_level: Stagnation level (0.0-1.0)
            mutation_rate: Current mutation rate

        Returns:
            GeneticHealth assessment
        """
        health = GeneticHealth(
            timestamp=time.time(),
            population_size=population_size,
            genetic_diversity=genetic_diversity,
            avg_fitness=avg_fitness,
            stagnation_level=stagnation_level,
            dangerous_mutations=len(self.dangerous_mutations),
            beneficial_mutations=len(self.beneficial_mutations),
            mutation_rate=mutation_rate,
        )

        # Generate recommendation
        if genetic_diversity < self.diversity_threshold:
            health.recommendation = (
                "INCREASE_DIVERSITY: Population lacks genetic diversity"
            )
            self._intervene_diversity(health)
        elif stagnation_level > self.stagnation_threshold:
            health.recommendation = (
                "INCREASE_MUTATION: Population is stagnating"
            )
            self._intervene_stagnation(health)
        elif len(self.dangerous_mutations) > 5:
            health.recommendation = (
                "CULL_DANGEROUS: Too many dangerous mutations"
            )
            self._intervene_dangerous_mutations()
        else:
            health.recommendation = "HEALTHY: Genetic health is good"

        self.health_history.append(health)
        self.total_assessments += 1

        logger.info(
            f"Genetic health assessed: diversity={genetic_diversity:.2f}, "
            f"stagnation={stagnation_level:.2f}, fitness={avg_fitness:.2f}"
        )
        logger.info(f"  Recommendation: {health.recommendation}")

        return health

    def track_mutation(
        self,
        mutation_id: str,
        mutation_type: str,
        is_beneficial: bool,
        description: str,
        impact: float,
    ):
        """
        Track a mutation in the gene pool.

        Args:
            mutation_id: Unique mutation identifier
            mutation_type: Type of mutation
            is_beneficial: Whether mutation is beneficial
            description: Description of mutation
            impact: Impact magnitude (0.0-1.0)
        """
        mutation = {
            "mutation_id": mutation_id,
            "timestamp": time.time(),
            "mutation_type": mutation_type,
            "description": description,
            "impact": impact,
        }

        if is_beneficial:
            self.beneficial_mutations.append(mutation)
            logger.info(
                f"Beneficial mutation tracked: {mutation_id} (impact={impact:.2f})"
            )
        else:
            self.dangerous_mutations.append(mutation)
            logger.warning(
                f"Dangerous mutation tracked: {mutation_id} (impact={impact:.2f})"
            )

    def create_evolutionary_sandbox(
        self,
        sandbox_id: str,
        description: str,
        population_size: int = 10,
        mutation_rate: float = 0.2,
    ) -> Dict:
        """
        Create an evolutionary sandbox to test radical new designs.

        Args:
            sandbox_id: Unique sandbox identifier
            description: Description of sandbox purpose
            population_size: Population size in sandbox
            mutation_rate: Mutation rate for sandbox

        Returns:
            Sandbox configuration
        """
        sandbox = {
            "sandbox_id": sandbox_id,
            "created_at": time.time(),
            "description": description,
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "generations": 0,
            "best_fitness": 0.0,
            "active": True,
        }

        self.sandboxes[sandbox_id] = sandbox
        self.total_sandboxes_created += 1

        logger.info(f"Evolutionary sandbox created: {sandbox_id}")
        logger.info(f"  Description: {description}")
        logger.info(
            f"  Population: {population_size}, Mutation rate: {mutation_rate}"
        )

        return sandbox

    def _intervene_diversity(self, health: GeneticHealth):
        """Intervene to increase genetic diversity"""
        logger.info("Intervening to increase genetic diversity")

        if self.evolution_engine:
            # In production, this would call evolution engine to increase diversity
            # For example: inject new genetic material, increase crossover rate
            logger.info("  Would increase crossover rate and inject new genes")

        self.total_interventions += 1

    def _intervene_stagnation(self, health: GeneticHealth):
        """Intervene to address stagnation"""
        logger.info("Intervening to address genetic stagnation")

        if self.evolution_engine:
            # In production, this would increase mutation rate
            logger.info(
                f"  Would increase mutation rate from {health.mutation_rate}"
            )

        self.total_interventions += 1

    def _intervene_dangerous_mutations(self):
        """Intervene to cull dangerous mutations"""
        logger.info("Intervening to cull dangerous mutations")

        if self.evolution_engine:
            # In production, this would remove agents with dangerous mutations
            logger.info(
                f"  Would cull {len(self.dangerous_mutations)} dangerous mutations"
            )

        # Clear dangerous mutations list after intervention
        self.dangerous_mutations.clear()

        self.total_interventions += 1

    def get_statistics(self) -> Dict:
        """Get biologist statistics"""
        recent_health = (
            list(self.health_history)[-10:] if self.health_history else []
        )

        avg_diversity = (
            sum(h.genetic_diversity for h in recent_health)
            / len(recent_health)
            if recent_health
            else 0.0
        )
        avg_fitness = (
            sum(h.avg_fitness for h in recent_health) / len(recent_health)
            if recent_health
            else 0.0
        )

        return {
            "total_assessments": self.total_assessments,
            "total_interventions": self.total_interventions,
            "total_sandboxes_created": self.total_sandboxes_created,
            "beneficial_mutations": len(self.beneficial_mutations),
            "dangerous_mutations": len(self.dangerous_mutations),
            "active_sandboxes": sum(
                1 for s in self.sandboxes.values() if s["active"]
            ),
            "avg_diversity": avg_diversity,
            "avg_fitness": avg_fitness,
        }


class CouncilOfProfessors:
    """
    The Council of Professors - the governance layer.

    Coordinates the three professor agents to provide comprehensive
    oversight of the digital polity.
    """

    def __init__(
        self,
        knowledge_graph: Optional[object] = None,
        evolution_engine: Optional[object] = None,
    ):
        """
        Initialize Council of Professors.

        Args:
            knowledge_graph: Optional KnowledgeGraph for integration
            evolution_engine: Optional EvolutionEngine for integration
        """
        self.pathologist = SystemicPathologist(knowledge_graph)
        self.immunologist = StrategicImmunologist()
        self.biologist = EvolutionaryBiologist(evolution_engine)

        self.created_at = time.time()

        logger.info("Council of Professors assembled")

    def get_statistics(self) -> Dict:
        """Get statistics for entire council"""
        return {
            "pathologist": self.pathologist.get_statistics(),
            "immunologist": self.immunologist.get_statistics(),
            "biologist": self.biologist.get_statistics(),
            "council_age": time.time() - self.created_at,
        }
