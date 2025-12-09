"""
Adaptive Immune Response - Digital White Blood Cells

This module implements a multi-role immune system inspired by biological
immune responses:
- First Responders (Neutrophils): Fast containment
- Specialist Healers (Macrophages): Targeted repair
- Reinforcement Squads: Adaptive response under sustained stress
- Memory Cells (B-Cells): Long-term threat memory
"""

import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class ImmuneAgentType(str, Enum):
    """Types of immune system agents."""
    NEUTROPHIL = "neutrophil"       # First responder
    MACROPHAGE = "macrophage"       # Specialist healer
    REINFORCEMENT = "reinforcement"  # Adaptive response
    B_CELL = "b_cell"               # Memory cell


class ThreatLevel(str, Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseStatus(str, Enum):
    """Status of immune response."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass
class ThreatSignature:
    """
    A unique signature identifying a threat.
    
    Attributes:
        signature_id: Unique identifier
        signature_hash: Hash of threat characteristics
        threat_type: Type of threat
        features: Key features of the threat
        first_seen: When first encountered
        last_seen: When last encountered
        occurrence_count: Number of times seen
        severity: Average severity (0.0-1.0)
    """
    signature_id: str
    signature_hash: str
    threat_type: str
    features: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    occurrence_count: int = 1
    severity: float = 0.5
    
    def update_occurrence(self, severity: float) -> None:
        """Update occurrence and severity."""
        self.occurrence_count += 1
        self.last_seen = time.time()
        # Running average of severity
        self.severity = (self.severity * (self.occurrence_count - 1) + severity) / self.occurrence_count


@dataclass
class ImmuneResponse:
    """
    A coordinated immune response to a threat.
    
    Attributes:
        response_id: Unique identifier
        threat_signature: Signature of the threat
        target_node: Node under threat
        threat_level: Severity level
        status: Current status
        agents_deployed: IDs of deployed agents
        start_time: When response started
        resolution_time: When resolved (None if ongoing)
        success: Whether response was successful
    """
    response_id: str
    threat_signature: str
    target_node: str
    threat_level: ThreatLevel
    status: ResponseStatus = ResponseStatus.INITIATED
    agents_deployed: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    resolution_time: Optional[float] = None
    success: Optional[bool] = None
    
    def get_response_duration(self) -> float:
        """Get duration of response in seconds."""
        if self.resolution_time:
            return self.resolution_time - self.start_time
        return time.time() - self.start_time


class NeutrophilAgent:
    """
    First Responder (Neutrophil) - Fast containment agent.
    
    These agents are fast and lightweight. Their job is not to solve
    the problem, but to contain it - isolating the affected node and
    preventing the threat from spreading.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize neutrophil agent.
        
        Args:
            agent_id: Unique identifier
        """
        self.agent_id = agent_id
        self.agent_type = ImmuneAgentType.NEUTROPHIL
        self.deployments = 0
        self.successful_containments = 0
        self.current_location: Optional[str] = None
        self.response_time = 0.5  # Fast response
        
    def respond(self, threat_node: str, threat_level: ThreatLevel) -> Dict[str, Any]:
        """
        Respond to a threat by containing it.
        
        Args:
            threat_node: Node under threat
            threat_level: Severity of threat
            
        Returns:
            Containment result
        """
        self.deployments += 1
        self.current_location = threat_node
        
        # Perform containment actions
        actions = [
            "isolate_node",
            "block_incoming_traffic",
            "prevent_spread"
        ]
        
        # Higher threat levels require more aggressive containment
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            actions.extend([
                "quarantine_neighbors",
                "suspend_services"
            ])
            
        success_probability = 0.9 if threat_level != ThreatLevel.CRITICAL else 0.7
        
        result = {
            'agent_id': self.agent_id,
            'actions_taken': actions,
            'containment_level': 'aggressive' if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] else 'standard',
            'estimated_success': success_probability,
            'response_time': self.response_time
        }
        
        return result
    
    def record_success(self) -> None:
        """Record a successful containment."""
        self.successful_containments += 1


class MacrophageAgent:
    """
    Specialist Healer (Macrophage) - Targeted repair agent.
    
    These are more powerful effector agents dispatched to a contained
    incident. They use the Adaptive Memory (Knowledge Graph) to select
    the best-known tool or strategy to perform the actual repair.
    """
    
    def __init__(self, agent_id: str, knowledge_graph: Any = None):
        """
        Initialize macrophage agent.
        
        Args:
            agent_id: Unique identifier
            knowledge_graph: Reference to knowledge graph for strategy selection
        """
        self.agent_id = agent_id
        self.agent_type = ImmuneAgentType.MACROPHAGE
        self.knowledge_graph = knowledge_graph
        self.deployments = 0
        self.successful_repairs = 0
        self.current_location: Optional[str] = None
        self.response_time = 2.0  # Slower but more thorough
        
    def respond(self, 
                threat_node: str,
                threat_signature: str,
                threat_level: ThreatLevel,
                pattern_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Respond to a contained threat by performing repair.
        
        Args:
            threat_node: Node to repair
            threat_signature: Signature of the threat
            threat_level: Severity
            pattern_id: Optional problem pattern ID from knowledge graph
            
        Returns:
            Repair result
        """
        self.deployments += 1
        self.current_location = threat_node
        
        # Select strategy from knowledge graph
        strategy = None
        if self.knowledge_graph and pattern_id:
            best_strategies = self.knowledge_graph.get_best_strategy(pattern_id, top_k=1)
            if best_strategies:
                strategy_id, weight = best_strategies[0]
                strategy = self.knowledge_graph.solution_strategies.get(strategy_id)
                
        # Default repair actions if no strategy found
        actions = strategy.actions if strategy else [
            "scan_for_threats",
            "remove_malicious_code",
            "restore_configuration",
            "restart_services",
            "verify_integrity"
        ]
        
        result = {
            'agent_id': self.agent_id,
            'strategy_used': strategy.name if strategy else 'default_repair',
            'actions_taken': actions,
            'repair_level': 'comprehensive',
            'knowledge_guided': strategy is not None,
            'response_time': self.response_time
        }
        
        return result
    
    def record_success(self) -> None:
        """Record a successful repair."""
        self.successful_repairs += 1


class ReinforcementSquad:
    """
    Reinforcement Squad - Adaptive response under sustained stress.
    
    When a node or region is under sustained stress, the system
    dynamically dispatches a squad of general-purpose agents to
    reinforce it, providing extra computational resources, handling
    excess traffic, or stabilizing neighbors.
    """
    
    def __init__(self, squad_id: str, size: int = 3):
        """
        Initialize reinforcement squad.
        
        Args:
            squad_id: Unique identifier
            size: Number of agents in squad
        """
        self.squad_id = squad_id
        self.agent_type = ImmuneAgentType.REINFORCEMENT
        self.size = size
        self.deployments = 0
        self.current_location: Optional[str] = None
        self.response_time = 3.0  # Takes time to mobilize
        
    def respond(self, stress_node: str, stress_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Respond to sustained stress by providing reinforcement.
        
        Args:
            stress_node: Node under stress
            stress_metrics: Metrics indicating stress type and level
            
        Returns:
            Reinforcement result
        """
        self.deployments += 1
        self.current_location = stress_node
        
        # Determine reinforcement strategy based on stress type
        actions = []
        
        cpu_stress = stress_metrics.get('cpu_load', 0.0)
        memory_stress = stress_metrics.get('memory_usage', 0.0)
        network_stress = stress_metrics.get('network_load', 0.0)
        
        if cpu_stress > 0.8:
            actions.extend([
                "provision_additional_compute",
                "distribute_workload",
                "optimize_processes"
            ])
            
        if memory_stress > 0.8:
            actions.extend([
                "provision_additional_memory",
                "clear_caches",
                "migrate_low_priority_tasks"
            ])
            
        if network_stress > 0.8:
            actions.extend([
                "provision_bandwidth",
                "reroute_traffic",
                "enable_compression"
            ])
            
        result = {
            'squad_id': self.squad_id,
            'squad_size': self.size,
            'actions_taken': actions,
            'reinforcement_type': 'resource_provision',
            'estimated_capacity_increase': self.size * 0.3,  # 30% per agent
            'response_time': self.response_time
        }
        
        return result


class BCellAgent:
    """
    Memory Cell (B-Cell) - Long-term threat memory.
    
    Once a new threat is successfully neutralized, a Memory Cell is
    created that is genetically predisposed to "see" and "smell" that
    specific threat signature, ensuring a much faster response the
    next time it appears.
    """
    
    def __init__(self, agent_id: str, threat_signature: ThreatSignature):
        """
        Initialize B-cell agent tuned to a specific threat.
        
        Args:
            agent_id: Unique identifier
            threat_signature: The threat this cell remembers
        """
        self.agent_id = agent_id
        self.agent_type = ImmuneAgentType.B_CELL
        self.threat_signature = threat_signature
        self.detections = 0
        self.false_positives = 0
        self.created_at = time.time()
        
    def recognize_threat(self, observed_features: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if observed features match remembered threat.
        
        Args:
            observed_features: Features to check
            
        Returns:
            (is_match, confidence) tuple
        """
        # Calculate similarity between observed and remembered features
        matches = 0
        total = len(self.threat_signature.features)
        
        if total == 0:
            return (False, 0.0)
            
        for key, expected_value in self.threat_signature.features.items():
            if key in observed_features:
                observed_value = observed_features[key]
                if observed_value == expected_value:
                    matches += 1
                elif isinstance(expected_value, (int, float)) and isinstance(observed_value, (int, float)):
                    # Numerical similarity
                    diff = abs(expected_value - observed_value)
                    if diff < expected_value * 0.1:  # Within 10%
                        matches += 0.8
                        
        similarity = matches / total
        is_match = similarity > 0.7  # 70% threshold
        
        if is_match:
            self.detections += 1
            
        return (is_match, similarity)
    
    def get_memory_age(self) -> float:
        """Get age of this memory in seconds."""
        return time.time() - self.created_at


class AdaptiveImmuneSystem:
    """
    Adaptive Immune System - Coordinates all immune responses.
    
    This system manages the full lifecycle of immune responses:
    - Deploying neutrophils for containment
    - Dispatching macrophages for repair
    - Mobilizing reinforcement squads
    - Creating and maintaining memory cells
    """
    
    def __init__(self, knowledge_graph: Any = None):
        """
        Initialize the adaptive immune system.
        
        Args:
            knowledge_graph: Optional knowledge graph for strategy selection
        """
        self.knowledge_graph = knowledge_graph
        
        # Agent pools
        self.neutrophils: List[NeutrophilAgent] = []
        self.macrophages: List[MacrophageAgent] = []
        self.reinforcement_squads: List[ReinforcementSquad] = []
        self.b_cells: List[BCellAgent] = []
        
        # Threat tracking
        self.known_threats: Dict[str, ThreatSignature] = {}
        self.active_responses: Dict[str, ImmuneResponse] = {}
        self.completed_responses: List[ImmuneResponse] = []
        
        # Statistics
        self.total_responses = 0
        self.successful_responses = 0
        self.failed_responses = 0
        
    def initialize_agents(self,
                         neutrophil_count: int = 5,
                         macrophage_count: int = 3,
                         reinforcement_squads: int = 2) -> None:
        """
        Initialize immune system agents.
        
        Args:
            neutrophil_count: Number of neutrophils to create
            macrophage_count: Number of macrophages to create
            reinforcement_squads: Number of reinforcement squads to create
        """
        # Create neutrophils
        for i in range(neutrophil_count):
            self.neutrophils.append(NeutrophilAgent(f"neutrophil_{i}"))
            
        # Create macrophages
        for i in range(macrophage_count):
            self.macrophages.append(MacrophageAgent(
                f"macrophage_{i}",
                self.knowledge_graph
            ))
            
        # Create reinforcement squads
        for i in range(reinforcement_squads):
            self.reinforcement_squads.append(ReinforcementSquad(f"squad_{i}"))
            
        logger.info(f"Initialized immune system: {neutrophil_count} neutrophils, "
                   f"{macrophage_count} macrophages, {reinforcement_squads} squads")
    
    def detect_threat(self, 
                     threat_features: Dict[str, Any],
                     threat_type: str,
                     severity: float) -> Optional[str]:
        """
        Detect and classify a threat, checking against memory cells.
        
        Args:
            threat_features: Features of the threat
            threat_type: Type of threat
            severity: Severity level (0.0-1.0)
            
        Returns:
            Threat signature ID if recognized, None if novel
        """
        # Check memory cells first (faster recognition)
        for b_cell in self.b_cells:
            is_match, confidence = b_cell.recognize_threat(threat_features)
            if is_match:
                logger.info(f"Memory cell {b_cell.agent_id} recognized threat "
                           f"(confidence: {confidence:.2f})")
                # Update known threat
                sig = b_cell.threat_signature
                sig.update_occurrence(severity)
                return sig.signature_id
                
        # Novel threat - create new signature
        feature_str = f"{threat_type}_{str(sorted(threat_features.items()))}"
        signature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]
        signature_id = f"threat_{signature_hash}"
        
        signature = ThreatSignature(
            signature_id=signature_id,
            signature_hash=signature_hash,
            threat_type=threat_type,
            features=threat_features,
            severity=severity
        )
        
        self.known_threats[signature_id] = signature
        logger.info(f"Novel threat detected: {signature_id}")
        
        return signature_id
    
    def initiate_response(self,
                         threat_signature: str,
                         target_node: str,
                         threat_level: ThreatLevel) -> str:
        """
        Initiate a coordinated immune response.
        
        Args:
            threat_signature: Signature of the threat
            target_node: Node under threat
            threat_level: Severity level
            
        Returns:
            Response ID
        """
        self.total_responses += 1
        response_id = f"response_{int(time.time() * 1000)}_{self.total_responses}"
        
        response = ImmuneResponse(
            response_id=response_id,
            threat_signature=threat_signature,
            target_node=target_node,
            threat_level=threat_level
        )
        
        self.active_responses[response_id] = response
        logger.info(f"Initiated immune response {response_id} for {target_node}")
        
        # Phase 1: Deploy neutrophils for containment
        self._deploy_neutrophils(response)
        
        return response_id
    
    def _deploy_neutrophils(self, response: ImmuneResponse) -> None:
        """Deploy neutrophils for containment."""
        # Select available neutrophils
        available = [n for n in self.neutrophils if n.current_location is None]
        
        # Deploy 1-2 neutrophils based on threat level
        count = 2 if response.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] else 1
        count = min(count, len(available))
        
        for neutrophil in available[:count]:
            result = neutrophil.respond(response.target_node, response.threat_level)
            response.agents_deployed.append(neutrophil.agent_id)
            logger.debug(f"Deployed {neutrophil.agent_id} for containment")
            
        response.status = ResponseStatus.IN_PROGRESS
    
    def escalate_to_repair(self, response_id: str, pattern_id: Optional[str] = None) -> None:
        """
        Escalate to repair phase with macrophages.
        
        Args:
            response_id: Response to escalate
            pattern_id: Optional problem pattern ID from knowledge graph
        """
        if response_id not in self.active_responses:
            logger.warning(f"Response {response_id} not found")
            return
            
        response = self.active_responses[response_id]
        
        # Select available macrophage
        available = [m for m in self.macrophages if m.current_location is None]
        
        if not available:
            logger.warning("No macrophages available for repair")
            return
            
        macrophage = available[0]
        result = macrophage.respond(
            response.target_node,
            response.threat_signature,
            response.threat_level,
            pattern_id
        )
        
        response.agents_deployed.append(macrophage.agent_id)
        response.status = ResponseStatus.CONTAINED
        logger.info(f"Escalated {response_id} to repair phase with {macrophage.agent_id}")
    
    def deploy_reinforcements(self, target_node: str, stress_metrics: Dict[str, float]) -> str:
        """
        Deploy reinforcement squad to a stressed node.
        
        Args:
            target_node: Node under stress
            stress_metrics: Stress metrics
            
        Returns:
            Squad ID deployed
        """
        # Select available squad
        available = [s for s in self.reinforcement_squads if s.current_location is None]
        
        if not available:
            logger.warning("No reinforcement squads available")
            return ""
            
        squad = available[0]
        result = squad.respond(target_node, stress_metrics)
        logger.info(f"Deployed reinforcement squad {squad.squad_id} to {target_node}")
        
        return squad.squad_id
    
    def complete_response(self, response_id: str, success: bool) -> None:
        """
        Complete an immune response.
        
        Args:
            response_id: Response ID
            success: Whether response was successful
        """
        if response_id not in self.active_responses:
            return
            
        response = self.active_responses.pop(response_id)
        response.resolution_time = time.time()
        response.success = success
        response.status = ResponseStatus.RESOLVED if success else ResponseStatus.FAILED
        
        # Update statistics
        if success:
            self.successful_responses += 1
            
            # Create memory cell for successful novel threat neutralization
            if response.threat_signature in self.known_threats:
                threat_sig = self.known_threats[response.threat_signature]
                if threat_sig.occurrence_count == 1:  # Novel threat
                    self._create_memory_cell(threat_sig)
        else:
            self.failed_responses += 1
            
        # Release agents
        for agent_id in response.agents_deployed:
            self._release_agent(agent_id)
            
        self.completed_responses.append(response)
        
        duration = response.get_response_duration()
        logger.info(f"Response {response_id} completed in {duration:.1f}s "
                   f"({'success' if success else 'failed'})")
    
    def _create_memory_cell(self, threat_signature: ThreatSignature) -> None:
        """Create a B-cell memory for a threat."""
        b_cell_id = f"bcell_{threat_signature.signature_hash}"
        b_cell = BCellAgent(b_cell_id, threat_signature)
        self.b_cells.append(b_cell)
        logger.info(f"Created memory cell {b_cell_id} for {threat_signature.threat_type}")
    
    def _release_agent(self, agent_id: str) -> None:
        """Release an agent back to the pool."""
        # Check neutrophils
        for neutrophil in self.neutrophils:
            if neutrophil.agent_id == agent_id:
                neutrophil.current_location = None
                return
                
        # Check macrophages
        for macrophage in self.macrophages:
            if macrophage.agent_id == agent_id:
                macrophage.current_location = None
                return
                
        # Check squads
        for squad in self.reinforcement_squads:
            if squad.squad_id == agent_id:
                squad.current_location = None
                return
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive immune system statistics."""
        return {
            'total_responses': self.total_responses,
            'successful_responses': self.successful_responses,
            'failed_responses': self.failed_responses,
            'success_rate': self.successful_responses / max(1, self.total_responses),
            'active_responses': len(self.active_responses),
            'neutrophils': len(self.neutrophils),
            'macrophages': len(self.macrophages),
            'reinforcement_squads': len(self.reinforcement_squads),
            'memory_cells': len(self.b_cells),
            'known_threats': len(self.known_threats),
            'avg_response_time': self._calculate_avg_response_time()
        }
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time for completed responses."""
        if not self.completed_responses:
            return 0.0
            
        total_time = sum(r.get_response_duration() for r in self.completed_responses)
        return total_time / len(self.completed_responses)
