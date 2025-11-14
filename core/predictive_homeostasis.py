"""
Predictive Homeostasis & Systemic Resilience for Phase 2

This module implements predictive capabilities to anticipate and prevent system failures
before they cascade, and adaptive resource management for system-wide resilience.

Key Features:
- Emergent pattern recognition for failure precursors
- Predictive models for cascading failures
- Adaptive resource allocation and migration
- System-wide stress response coordination
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    """Types of system failures that can be predicted"""
    CASCADING_OVERLOAD = "cascading_overload"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    SECURITY_BREACH = "security_breach"
    SERVICE_DEGRADATION = "service_degradation"


class StressType(str, Enum):
    """Types of system-wide stress"""
    CPU_PRESSURE = "cpu_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    NETWORK_CONGESTION = "network_congestion"
    ENERGY_CRISIS = "energy_crisis"
    SECURITY_ALERT = "security_alert"


@dataclass
class FailurePrecursor:
    """
    A pattern that precedes system failure.
    
    Learned through observation of correlation between events.
    """
    precursor_id: str
    failure_type: FailureType
    
    # Pattern definition
    indicators: Dict[str, Any]  # e.g., {"latency_edge_A": "> 100ms", "cpu_node_B": "> 0.9"}
    time_window: float  # How far ahead this predicts (seconds)
    
    # Confidence metrics
    observed_count: int = 0
    prediction_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    
    # Learned correlations
    correlations: Dict[str, float] = field(default_factory=dict)  # metric -> correlation strength
    
    def confidence(self) -> float:
        """Calculate prediction confidence (precision)"""
        if self.prediction_count == 0:
            return 0.0
        return self.true_positives / self.prediction_count
    
    def update_statistics(self, was_true_positive: bool):
        """Update prediction statistics"""
        self.prediction_count += 1
        if was_true_positive:
            self.true_positives += 1
        else:
            self.false_positives += 1


@dataclass
class ResourceAllocation:
    """Resource allocation for an agent or service"""
    entity_id: str  # Agent or service ID
    node_id: str
    
    # Allocated resources
    cpu_allocation: float = 0.0      # CPU share (0-1)
    memory_allocation: float = 0.0   # Memory share (0-1)
    energy_allocation: float = 0.0   # Energy units
    
    # Usage metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    energy_consumption: float = 0.0
    
    # Priority and constraints
    priority: float = 0.5  # 0-1, higher = more important
    min_allocation: float = 0.1  # Minimum required
    max_allocation: float = 1.0  # Maximum allowed
    
    # Performance
    efficiency: float = 1.0  # Usage / allocation
    satisfaction: float = 1.0  # Needs met / needs requested
    
    def update_efficiency(self):
        """Calculate resource efficiency"""
        total_alloc = self.cpu_allocation + self.memory_allocation + self.energy_allocation
        total_usage = self.cpu_usage + self.memory_usage + self.energy_consumption
        
        if total_alloc > 0:
            self.efficiency = total_usage / total_alloc
        else:
            self.efficiency = 0.0


@dataclass
class SystemStressState:
    """System-wide stress state"""
    stress_type: StressType
    severity: float  # 0-1, higher = more severe
    affected_nodes: Set[str] = field(default_factory=set)
    affected_regions: Set[str] = field(default_factory=set)
    
    # Timeline
    start_time: float = field(default_factory=time.time)
    peak_severity: float = 0.0
    
    # Response
    agents_responding: Set[int] = field(default_factory=set)
    mitigation_actions: List[str] = field(default_factory=list)
    
    # Outcome
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def update_severity(self, new_severity: float):
        """Update severity and track peak"""
        self.severity = new_severity
        self.peak_severity = max(self.peak_severity, new_severity)


class PredictiveHomeostasisSystem:
    """
    System for predictive failure prevention and adaptive resource management.
    
    Learns patterns that precede failures and coordinates preemptive actions.
    """
    
    def __init__(self, prediction_window: float = 60.0):
        """
        Initialize predictive homeostasis system.
        
        Args:
            prediction_window: How far ahead to predict (seconds)
        """
        self.prediction_window = prediction_window
        
        # Failure precursors (learned patterns)
        self.precursors: Dict[FailureType, List[FailurePrecursor]] = defaultdict(list)
        
        # Time-series data for pattern recognition
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Resource allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # System stress states
        self.active_stresses: Dict[str, SystemStressState] = {}
        
        # Predictions
        self.active_predictions: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.predictions_made = 0
        self.predictions_correct = 0
        self.failures_prevented = 0
        self.resources_reallocated = 0
        self.agents_migrated = 0
        
        logger.info(f"PredictiveHomeostasisSystem initialized (window: {prediction_window}s)")
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """
        Record a metric value for pattern recognition.
        
        Args:
            metric_name: Name of metric (e.g., "latency_edge_A", "cpu_node_B")
            value: Metric value
            timestamp: Optional timestamp (default: now)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.metric_history[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
    
    def detect_correlation(
        self,
        metric1: str,
        metric2: str,
        lag: float = 0.0
    ) -> float:
        """
        Detect correlation between two metrics.
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            lag: Time lag to consider (seconds)
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        history1 = self.metric_history.get(metric1, [])
        history2 = self.metric_history.get(metric2, [])
        
        if len(history1) < 10 or len(history2) < 10:
            return 0.0
        
        # Extract values
        values1 = np.array([h["value"] for h in history1])
        values2 = np.array([h["value"] for h in history2])
        
        # Handle different lengths
        min_len = min(len(values1), len(values2))
        values1 = values1[-min_len:]
        values2 = values2[-min_len:]
        
        # Compute correlation
        if len(values1) > 1 and len(values2) > 1:
            correlation = np.corrcoef(values1, values2)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def learn_failure_precursor(
        self,
        failure_type: FailureType,
        indicators: Dict[str, Any],
        time_before_failure: float
    ) -> str:
        """
        Learn a new failure precursor pattern.
        
        Args:
            failure_type: Type of failure
            indicators: Observable indicators before failure
            time_before_failure: How long before failure these appeared
            
        Returns:
            Precursor ID
        """
        precursor_id = f"precursor_{len(self.precursors[failure_type])}"
        
        precursor = FailurePrecursor(
            precursor_id=precursor_id,
            failure_type=failure_type,
            indicators=indicators,
            time_window=time_before_failure
        )
        
        # Compute correlations between indicators
        metric_names = list(indicators.keys())
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                corr = self.detect_correlation(metric1, metric2)
                precursor.correlations[f"{metric1}_{metric2}"] = corr
        
        precursor.observed_count = 1
        self.precursors[failure_type].append(precursor)
        
        logger.info(f"Learned failure precursor for {failure_type.value}: {precursor_id}")
        return precursor_id
    
    def predict_failure(
        self,
        current_metrics: Dict[str, float],
        confidence_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Predict potential failures based on current metrics.
        
        Args:
            current_metrics: Current system metrics
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            List of predictions with type, confidence, and time estimate
        """
        predictions = []
        
        for failure_type, precursors in self.precursors.items():
            for precursor in precursors:
                # Check if current metrics match precursor pattern
                match_score = self._match_precursor(current_metrics, precursor.indicators)
                
                if match_score > 0.7 and precursor.confidence() >= confidence_threshold:
                    prediction = {
                        "failure_type": failure_type.value,
                        "precursor_id": precursor.precursor_id,
                        "confidence": precursor.confidence(),
                        "match_score": match_score,
                        "time_to_failure": precursor.time_window,
                        "timestamp": time.time()
                    }
                    predictions.append(prediction)
                    self.predictions_made += 1
                    
                    logger.warning(
                        f"Predicted {failure_type.value} in {precursor.time_window}s "
                        f"(confidence: {precursor.confidence():.2f})"
                    )
        
        # Store active predictions
        self.active_predictions.extend(predictions)
        return predictions
    
    def _match_precursor(
        self,
        current_metrics: Dict[str, float],
        precursor_indicators: Dict[str, Any]
    ) -> float:
        """
        Calculate how well current metrics match precursor pattern.
        
        Returns:
            Match score (0-1)
        """
        if not precursor_indicators:
            return 0.0
        
        matches = 0
        for indicator, condition in precursor_indicators.items():
            if indicator not in current_metrics:
                continue
            
            value = current_metrics[indicator]
            
            # Parse condition (simple threshold for now)
            if isinstance(condition, str):
                if ">" in condition:
                    threshold = float(condition.split(">")[1].strip().rstrip("ms%"))
                    if value > threshold:
                        matches += 1
                elif "<" in condition:
                    threshold = float(condition.split("<")[1].strip().rstrip("ms%"))
                    if value < threshold:
                        matches += 1
            elif isinstance(condition, (int, float)):
                # Direct threshold
                if value > condition:
                    matches += 1
        
        return matches / len(precursor_indicators) if precursor_indicators else 0.0
    
    def allocate_resources(
        self,
        entity_id: str,
        node_id: str,
        priority: float = 0.5,
        min_allocation: float = 0.1
    ) -> ResourceAllocation:
        """
        Allocate resources to an entity.
        
        Args:
            entity_id: Agent or service ID
            node_id: Node where entity runs
            priority: Priority level (0-1)
            min_allocation: Minimum required resources
            
        Returns:
            ResourceAllocation object
        """
        allocation = ResourceAllocation(
            entity_id=entity_id,
            node_id=node_id,
            priority=priority,
            min_allocation=min_allocation
        )
        
        # Initial allocation based on priority
        allocation.cpu_allocation = min_allocation + (1.0 - min_allocation) * priority
        allocation.memory_allocation = min_allocation + (1.0 - min_allocation) * priority
        allocation.energy_allocation = 1.0 * priority
        
        self.allocations[entity_id] = allocation
        return allocation
    
    def adaptive_reallocation(
        self,
        stress_type: StressType,
        affected_nodes: Set[str]
    ) -> Dict[str, Any]:
        """
        Adaptively reallocate resources in response to system stress.
        
        Args:
            stress_type: Type of stress
            affected_nodes: Nodes under stress
            
        Returns:
            Reallocation statistics
        """
        reallocations = []
        
        for entity_id, allocation in self.allocations.items():
            if allocation.node_id in affected_nodes:
                # Reduce allocation for low-priority entities
                if allocation.priority < 0.5:
                    old_alloc = allocation.cpu_allocation
                    allocation.cpu_allocation *= 0.8
                    allocation.memory_allocation *= 0.8
                    allocation.energy_allocation *= 0.8
                    
                    reallocations.append({
                        "entity_id": entity_id,
                        "old_allocation": old_alloc,
                        "new_allocation": allocation.cpu_allocation,
                        "reduction": old_alloc - allocation.cpu_allocation
                    })
        
        self.resources_reallocated += len(reallocations)
        
        logger.info(
            f"Reallocated resources for {len(reallocations)} entities "
            f"due to {stress_type.value}"
        )
        
        return {
            "stress_type": stress_type.value,
            "affected_nodes": list(affected_nodes),
            "reallocations": reallocations,
            "total_freed": sum(r["reduction"] for r in reallocations)
        }
    
    def migrate_agent(
        self,
        agent_id: str,
        from_node: str,
        to_node: str,
        reason: str
    ) -> bool:
        """
        Migrate an agent from one node to another.
        
        Args:
            agent_id: Agent to migrate
            from_node: Source node
            to_node: Destination node
            reason: Migration reason
            
        Returns:
            True if migration successful
        """
        # Update allocation if it exists
        if agent_id in self.allocations:
            self.allocations[agent_id].node_id = to_node
        
        self.agents_migrated += 1
        
        logger.info(f"Migrated agent {agent_id} from {from_node} to {to_node}: {reason}")
        return True
    
    def detect_system_stress(
        self,
        node_metrics: Dict[str, Dict[str, float]]
    ) -> List[SystemStressState]:
        """
        Detect system-wide stress conditions.
        
        Args:
            node_metrics: Metrics for each node {node_id: {metric: value}}
            
        Returns:
            List of detected stress states
        """
        stress_states = []
        
        # CPU pressure detection
        high_cpu_nodes = {
            node_id for node_id, metrics in node_metrics.items()
            if metrics.get("cpu_load", 0) > 0.85
        }
        
        if len(high_cpu_nodes) >= 3:  # Multiple nodes stressed
            stress_id = f"cpu_stress_{int(time.time())}"
            if stress_id not in self.active_stresses:
                severity = len(high_cpu_nodes) / len(node_metrics) if node_metrics else 0
                stress = SystemStressState(
                    stress_type=StressType.CPU_PRESSURE,
                    severity=severity,
                    affected_nodes=high_cpu_nodes
                )
                self.active_stresses[stress_id] = stress
                stress_states.append(stress)
                
                logger.warning(f"Detected CPU pressure affecting {len(high_cpu_nodes)} nodes")
        
        # Memory pressure detection
        high_mem_nodes = {
            node_id for node_id, metrics in node_metrics.items()
            if metrics.get("memory_usage", 0) > 0.90
        }
        
        if len(high_mem_nodes) >= 2:
            stress_id = f"mem_stress_{int(time.time())}"
            if stress_id not in self.active_stresses:
                severity = len(high_mem_nodes) / len(node_metrics) if node_metrics else 0
                stress = SystemStressState(
                    stress_type=StressType.MEMORY_PRESSURE,
                    severity=severity,
                    affected_nodes=high_mem_nodes
                )
                self.active_stresses[stress_id] = stress
                stress_states.append(stress)
        
        # Energy crisis detection
        low_energy_nodes = {
            node_id for node_id, metrics in node_metrics.items()
            if metrics.get("energy", 0) < 0.2
        }
        
        if len(low_energy_nodes) >= 2:
            stress_id = f"energy_crisis_{int(time.time())}"
            if stress_id not in self.active_stresses:
                severity = len(low_energy_nodes) / len(node_metrics) if node_metrics else 0
                stress = SystemStressState(
                    stress_type=StressType.ENERGY_CRISIS,
                    severity=severity,
                    affected_nodes=low_energy_nodes
                )
                self.active_stresses[stress_id] = stress
                stress_states.append(stress)
        
        return stress_states
    
    def coordinate_stress_response(
        self,
        stress_id: str,
        responding_agents: Set[int],
        actions: List[str]
    ):
        """Record coordinated response to system stress"""
        if stress_id in self.active_stresses:
            stress = self.active_stresses[stress_id]
            stress.agents_responding.update(responding_agents)
            stress.mitigation_actions.extend(actions)
    
    def resolve_stress(self, stress_id: str):
        """Mark a stress condition as resolved"""
        if stress_id in self.active_stresses:
            stress = self.active_stresses[stress_id]
            stress.resolved = True
            stress.resolution_time = time.time() - stress.start_time
            
            logger.info(
                f"Resolved {stress.stress_type.value} in {stress.resolution_time:.1f}s "
                f"with {len(stress.agents_responding)} agents"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictive homeostasis statistics"""
        total_precursors = sum(len(p) for p in self.precursors.values())
        active_stress = len([s for s in self.active_stresses.values() if not s.resolved])
        
        return {
            "learned_precursors": total_precursors,
            "predictions_made": self.predictions_made,
            "predictions_correct": self.predictions_correct,
            "prediction_accuracy": (
                self.predictions_correct / self.predictions_made 
                if self.predictions_made > 0 else 0.0
            ),
            "failures_prevented": self.failures_prevented,
            "active_allocations": len(self.allocations),
            "resources_reallocated": self.resources_reallocated,
            "agents_migrated": self.agents_migrated,
            "active_stress_conditions": active_stress,
            "total_stress_events": len(self.active_stresses)
        }
