"""
Adaptive Memory & Learning Core - The Hippocampus

This module implements an advanced Knowledge Graph that serves as the
collective's long-term memory and learning center. It learns from both
successes and failures, performs root cause analysis, and serves as an
experience buffer for reinforcement learning.
"""

import logging
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict, Counter
import hashlib

logger = logging.getLogger(__name__)


class OutcomeType(str, Enum):
    """Outcome types for incident responses."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    IN_PROGRESS = "in_progress"
    UNKNOWN = "unknown"


class EdgeWeight(str, Enum):
    """Knowledge graph edge weight types."""
    STRONG_POSITIVE = "strong_positive"      # +1.0
    POSITIVE = "positive"                    # +0.5
    NEUTRAL = "neutral"                      # 0.0
    NEGATIVE = "negative"                    # -0.5
    STRONG_NEGATIVE = "strong_negative"      # -1.0


@dataclass
class ProblemPattern:
    """
    A recognized problem pattern in the system.
    
    Attributes:
        pattern_id: Unique identifier
        signature: Hash signature of the problem
        description: Human-readable description
        features: Key features that define this pattern
        severity: Severity level (0.0-1.0)
        occurrence_count: Number of times seen
        first_seen: Timestamp of first occurrence
        last_seen: Timestamp of last occurrence
    """
    pattern_id: str
    signature: str
    description: str
    features: Dict[str, Any] = field(default_factory=dict)
    severity: float = 0.5
    occurrence_count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def update_occurrence(self) -> None:
        """Update occurrence tracking."""
        self.occurrence_count += 1
        self.last_seen = time.time()


@dataclass
class SolutionStrategy:
    """
    A solution strategy that can be applied to problems.
    
    Attributes:
        strategy_id: Unique identifier
        name: Strategy name
        description: What the strategy does
        actions: List of actions to execute
        resource_cost: Estimated resource cost
        success_count: Number of successful applications
        failure_count: Number of failed applications
        average_resolution_time: Average time to resolve (seconds)
    """
    strategy_id: str
    name: str
    description: str
    actions: List[str] = field(default_factory=list)
    resource_cost: float = 1.0
    success_count: int = 0
    failure_count: int = 0
    average_resolution_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def record_outcome(self, outcome: OutcomeType, resolution_time: float) -> None:
        """Record an outcome for this strategy."""
        if outcome == OutcomeType.SUCCESS:
            self.success_count += 1
            # Update average resolution time
            total_count = self.success_count + self.failure_count
            self.average_resolution_time = (
                (self.average_resolution_time * (total_count - 1) + resolution_time) / total_count
            )
        elif outcome == OutcomeType.FAILURE:
            self.failure_count += 1
        elif outcome == OutcomeType.PARTIAL_SUCCESS:
            # Count as 0.5 success
            self.success_count += 0.5
            self.failure_count += 0.5


@dataclass
class KnowledgeEdge:
    """
    An edge in the knowledge graph connecting problems to solutions.
    
    Attributes:
        source_pattern_id: Problem pattern ID
        target_strategy_id: Solution strategy ID
        weight: Edge weight (strength of association)
        confidence: Confidence level (0.0-1.0)
        sample_count: Number of observations
        last_updated: When this edge was last updated
    """
    source_pattern_id: str
    target_strategy_id: str
    weight: float = 0.0
    confidence: float = 0.5
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def update_weight(self, outcome: OutcomeType, learning_rate: float = 0.1) -> None:
        """
        Update edge weight based on outcome.
        
        Args:
            outcome: The outcome of applying this strategy to this problem
            learning_rate: How quickly to adjust weights
        """
        self.sample_count += 1
        self.last_updated = time.time()
        
        # Adjust weight based on outcome
        if outcome == OutcomeType.SUCCESS:
            # Positive reinforcement
            self.weight = min(1.0, self.weight + learning_rate)
            self.confidence = min(1.0, self.confidence + 0.05)
        elif outcome == OutcomeType.FAILURE:
            # Negative reinforcement
            self.weight = max(-1.0, self.weight - learning_rate)
            self.confidence = min(1.0, self.confidence + 0.05)
        elif outcome == OutcomeType.PARTIAL_SUCCESS:
            # Mild positive reinforcement
            self.weight = min(1.0, self.weight + learning_rate * 0.3)
            self.confidence = min(1.0, self.confidence + 0.02)
            
    def get_weight_type(self) -> EdgeWeight:
        """Get categorical weight type."""
        if self.weight >= 0.75:
            return EdgeWeight.STRONG_POSITIVE
        elif self.weight >= 0.25:
            return EdgeWeight.POSITIVE
        elif self.weight > -0.25:
            return EdgeWeight.NEUTRAL
        elif self.weight > -0.75:
            return EdgeWeight.NEGATIVE
        else:
            return EdgeWeight.STRONG_NEGATIVE


@dataclass
class IncidentExperience:
    """
    A complete experience for reinforcement learning.
    
    This represents (state, action, reward, next_state) tuple
    for RL algorithms.
    """
    experience_id: str
    timestamp: float
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    outcome: OutcomeType
    pattern_id: Optional[str] = None
    strategy_id: Optional[str] = None
    
    def to_rl_tuple(self) -> Tuple[Dict, str, float, Dict]:
        """Convert to standard RL tuple format."""
        return (self.state, self.action, self.reward, self.next_state)


@dataclass
class RootCauseAnalysis:
    """
    Root cause analysis for a failure.
    
    Attributes:
        analysis_id: Unique identifier
        incident_id: Related incident ID
        pattern_id: Problem pattern ID
        strategy_id: Failed strategy ID
        timestamp: When analysis was performed
        suspected_causes: List of suspected root causes
        confidence: Confidence in the analysis
        recommendations: Recommendations to avoid future failures
    """
    analysis_id: str
    incident_id: str
    pattern_id: str
    strategy_id: str
    timestamp: float
    suspected_causes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    recommendations: List[str] = field(default_factory=list)


class KnowledgeGraph:
    """
    Advanced Knowledge Graph for collective learning and memory.
    
    This serves as the "Hippocampus" - the long-term memory and learning
    center of the agent collective. It maintains:
    - Problem patterns and their features
    - Solution strategies and their effectiveness
    - Weighted edges representing learned associations
    - RL experience buffer for training
    - Root cause analyses for failures
    """
    
    def __init__(self, learning_rate: float = 0.1, max_experiences: int = 10000):
        """
        Initialize the knowledge graph.
        
        Args:
            learning_rate: How quickly to update edge weights
            max_experiences: Maximum experiences to retain for RL
        """
        self.learning_rate = learning_rate
        self.max_experiences = max_experiences
        
        # Core graph structures
        self.problem_patterns: Dict[str, ProblemPattern] = {}
        self.solution_strategies: Dict[str, SolutionStrategy] = {}
        self.knowledge_edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        
        # RL experience buffer
        self.experiences: List[IncidentExperience] = []
        
        # Root cause analyses
        self.root_cause_analyses: List[RootCauseAnalysis] = []
        
        # Statistics
        self.total_incidents = 0
        self.successful_resolutions = 0
        self.failed_resolutions = 0
        self.partial_successes = 0
        
    def recognize_pattern(self, 
                         description: str,
                         features: Dict[str, Any],
                         severity: float = 0.5) -> str:
        """
        Recognize or create a problem pattern.
        
        Args:
            description: Problem description
            features: Key features of the problem
            severity: Severity level
            
        Returns:
            Pattern ID
        """
        # Generate signature from features
        feature_str = json.dumps(features, sort_keys=True)
        signature = hashlib.md5(feature_str.encode()).hexdigest()[:16]
        
        # Check if pattern exists
        for pattern in self.problem_patterns.values():
            if pattern.signature == signature:
                pattern.update_occurrence()
                logger.debug(f"Recognized existing pattern: {pattern.pattern_id}")
                return pattern.pattern_id
                
        # Create new pattern
        pattern_id = f"pattern_{signature}"
        pattern = ProblemPattern(
            pattern_id=pattern_id,
            signature=signature,
            description=description,
            features=features,
            severity=severity
        )
        self.problem_patterns[pattern_id] = pattern
        logger.info(f"Created new problem pattern: {pattern_id}")
        return pattern_id
    
    def register_strategy(self,
                         name: str,
                         description: str,
                         actions: List[str],
                         resource_cost: float = 1.0) -> str:
        """
        Register a solution strategy.
        
        Args:
            name: Strategy name
            description: What it does
            actions: List of actions
            resource_cost: Resource cost estimate
            
        Returns:
            Strategy ID
        """
        strategy_id = f"strategy_{hashlib.md5(name.encode()).hexdigest()[:16]}"
        
        if strategy_id not in self.solution_strategies:
            strategy = SolutionStrategy(
                strategy_id=strategy_id,
                name=name,
                description=description,
                actions=actions,
                resource_cost=resource_cost
            )
            self.solution_strategies[strategy_id] = strategy
            logger.info(f"Registered new strategy: {name}")
            
        return strategy_id
    
    def record_incident(self,
                       pattern_id: str,
                       strategy_id: str,
                       outcome: OutcomeType,
                       resolution_time: float,
                       state: Optional[Dict[str, Any]] = None,
                       next_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Record an incident response and its outcome.
        
        This is the core learning mechanism - every incident teaches
        the system whether a strategy works for a problem pattern.
        
        Args:
            pattern_id: The problem pattern
            strategy_id: The strategy applied
            outcome: The outcome
            resolution_time: Time to resolve (seconds)
            state: System state before action (for RL)
            next_state: System state after action (for RL)
            
        Returns:
            Experience ID
        """
        self.total_incidents += 1
        
        # Update strategy statistics
        if strategy_id in self.solution_strategies:
            strategy = self.solution_strategies[strategy_id]
            strategy.record_outcome(outcome, resolution_time)
            
        # Update or create knowledge edge
        edge_key = (pattern_id, strategy_id)
        if edge_key not in self.knowledge_edges:
            self.knowledge_edges[edge_key] = KnowledgeEdge(
                source_pattern_id=pattern_id,
                target_strategy_id=strategy_id
            )
            
        edge = self.knowledge_edges[edge_key]
        edge.update_weight(outcome, self.learning_rate)
        
        # Update outcome statistics
        if outcome == OutcomeType.SUCCESS:
            self.successful_resolutions += 1
        elif outcome == OutcomeType.FAILURE:
            self.failed_resolutions += 1
        elif outcome == OutcomeType.PARTIAL_SUCCESS:
            self.partial_successes += 1
            
        # Create RL experience
        if state is not None and next_state is not None:
            reward = self._calculate_reward(outcome, resolution_time)
            experience = IncidentExperience(
                experience_id=f"exp_{int(time.time() * 1000)}_{self.total_incidents}",
                timestamp=time.time(),
                state=state,
                action=strategy_id,
                reward=reward,
                next_state=next_state,
                outcome=outcome,
                pattern_id=pattern_id,
                strategy_id=strategy_id
            )
            self.experiences.append(experience)
            
            # Limit experience buffer size
            if len(self.experiences) > self.max_experiences:
                self.experiences = self.experiences[-self.max_experiences:]
                
            logger.debug(f"Recorded experience with reward {reward:.3f}")
            
        # Trigger root cause analysis for failures
        if outcome == OutcomeType.FAILURE:
            self._perform_root_cause_analysis(pattern_id, strategy_id)
            
        return f"incident_{self.total_incidents}"
    
    def _calculate_reward(self, outcome: OutcomeType, resolution_time: float) -> float:
        """
        Calculate RL reward based on outcome.
        
        Args:
            outcome: The outcome
            resolution_time: Time to resolve
            
        Returns:
            Reward value (typically -1.0 to 1.0)
        """
        if outcome == OutcomeType.SUCCESS:
            # High reward, penalized slightly for slow resolution
            base_reward = 1.0
            time_penalty = min(0.3, resolution_time / 300.0)  # Max 300 seconds
            return base_reward - time_penalty
        elif outcome == OutcomeType.FAILURE:
            # Strong negative reward
            return -1.0
        elif outcome == OutcomeType.PARTIAL_SUCCESS:
            # Mild positive reward
            return 0.3
        else:
            return 0.0
    
    def _perform_root_cause_analysis(self, pattern_id: str, strategy_id: str) -> str:
        """
        Perform root cause analysis for a failed strategy.
        
        Args:
            pattern_id: Problem pattern
            strategy_id: Failed strategy
            
        Returns:
            Analysis ID
        """
        analysis_id = f"rca_{int(time.time() * 1000)}"
        
        # Analyze why the strategy failed
        suspected_causes = []
        recommendations = []
        
        pattern = self.problem_patterns.get(pattern_id)
        strategy = self.solution_strategies.get(strategy_id)
        
        if pattern and strategy:
            # Check if strategy has high failure rate
            if strategy.failure_count > strategy.success_count:
                suspected_causes.append("Strategy has historically high failure rate")
                recommendations.append("Consider alternative strategies with better track record")
                
            # Check if problem severity is too high for this strategy
            if pattern.severity > 0.8 and strategy.resource_cost < 2.0:
                suspected_causes.append("Problem severity may exceed strategy capability")
                recommendations.append("Apply more resource-intensive strategies for severe problems")
                
            # Check if pattern is novel
            if pattern.occurrence_count <= 2:
                suspected_causes.append("Novel problem pattern with limited data")
                recommendations.append("Gather more data and try diverse strategies")
                
        analysis = RootCauseAnalysis(
            analysis_id=analysis_id,
            incident_id=f"incident_{self.total_incidents}",
            pattern_id=pattern_id,
            strategy_id=strategy_id,
            timestamp=time.time(),
            suspected_causes=suspected_causes,
            confidence=0.7,
            recommendations=recommendations
        )
        
        self.root_cause_analyses.append(analysis)
        logger.warning(f"Root cause analysis: {len(suspected_causes)} causes identified")
        
        return analysis_id
    
    def get_best_strategy(self, pattern_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the best strategies for a problem pattern.
        
        Args:
            pattern_id: Problem pattern ID
            top_k: Number of top strategies to return
            
        Returns:
            List of (strategy_id, weight) tuples, sorted by weight
        """
        # Find all edges from this pattern
        relevant_edges = [
            (edge.target_strategy_id, edge.weight)
            for edge in self.knowledge_edges.values()
            if edge.source_pattern_id == pattern_id
        ]
        
        # Sort by weight (descending)
        relevant_edges.sort(key=lambda x: x[1], reverse=True)
        
        return relevant_edges[:top_k]
    
    def get_experiences_for_pattern(self, pattern_id: str) -> List[IncidentExperience]:
        """Get all RL experiences related to a pattern."""
        return [exp for exp in self.experiences if exp.pattern_id == pattern_id]
    
    def get_rl_buffer(self) -> List[Tuple[Dict, str, float, Dict]]:
        """
        Get the RL experience buffer in standard format.
        
        Returns:
            List of (state, action, reward, next_state) tuples
        """
        return [exp.to_rl_tuple() for exp in self.experiences]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        return {
            'total_patterns': len(self.problem_patterns),
            'total_strategies': len(self.solution_strategies),
            'total_edges': len(self.knowledge_edges),
            'total_incidents': self.total_incidents,
            'successful_resolutions': self.successful_resolutions,
            'failed_resolutions': self.failed_resolutions,
            'partial_successes': self.partial_successes,
            'success_rate': self.successful_resolutions / max(1, self.total_incidents),
            'experiences_recorded': len(self.experiences),
            'root_cause_analyses': len(self.root_cause_analyses),
            'most_common_patterns': self._get_most_common_patterns(5),
            'best_strategies': self._get_best_strategies(5)
        }
    
    def _get_most_common_patterns(self, limit: int) -> List[Dict[str, Any]]:
        """Get most frequently occurring patterns."""
        patterns = sorted(
            self.problem_patterns.values(),
            key=lambda p: p.occurrence_count,
            reverse=True
        )[:limit]
        
        return [
            {
                'pattern_id': p.pattern_id,
                'description': p.description,
                'occurrence_count': p.occurrence_count,
                'severity': p.severity
            }
            for p in patterns
        ]
    
    def _get_best_strategies(self, limit: int) -> List[Dict[str, Any]]:
        """Get strategies with best success rates."""
        strategies = sorted(
            self.solution_strategies.values(),
            key=lambda s: s.success_rate,
            reverse=True
        )[:limit]
        
        return [
            {
                'strategy_id': s.strategy_id,
                'name': s.name,
                'success_rate': s.success_rate,
                'success_count': s.success_count,
                'failure_count': s.failure_count
            }
            for s in strategies
        ]
    
    def export_graph(self, filepath: str) -> None:
        """Export knowledge graph to JSON file."""
        data = {
            'patterns': {k: asdict(v) for k, v in self.problem_patterns.items()},
            'strategies': {k: asdict(v) for k, v in self.solution_strategies.items()},
            'edges': {
                f"{k[0]}_{k[1]}": asdict(v) 
                for k, v in self.knowledge_edges.items()
            },
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exported knowledge graph to {filepath}")
    
    def import_graph(self, filepath: str) -> None:
        """Import knowledge graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Import patterns
        for pattern_data in data.get('patterns', {}).values():
            pattern = ProblemPattern(**pattern_data)
            self.problem_patterns[pattern.pattern_id] = pattern
            
        # Import strategies
        for strategy_data in data.get('strategies', {}).values():
            strategy = SolutionStrategy(**strategy_data)
            self.solution_strategies[strategy.strategy_id] = strategy
            
        # Import edges
        for edge_data in data.get('edges', {}).values():
            edge = KnowledgeEdge(**edge_data)
            key = (edge.source_pattern_id, edge.target_strategy_id)
            self.knowledge_edges[key] = edge
            
        logger.info(f"Imported knowledge graph from {filepath}")
