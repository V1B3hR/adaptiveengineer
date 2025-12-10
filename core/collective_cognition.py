"""
Collective Cognition Engine - The Prefrontal Cortex

This module implements high-level meta-learning and creative synthesis
for the agent collective. It analyzes patterns across the Knowledge Graph
to propose novel, hybrid strategies for zero-day threats.
"""

import logging
import time
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """Types of insights discovered by the cognition engine."""
    PATTERN_CLUSTER = "pattern_cluster"           # Similar problems grouped
    STRATEGY_COMBINATION = "strategy_combination"  # Hybrid strategy proposal
    FAILURE_CORRELATION = "failure_correlation"    # Common failure causes
    SUCCESS_PATTERN = "success_pattern"            # Successful approach patterns
    NOVEL_THREAT = "novel_threat"                  # New threat type identified
    OPTIMIZATION = "optimization"                  # Resource optimization opportunity


@dataclass
class CognitiveInsight:
    """
    An insight discovered through meta-learning.
    
    Attributes:
        insight_id: Unique identifier
        insight_type: Type of insight
        timestamp: When discovered
        description: Human-readable description
        confidence: Confidence level (0.0-1.0)
        supporting_evidence: Evidence supporting this insight
        recommendation: Recommended action
        priority: Priority level (0.0-1.0)
    """
    insight_id: str
    insight_type: InsightType
    timestamp: float
    description: str
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    recommendation: str = ""
    priority: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'insight_id': self.insight_id,
            'insight_type': self.insight_type.value,
            'timestamp': self.timestamp,
            'description': self.description,
            'confidence': self.confidence,
            'supporting_evidence': self.supporting_evidence,
            'recommendation': self.recommendation,
            'priority': self.priority
        }


@dataclass
class HybridStrategy:
    """
    A novel strategy synthesized from existing strategies.
    
    Attributes:
        strategy_id: Unique identifier
        name: Strategy name
        parent_strategies: IDs of strategies combined
        description: What this strategy does
        actions: Combined action sequence
        estimated_effectiveness: Predicted effectiveness
        resource_cost: Estimated resource cost
        novelty_score: How novel this strategy is
    """
    strategy_id: str
    name: str
    parent_strategies: List[str]
    description: str
    actions: List[str] = field(default_factory=list)
    estimated_effectiveness: float = 0.5
    resource_cost: float = 1.0
    novelty_score: float = 0.5


class CollectiveCognitionEngine:
    """
    Collective Cognition Engine - Meta-learning and creative synthesis.
    
    This engine runs periodic analysis on the Knowledge Graph to:
    - Identify patterns across hundreds of incidents
    - Propose novel hybrid strategies
    - Detect emerging threats
    - Optimize resource allocation
    - Generate creative solutions for zero-day problems
    """
    
    def __init__(self, knowledge_graph: Any, evolution_engine: Any = None):
        """
        Initialize the cognition engine.
        
        Args:
            knowledge_graph: Reference to the Knowledge Graph
            evolution_engine: Optional evolution engine for seeding strategies
        """
        self.knowledge_graph = knowledge_graph
        self.evolution_engine = evolution_engine
        
        # Cognitive state
        self.insights: List[CognitiveInsight] = []
        self.hybrid_strategies: List[HybridStrategy] = []
        self.pattern_clusters: Dict[str, List[str]] = defaultdict(list)
        
        # Analysis tracking
        self.last_analysis_time = 0.0
        self.analysis_count = 0
        self.insights_generated = 0
        self.strategies_synthesized = 0
        
        # Configuration
        self.min_incidents_for_analysis = 10
        self.analysis_interval = 300.0  # 5 minutes
        
    def should_analyze(self) -> bool:
        """
        Check if it's time to perform meta-learning analysis.
        
        Returns:
            True if analysis should run
        """
        # Check if enough time has passed
        time_since_last = time.time() - self.last_analysis_time
        if time_since_last < self.analysis_interval:
            return False
            
        # Check if we have enough data
        if self.knowledge_graph.total_incidents < self.min_incidents_for_analysis:
            return False
            
        return True
    
    def perform_meta_learning(self) -> List[CognitiveInsight]:
        """
        Perform comprehensive meta-learning analysis.
        
        This is the main cognitive function that analyzes the entire
        Knowledge Graph to discover patterns, insights, and opportunities.
        
        Returns:
            List of newly discovered insights
        """
        logger.info("=== Starting Meta-Learning Analysis ===")
        self.analysis_count += 1
        self.last_analysis_time = time.time()
        
        new_insights = []
        
        # 1. Cluster similar problem patterns
        pattern_insights = self._cluster_problem_patterns()
        new_insights.extend(pattern_insights)
        
        # 2. Analyze failure correlations
        failure_insights = self._analyze_failure_correlations()
        new_insights.extend(failure_insights)
        
        # 3. Identify success patterns
        success_insights = self._identify_success_patterns()
        new_insights.extend(success_insights)
        
        # 4. Synthesize hybrid strategies
        hybrid_insights = self._synthesize_hybrid_strategies()
        new_insights.extend(hybrid_insights)
        
        # 5. Detect novel threats
        threat_insights = self._detect_novel_threats()
        new_insights.extend(threat_insights)
        
        # 6. Identify optimization opportunities
        optimization_insights = self._identify_optimizations()
        new_insights.extend(optimization_insights)
        
        # Store insights
        self.insights.extend(new_insights)
        self.insights_generated += len(new_insights)
        
        logger.info(f"Meta-learning complete: {len(new_insights)} new insights generated")
        
        return new_insights
    
    def _cluster_problem_patterns(self) -> List[CognitiveInsight]:
        """
        Cluster similar problem patterns together.
        
        This helps identify recurring issue types and their variations.
        """
        insights = []
        
        patterns = list(self.knowledge_graph.problem_patterns.values())
        if len(patterns) < 3:
            return insights
            
        # Simple clustering by feature similarity
        clusters: Dict[str, List[str]] = defaultdict(list)
        
        for pattern in patterns:
            # Create cluster key from dominant features
            feature_keys = sorted(pattern.features.keys())[:3]  # Top 3 features
            cluster_key = "_".join(feature_keys)
            clusters[cluster_key].append(pattern.pattern_id)
            
        # Identify significant clusters
        for cluster_key, pattern_ids in clusters.items():
            if len(pattern_ids) >= 3:  # At least 3 similar patterns
                self.pattern_clusters[cluster_key] = pattern_ids
                
                # Calculate average severity
                avg_severity = sum(
                    self.knowledge_graph.problem_patterns[pid].severity
                    for pid in pattern_ids
                ) / len(pattern_ids)
                
                insight = CognitiveInsight(
                    insight_id=f"insight_cluster_{self.analysis_count}_{cluster_key}",
                    insight_type=InsightType.PATTERN_CLUSTER,
                    timestamp=time.time(),
                    description=f"Identified cluster of {len(pattern_ids)} similar problems",
                    confidence=0.8,
                    supporting_evidence=pattern_ids,
                    recommendation=f"Develop specialized strategy for this problem class",
                    priority=avg_severity
                )
                
                insights.append(insight)
                logger.debug(f"Discovered pattern cluster: {cluster_key} ({len(pattern_ids)} patterns)")
                
        return insights
    
    def _analyze_failure_correlations(self) -> List[CognitiveInsight]:
        """
        Analyze correlations between failures.
        
        Identifies strategies that consistently fail for certain problem types.
        """
        insights = []
        
        # Find edges with strong negative weights (failures)
        failing_combinations = [
            (edge.source_pattern_id, edge.target_strategy_id, edge.weight)
            for edge in self.knowledge_graph.knowledge_edges.values()
            if edge.weight < -0.5 and edge.sample_count >= 3
        ]
        
        if not failing_combinations:
            return insights
            
        # Group by strategy to find strategies with high failure rates
        strategy_failures: Dict[str, List[str]] = defaultdict(list)
        for pattern_id, strategy_id, weight in failing_combinations:
            strategy_failures[strategy_id].append(pattern_id)
            
        for strategy_id, pattern_ids in strategy_failures.items():
            if len(pattern_ids) >= 2:
                strategy = self.knowledge_graph.solution_strategies.get(strategy_id)
                if not strategy:
                    continue
                    
                insight = CognitiveInsight(
                    insight_id=f"insight_failure_{self.analysis_count}_{strategy_id}",
                    insight_type=InsightType.FAILURE_CORRELATION,
                    timestamp=time.time(),
                    description=f"Strategy '{strategy.name}' consistently fails for {len(pattern_ids)} problem types",
                    confidence=0.85,
                    supporting_evidence=[f"pattern_{pid}" for pid in pattern_ids],
                    recommendation=f"Avoid using '{strategy.name}' for these problem types; develop alternative approach",
                    priority=0.8
                )
                
                insights.append(insight)
                
        return insights
    
    def _identify_success_patterns(self) -> List[CognitiveInsight]:
        """
        Identify patterns in successful responses.
        
        Finds strategies that consistently succeed for certain problem types.
        """
        insights = []
        
        # Find edges with strong positive weights (successes)
        successful_combinations = [
            (edge.source_pattern_id, edge.target_strategy_id, edge.weight)
            for edge in self.knowledge_graph.knowledge_edges.values()
            if edge.weight > 0.7 and edge.sample_count >= 3
        ]
        
        if not successful_combinations:
            return insights
            
        # Group by pattern to find reliable solutions
        pattern_solutions: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for pattern_id, strategy_id, weight in successful_combinations:
            pattern_solutions[pattern_id].append((strategy_id, weight))
            
        for pattern_id, solutions in pattern_solutions.items():
            if len(solutions) >= 2:  # Multiple successful strategies
                pattern = self.knowledge_graph.problem_patterns.get(pattern_id)
                if not pattern:
                    continue
                    
                best_strategy_id = max(solutions, key=lambda x: x[1])[0]
                strategy = self.knowledge_graph.solution_strategies.get(best_strategy_id)
                
                insight = CognitiveInsight(
                    insight_id=f"insight_success_{self.analysis_count}_{pattern_id}",
                    insight_type=InsightType.SUCCESS_PATTERN,
                    timestamp=time.time(),
                    description=f"Found reliable solution for '{pattern.description}'",
                    confidence=0.9,
                    supporting_evidence=[s[0] for s in solutions],
                    recommendation=f"Prefer strategy '{strategy.name if strategy else 'unknown'}' for this problem type",
                    priority=0.7
                )
                
                insights.append(insight)
                
        return insights
    
    def _synthesize_hybrid_strategies(self) -> List[CognitiveInsight]:
        """
        Synthesize novel hybrid strategies.
        
        This is the core creative function - combining successful strategies
        to create new approaches for unseen problems.
        """
        insights = []
        
        # Get strategies with good success rates
        successful_strategies = [
            s for s in self.knowledge_graph.solution_strategies.values()
            if s.success_rate > 0.6 and s.success_count >= 3
        ]
        
        if len(successful_strategies) < 2:
            return insights
            
        # Generate hybrid combinations
        max_combinations = 3
        combinations_tried = 0
        
        for i in range(len(successful_strategies)):
            for j in range(i + 1, len(successful_strategies)):
                if combinations_tried >= max_combinations:
                    break
                    
                strategy_a = successful_strategies[i]
                strategy_b = successful_strategies[j]
                
                # Check if strategies are complementary
                if self._are_complementary(strategy_a, strategy_b):
                    hybrid = self._create_hybrid_strategy(strategy_a, strategy_b)
                    self.hybrid_strategies.append(hybrid)
                    self.strategies_synthesized += 1
                    
                    # Seed into evolution engine if available
                    if self.evolution_engine:
                        self._seed_into_evolution(hybrid)
                    
                    insight = CognitiveInsight(
                        insight_id=f"insight_hybrid_{self.analysis_count}_{hybrid.strategy_id}",
                        insight_type=InsightType.STRATEGY_COMBINATION,
                        timestamp=time.time(),
                        description=f"Synthesized hybrid strategy: {hybrid.name}",
                        confidence=hybrid.estimated_effectiveness,
                        supporting_evidence=[strategy_a.strategy_id, strategy_b.strategy_id],
                        recommendation=f"Test this hybrid strategy on novel problems",
                        priority=hybrid.novelty_score
                    )
                    
                    insights.append(insight)
                    logger.info(f"Synthesized hybrid strategy: {hybrid.name}")
                    
                    combinations_tried += 1
                    
        return insights
    
    def _are_complementary(self, strategy_a: Any, strategy_b: Any) -> bool:
        """
        Check if two strategies are complementary.
        
        Strategies are complementary if they address different aspects
        of a problem (e.g., detection + mitigation, or containment + recovery).
        """
        # Simple heuristic: check if action sets have low overlap
        actions_a = set(strategy_a.actions)
        actions_b = set(strategy_b.actions)
        
        if not actions_a or not actions_b:
            return False
            
        overlap = len(actions_a & actions_b) / min(len(actions_a), len(actions_b))
        
        # Complementary if less than 30% overlap
        return overlap < 0.3
    
    def _create_hybrid_strategy(self, strategy_a: Any, strategy_b: Any) -> HybridStrategy:
        """
        Create a hybrid strategy from two parent strategies.
        
        Args:
            strategy_a: First parent strategy
            strategy_b: Second parent strategy
            
        Returns:
            New hybrid strategy
        """
        # Combine names
        name = f"Hybrid_{strategy_a.name}_{strategy_b.name}"
        
        # Merge actions intelligently
        # Phase 1: Actions from strategy A
        # Phase 2: Actions from strategy B
        actions = list(strategy_a.actions) + list(strategy_b.actions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
                
        # Estimate effectiveness (average of parents)
        est_effectiveness = (strategy_a.success_rate + strategy_b.success_rate) / 2
        
        # Combined resource cost
        resource_cost = (strategy_a.resource_cost + strategy_b.resource_cost) * 0.7  # Some efficiency
        
        # Novelty based on how different the parents are
        novelty = 1.0 - (len(set(strategy_a.actions) & set(strategy_b.actions)) / 
                        max(1, min(len(strategy_a.actions), len(strategy_b.actions))))
        
        hybrid_id = f"hybrid_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        
        return HybridStrategy(
            strategy_id=hybrid_id,
            name=name,
            parent_strategies=[strategy_a.strategy_id, strategy_b.strategy_id],
            description=f"Combined approach: {strategy_a.description} + {strategy_b.description}",
            actions=unique_actions,
            estimated_effectiveness=est_effectiveness,
            resource_cost=resource_cost,
            novelty_score=novelty
        )
    
    def _seed_into_evolution(self, hybrid: HybridStrategy) -> None:
        """
        Seed a hybrid strategy into the evolution engine.
        
        This allows the GA to test and evolve the synthesized strategy.
        """
        if not self.evolution_engine:
            return
            
        # Convert hybrid to evolution engine strategy format
        # This would depend on the evolution engine's API
        logger.debug(f"Seeded hybrid strategy {hybrid.name} into evolution engine")
    
    def _detect_novel_threats(self) -> List[CognitiveInsight]:
        """
        Detect novel threat patterns.
        
        Identifies threats that don't match known patterns well.
        """
        insights = []
        
        # Look for patterns with low occurrence but high severity
        novel_patterns = [
            p for p in self.knowledge_graph.problem_patterns.values()
            if p.occurrence_count <= 2 and p.severity > 0.7
        ]
        
        for pattern in novel_patterns:
            # Check if we have any successful strategies for this pattern
            best_strategies = self.knowledge_graph.get_best_strategy(pattern.pattern_id, top_k=1)
            
            if not best_strategies or best_strategies[0][1] < 0.3:  # No good solution
                insight = CognitiveInsight(
                    insight_id=f"insight_novel_{self.analysis_count}_{pattern.pattern_id}",
                    insight_type=InsightType.NOVEL_THREAT,
                    timestamp=time.time(),
                    description=f"Novel threat detected: {pattern.description}",
                    confidence=0.75,
                    supporting_evidence=[pattern.pattern_id],
                    recommendation="Develop new strategies; consider hybrid approaches",
                    priority=pattern.severity
                )
                
                insights.append(insight)
                logger.warning(f"Novel threat detected: {pattern.description}")
                
        return insights
    
    def _identify_optimizations(self) -> List[CognitiveInsight]:
        """
        Identify resource optimization opportunities.
        
        Finds cases where cheaper strategies could be used.
        """
        insights = []
        
        # Look for patterns with multiple successful strategies
        for pattern_id in self.knowledge_graph.problem_patterns.keys():
            strategies = self.knowledge_graph.get_best_strategy(pattern_id, top_k=5)
            
            if len(strategies) >= 2:
                # Find the most cost-effective (good weight, low cost)
                strategy_costs = []
                for strategy_id, weight in strategies:
                    if weight > 0.5:  # Only consider successful strategies
                        strategy = self.knowledge_graph.solution_strategies.get(strategy_id)
                        if strategy:
                            # Cost-effectiveness: weight / resource_cost
                            cost_effectiveness = weight / max(0.1, strategy.resource_cost)
                            strategy_costs.append((strategy_id, cost_effectiveness, strategy))
                            
                if len(strategy_costs) >= 2:
                    # Sort by cost-effectiveness
                    strategy_costs.sort(key=lambda x: x[1], reverse=True)
                    
                    best = strategy_costs[0]
                    insight = CognitiveInsight(
                        insight_id=f"insight_opt_{self.analysis_count}_{pattern_id}",
                        insight_type=InsightType.OPTIMIZATION,
                        timestamp=time.time(),
                        description=f"Resource optimization available for pattern",
                        confidence=0.8,
                        supporting_evidence=[best[0]],
                        recommendation=f"Prefer cost-effective strategy '{best[2].name}'",
                        priority=0.6
                    )
                    
                    insights.append(insight)
                    
        return insights
    
    def get_top_insights(self, limit: int = 10, min_priority: float = 0.5) -> List[CognitiveInsight]:
        """
        Get top priority insights.
        
        Args:
            limit: Maximum number of insights to return
            min_priority: Minimum priority threshold
            
        Returns:
            List of top insights
        """
        # Filter by priority
        relevant = [i for i in self.insights if i.priority >= min_priority]
        
        # Sort by priority * confidence
        relevant.sort(key=lambda i: i.priority * i.confidence, reverse=True)
        
        return relevant[:limit]
    
    def get_hybrid_strategies(self) -> List[HybridStrategy]:
        """Get all synthesized hybrid strategies."""
        return self.hybrid_strategies
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cognitive engine statistics."""
        return {
            'analysis_count': self.analysis_count,
            'insights_generated': self.insights_generated,
            'strategies_synthesized': self.strategies_synthesized,
            'pattern_clusters': len(self.pattern_clusters),
            'last_analysis_time': self.last_analysis_time,
            'insights_by_type': self._count_insights_by_type(),
            'avg_insight_confidence': self._avg_insight_confidence(),
            'high_priority_insights': len([i for i in self.insights if i.priority > 0.7])
        }
    
    def _count_insights_by_type(self) -> Dict[str, int]:
        """Count insights by type."""
        counts = Counter(i.insight_type.value for i in self.insights)
        return dict(counts)
    
    def _avg_insight_confidence(self) -> float:
        """Calculate average insight confidence."""
        if not self.insights:
            return 0.0
        return sum(i.confidence for i in self.insights) / len(self.insights)
