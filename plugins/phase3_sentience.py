"""
Phase 3: Collective Sentience & Proactive Intelligence Plugin

Integrates:
- Digital Sensory Cortex
- Adaptive Memory & Learning Core (Knowledge Graph)
- Adaptive Immune Response
- Collective Cognition Engine
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from core.plugin_base import PluginBase
from core.sensory_cortex import (
    SensoryCortex,
    SenseType,
    AnomalyLevel,
    SensoryInput,
)
from core.knowledge_graph import (
    KnowledgeGraph,
    OutcomeType,
    ProblemPattern,
    SolutionStrategy,
)
from core.immune_system import (
    AdaptiveImmuneSystem,
    ThreatLevel,
    ResponseStatus,
)
from core.collective_cognition import (
    CollectiveCognitionEngine,
    InsightType,
    CognitiveInsight,
)

logger = logging.getLogger(__name__)


class Phase3SentiencePlugin(PluginBase):
    """
    Phase 3 Plugin - Collective Sentience & Proactive Intelligence

    This plugin integrates all Phase 3 features into the agent collective:
    - Multi-modal sensory system
    - Knowledge graph for learning
    - Immune system for threat response
    - Collective cognition for meta-learning
    """

    def __init__(self):
        """Initialize Phase 3 plugin."""
        super().__init__("Phase3Sentience")

        # Core systems
        self.sensory_cortex: Optional[SensoryCortex] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.immune_system: Optional[AdaptiveImmuneSystem] = None
        self.cognition_engine: Optional[CollectiveCognitionEngine] = None

        # Per-node tracking
        self.node_sensory_data: Dict[int, List[SensoryInput]] = {}
        self.node_threat_levels: Dict[int, float] = {}
        self.node_response_ids: Dict[int, List[str]] = {}

        # System-wide state
        self.last_cognition_time = 0.0
        self.total_threats_detected = 0
        self.total_responses_initiated = 0

    def initialize(self, node: Any) -> None:
        """
        Initialize Phase 3 systems for a node.

        Args:
            node: The agent node
        """
        node_id = getattr(node, "node_id", 0)

        # Initialize on first node only (shared systems)
        if not self.sensory_cortex:
            logger.info("Initializing Phase 3 Sentience systems...")

            # Initialize sensory cortex
            self.sensory_cortex = SensoryCortex()
            self.sensory_cortex.deploy_sensors(
                sight_count=2,
                hearing_count=2,
                smell_count=2,
                taste_count=1,
                touch_count=3,
            )

            # Initialize knowledge graph
            self.knowledge_graph = KnowledgeGraph(
                learning_rate=0.1, max_experiences=10000
            )

            # Register some default strategies
            self._register_default_strategies()

            # Initialize immune system
            self.immune_system = AdaptiveImmuneSystem(self.knowledge_graph)
            self.immune_system.initialize_agents(
                neutrophil_count=5, macrophage_count=3, reinforcement_squads=2
            )

            # Initialize cognition engine
            self.cognition_engine = CollectiveCognitionEngine(
                self.knowledge_graph,
                evolution_engine=None,  # Can be connected later
            )

            logger.info("Phase 3 Sentience systems initialized successfully")

        # Initialize per-node tracking
        self.node_sensory_data[node_id] = []
        self.node_threat_levels[node_id] = 0.0
        self.node_response_ids[node_id] = []

    def _register_default_strategies(self) -> None:
        """Register default solution strategies."""
        strategies = [
            {
                "name": "Isolate and Scan",
                "description": "Isolate node and perform security scan",
                "actions": [
                    "isolate_node",
                    "run_security_scan",
                    "remove_threats",
                ],
                "resource_cost": 1.5,
            },
            {
                "name": "Restart and Restore",
                "description": "Restart services and restore from backup",
                "actions": [
                    "stop_services",
                    "restore_backup",
                    "restart_services",
                ],
                "resource_cost": 2.0,
            },
            {
                "name": "Traffic Reroute",
                "description": "Reroute traffic around affected node",
                "actions": [
                    "update_routing",
                    "redirect_traffic",
                    "monitor_flow",
                ],
                "resource_cost": 1.0,
            },
            {
                "name": "Resource Scale",
                "description": "Scale resources to handle load",
                "actions": [
                    "provision_resources",
                    "distribute_load",
                    "optimize_config",
                ],
                "resource_cost": 2.5,
            },
            {
                "name": "Containment Protocol",
                "description": "Contain and quarantine threat",
                "actions": [
                    "quarantine_node",
                    "block_connections",
                    "alert_neighbors",
                ],
                "resource_cost": 1.2,
            },
        ]

        for strat in strategies:
            self.knowledge_graph.register_strategy(
                name=strat["name"],
                description=strat["description"],
                actions=strat["actions"],
                resource_cost=strat["resource_cost"],
            )

    def get_state_variables(self) -> Dict[str, Any]:
        """
        Get Phase 3 state variables.

        Returns:
            Dictionary of state variable names and their types
        """
        return {
            # Sensory state
            "total_sensors": float,
            "sensory_detections": float,
            "anomaly_level": float,
            "sight_detections": float,
            "hearing_detections": float,
            "smell_detections": float,
            "taste_detections": float,
            "touch_detections": float,
            # Knowledge state
            "patterns_learned": float,
            "strategies_known": float,
            "total_incidents": float,
            "success_rate": float,
            "experiences_recorded": float,
            # Immune state
            "active_responses": float,
            "threat_level": float,
            "neutrophils_active": float,
            "macrophages_active": float,
            "memory_cells": float,
            # Cognitive state
            "insights_generated": float,
            "hybrid_strategies": float,
            "last_meta_learning": float,
            "cognitive_priority": float,
        }

    def update(self, delta_time: float) -> None:
        """
        Update Phase 3 systems.

        Args:
            delta_time: Time since last update
        """
        if not self.sensory_cortex or not self.knowledge_graph:
            return

        # Periodic meta-learning analysis
        if self.cognition_engine and self.cognition_engine.should_analyze():
            logger.info("Performing meta-learning analysis...")
            insights = self.cognition_engine.perform_meta_learning()
            logger.info(f"Generated {len(insights)} new insights")

            # Process high-priority insights
            for insight in insights[:3]:  # Top 3
                self._process_insight(insight)

    def _process_insight(self, insight: CognitiveInsight) -> None:
        """Process a cognitive insight."""
        if insight.insight_type == InsightType.STRATEGY_COMBINATION:
            logger.info(
                f"New hybrid strategy available: {insight.description}"
            )
        elif insight.insight_type == InsightType.NOVEL_THREAT:
            logger.warning(f"Novel threat detected: {insight.description}")
        elif insight.insight_type == InsightType.FAILURE_CORRELATION:
            logger.warning(
                f"Failure pattern identified: {insight.description}"
            )

    def get_available_actions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available Phase 3 actions.

        Returns:
            Dictionary of action names and their specifications
        """
        return {
            "sense_environment": {
                "description": "Perform multi-modal sensory scan",
                "parameters": {"target_node": str, "environment": Any},
            },
            "detect_threat": {
                "description": "Detect and classify a threat",
                "parameters": {
                    "threat_features": Dict[str, Any],
                    "threat_type": str,
                    "severity": float,
                },
            },
            "initiate_response": {
                "description": "Initiate immune response to threat",
                "parameters": {
                    "threat_signature": str,
                    "target_node": str,
                    "threat_level": str,
                },
            },
            "learn_from_incident": {
                "description": "Record incident outcome for learning",
                "parameters": {
                    "pattern_description": str,
                    "pattern_features": Dict[str, Any],
                    "strategy_name": str,
                    "outcome": str,
                    "resolution_time": float,
                },
            },
            "request_best_strategy": {
                "description": "Get best known strategy for a problem",
                "parameters": {
                    "problem_description": str,
                    "problem_features": Dict[str, Any],
                },
            },
            "deploy_reinforcements": {
                "description": "Deploy reinforcement squad to stressed node",
                "parameters": {
                    "target_node": str,
                    "stress_metrics": Dict[str, float],
                },
            },
            "get_cognitive_insights": {
                "description": "Get top cognitive insights",
                "parameters": {"limit": int, "min_priority": float},
            },
        }

    def sense_environment(
        self, node: Any, target_node: str, environment: Any
    ) -> List[SensoryInput]:
        """
        Perform multi-modal sensory scan.

        Args:
            node: The sensing agent
            target_node: Node to sense
            environment: The Living Graph environment

        Returns:
            List of sensory inputs detected
        """
        if not self.sensory_cortex:
            return []

        node_id = getattr(node, "node_id", 0)

        # Perform sensing
        inputs = self.sensory_cortex.sense_node(environment, target_node)

        # Record detections
        self.node_sensory_data[node_id].extend(inputs)

        # Update threat level
        if inputs:
            max_threat = max(
                self._anomaly_to_threat_score(inp.anomaly_level)
                for inp in inputs
            )
            self.node_threat_levels[node_id] = max(
                self.node_threat_levels.get(node_id, 0.0), max_threat
            )

        return inputs

    def _anomaly_to_threat_score(self, anomaly_level: AnomalyLevel) -> float:
        """Convert anomaly level to threat score."""
        mapping = {
            AnomalyLevel.NONE: 0.0,
            AnomalyLevel.LOW: 0.3,
            AnomalyLevel.MEDIUM: 0.6,
            AnomalyLevel.HIGH: 0.8,
            AnomalyLevel.CRITICAL: 1.0,
        }
        return mapping.get(anomaly_level, 0.0)

    def _anomaly_to_threat_level(
        self, anomaly_level: AnomalyLevel
    ) -> ThreatLevel:
        """Convert anomaly level to threat level."""
        mapping = {
            AnomalyLevel.NONE: ThreatLevel.LOW,
            AnomalyLevel.LOW: ThreatLevel.LOW,
            AnomalyLevel.MEDIUM: ThreatLevel.MEDIUM,
            AnomalyLevel.HIGH: ThreatLevel.HIGH,
            AnomalyLevel.CRITICAL: ThreatLevel.CRITICAL,
        }
        return mapping.get(anomaly_level, ThreatLevel.LOW)

    def detect_threat(
        self,
        node: Any,
        threat_features: Dict[str, Any],
        threat_type: str,
        severity: float,
    ) -> Optional[str]:
        """
        Detect and classify a threat.

        Args:
            node: The detecting agent
            threat_features: Features of the threat
            threat_type: Type of threat
            severity: Severity (0.0-1.0)

        Returns:
            Threat signature ID
        """
        if not self.immune_system:
            return None

        self.total_threats_detected += 1

        # Detect threat (checks memory cells)
        threat_sig = self.immune_system.detect_threat(
            threat_features, threat_type, severity
        )

        return threat_sig

    def initiate_response(
        self,
        node: Any,
        threat_signature: str,
        target_node: str,
        threat_level: str,
    ) -> str:
        """
        Initiate immune response to threat.

        Args:
            node: The initiating agent
            threat_signature: Threat signature
            target_node: Node under threat
            threat_level: Threat level (low/medium/high/critical)

        Returns:
            Response ID
        """
        if not self.immune_system:
            return ""

        node_id = getattr(node, "node_id", 0)

        # Convert string to enum
        level_map = {
            "low": ThreatLevel.LOW,
            "medium": ThreatLevel.MEDIUM,
            "high": ThreatLevel.HIGH,
            "critical": ThreatLevel.CRITICAL,
        }
        threat_level_enum = level_map.get(
            threat_level.lower(), ThreatLevel.MEDIUM
        )

        # Initiate response
        response_id = self.immune_system.initiate_response(
            threat_signature, target_node, threat_level_enum
        )

        # Track response
        self.node_response_ids[node_id].append(response_id)
        self.total_responses_initiated += 1

        return response_id

    def learn_from_incident(
        self,
        node: Any,
        pattern_description: str,
        pattern_features: Dict[str, Any],
        strategy_name: str,
        outcome: str,
        resolution_time: float,
    ) -> str:
        """
        Record incident outcome for learning.

        Args:
            node: The learning agent
            pattern_description: Description of the problem
            pattern_features: Problem features
            strategy_name: Strategy used
            outcome: Outcome (success/failure/partial_success)
            resolution_time: Time to resolve

        Returns:
            Incident ID
        """
        if not self.knowledge_graph:
            return ""

        # Recognize or create pattern
        pattern_id = self.knowledge_graph.recognize_pattern(
            pattern_description,
            pattern_features,
            severity=pattern_features.get("severity", 0.5),
        )

        # Get strategy ID (find by name)
        strategy_id = None
        for sid, strat in self.knowledge_graph.solution_strategies.items():
            if strat.name == strategy_name:
                strategy_id = sid
                break

        if not strategy_id:
            # Register new strategy if not found
            strategy_id = self.knowledge_graph.register_strategy(
                name=strategy_name,
                description=f"Strategy: {strategy_name}",
                actions=[strategy_name],
                resource_cost=1.0,
            )

        # Convert outcome string to enum
        outcome_map = {
            "success": OutcomeType.SUCCESS,
            "failure": OutcomeType.FAILURE,
            "partial_success": OutcomeType.PARTIAL_SUCCESS,
            "partial": OutcomeType.PARTIAL_SUCCESS,
        }
        outcome_enum = outcome_map.get(outcome.lower(), OutcomeType.UNKNOWN)

        # Record incident
        incident_id = self.knowledge_graph.record_incident(
            pattern_id, strategy_id, outcome_enum, resolution_time
        )

        return incident_id

    def request_best_strategy(
        self,
        node: Any,
        problem_description: str,
        problem_features: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Get best known strategy for a problem.

        Args:
            node: The requesting agent
            problem_description: Description of the problem
            problem_features: Problem features

        Returns:
            Strategy information or None
        """
        if not self.knowledge_graph:
            return None

        # Recognize pattern
        pattern_id = self.knowledge_graph.recognize_pattern(
            problem_description,
            problem_features,
            severity=problem_features.get("severity", 0.5),
        )

        # Get best strategies
        best_strategies = self.knowledge_graph.get_best_strategy(
            pattern_id, top_k=1
        )

        if not best_strategies:
            return None

        strategy_id, weight = best_strategies[0]
        strategy = self.knowledge_graph.solution_strategies.get(strategy_id)

        if not strategy:
            return None

        return {
            "strategy_id": strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "actions": strategy.actions,
            "weight": weight,
            "success_rate": strategy.success_rate,
            "resource_cost": strategy.resource_cost,
        }

    def deploy_reinforcements(
        self, node: Any, target_node: str, stress_metrics: Dict[str, float]
    ) -> str:
        """
        Deploy reinforcement squad to stressed node.

        Args:
            node: The requesting agent
            target_node: Node under stress
            stress_metrics: Stress metrics

        Returns:
            Squad ID
        """
        if not self.immune_system:
            return ""

        squad_id = self.immune_system.deploy_reinforcements(
            target_node, stress_metrics
        )

        return squad_id

    def get_cognitive_insights(
        self, node: Any, limit: int = 10, min_priority: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get top cognitive insights.

        Args:
            node: The requesting agent
            limit: Maximum number of insights
            min_priority: Minimum priority threshold

        Returns:
            List of insights
        """
        if not self.cognition_engine:
            return []

        insights = self.cognition_engine.get_top_insights(limit, min_priority)

        return [insight.to_dict() for insight in insights]

    def get_state(self, node: Any) -> Dict[str, Any]:
        """
        Get current Phase 3 state for a node.

        Args:
            node: The agent node

        Returns:
            Current state values
        """
        node_id = getattr(node, "node_id", 0)

        # Get system statistics
        sensory_stats = (
            self.sensory_cortex.get_statistics() if self.sensory_cortex else {}
        )
        knowledge_stats = (
            self.knowledge_graph.get_statistics()
            if self.knowledge_graph
            else {}
        )
        immune_stats = (
            self.immune_system.get_statistics() if self.immune_system else {}
        )
        cognitive_stats = (
            self.cognition_engine.get_statistics()
            if self.cognition_engine
            else {}
        )

        return {
            # Sensory state
            "total_sensors": sensory_stats.get("total_sensors", 0),
            "sensory_detections": sensory_stats.get("total_detections", 0),
            "anomaly_level": self.node_threat_levels.get(node_id, 0.0),
            "sight_detections": sensory_stats.get("sight_detections", 0),
            "hearing_detections": sensory_stats.get("hearing_detections", 0),
            "smell_detections": sensory_stats.get("smell_detections", 0),
            "taste_detections": sensory_stats.get("taste_detections", 0),
            "touch_detections": sensory_stats.get("touch_detections", 0),
            # Knowledge state
            "patterns_learned": knowledge_stats.get("total_patterns", 0),
            "strategies_known": knowledge_stats.get("total_strategies", 0),
            "total_incidents": knowledge_stats.get("total_incidents", 0),
            "success_rate": knowledge_stats.get("success_rate", 0.0),
            "experiences_recorded": knowledge_stats.get(
                "experiences_recorded", 0
            ),
            # Immune state
            "active_responses": immune_stats.get("active_responses", 0),
            "threat_level": self.node_threat_levels.get(node_id, 0.0),
            "neutrophils_active": immune_stats.get("neutrophils", 0),
            "macrophages_active": immune_stats.get("macrophages", 0),
            "memory_cells": immune_stats.get("memory_cells", 0),
            # Cognitive state
            "insights_generated": cognitive_stats.get("insights_generated", 0),
            "hybrid_strategies": cognitive_stats.get(
                "strategies_synthesized", 0
            ),
            "last_meta_learning": cognitive_stats.get(
                "last_analysis_time", 0.0
            ),
            "cognitive_priority": len(
                cognitive_stats.get("insights_by_type", {})
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive Phase 3 summary.

        Returns:
            Summary of all Phase 3 systems
        """
        summary = {
            "plugin_name": self.name,
            "systems_active": bool(
                self.sensory_cortex
                and self.knowledge_graph
                and self.immune_system
                and self.cognition_engine
            ),
        }

        if self.sensory_cortex:
            summary["sensory_cortex"] = self.sensory_cortex.get_statistics()

        if self.knowledge_graph:
            summary["knowledge_graph"] = self.knowledge_graph.get_statistics()

        if self.immune_system:
            summary["immune_system"] = self.immune_system.get_statistics()

        if self.cognition_engine:
            summary["cognition_engine"] = (
                self.cognition_engine.get_statistics()
            )

        summary["total_threats_detected"] = self.total_threats_detected
        summary["total_responses_initiated"] = self.total_responses_initiated

        return summary
