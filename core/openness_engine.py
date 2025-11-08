"""
Openness Engine: Environmental Adaptation and Reorganization (AL Principle #5)

Enables agents to adapt, evolve, and reorganize in response to unpredictable,
open-ended environments. Supports dynamic role changes, environmental learning,
and self-organization without predefined structures.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import random

logger = logging.getLogger('openness_engine')


class EnvironmentType(str, Enum):
    """Types of environments agents can encounter"""
    STABLE = "stable"
    DYNAMIC = "dynamic"
    CHAOTIC = "chaotic"
    ADVERSARIAL = "adversarial"
    NOVEL = "novel"  # Completely new, unseen conditions


class AdaptationStrategy(str, Enum):
    """Strategies for adapting to environmental changes"""
    CONSERVATIVE = "conservative"  # Minimal changes, preserve existing structure
    MODERATE = "moderate"  # Balanced adaptation
    AGGRESSIVE = "aggressive"  # Rapid reorganization
    EXPLORATORY = "exploratory"  # Try novel approaches


class OrganizationPattern(str, Enum):
    """Self-organization patterns that can emerge"""
    HIERARCHICAL = "hierarchical"
    DISTRIBUTED = "distributed"
    MESH = "mesh"
    CLUSTERED = "clustered"
    HYBRID = "hybrid"


@dataclass
class EnvironmentalState:
    """Represents current environmental conditions"""
    environment_type: EnvironmentType
    volatility: float  # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0
    novelty: float  # 0.0 to 1.0 (how different from known patterns)
    threat_level: float  # 0.0 to 1.0
    resource_availability: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    
    def get_stability_score(self) -> float:
        """Calculate overall environmental stability (0=chaotic, 1=stable)"""
        return 1.0 - (self.volatility + self.threat_level) / 2.0


@dataclass
class AdaptationResult:
    """Result of an adaptation attempt"""
    strategy_used: AdaptationStrategy
    success: bool
    confidence: float
    new_organization: Optional[OrganizationPattern]
    changes_made: List[str]
    adaptation_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class EnvironmentalPattern:
    """Learned pattern about environmental conditions"""
    pattern_id: str
    environment_type: EnvironmentType
    characteristics: Dict[str, float]
    successful_adaptations: List[AdaptationStrategy]
    failed_adaptations: List[AdaptationStrategy]
    encounter_count: int
    last_seen: float
    
    def update_success(self, strategy: AdaptationStrategy):
        """Record successful adaptation strategy"""
        if strategy not in self.successful_adaptations:
            self.successful_adaptations.append(strategy)
        self.encounter_count += 1
        self.last_seen = time.time()
    
    def update_failure(self, strategy: AdaptationStrategy):
        """Record failed adaptation strategy"""
        if strategy not in self.failed_adaptations:
            self.failed_adaptations.append(strategy)
        self.encounter_count += 1
        self.last_seen = time.time()


class OpennessEngine:
    """
    Manages agent adaptation to unpredictable, open-ended environments.
    
    Key capabilities:
    - Environmental sensing and classification
    - Adaptive strategy selection
    - Self-reorganization without predefined structures
    - Learning from environmental patterns
    - Continuous evolution and adaptation
    """
    
    def __init__(
        self,
        node_id: int,
        adaptation_threshold: float = 0.6,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.2
    ):
        """
        Initialize the openness engine.
        
        Args:
            node_id: Unique identifier for this node
            adaptation_threshold: Threshold for triggering adaptation (0.0-1.0)
            learning_rate: Rate at which patterns are learned (0.0-1.0)
            exploration_rate: Probability of trying novel strategies (0.0-1.0)
        """
        self.node_id = node_id
        self.adaptation_threshold = max(0.0, min(1.0, adaptation_threshold))
        self.learning_rate = max(0.0, min(1.0, learning_rate))
        self.exploration_rate = max(0.0, min(1.0, exploration_rate))
        
        # Current state
        self.current_environment: Optional[EnvironmentalState] = None
        self.current_organization: OrganizationPattern = OrganizationPattern.DISTRIBUTED
        self.adaptation_history: List[AdaptationResult] = []
        
        # Learned patterns
        self.environmental_patterns: Dict[str, EnvironmentalPattern] = {}
        
        # Metrics
        self.environments_encountered = 0
        self.adaptations_performed = 0
        self.adaptations_successful = 0
        self.reorganizations_performed = 0
        
        logger.info(
            f"Openness engine initialized for node {node_id} "
            f"(adapt_thresh={adaptation_threshold:.2f}, "
            f"explore_rate={exploration_rate:.2f})"
        )
    
    def sense_environment(
        self,
        volatility: float,
        complexity: float,
        novelty: float,
        threat_level: float,
        resource_availability: float
    ) -> EnvironmentalState:
        """
        Sense and classify the current environment.
        
        Args:
            volatility: Environmental volatility (0.0-1.0)
            complexity: Environmental complexity (0.0-1.0)
            novelty: How novel/unknown the environment is (0.0-1.0)
            threat_level: Current threat level (0.0-1.0)
            resource_availability: Available resources (0.0-1.0)
        
        Returns:
            EnvironmentalState object
        """
        # Classify environment type based on characteristics
        if novelty > 0.8:
            env_type = EnvironmentType.NOVEL
        elif threat_level > 0.7:
            env_type = EnvironmentType.ADVERSARIAL
        elif volatility > 0.7 and complexity > 0.7:
            env_type = EnvironmentType.CHAOTIC
        elif volatility > 0.4:
            env_type = EnvironmentType.DYNAMIC
        else:
            env_type = EnvironmentType.STABLE
        
        self.current_environment = EnvironmentalState(
            environment_type=env_type,
            volatility=max(0.0, min(1.0, volatility)),
            complexity=max(0.0, min(1.0, complexity)),
            novelty=max(0.0, min(1.0, novelty)),
            threat_level=max(0.0, min(1.0, threat_level)),
            resource_availability=max(0.0, min(1.0, resource_availability))
        )
        
        self.environments_encountered += 1
        
        logger.debug(
            f"Environment sensed: {env_type} "
            f"(volatility={volatility:.2f}, complexity={complexity:.2f}, "
            f"novelty={novelty:.2f})"
        )
        
        return self.current_environment
    
    def should_adapt(self, environment: Optional[EnvironmentalState] = None) -> bool:
        """
        Determine if adaptation is needed.
        
        Args:
            environment: Environment to evaluate (uses current if None)
        
        Returns:
            True if adaptation should be triggered
        """
        env = environment or self.current_environment
        if not env:
            return False
        
        # Adapt if environment is unstable or novel
        instability = (env.volatility + env.complexity + env.novelty) / 3.0
        
        return instability > self.adaptation_threshold
    
    def select_adaptation_strategy(
        self,
        environment: EnvironmentalState
    ) -> AdaptationStrategy:
        """
        Select appropriate adaptation strategy for the environment.
        
        Args:
            environment: Current environmental state
        
        Returns:
            Selected adaptation strategy
        """
        # Check if we've seen similar environments before
        pattern = self._find_matching_pattern(environment)
        
        # Exploration: Try novel approaches occasionally
        if random.random() < self.exploration_rate:
            return AdaptationStrategy.EXPLORATORY
        
        # Use learned patterns if available
        if pattern and pattern.successful_adaptations:
            # Prefer previously successful strategies
            return random.choice(pattern.successful_adaptations)
        
        # Select strategy based on environment characteristics
        stability = environment.get_stability_score()
        
        if stability < 0.3:
            # Chaotic environment - aggressive adaptation
            return AdaptationStrategy.AGGRESSIVE
        elif stability < 0.5 or environment.novelty > 0.6:
            # Dynamic or novel environment - exploratory
            return AdaptationStrategy.EXPLORATORY
        elif stability < 0.7:
            # Moderately stable - moderate adaptation
            return AdaptationStrategy.MODERATE
        else:
            # Stable environment - conservative changes
            return AdaptationStrategy.CONSERVATIVE
    
    def adapt_to_environment(
        self,
        environment: Optional[EnvironmentalState] = None
    ) -> AdaptationResult:
        """
        Adapt agent behavior to current environment.
        
        Args:
            environment: Environment to adapt to (uses current if None)
        
        Returns:
            AdaptationResult with details of adaptation
        """
        env = environment or self.current_environment
        if not env:
            raise ValueError("No environment to adapt to")
        
        start_time = time.time()
        
        # Select adaptation strategy
        strategy = self.select_adaptation_strategy(env)
        
        # Perform adaptation based on strategy
        changes = []
        new_org = None
        success = False
        confidence = 0.0
        
        try:
            if strategy == AdaptationStrategy.CONSERVATIVE:
                changes = self._conservative_adaptation(env)
                confidence = 0.7
                success = True
            
            elif strategy == AdaptationStrategy.MODERATE:
                changes = self._moderate_adaptation(env)
                new_org = self._suggest_reorganization(env)
                confidence = 0.6
                success = True
            
            elif strategy == AdaptationStrategy.AGGRESSIVE:
                changes = self._aggressive_adaptation(env)
                new_org = self._suggest_reorganization(env)
                if new_org:
                    self.reorganize(new_org)
                confidence = 0.5
                success = True
            
            elif strategy == AdaptationStrategy.EXPLORATORY:
                changes = self._exploratory_adaptation(env)
                new_org = self._suggest_reorganization(env)
                confidence = 0.4
                success = random.random() < 0.7  # Exploratory has lower success rate
            
            # Update metrics and learning
            self.adaptations_performed += 1
            if success:
                self.adaptations_successful += 1
            
            # Learn from this adaptation
            self._learn_from_adaptation(env, strategy, success)
            
            adaptation_time = time.time() - start_time
            
            result = AdaptationResult(
                strategy_used=strategy,
                success=success,
                confidence=confidence,
                new_organization=new_org,
                changes_made=changes,
                adaptation_time=adaptation_time
            )
            
            self.adaptation_history.append(result)
            
            if success:
                logger.info(
                    f"Successfully adapted using {strategy} "
                    f"(confidence={confidence:.2f}, changes={len(changes)})"
                )
            else:
                logger.warning(
                    f"Adaptation attempt using {strategy} failed"
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Adaptation failed: {e}")
            return AdaptationResult(
                strategy_used=strategy,
                success=False,
                confidence=0.0,
                new_organization=None,
                changes_made=[],
                adaptation_time=time.time() - start_time
            )
    
    def reorganize(self, new_pattern: OrganizationPattern) -> bool:
        """
        Reorganize agent structure to new pattern.
        
        Args:
            new_pattern: New organization pattern to adopt
        
        Returns:
            True if reorganization successful
        """
        old_pattern = self.current_organization
        self.current_organization = new_pattern
        self.reorganizations_performed += 1
        
        logger.info(
            f"Reorganized from {old_pattern} to {new_pattern}"
        )
        
        return True
    
    def _conservative_adaptation(self, env: EnvironmentalState) -> List[str]:
        """Make minimal, conservative changes"""
        changes = [
            "Adjusted threshold sensitivity by 5%",
            "Updated monitoring frequency"
        ]
        return changes
    
    def _moderate_adaptation(self, env: EnvironmentalState) -> List[str]:
        """Make balanced, moderate changes"""
        changes = [
            "Reconfigured detection parameters",
            "Adjusted response priorities",
            "Updated resource allocation"
        ]
        return changes
    
    def _aggressive_adaptation(self, env: EnvironmentalState) -> List[str]:
        """Make significant, rapid changes"""
        changes = [
            "Complete strategy overhaul",
            "Rebuilt detection models",
            "Reorganized agent structure",
            "Updated all operational parameters"
        ]
        return changes
    
    def _exploratory_adaptation(self, env: EnvironmentalState) -> List[str]:
        """Try novel, experimental approaches"""
        changes = [
            "Enabled experimental feature: adaptive learning",
            "Tried novel detection algorithm",
            "Tested new coordination pattern",
            "Experimented with hybrid organization"
        ]
        return changes
    
    def _suggest_reorganization(
        self,
        env: EnvironmentalState
    ) -> Optional[OrganizationPattern]:
        """Suggest new organization pattern for environment"""
        stability = env.get_stability_score()
        
        if env.threat_level > 0.7:
            # High threat - hierarchical for fast response
            return OrganizationPattern.HIERARCHICAL
        elif stability < 0.3:
            # Chaotic - distributed for resilience
            return OrganizationPattern.DISTRIBUTED
        elif env.complexity > 0.7:
            # Complex - mesh for flexibility
            return OrganizationPattern.MESH
        elif env.resource_availability < 0.3:
            # Low resources - clustered for efficiency
            return OrganizationPattern.CLUSTERED
        else:
            # Hybrid for balanced approach
            return OrganizationPattern.HYBRID
    
    def _find_matching_pattern(
        self,
        env: EnvironmentalState
    ) -> Optional[EnvironmentalPattern]:
        """Find learned pattern matching current environment"""
        for pattern in self.environmental_patterns.values():
            if pattern.environment_type == env.environment_type:
                # Check similarity of characteristics
                similarity = self._calculate_similarity(pattern, env)
                if similarity > 0.7:
                    return pattern
        return None
    
    def _calculate_similarity(
        self,
        pattern: EnvironmentalPattern,
        env: EnvironmentalState
    ) -> float:
        """Calculate similarity between pattern and environment"""
        char = pattern.characteristics
        similarity = (
            1.0 - abs(char.get('volatility', 0.5) - env.volatility) +
            1.0 - abs(char.get('complexity', 0.5) - env.complexity) +
            1.0 - abs(char.get('novelty', 0.5) - env.novelty) +
            1.0 - abs(char.get('threat_level', 0.5) - env.threat_level)
        ) / 4.0
        return similarity
    
    def _learn_from_adaptation(
        self,
        env: EnvironmentalState,
        strategy: AdaptationStrategy,
        success: bool
    ):
        """Learn from adaptation attempt"""
        # Find or create pattern
        pattern_id = f"{env.environment_type}_{self.node_id}"
        
        if pattern_id not in self.environmental_patterns:
            self.environmental_patterns[pattern_id] = EnvironmentalPattern(
                pattern_id=pattern_id,
                environment_type=env.environment_type,
                characteristics={
                    'volatility': env.volatility,
                    'complexity': env.complexity,
                    'novelty': env.novelty,
                    'threat_level': env.threat_level
                },
                successful_adaptations=[],
                failed_adaptations=[],
                encounter_count=0,
                last_seen=time.time()
            )
        
        pattern = self.environmental_patterns[pattern_id]
        
        # Update pattern with learning
        if success:
            pattern.update_success(strategy)
        else:
            pattern.update_failure(strategy)
        
        # Update characteristics with running average
        for key in ['volatility', 'complexity', 'novelty', 'threat_level']:
            old_val = pattern.characteristics.get(key, 0.5)
            new_val = getattr(env, key)
            pattern.characteristics[key] = (
                old_val * (1 - self.learning_rate) +
                new_val * self.learning_rate
            )
    
    def get_openness_metrics(self) -> Dict[str, Any]:
        """Get metrics about openness and adaptation"""
        success_rate = (
            self.adaptations_successful / self.adaptations_performed
            if self.adaptations_performed > 0 else 0.0
        )
        
        return {
            'node_id': self.node_id,
            'current_organization': self.current_organization.value,
            'environments_encountered': self.environments_encountered,
            'adaptations_performed': self.adaptations_performed,
            'adaptations_successful': self.adaptations_successful,
            'adaptation_success_rate': success_rate,
            'reorganizations_performed': self.reorganizations_performed,
            'patterns_learned': len(self.environmental_patterns),
            'current_environment': (
                self.current_environment.environment_type.value
                if self.current_environment else None
            )
        }
    
    def get_learned_patterns(self) -> List[Dict[str, Any]]:
        """Get all learned environmental patterns"""
        patterns = []
        for pattern in self.environmental_patterns.values():
            patterns.append({
                'pattern_id': pattern.pattern_id,
                'environment_type': pattern.environment_type.value,
                'encounter_count': pattern.encounter_count,
                'successful_strategies': [s.value for s in pattern.successful_adaptations],
                'failed_strategies': [s.value for s in pattern.failed_adaptations],
                'last_seen': pattern.last_seen
            })
        return patterns
