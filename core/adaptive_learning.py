"""
Adaptive learning system for auto-tuning thresholds and learning normal behavior.

This module provides mechanisms to learn "normal" service, traffic, and error
behavior patterns and automatically tune detection thresholds based on observed data.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class BehaviorType(str, Enum):
    """Types of behavior to learn."""
    SERVICE = "service"
    TRAFFIC = "traffic"
    ERROR = "error"
    RESOURCE = "resource"
    PERFORMANCE = "performance"


@dataclass
class Observation:
    """A single observation of system behavior."""
    timestamp: float
    value: float
    behavior_type: BehaviorType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorProfile:
    """
    Statistical profile of normal behavior.
    
    Maintains running statistics to characterize normal behavior patterns
    and detect anomalies.
    """
    behavior_type: BehaviorType
    observations: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Running statistics
    mean: float = 0.0
    variance: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    
    # Adaptive thresholds
    lower_threshold: float = 0.0
    upper_threshold: float = 1.0
    
    # Learning parameters
    learning_rate: float = 0.01
    adaptation_count: int = 0
    last_update: float = field(default_factory=time.time)
    
    def update(self, observation: Observation) -> None:
        """
        Update profile with new observation.
        
        Uses incremental learning to update statistics without
        storing all historical data.
        """
        value = observation.value
        self.observations.append(observation)
        
        # Update min/max
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        
        # Incremental mean and variance update (Welford's algorithm)
        n = self.adaptation_count + 1
        delta = value - self.mean
        self.mean += delta / n
        delta2 = value - self.mean
        self.variance += (delta * delta2 - self.variance) / n
        
        self.adaptation_count += 1
        self.last_update = time.time()
    
    def auto_tune_thresholds(
        self,
        num_std_devs: float = 2.0,
        min_observations: int = 10
    ) -> Tuple[float, float]:
        """
        Automatically tune thresholds based on observed behavior.
        
        Args:
            num_std_devs: Number of standard deviations for threshold
            min_observations: Minimum observations before auto-tuning
            
        Returns:
            (lower_threshold, upper_threshold)
        """
        if self.adaptation_count < min_observations:
            return self.lower_threshold, self.upper_threshold
        
        std_dev = self.variance ** 0.5
        
        # Set thresholds based on mean ± N standard deviations
        self.lower_threshold = max(0.0, self.mean - num_std_devs * std_dev)
        self.upper_threshold = min(1.0, self.mean + num_std_devs * std_dev)
        
        return self.lower_threshold, self.upper_threshold
    
    def is_anomaly(self, value: float, sensitivity: float = 1.0) -> Tuple[bool, float]:
        """
        Check if a value is anomalous.
        
        Args:
            value: Value to check
            sensitivity: Multiplier for threshold sensitivity (higher = more sensitive)
            
        Returns:
            (is_anomaly, anomaly_score)
        """
        if self.adaptation_count < 2:
            return False, 0.0
        
        # Calculate z-score
        std_dev = self.variance ** 0.5
        if std_dev == 0:
            return False, 0.0
        
        z_score = abs(value - self.mean) / std_dev
        
        # Anomaly if beyond threshold (adjusted by sensitivity)
        threshold = 2.0 / sensitivity
        is_anomaly = z_score > threshold
        
        # Anomaly score normalized to 0-1
        anomaly_score = min(1.0, z_score / (threshold * 2))
        
        return is_anomaly, anomaly_score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'behavior_type': self.behavior_type,
            'mean': self.mean,
            'variance': self.variance,
            'std_dev': self.variance ** 0.5,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'lower_threshold': self.lower_threshold,
            'upper_threshold': self.upper_threshold,
            'adaptation_count': self.adaptation_count,
            'observations': len(self.observations)
        }


class AdaptiveLearningSystem:
    """
    Adaptive learning system that learns normal behavior and auto-tunes thresholds.
    
    Implements:
    - Continual learning from observations
    - Statistical profiling of normal behavior
    - Automatic threshold tuning
    - Anomaly detection
    - Multi-variate behavior tracking
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        profile_window_size: int = 1000,
        auto_tune_interval: float = 300.0  # 5 minutes
    ):
        """
        Initialize adaptive learning system.
        
        Args:
            learning_rate: Rate of adaptation to new observations
            profile_window_size: Number of observations to keep per profile
            auto_tune_interval: Seconds between automatic threshold tuning
        """
        self.learning_rate = learning_rate
        self.profile_window_size = profile_window_size
        self.auto_tune_interval = auto_tune_interval
        
        self.profiles: Dict[BehaviorType, BehaviorProfile] = {}
        self.last_auto_tune: Dict[BehaviorType, float] = {}
        
        # Learning metrics
        self.total_observations = 0
        self.anomalies_detected = 0
        self.threshold_adjustments = 0
        
        logger.info(f"Adaptive learning system initialized (lr={learning_rate}, "
                   f"window={profile_window_size})")
    
    def observe(
        self,
        behavior_type: BehaviorType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record an observation and update learning.
        
        Args:
            behavior_type: Type of behavior observed
            value: Observed value
            metadata: Optional metadata about the observation
            
        Returns:
            Observation results including anomaly status
        """
        # Get or create profile
        if behavior_type not in self.profiles:
            self.profiles[behavior_type] = BehaviorProfile(
                behavior_type=behavior_type,
                learning_rate=self.learning_rate
            )
            self.last_auto_tune[behavior_type] = time.time()
        
        profile = self.profiles[behavior_type]
        
        # Check if anomaly before update
        is_anomaly, anomaly_score = profile.is_anomaly(value)
        
        # Create observation
        observation = Observation(
            timestamp=time.time(),
            value=value,
            behavior_type=behavior_type,
            metadata=metadata or {}
        )
        
        # Update profile (learn from this observation)
        profile.update(observation)
        self.total_observations += 1
        
        if is_anomaly:
            self.anomalies_detected += 1
        
        # Auto-tune thresholds periodically
        time_since_tune = time.time() - self.last_auto_tune[behavior_type]
        if time_since_tune >= self.auto_tune_interval:
            profile.auto_tune_thresholds()
            self.last_auto_tune[behavior_type] = time.time()
            self.threshold_adjustments += 1
        
        return {
            'behavior_type': behavior_type,
            'value': value,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'profile_mean': profile.mean,
            'lower_threshold': profile.lower_threshold,
            'upper_threshold': profile.upper_threshold
        }
    
    def force_auto_tune(self, behavior_type: Optional[BehaviorType] = None) -> Dict[str, Any]:
        """
        Force immediate auto-tuning of thresholds.
        
        Args:
            behavior_type: Specific type to tune, or None for all
            
        Returns:
            Tuning results
        """
        results = {}
        
        types_to_tune = [behavior_type] if behavior_type else list(self.profiles.keys())
        
        for btype in types_to_tune:
            if btype in self.profiles:
                profile = self.profiles[btype]
                lower, upper = profile.auto_tune_thresholds()
                self.last_auto_tune[btype] = time.time()
                self.threshold_adjustments += 1
                
                results[btype] = {
                    'lower_threshold': lower,
                    'upper_threshold': upper,
                    'mean': profile.mean,
                    'std_dev': profile.variance ** 0.5
                }
        
        logger.info(f"Auto-tuned thresholds for {len(results)} behavior types")
        return results
    
    def get_profile(self, behavior_type: BehaviorType) -> Optional[BehaviorProfile]:
        """Get behavior profile for a type."""
        return self.profiles.get(behavior_type)
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all profiles."""
        profile_stats = {}
        for btype, profile in self.profiles.items():
            profile_stats[btype] = profile.get_statistics()
        
        return {
            'total_observations': self.total_observations,
            'anomalies_detected': self.anomalies_detected,
            'threshold_adjustments': self.threshold_adjustments,
            'profiles': profile_stats,
            'learning_rate': self.learning_rate
        }
    
    def predict_normal_range(
        self,
        behavior_type: BehaviorType,
        confidence: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """
        Predict the normal range for a behavior type.
        
        Args:
            behavior_type: Type of behavior
            confidence: Confidence level (0-1)
            
        Returns:
            (lower_bound, upper_bound) or None if insufficient data
        """
        profile = self.profiles.get(behavior_type)
        if not profile or profile.adaptation_count < 10:
            return None
        
        # Use confidence interval based on normal distribution
        from math import sqrt
        
        std_dev = sqrt(profile.variance)
        # z-score for confidence level (approximation)
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        
        lower = profile.mean - z * std_dev
        upper = profile.mean + z * std_dev
        
        return max(0.0, lower), min(1.0, upper)
    
    def reset_profile(self, behavior_type: BehaviorType) -> None:
        """Reset a behavior profile to start fresh learning."""
        if behavior_type in self.profiles:
            del self.profiles[behavior_type]
            del self.last_auto_tune[behavior_type]
            logger.info(f"Reset profile for {behavior_type}")
    
    def export_learned_parameters(self) -> Dict[str, Any]:
        """
        Export learned parameters for persistence or transfer.
        
        Returns:
            Dictionary of learned parameters
        """
        export = {
            'learning_rate': self.learning_rate,
            'total_observations': self.total_observations,
            'anomalies_detected': self.anomalies_detected,
            'threshold_adjustments': self.threshold_adjustments,
            'profiles': {}
        }
        
        for btype, profile in self.profiles.items():
            export['profiles'][btype] = {
                'mean': profile.mean,
                'variance': profile.variance,
                'min_value': profile.min_value,
                'max_value': profile.max_value,
                'lower_threshold': profile.lower_threshold,
                'upper_threshold': profile.upper_threshold,
                'adaptation_count': profile.adaptation_count
            }
        
        return export
    
    def import_learned_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Import previously learned parameters.
        
        Args:
            parameters: Dictionary of learned parameters from export
        """
        self.learning_rate = parameters.get('learning_rate', self.learning_rate)
        self.total_observations = parameters.get('total_observations', 0)
        self.anomalies_detected = parameters.get('anomalies_detected', 0)
        self.threshold_adjustments = parameters.get('threshold_adjustments', 0)
        
        for btype_str, profile_data in parameters.get('profiles', {}).items():
            btype = BehaviorType(btype_str)
            profile = BehaviorProfile(behavior_type=btype, learning_rate=self.learning_rate)
            
            profile.mean = profile_data['mean']
            profile.variance = profile_data['variance']
            profile.min_value = profile_data['min_value']
            profile.max_value = profile_data['max_value']
            profile.lower_threshold = profile_data['lower_threshold']
            profile.upper_threshold = profile_data['upper_threshold']
            profile.adaptation_count = profile_data['adaptation_count']
            
            self.profiles[btype] = profile
            self.last_auto_tune[btype] = time.time()
        
        logger.info(f"Imported parameters for {len(self.profiles)} behavior profiles")
