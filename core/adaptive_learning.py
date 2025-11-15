"""
Adaptive learning system for auto-tuning thresholds and learning normal behavior.

This module provides mechanisms to learn "normal" service, traffic, and error
behavior patterns and automatically tune detection thresholds based on observed data.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Tuple, Optional
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

    Improvements:
    - Adds 'motivation' (float 0..1) which biases learning speed and threshold tightness.
    - Adds 'joy' (float 0..1) which increases when anomalies are resolved.
    - Uses standard Welford algorithm internally (mean, m2 -> variance).
    - Keeps an exponential moving average (ema) updated with an effective learning rate
      scaled by motivation to give an alternative, motivation-sensitive estimate.
    - Persistence via to_dict()/from_dict().
    - resolve_anomaly() to be called when an anomaly is handled/resolved to increase joy/motivation.
    """

    behavior_type: Any
    observations: Deque = field(default_factory=lambda: deque(maxlen=1000))

    # Running statistics (Welford)
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences from the current mean
    # derived variance = m2 / (n-1) when n > 1

    # Optional EMA (motivation will bias this)
    ema: Optional[float] = None

    min_value: float = float('inf')
    max_value: float = float('-inf')

    # Adaptive thresholds (in [0,1])
    lower_threshold: float = 0.0
    upper_threshold: float = 1.0

    # Learning parameters
    learning_rate: float = 0.01  # base for EMA updates
    adaptation_count: int = 0
    last_update: float = field(default_factory=time.time)

    # Motivation & Joy
    motivation: float = 0.5  # 0.0 (low) .. 1.0 (high)
    joy: float = 0.5         # 0.0 .. 1.0, increases when anomalies are resolved

    # Sensitivity tuning (how strongly motivation affects learning & thresholds)
    motivation_sensitivity: float = 0.5   # affects learning_rate scaling
    threshold_sensitivity: float = 0.5    # affects threshold tightening/loosening

    # Behavior tuning for resolve/decay
    joy_gain: float = 0.05          # base gain to joy per resolved anomaly (scaled)
    motivation_gain_on_resolve: float = 0.02
    motivation_decay_per_sec: float = 1e-4  # small continuous decay
    joy_decay_per_sec: float = 1e-4

    def _clamp01(self, v: float) -> float:
        return max(0.0, min(1.0, v))

    def _effective_learning_rate(self) -> float:
        """
        Scale base learning_rate by motivation.
        motivation > 0.5 -> increase effective lr (faster adaptation)
        motivation < 0.5 -> decrease effective lr (slower adaptation)
        """
        multiplier = 1.0 + (self.motivation - 0.5) * 2.0 * self.motivation_sensitivity
        eff = self.learning_rate * multiplier
        # keep within reasonable bounds
        return max(1e-6, min(1.0, eff))

    def _effective_num_std_devs(self, base_num_std_devs: float) -> float:
        """
        Reduce the number of std devs (tighten thresholds) when motivation is high.
        motivation=1 => tighter thresholds, motivation=0 => looser thresholds.
        """
        # map motivation in [0,1] to factor in [1 + threshold_sensitivity, 1 - threshold_sensitivity]
        adjustment = (self.motivation - 0.5) * 2.0 * self.threshold_sensitivity
        # negative adjustment -> larger num_std_devs (looser); positive -> smaller (tighter)
        eff = base_num_std_devs * (1.0 - adjustment)
        # Avoid non-sensical very small or negative numbers
        return max(0.1, eff)

    def _decay_motivation_and_joy(self) -> None:
        """Apply small decay to motivation and joy based on elapsed time since last_update."""
        now = time.time()
        elapsed = max(0.0, now - self.last_update)
        if elapsed <= 0:
            return
        # exponential-like discrete decay
        self.motivation = self._clamp01(self.motivation * (1.0 - self.motivation_decay_per_sec) ** elapsed)
        self.joy = self._clamp01(self.joy * (1.0 - self.joy_decay_per_sec) ** elapsed)
        # update last_update here only for decay bookkeeping, other methods will set it again
        self.last_update = now

    def update(self, observation: Any) -> None:
        """
        Update profile with a new observation.

        Expects observation to have a numeric .value attribute (or be numeric itself).
        Uses Welford's algorithm for numerically stable incremental updates and also
        updates an EMA that is biased by 'motivation'.
        """
        # normalize observation value extraction
        value = observation.value if hasattr(observation, "value") else float(observation)

        # decay first so motivation/joy remain time-aware
        self._decay_motivation_and_joy()

        # store observation
        self.observations.append(observation)

        # update min/max
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

        # Welford update
        n = self.adaptation_count + 1
        delta = value - self.mean
        self.mean += delta / n
        delta2 = value - self.mean
        self.m2 += delta * delta2

        # update variance derived if needed (kept via getter)
        self.adaptation_count = n
        self.last_update = time.time()

        # update EMA using motivation-scaled learning rate
        eff_lr = self._effective_learning_rate()
        if self.ema is None:
            self.ema = value
        else:
            self.ema = eff_lr * value + (1.0 - eff_lr) * self.ema

    @property
    def variance(self) -> float:
        if self.adaptation_count < 2:
            return 0.0
        return self.m2 / (self.adaptation_count - 1)

    def auto_tune_thresholds(
        self,
        num_std_devs: float = 2.0,
        min_observations: int = 10
    ) -> Tuple[float, float]:
        """
        Automatically tune thresholds based on observed behavior and current motivation.

        Higher motivation will generally tighten thresholds (smaller number of std devs).
        """
        if self.adaptation_count < min_observations:
            return self.lower_threshold, self.upper_threshold

        std_dev = self.variance ** 0.5
        eff_num = self._effective_num_std_devs(num_std_devs)

        lower = self.mean - eff_num * std_dev
        upper = self.mean + eff_num * std_dev

        # clamp thresholds into [0,1] domain if appropriate for your data; keep numeric safety
        self.lower_threshold = max(0.0, min(1.0, lower))
        self.upper_threshold = max(0.0, min(1.0, upper))

        # ensure lower <= upper
        if self.lower_threshold > self.upper_threshold:
            mid = (self.lower_threshold + self.upper_threshold) / 2.0
            self.lower_threshold = mid
            self.upper_threshold = mid

        return self.lower_threshold, self.upper_threshold

    def is_anomaly(self, value: float, sensitivity: float = 1.0) -> Tuple[bool, float]:
        """
        Check if a value is anomalous.

        sensitivity: multiplier for detection sensitivity (higher -> more sensitive).
        Returns (is_anomaly, anomaly_score) where anomaly_score is 0..1.
        """
        if self.adaptation_count < 2:
            return False, 0.0

        std_dev = self.variance ** 0.5
        if std_dev == 0:
            return False, 0.0

        # compute z-score
        z_score = abs(value - self.mean) / std_dev

        # use motivation to tighten or loosen threshold
        base_threshold = 2.0
        eff_threshold = self._effective_num_std_devs(base_threshold) / max(1e-6, sensitivity)

        is_anom = z_score > eff_threshold

        # normalize anomaly score: how far beyond the threshold (clamped)
        anomaly_score = max(0.0, min(1.0, (z_score - eff_threshold) / (eff_threshold * 2.0)))

        return is_anom, anomaly_score

    def resolve_anomaly(self, anomaly_score: float) -> None:
        """
        Call this after handling or resolving an anomaly. This will:
        - increase joy proportional to anomaly_score,
        - modestly increase motivation,
        - update last_update timestamp.

        This models 'satisfaction' (joy) when issues are fixed, which in turn can
        influence future learning/thresholding via motivation.
        """
        self._decay_motivation_and_joy()

        # scale gains with anomaly_score (0..1)
        gain = self._clamp01(anomaly_score)
        self.joy = self._clamp01(self.joy + gain * self.joy_gain)
        self.motivation = self._clamp01(self.motivation + gain * self.motivation_gain_on_resolve)

        self.last_update = time.time()

    def set_motivation(self, value: float) -> None:
        self.motivation = self._clamp01(value)

    def set_joy(self, value: float) -> None:
        self.joy = self._clamp01(value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile state for persistence or export (observations excluded)."""
        return {
            "behavior_type": self.behavior_type,
            "mean": self.mean,
            "m2": self.m2,
            "variance": self.variance,
            "ema": self.ema,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "lower_threshold": self.lower_threshold,
            "upper_threshold": self.upper_threshold,
            "learning_rate": self.learning_rate,
            "adaptation_count": self.adaptation_count,
            "last_update": self.last_update,
            "motivation": self.motivation,
            "joy": self.joy,
            "motivation_sensitivity": self.motivation_sensitivity,
            "threshold_sensitivity": self.threshold_sensitivity,
            "joy_gain": self.joy_gain,
            "motivation_gain_on_resolve": self.motivation_gain_on_resolve,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorProfile":
        """Restore a BehaviorProfile from a dictionary previously returned by to_dict()."""
        bp = cls(behavior_type=data.get("behavior_type"))
        bp.mean = float(data.get("mean", 0.0))
        bp.m2 = float(data.get("m2", 0.0))
        bp.ema = data.get("ema", None)
        bp.min_value = float(data.get("min_value", float("inf")))
        bp.max_value = float(data.get("max_value", float("-inf")))
        bp.lower_threshold = float(data.get("lower_threshold", 0.0))
        bp.upper_threshold = float(data.get("upper_threshold", 1.0))
        bp.learning_rate = float(data.get("learning_rate", 0.01))
        bp.adaptation_count = int(data.get("adaptation_count", 0))
        bp.last_update = float(data.get("last_update", time.time()))
        bp.motivation = bp._clamp01(float(data.get("motivation", 0.5)))
        bp.joy = bp._clamp01(float(data.get("joy", 0.5)))
        bp.motivation_sensitivity = float(data.get("motivation_sensitivity", bp.motivation_sensitivity))
        bp.threshold_sensitivity = float(data.get("threshold_sensitivity", bp.threshold_sensitivity))
        bp.joy_gain = float(data.get("joy_gain", bp.joy_gain))
        bp.motivation_gain_on_resolve = float(data.get("motivation_gain_on_resolve", bp.motivation_gain_on_resolve))
        return bp

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "behavior_type": self.behavior_type,
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.variance ** 0.5,
            "ema": self.ema,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "lower_threshold": self.lower_threshold,
            "upper_threshold": self.upper_threshold,
            "adaptation_count": self.adaptation_count,
            "observations": len(self.observations),
            "motivation": self.motivation,
            "joy": self.joy,
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
