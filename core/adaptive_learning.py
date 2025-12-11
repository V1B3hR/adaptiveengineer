"""
Adaptive learning system for auto-tuning thresholds and learning normal behavior.
Updated with Emotional Intelligence: Frustration, Curiosity, and Aggression modes
to support Symbiotic Evolution.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Tuple, Optional
from enum import Enum
import math

logger = logging.getLogger(__name__)


class BehaviorType(str, Enum):
    """Types of behavior to learn."""

    SERVICE = "service"
    TRAFFIC = "traffic"
    ERROR = "error"
    RESOURCE = "resource"
    PERFORMANCE = "performance"
    SYMBIOTIC_TEST = "symbiotic_test"


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
    Statistical profile of normal behavior with Emotional Intelligence.

    Now includes:
    - Motivation & Joy (Legacy)
    - Calmness, Curiosity, Frustration (New)
    - Aggressive Mode handling
    - Novelty detection
    """

    behavior_type: Any
    observations: Deque = field(default_factory=lambda: deque(maxlen=1000))

    # Running statistics (Welford)
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences from the current mean
    ema: Optional[float] = None

    min_value: float = float("inf")
    max_value: float = float("-inf")

    # Adaptive thresholds
    lower_threshold: float = 0.0
    upper_threshold: float = 1.0

    # Learning parameters
    learning_rate: float = 0.01
    adaptation_count: int = 0
    last_update: float = field(default_factory=time.time)

    # --- EMOTIONAL STATE ---

    # Core drives (0.0 - 1.0)
    motivation: float = 0.5
    joy: float = 0.5

    # Personality Traits (DNA from Evolution)
    calmness: float = 0.5  # 1.0 = Zen master, 0.0 = Volatile
    curiosity_drive: float = 0.1  # How much joy from novelty
    motivation_memory: float = 0.95  # Retention of motivation over time

    # Dynamic Emotional State
    frustration: float = 0.0  # Accumulates with unresolved/severe anomalies
    aggressive_mode: bool = False  # Triggered by high frustration

    # Sensitivities
    motivation_sensitivity: float = 0.5
    threshold_sensitivity: float = 0.5
    joy_gain: float = 0.05
    motivation_gain_on_resolve: float = 0.02

    # Decay factors
    motivation_decay_per_sec: float = 1e-4
    joy_decay_per_sec: float = 1e-4

    # Internal metrics for behavior tracking
    _resolve_ema: float = 0.0  # Tracks rate of resolutions (spam protection)

    def _clamp01(self, v: float) -> float:
        return max(0.0, min(1.0, v))

    def _effective_learning_rate(self) -> float:
        # Aggressive mode forces high learning rate
        if self.aggressive_mode:
            return min(1.0, self.learning_rate * 2.0)

        multiplier = (
            1.0 + (self.motivation - 0.5) * 2.0 * self.motivation_sensitivity
        )
        eff = self.learning_rate * multiplier
        return max(1e-6, min(1.0, eff))

    def _effective_num_std_devs(self, base_num_std_devs: float) -> float:
        # Aggressive mode tightens thresholds significantly
        if self.aggressive_mode:
            return max(0.1, base_num_std_devs * 0.5)

        adjustment = (self.motivation - 0.5) * 2.0 * self.threshold_sensitivity
        eff = base_num_std_devs * (1.0 - adjustment)
        return max(0.1, eff)

    def _decay_motivation_and_joy(self) -> None:
        now = time.time()
        elapsed = max(0.0, now - self.last_update)
        if elapsed <= 0:
            return

        # Use motivation_memory if available, else default decay
        decay_rate = self.motivation_decay_per_sec

        self.motivation = self._clamp01(
            self.motivation * (1.0 - decay_rate) ** elapsed
        )
        self.joy = self._clamp01(
            self.joy * (1.0 - self.joy_decay_per_sec) ** elapsed
        )

        # Frustration decays over time if not fed (calmness speeds this up)
        frustration_decay = 0.05 * (1.0 + self.calmness)
        self.frustration = max(
            0.0, self.frustration - (frustration_decay * elapsed)
        )

        # Exit aggressive mode if frustration drops
        if self.aggressive_mode and self.frustration < 0.5:
            self.aggressive_mode = False

        self.last_update = now

    def update(self, observation: Any) -> None:
        value = (
            observation.value
            if hasattr(observation, "value")
            else float(observation)
        )
        self._decay_motivation_and_joy()
        self.observations.append(observation)

        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

        n = self.adaptation_count + 1
        delta = value - self.mean
        self.mean += delta / n
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.adaptation_count = n
        self.last_update = time.time()

        eff_lr = self._effective_learning_rate()
        if self.ema is None:
            self.ema = value
        else:
            self.ema = eff_lr * value + (1.0 - eff_lr) * self.ema

        # Decay resolve rate tracking (part of spam protection)
        self._resolve_ema = self._resolve_ema * 0.95

    @property
    def variance(self) -> float:
        if self.adaptation_count < 2:
            return 0.0
        return self.m2 / (self.adaptation_count - 1)

    @property
    def std_dev(self) -> float:
        return self.variance**0.5

    def auto_tune_thresholds(
        self, num_std_devs: float = 2.0, min_observations: int = 10
    ) -> Tuple[float, float]:
        if self.adaptation_count < min_observations:
            return self.lower_threshold, self.upper_threshold

        std = self.std_dev
        eff_num = self._effective_num_std_devs(num_std_devs)

        lower = self.mean - eff_num * std
        upper = self.mean + eff_num * std

        self.lower_threshold = max(0.0, min(1.0, lower))
        self.upper_threshold = max(0.0, min(1.0, upper))

        if self.lower_threshold > self.upper_threshold:
            mid = (self.lower_threshold + self.upper_threshold) / 2.0
            self.lower_threshold = mid
            self.upper_threshold = mid

        return self.lower_threshold, self.upper_threshold

    def is_anomaly(
        self, value: float, sensitivity: float = 1.0
    ) -> Tuple[bool, float]:
        if self.adaptation_count < 2:
            return False, 0.0

        std = self.std_dev
        if std == 0:
            return False, 0.0

        z_score = abs(value - self.mean) / std
        base_threshold = 2.0
        eff_threshold = self._effective_num_std_devs(base_threshold) / max(
            1e-6, sensitivity
        )

        is_anom = z_score > eff_threshold
        anomaly_score = max(
            0.0, min(1.0, (z_score - eff_threshold) / (eff_threshold * 2.0))
        )

        return is_anom, anomaly_score

    # --- NEW SYMBIOTIC METHODS ---

    def novelty_score(self, observation: Any) -> float:
        """
        Calculates how 'novel' or surprising an observation is.
        Used by Symbiotic Bridge to reward curiosity.
        """
        value = (
            observation.value
            if hasattr(observation, "value")
            else float(observation)
        )
        if self.std_dev == 0:
            return 0.0

        z_score = abs(value - self.mean) / self.std_dev
        # Novelty is finding something just on the edge of known (1-3 std devs),
        # not necessarily extreme chaos.
        novelty = max(0.0, min(1.0, z_score / 4.0))
        return novelty

    def enter_aggressive_mode(self) -> None:
        """
        Triggered when frustration peaks.
        Maximizes motivation and tightens thresholds to survive.
        """
        self.aggressive_mode = True
        self.motivation = 1.0  # Full focus
        # Frustration stays high until cooled down by time
        logger.warning(
            f"Profile {self.behavior_type} entering AGGRESSIVE MODE"
        )

    def apply_overactive_penalty(self, penalty: float) -> None:
        """
        Punishes the system for spamming resolves (hyperactivity).
        """
        self.joy = max(0.0, self.joy - penalty)
        self.motivation = max(0.0, self.motivation - penalty)

    @property
    def recent_resolve_rate(self) -> float:
        """
        Returns an estimated rate (0.0-1.0) of recent resolve actions.
        """
        return self._resolve_ema

    def resolve_anomaly(self, anomaly_score: float) -> None:
        """
        Call this after handling or resolving an anomaly.
        """
        self._decay_motivation_and_joy()

        gain = self._clamp01(anomaly_score)
        self.joy = self._clamp01(self.joy + gain * self.joy_gain)
        self.motivation = self._clamp01(
            self.motivation + gain * self.motivation_gain_on_resolve
        )

        # Decrease frustration on successful resolve
        self.frustration = max(0.0, self.frustration - 0.2)

        # Update resolve rate tracker (bump up)
        self._resolve_ema = 0.1 * 1.0 + 0.9 * self._resolve_ema

        self.last_update = time.time()

    def set_motivation(self, value: float) -> None:
        self.motivation = self._clamp01(value)

    def set_joy(self, value: float) -> None:
        self.joy = self._clamp01(value)

    def to_dict(self) -> Dict[str, Any]:
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
            # Emotional State
            "motivation": self.motivation,
            "joy": self.joy,
            "frustration": self.frustration,
            "aggressive_mode": self.aggressive_mode,
            # Traits
            "calmness": self.calmness,
            "curiosity_drive": self.curiosity_drive,
            "motivation_memory": self.motivation_memory,
            # Tuning
            "motivation_sensitivity": self.motivation_sensitivity,
            "threshold_sensitivity": self.threshold_sensitivity,
            "joy_gain": self.joy_gain,
            "motivation_gain_on_resolve": self.motivation_gain_on_resolve,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorProfile":
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
        bp.frustration = float(data.get("frustration", 0.0))
        bp.aggressive_mode = bool(data.get("aggressive_mode", False))

        bp.calmness = float(data.get("calmness", 0.5))
        bp.curiosity_drive = float(data.get("curiosity_drive", 0.1))
        bp.motivation_memory = float(data.get("motivation_memory", 0.95))

        bp.motivation_sensitivity = float(
            data.get("motivation_sensitivity", bp.motivation_sensitivity)
        )
        bp.threshold_sensitivity = float(
            data.get("threshold_sensitivity", bp.threshold_sensitivity)
        )
        bp.joy_gain = float(data.get("joy_gain", bp.joy_gain))
        bp.motivation_gain_on_resolve = float(
            data.get(
                "motivation_gain_on_resolve", bp.motivation_gain_on_resolve
            )
        )
        return bp

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "behavior_type": self.behavior_type,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "ema": self.ema,
            "adaptation_count": self.adaptation_count,
            "observations": len(self.observations),
            "motivation": self.motivation,
            "joy": self.joy,
            "frustration": self.frustration,
            "aggressive_mode": self.aggressive_mode,
        }


class AdaptiveLearningSystem:
    """
    Adaptive learning system that learns normal behavior and auto-tunes thresholds.
    Coordinates multiple BehaviorProfiles.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        profile_window_size: int = 1000,
        auto_tune_interval: float = 300.0,
    ):
        self.learning_rate = learning_rate
        self.profile_window_size = profile_window_size
        self.auto_tune_interval = auto_tune_interval

        self.profiles: Dict[BehaviorType, BehaviorProfile] = {}
        self.last_auto_tune: Dict[BehaviorType, float] = {}

        self.total_observations = 0
        self.anomalies_detected = 0
        self.threshold_adjustments = 0

        logger.info(
            f"Adaptive learning system initialized (lr={learning_rate})"
        )

    def observe(
        self,
        behavior_type: BehaviorType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if behavior_type not in self.profiles:
            self.profiles[behavior_type] = BehaviorProfile(
                behavior_type=behavior_type, learning_rate=self.learning_rate
            )
            self.last_auto_tune[behavior_type] = time.time()

        profile = self.profiles[behavior_type]

        is_anomaly, anomaly_score = profile.is_anomaly(value)

        observation = Observation(
            timestamp=time.time(),
            value=value,
            behavior_type=behavior_type,
            metadata=metadata or {},
        )

        profile.update(observation)
        self.total_observations += 1

        if is_anomaly:
            self.anomalies_detected += 1

        time_since_tune = time.time() - self.last_auto_tune[behavior_type]
        if time_since_tune >= self.auto_tune_interval:
            profile.auto_tune_thresholds()
            self.last_auto_tune[behavior_type] = time.time()
            self.threshold_adjustments += 1

        return {
            "behavior_type": behavior_type,
            "value": value,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "profile_mean": profile.mean,
            "lower_threshold": profile.lower_threshold,
            "upper_threshold": profile.upper_threshold,
            "joy": profile.joy,
            "frustration": profile.frustration,
        }

    def force_auto_tune(
        self, behavior_type: Optional[BehaviorType] = None
    ) -> Dict[str, Any]:
        results = {}
        types_to_tune = (
            [behavior_type] if behavior_type else list(self.profiles.keys())
        )

        for btype in types_to_tune:
            if btype in self.profiles:
                profile = self.profiles[btype]
                lower, upper = profile.auto_tune_thresholds()
                self.last_auto_tune[btype] = time.time()
                self.threshold_adjustments += 1
                results[btype] = {
                    "lower_threshold": lower,
                    "upper_threshold": upper,
                    "mean": profile.mean,
                }
        return results

    def get_profile(
        self, behavior_type: BehaviorType
    ) -> Optional[BehaviorProfile]:
        return self.profiles.get(behavior_type)

    def get_all_statistics(self) -> Dict[str, Any]:
        profile_stats = {}
        for btype, profile in self.profiles.items():
            profile_stats[btype] = profile.get_statistics()

        return {
            "total_observations": self.total_observations,
            "anomalies_detected": self.anomalies_detected,
            "profiles": profile_stats,
        }

    def reset_profile(self, behavior_type: BehaviorType) -> None:
        if behavior_type in self.profiles:
            del self.profiles[behavior_type]
            del self.last_auto_tune[behavior_type]
            logger.info(f"Reset profile for {behavior_type}")

    def export_learned_parameters(self) -> Dict[str, Any]:
        export = {
            "learning_rate": self.learning_rate,
            "total_observations": self.total_observations,
            "profiles": {},
        }
        for btype, profile in self.profiles.items():
            export["profiles"][str(btype)] = profile.to_dict()
        return export

    def import_learned_parameters(self, parameters: Dict[str, Any]) -> None:
        self.learning_rate = parameters.get(
            "learning_rate", self.learning_rate
        )
        for btype_str, profile_data in parameters.get("profiles", {}).items():
            # Try to match Enum if possible, else string
            try:
                btype = BehaviorType(btype_str)
            except ValueError:
                btype = btype_str  # Fallback

            profile = BehaviorProfile.from_dict(profile_data)
            self.profiles[btype] = profile
            self.last_auto_tune[btype] = time.time()
