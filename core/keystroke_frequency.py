"""
Keystroke Frequency Analyzer

Analyzes typing rhythm for behavioral biometrics and continuous authentication.
Uses FFT-based rhythm signatures to detect impersonation and authenticate users.

Features:
- Dwell time and flight time measurements
- FFT-based rhythm signature (first 20 coefficients)
- Impersonation detection via baseline comparison
- Continuous authentication capability
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

logger = logging.getLogger(__name__)


class AuthenticationStatus(str, Enum):
    """Authentication status results."""
    AUTHENTICATED = "authenticated"
    SUSPICIOUS = "suspicious"
    REJECTED = "rejected"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class KeystrokeEvent:
    """Represents a keystroke event."""
    key: str
    press_time: float
    release_time: float
    user_id: Optional[str] = None
    
    @property
    def dwell_time(self) -> float:
        """Time key was held down."""
        return self.release_time - self.press_time
    
    def flight_time_to(self, next_event: 'KeystrokeEvent') -> float:
        """Time between release of this key and press of next key."""
        return next_event.press_time - self.release_time


@dataclass
class TypingSignature:
    """Represents a user's typing signature."""
    user_id: str
    fft_coefficients: np.ndarray  # First 20 FFT coefficients
    mean_dwell_time: float
    std_dwell_time: float
    mean_flight_time: float
    std_flight_time: float
    timestamp: float
    sample_count: int


@dataclass
class AuthenticationResult:
    """Result of keystroke authentication."""
    user_id: str
    status: AuthenticationStatus
    confidence: float
    timestamp: float
    description: str
    distance_from_baseline: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeystrokeFrequencyAnalyzer:
    """
    Keystroke Frequency Analyzer for behavioral biometrics.
    
    Analyzes typing rhythm patterns using frequency domain analysis to create
    unique behavioral signatures for each user. Can detect impersonation attempts
    and provide continuous authentication.
    
    Attributes:
        min_samples_for_baseline: Minimum keystrokes to establish baseline
        authentication_threshold: Distance threshold for authentication
        fft_coefficients: Number of FFT coefficients to use (default 20)
    """
    
    def __init__(
        self,
        min_samples_for_baseline: int = 100,
        authentication_threshold: float = 0.3,
        fft_coefficients: int = 20
    ):
        """
        Initialize Keystroke Frequency Analyzer.
        
        Args:
            min_samples_for_baseline: Minimum keystrokes to establish user baseline
            authentication_threshold: Distance threshold for authentication
            fft_coefficients: Number of FFT coefficients in signature
        """
        self.min_samples_for_baseline = min_samples_for_baseline
        self.authentication_threshold = authentication_threshold
        self.fft_coefficients = fft_coefficients
        
        # User keystroke buffers
        self.user_keystrokes: Dict[str, deque] = {}
        
        # User typing signatures (baselines)
        self.user_signatures: Dict[str, TypingSignature] = {}
        
        # Authentication results
        self.authentication_results: List[AuthenticationResult] = []
        
        logger.info(f"Keystroke Analyzer initialized: min_samples={min_samples_for_baseline}, "
                   f"threshold={authentication_threshold}")
    
    def add_keystroke(
        self,
        event: KeystrokeEvent
    ):
        """
        Add a keystroke event.
        
        Args:
            event: Keystroke event to add
        """
        if not event.user_id:
            logger.warning("Keystroke event missing user_id")
            return
        
        if event.user_id not in self.user_keystrokes:
            self.user_keystrokes[event.user_id] = deque(maxlen=500)
        
        self.user_keystrokes[event.user_id].append(event)
    
    def establish_baseline(
        self,
        user_id: str,
        keystrokes: Optional[List[KeystrokeEvent]] = None
    ) -> Optional[TypingSignature]:
        """
        Establish typing baseline for a user.
        
        Args:
            user_id: User identifier
            keystrokes: Keystroke events (uses buffer if None)
            
        Returns:
            Typing signature if sufficient data, None otherwise
        """
        # Get keystrokes
        if keystrokes is not None:
            keystroke_list = keystrokes
        elif user_id in self.user_keystrokes:
            keystroke_list = list(self.user_keystrokes[user_id])
        else:
            return None
        
        if len(keystroke_list) < self.min_samples_for_baseline:
            logger.info(f"Insufficient keystrokes for {user_id}: {len(keystroke_list)}/{self.min_samples_for_baseline}")
            return None
        
        # Extract timing features
        dwell_times = [k.dwell_time for k in keystroke_list]
        
        # Calculate flight times
        flight_times = []
        for i in range(len(keystroke_list) - 1):
            flight_time = keystroke_list[i].flight_time_to(keystroke_list[i + 1])
            if flight_time >= 0:  # Ignore negative (overlapping keys)
                flight_times.append(flight_time)
        
        if len(flight_times) < 10:
            return None
        
        # Calculate statistical features
        mean_dwell = np.mean(dwell_times)
        std_dwell = np.std(dwell_times)
        mean_flight = np.mean(flight_times)
        std_flight = np.std(flight_times)
        
        # Compute FFT-based signature
        fft_coeffs = self._compute_typing_fft_signature(dwell_times, flight_times)
        
        signature = TypingSignature(
            user_id=user_id,
            fft_coefficients=fft_coeffs,
            mean_dwell_time=mean_dwell,
            std_dwell_time=std_dwell,
            mean_flight_time=mean_flight,
            std_flight_time=std_flight,
            timestamp=time.time(),
            sample_count=len(keystroke_list)
        )
        
        self.user_signatures[user_id] = signature
        
        logger.info(f"Established baseline for {user_id}: "
                   f"dwell={mean_dwell*1000:.1f}ms, flight={mean_flight*1000:.1f}ms")
        
        return signature
    
    def _compute_typing_fft_signature(
        self,
        dwell_times: List[float],
        flight_times: List[float]
    ) -> np.ndarray:
        """
        Compute FFT-based typing rhythm signature.
        
        Combines dwell and flight times into a frequency domain signature
        that captures the user's unique typing rhythm.
        
        Args:
            dwell_times: Key dwell times
            flight_times: Inter-key flight times
            
        Returns:
            Array of FFT coefficients
        """
        # Combine dwell and flight times into single time series
        # Alternating pattern: dwell, flight, dwell, flight, ...
        combined = []
        for i in range(len(flight_times)):
            combined.append(dwell_times[i])
            combined.append(flight_times[i])
        if len(dwell_times) > len(flight_times):
            combined.append(dwell_times[-1])
        
        # Convert to numpy array
        timing_series = np.array(combined)
        
        # Remove mean
        timing_centered = timing_series - np.mean(timing_series)
        
        # Apply window
        window = np.hanning(len(timing_centered))
        windowed = timing_centered * window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        
        # Take magnitude and normalize
        fft_mag = np.abs(fft_result)
        
        # Use first N coefficients as signature
        num_coeffs = min(self.fft_coefficients, len(fft_mag))
        signature = fft_mag[:num_coeffs]
        
        # Normalize to unit vector
        norm = np.linalg.norm(signature)
        if norm > 0:
            signature = signature / norm
        
        # Pad with zeros if needed
        if len(signature) < self.fft_coefficients:
            signature = np.pad(signature, (0, self.fft_coefficients - len(signature)))
        
        return signature
    
    def authenticate(
        self,
        user_id: str,
        keystrokes: Optional[List[KeystrokeEvent]] = None,
        min_keystrokes: int = 20
    ) -> AuthenticationResult:
        """
        Authenticate user based on typing rhythm.
        
        Args:
            user_id: User claiming identity
            keystrokes: Recent keystrokes (uses buffer if None)
            min_keystrokes: Minimum keystrokes needed for authentication
            
        Returns:
            Authentication result
        """
        timestamp = time.time()
        
        # Check if baseline exists
        if user_id not in self.user_signatures:
            result = AuthenticationResult(
                user_id=user_id,
                status=AuthenticationStatus.INSUFFICIENT_DATA,
                confidence=0.0,
                timestamp=timestamp,
                description="No baseline signature for user",
                distance_from_baseline=float('inf')
            )
            self.authentication_results.append(result)
            return result
        
        # Get keystrokes
        if keystrokes is not None:
            keystroke_list = keystrokes
        elif user_id in self.user_keystrokes:
            keystroke_list = list(self.user_keystrokes[user_id])[-min_keystrokes:]
        else:
            result = AuthenticationResult(
                user_id=user_id,
                status=AuthenticationStatus.INSUFFICIENT_DATA,
                confidence=0.0,
                timestamp=timestamp,
                description="No keystroke data available",
                distance_from_baseline=float('inf')
            )
            self.authentication_results.append(result)
            return result
        
        if len(keystroke_list) < min_keystrokes:
            result = AuthenticationResult(
                user_id=user_id,
                status=AuthenticationStatus.INSUFFICIENT_DATA,
                confidence=0.0,
                timestamp=timestamp,
                description=f"Insufficient keystrokes: {len(keystroke_list)}/{min_keystrokes}",
                distance_from_baseline=float('inf')
            )
            self.authentication_results.append(result)
            return result
        
        # Compute current signature
        dwell_times = [k.dwell_time for k in keystroke_list]
        flight_times = []
        for i in range(len(keystroke_list) - 1):
            flight_time = keystroke_list[i].flight_time_to(keystroke_list[i + 1])
            if flight_time >= 0:
                flight_times.append(flight_time)
        
        if len(flight_times) < 5:
            result = AuthenticationResult(
                user_id=user_id,
                status=AuthenticationStatus.INSUFFICIENT_DATA,
                confidence=0.0,
                timestamp=timestamp,
                description="Insufficient timing data",
                distance_from_baseline=float('inf')
            )
            self.authentication_results.append(result)
            return result
        
        current_fft = self._compute_typing_fft_signature(dwell_times, flight_times)
        
        # Compare to baseline
        baseline = self.user_signatures[user_id]
        
        # Calculate Euclidean distance between FFT signatures
        fft_distance = np.linalg.norm(current_fft - baseline.fft_coefficients)
        
        # Calculate statistical distance
        current_mean_dwell = np.mean(dwell_times)
        current_mean_flight = np.mean(flight_times)
        
        # Minimum standard deviation threshold to avoid division by zero
        MIN_STD_THRESHOLD = 0.001
        
        # Calculate normalized distances
        dwell_distance = abs(current_mean_dwell - baseline.mean_dwell_time) / max(baseline.std_dwell_time, MIN_STD_THRESHOLD)
        flight_distance = abs(current_mean_flight - baseline.mean_flight_time) / max(baseline.std_flight_time, MIN_STD_THRESHOLD)
        
        # Combined distance (weighted)
        combined_distance = (0.6 * fft_distance + 0.2 * dwell_distance + 0.2 * flight_distance)
        
        # Determine authentication status
        if combined_distance < self.authentication_threshold:
            status = AuthenticationStatus.AUTHENTICATED
            confidence = 1.0 - (combined_distance / self.authentication_threshold)
            description = f"Authenticated: distance={combined_distance:.3f}"
        elif combined_distance < self.authentication_threshold * 2:
            status = AuthenticationStatus.SUSPICIOUS
            confidence = 0.5 - (combined_distance / (self.authentication_threshold * 2)) * 0.5
            description = f"Suspicious: distance={combined_distance:.3f}"
        else:
            status = AuthenticationStatus.REJECTED
            confidence = 0.0
            description = f"Rejected: distance={combined_distance:.3f}"
        
        result = AuthenticationResult(
            user_id=user_id,
            status=status,
            confidence=max(0.0, min(1.0, confidence)),
            timestamp=timestamp,
            description=description,
            distance_from_baseline=combined_distance,
            metadata={
                "fft_distance": float(fft_distance),
                "dwell_distance": float(dwell_distance),
                "flight_distance": float(flight_distance),
                "keystroke_count": len(keystroke_list)
            }
        )
        
        self.authentication_results.append(result)
        
        if status == AuthenticationStatus.REJECTED:
            logger.warning(f"Authentication rejected for {user_id}: distance={combined_distance:.3f}")
        elif status == AuthenticationStatus.SUSPICIOUS:
            logger.info(f"Suspicious authentication for {user_id}: distance={combined_distance:.3f}")
        
        return result
    
    def detect_impersonation(
        self,
        user_id: str,
        continuous_monitoring: bool = True,
        window_size: int = 30
    ) -> List[AuthenticationResult]:
        """
        Continuously monitor for impersonation attempts.
        
        Args:
            user_id: User to monitor
            continuous_monitoring: Check sliding windows
            window_size: Size of sliding window
            
        Returns:
            List of authentication results
        """
        if user_id not in self.user_keystrokes:
            return []
        
        keystrokes = list(self.user_keystrokes[user_id])
        
        if len(keystrokes) < window_size:
            return []
        
        results = []
        
        if continuous_monitoring:
            # Sliding window authentication
            for i in range(0, len(keystrokes) - window_size + 1, window_size // 2):
                window = keystrokes[i:i + window_size]
                result = self.authenticate(user_id, window, min_keystrokes=window_size)
                results.append(result)
        else:
            # Single authentication on recent keystrokes
            result = self.authenticate(user_id, keystrokes[-window_size:], min_keystrokes=window_size)
            results.append(result)
        
        return results
    
    def simulate_typing(
        self,
        user_id: str,
        text: str = "the quick brown fox jumps over the lazy dog",
        num_repetitions: int = 5,
        typing_speed_wpm: float = 60.0,
        add_noise: bool = True
    ) -> List[KeystrokeEvent]:
        """
        Simulate typing for testing.
        
        Args:
            user_id: User identifier
            text: Text to type
            num_repetitions: Number of times to repeat text
            typing_speed_wpm: Typing speed in words per minute
            add_noise: Add realistic timing noise
            
        Returns:
            List of simulated keystroke events
        """
        keystrokes = []
        
        # Calculate base timing from WPM
        # Average word length is 5 characters
        chars_per_second = (typing_speed_wpm * 5) / 60.0
        base_char_time = 1.0 / chars_per_second
        
        # Base dwell time (typical: 100-200ms)
        base_dwell = 0.15
        
        # User-specific rhythm (slight variations)
        user_factor = hash(user_id) % 100 / 500.0  # 0-0.2 variation
        
        current_time = time.time()
        
        for rep in range(num_repetitions):
            for char in (text + " "):  # Add space between repetitions
                # Dwell time with noise
                if add_noise:
                    dwell = base_dwell + user_factor + np.random.normal(0, 0.02)
                else:
                    dwell = base_dwell + user_factor
                
                dwell = max(0.05, min(0.3, dwell))  # Clamp to realistic range
                
                # Flight time with noise
                if add_noise:
                    flight = base_char_time - dwell + np.random.normal(0, 0.03)
                else:
                    flight = base_char_time - dwell
                
                flight = max(0.01, flight)  # Ensure positive
                
                press_time = current_time
                release_time = current_time + dwell
                
                event = KeystrokeEvent(
                    key=char,
                    press_time=press_time,
                    release_time=release_time,
                    user_id=user_id
                )
                
                keystrokes.append(event)
                self.add_keystroke(event)
                
                current_time = release_time + flight
        
        return keystrokes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "min_samples_for_baseline": self.min_samples_for_baseline,
            "authentication_threshold": self.authentication_threshold,
            "fft_coefficients": self.fft_coefficients,
            "users_tracked": len(self.user_keystrokes),
            "baselines_established": len(self.user_signatures),
            "authentication_attempts": len(self.authentication_results),
            "authentication_breakdown": self._get_auth_breakdown()
        }
    
    def _get_auth_breakdown(self) -> Dict[str, int]:
        """Get count of authentication results by status."""
        breakdown = {}
        for result in self.authentication_results:
            status = result.status.value
            breakdown[status] = breakdown.get(status, 0) + 1
        return breakdown
    
    def get_user_baseline(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get baseline information for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with baseline info or None
        """
        if user_id not in self.user_signatures:
            return None
        
        signature = self.user_signatures[user_id]
        
        return {
            "user_id": signature.user_id,
            "mean_dwell_time_ms": signature.mean_dwell_time * 1000,
            "std_dwell_time_ms": signature.std_dwell_time * 1000,
            "mean_flight_time_ms": signature.mean_flight_time * 1000,
            "std_flight_time_ms": signature.std_flight_time * 1000,
            "sample_count": signature.sample_count,
            "timestamp": signature.timestamp,
            "fft_signature_norm": float(np.linalg.norm(signature.fft_coefficients))
        }
