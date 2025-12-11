"""
Acoustic Frequency Analyzer Plugin

Performs FFT analysis of sound across infrasound, audible, and ultrasound ranges.
Detects acoustic signatures like gunshots, glass breaks, machinery, drones, and alarms.

Features:
- FFT analysis: infrasound (0-20Hz), audible (20-20kHz), ultrasound (20-100kHz)
- Signature recognition for security and environmental monitoring
- Machinery diagnostics via harmonics analysis
- Covert ultrasonic communication detection
- Sample rate: 44.1 kHz (CD quality) default, configurable up to 192 kHz
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class AcousticSignature(str, Enum):
    """Types of acoustic signatures that can be detected."""

    GUNSHOT = "gunshot"
    GLASS_BREAK = "glass_break"
    MACHINERY = "machinery"
    DRONE = "drone"
    ALARM = "alarm"
    EXPLOSION = "explosion"
    VEHICLE = "vehicle"
    HUMAN_VOICE = "human_voice"
    ULTRASONIC_COMM = "ultrasonic_communication"
    UNKNOWN = "unknown"


class FrequencyRange(str, Enum):
    """Acoustic frequency ranges."""

    INFRASOUND = "infrasound"  # 0-20 Hz
    LOW_AUDIO = "low_audio"  # 20-250 Hz
    MID_AUDIO = "mid_audio"  # 250-2000 Hz
    HIGH_AUDIO = "high_audio"  # 2-20 kHz
    ULTRASOUND = "ultrasound"  # 20-100 kHz


@dataclass
class AcousticEvent:
    """Represents a detected acoustic event."""

    signature: AcousticSignature
    timestamp: float
    frequency_hz: float
    amplitude: float
    duration_seconds: float
    confidence: float
    frequency_profile: np.ndarray = field(repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MachineryDiagnostic:
    """Diagnostic information for machinery based on acoustic analysis."""

    machine_id: str
    health_score: float  # 0.0 (critical) to 1.0 (healthy)
    fundamental_freq_hz: float
    harmonics: List[float]
    anomalies: List[str]
    timestamp: float


class AcousticFrequencyAnalyzer:
    """
    Acoustic Frequency Analyzer for sound analysis and signature recognition.

    Uses FFT to analyze acoustic signals across infrasound, audible, and ultrasound ranges.
    Detects security threats, machinery issues, and covert communications.

    Attributes:
        sample_rate_hz: ADC sample rate (default 44.1 kHz)
        fft_size: FFT window size
        overlap: Window overlap ratio for STFT
    """

    def __init__(
        self,
        sample_rate_hz: float = 44100.0,  # CD quality
        fft_size: int = 2048,
        overlap: float = 0.5,
    ):
        """
        Initialize Acoustic Frequency Analyzer.

        Args:
            sample_rate_hz: Sample rate for audio capture
            fft_size: FFT window size
            overlap: Overlap ratio for Short-Time Fourier Transform
        """
        self.sample_rate_hz = sample_rate_hz
        self.fft_size = fft_size
        self.overlap = overlap

        # Signature database
        self.signature_database = self._initialize_signature_database()

        # Detected events
        self.detected_events: List[AcousticEvent] = []

        # Machinery baselines for diagnostics
        self.machinery_baselines: Dict[str, MachineryDiagnostic] = {}

        # Ultrasonic communication detection
        self.ultrasonic_threshold_hz = 18000.0  # Above human hearing

        logger.info(
            f"Acoustic Analyzer initialized: {sample_rate_hz/1000:.1f} kHz sample rate, "
            f"FFT size: {fft_size}"
        )

    def _initialize_signature_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of known acoustic signatures."""
        return {
            "gunshot": {
                "freq_range": (150, 5000),  # Hz
                "duration_range": (0.001, 0.01),  # seconds (1-10ms)
                "peak_freq": 500,
                "energy_ratio": 0.9,  # High energy concentration
            },
            "glass_break": {
                "freq_range": (2000, 15000),
                "duration_range": (0.1, 0.5),
                "peak_freq": 5000,
                "energy_ratio": 0.7,
            },
            "alarm": {
                "freq_range": (800, 3000),
                "duration_range": (0.5, 5.0),
                "peak_freq": 1000,
                "periodicity": True,  # Repeating pattern
            },
            "drone": {
                "freq_range": (100, 2000),
                "duration_range": (1.0, 60.0),
                "peak_freq": 300,
                "harmonics": True,  # Multiple harmonics present
            },
            "explosion": {
                "freq_range": (20, 500),
                "duration_range": (0.1, 2.0),
                "peak_freq": 100,
                "energy_ratio": 0.95,
            },
        }

    def analyze_audio_sample(
        self,
        audio_data: Optional[np.ndarray] = None,
        duration_seconds: float = 1.0,
    ) -> List[AcousticEvent]:
        """
        Analyze audio sample for acoustic signatures.

        Args:
            audio_data: Audio samples (if None, generates synthetic data)
            duration_seconds: Duration of audio to analyze

        Returns:
            List of detected acoustic events
        """
        # Generate or use provided audio data
        if audio_data is None:
            audio_data = self._simulate_audio_sample(duration_seconds)

        # Perform FFT analysis
        freqs, power_spectrum = self._compute_fft(audio_data)

        # Detect signatures
        events = self._detect_signatures(freqs, power_spectrum, audio_data)

        # Detect ultrasonic communication
        ultrasonic_events = self._detect_ultrasonic_communication(
            freqs, power_spectrum
        )
        events.extend(ultrasonic_events)

        self.detected_events.extend(events)

        return events

    def _simulate_audio_sample(self, duration: float) -> np.ndarray:
        """
        Generate synthetic audio data for simulation.

        Args:
            duration: Duration in seconds

        Returns:
            Synthetic audio samples
        """
        num_samples = int(self.sample_rate_hz * duration)
        t = np.linspace(0, duration, num_samples)

        # Create background noise
        audio = np.random.normal(0, 0.01, num_samples)

        # Randomly add some events for testing
        if np.random.random() > 0.7:
            # Add a tone (could be machinery, alarm, etc.)
            freq = np.random.choice([300, 500, 1000, 2000])
            audio += 0.3 * np.sin(2 * np.pi * freq * t)

        return audio

    def _compute_fft(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of audio data.

        Args:
            audio_data: Audio samples

        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(audio_data))
        windowed_data = audio_data * window

        # Compute FFT
        fft_result = np.fft.rfft(windowed_data, n=self.fft_size)

        # Compute power spectrum
        power_spectrum = np.abs(fft_result) ** 2

        # Frequency bins
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate_hz)

        return freqs, power_spectrum

    def _detect_signatures(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray,
        audio_data: np.ndarray,
    ) -> List[AcousticEvent]:
        """
        Detect acoustic signatures from frequency spectrum.

        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            audio_data: Original audio data

        Returns:
            List of detected events
        """
        events = []
        timestamp = time.time()

        # Find peaks in spectrum
        peaks = self._find_spectrum_peaks(freqs, power_spectrum)

        if not peaks:
            return events

        # Check each peak against signature database
        for peak_freq, peak_power in peaks:
            # Match against known signatures
            best_match = None
            best_confidence = 0.0

            for sig_name, sig_info in self.signature_database.items():
                freq_min, freq_max = sig_info["freq_range"]

                if freq_min <= peak_freq <= freq_max:
                    # Calculate confidence based on frequency match
                    confidence = (
                        1.0 - abs(peak_freq - sig_info["peak_freq"]) / freq_max
                    )
                    confidence = max(0.0, min(1.0, confidence))

                    if confidence > best_confidence and confidence > 0.5:
                        best_confidence = confidence
                        best_match = sig_name

            if best_match:
                event = AcousticEvent(
                    signature=AcousticSignature(best_match),
                    timestamp=timestamp,
                    frequency_hz=peak_freq,
                    amplitude=np.sqrt(peak_power),
                    duration_seconds=len(audio_data) / self.sample_rate_hz,
                    confidence=best_confidence,
                    frequency_profile=power_spectrum,
                    metadata={"peak_power": float(peak_power)},
                )
                events.append(event)
                logger.info(
                    f"Detected {best_match}: {peak_freq:.1f} Hz, confidence: {best_confidence:.2f}"
                )

        return events

    def _find_spectrum_peaks(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray,
        threshold_factor: float = 3.0,
    ) -> List[Tuple[float, float]]:
        """
        Find peaks in power spectrum.

        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            threshold_factor: Factor above mean to consider a peak

        Returns:
            List of (frequency, power) tuples for peaks
        """
        # Calculate threshold
        mean_power = np.mean(power_spectrum)
        threshold = mean_power * threshold_factor

        peaks = []

        # Simple peak detection
        for i in range(1, len(power_spectrum) - 1):
            if (
                power_spectrum[i] > threshold
                and power_spectrum[i] > power_spectrum[i - 1]
                and power_spectrum[i] > power_spectrum[i + 1]
            ):
                peaks.append((freqs[i], power_spectrum[i]))

        # Sort by power
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Return top peaks
        return peaks[:10]

    def _detect_ultrasonic_communication(
        self, freqs: np.ndarray, power_spectrum: np.ndarray
    ) -> List[AcousticEvent]:
        """
        Detect covert ultrasonic communication.

        Ultrasonic signals above human hearing range may indicate covert channels.

        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum

        Returns:
            List of ultrasonic communication events
        """
        events = []

        # Check for significant energy in ultrasonic range
        ultrasonic_mask = freqs >= self.ultrasonic_threshold_hz

        if not np.any(ultrasonic_mask):
            return events

        ultrasonic_power = power_spectrum[ultrasonic_mask]
        ultrasonic_freqs = freqs[ultrasonic_mask]

        # Check if ultrasonic power exceeds threshold
        mean_audible_power = np.mean(
            power_spectrum[freqs < self.ultrasonic_threshold_hz]
        )
        mean_ultrasonic_power = np.mean(ultrasonic_power)

        if mean_ultrasonic_power > mean_audible_power * 2.0:
            # Potential ultrasonic communication
            peak_idx = np.argmax(ultrasonic_power)
            peak_freq = ultrasonic_freqs[peak_idx]

            event = AcousticEvent(
                signature=AcousticSignature.ULTRASONIC_COMM,
                timestamp=time.time(),
                frequency_hz=peak_freq,
                amplitude=np.sqrt(ultrasonic_power[peak_idx]),
                duration_seconds=0.0,  # Continuous
                confidence=0.8,
                frequency_profile=power_spectrum,
                metadata={
                    "ultrasonic_power_ratio": float(
                        mean_ultrasonic_power / mean_audible_power
                    )
                },
            )
            events.append(event)
            logger.warning(
                f"Ultrasonic communication detected at {peak_freq:.1f} Hz"
            )

        return events

    def diagnose_machinery(
        self,
        machine_id: str,
        audio_data: Optional[np.ndarray] = None,
        duration_seconds: float = 5.0,
    ) -> MachineryDiagnostic:
        """
        Diagnose machinery health from acoustic signature.

        Analyzes harmonics and frequency patterns to detect issues like
        bearing wear, imbalance, misalignment, etc.

        Args:
            machine_id: Unique identifier for machine
            audio_data: Audio samples from machine
            duration_seconds: Duration to analyze

        Returns:
            Machinery diagnostic report
        """
        if audio_data is None:
            audio_data = self._simulate_audio_sample(duration_seconds)

        # Compute FFT
        freqs, power_spectrum = self._compute_fft(audio_data)

        # Find fundamental frequency and harmonics
        peaks = self._find_spectrum_peaks(freqs, power_spectrum)

        if not peaks:
            # No clear signal
            return MachineryDiagnostic(
                machine_id=machine_id,
                health_score=0.5,
                fundamental_freq_hz=0.0,
                harmonics=[],
                anomalies=["No clear acoustic signature detected"],
                timestamp=time.time(),
            )

        fundamental_freq = peaks[0][0]

        # Find harmonics (multiples of fundamental)
        harmonics = []
        for freq, power in peaks[1:6]:  # Check next 5 peaks
            harmonic_ratio = freq / fundamental_freq
            if abs(harmonic_ratio - round(harmonic_ratio)) < 0.1:
                harmonics.append(freq)

        # Analyze for anomalies
        anomalies = []
        health_score = 1.0

        # Check for excessive harmonics (indicates wear)
        if len(harmonics) > 5:
            anomalies.append(
                "Excessive harmonics detected - possible bearing wear"
            )
            health_score -= 0.2

        # Check for sidebands (indicates modulation/issues)
        sidebands = self._detect_sidebands(
            freqs, power_spectrum, fundamental_freq
        )
        if sidebands:
            anomalies.append(
                f"Sidebands detected at {len(sidebands)} frequencies - possible misalignment"
            )
            health_score -= 0.15

        # Check for noise floor elevation
        noise_floor = np.median(power_spectrum)
        signal_to_noise = peaks[0][1] / noise_floor if noise_floor > 0 else 0

        if signal_to_noise < 10:
            anomalies.append(
                "Low signal-to-noise ratio - sensor issue or excessive vibration"
            )
            health_score -= 0.1

        health_score = max(0.0, health_score)

        diagnostic = MachineryDiagnostic(
            machine_id=machine_id,
            health_score=health_score,
            fundamental_freq_hz=fundamental_freq,
            harmonics=harmonics,
            anomalies=anomalies,
            timestamp=time.time(),
        )

        # Store baseline if first diagnostic
        if machine_id not in self.machinery_baselines:
            self.machinery_baselines[machine_id] = diagnostic

        logger.info(
            f"Machinery diagnostic for {machine_id}: health={health_score:.2f}, "
            f"freq={fundamental_freq:.1f} Hz, harmonics={len(harmonics)}"
        )

        return diagnostic

    def _detect_sidebands(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray,
        fundamental_freq: float,
        tolerance: float = 5.0,
    ) -> List[float]:
        """
        Detect sidebands around fundamental frequency.

        Sidebands indicate modulation which can reveal machinery issues.

        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            fundamental_freq: Fundamental frequency
            tolerance: Frequency tolerance in Hz

        Returns:
            List of sideband frequencies
        """
        sidebands = []

        # Find fundamental index
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))

        # Check for peaks near fundamental (but not at fundamental)
        search_range = int(tolerance * len(freqs) / (self.sample_rate_hz / 2))

        for offset in range(1, search_range):
            # Check both sides
            for idx in [fund_idx - offset, fund_idx + offset]:
                if 0 <= idx < len(power_spectrum):
                    if power_spectrum[idx] > power_spectrum[fund_idx] * 0.3:
                        sidebands.append(freqs[idx])

        return sidebands

    def get_frequency_range_analysis(
        self, audio_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Analyze energy distribution across frequency ranges.

        Args:
            audio_data: Audio samples

        Returns:
            Dictionary with energy per frequency range
        """
        if audio_data is None:
            audio_data = self._simulate_audio_sample(1.0)

        freqs, power_spectrum = self._compute_fft(audio_data)

        # Calculate energy in each range
        ranges = {
            "infrasound": (0, 20),
            "low_audio": (20, 250),
            "mid_audio": (250, 2000),
            "high_audio": (2000, 20000),
            "ultrasound": (20000, self.sample_rate_hz / 2),
        }

        energy_dist = {}
        total_energy = np.sum(power_spectrum)

        for range_name, (f_min, f_max) in ranges.items():
            mask = (freqs >= f_min) & (freqs <= f_max)
            range_energy = np.sum(power_spectrum[mask])
            energy_dist[range_name] = float(
                range_energy / total_energy if total_energy > 0 else 0
            )

        return energy_dist

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "fft_size": self.fft_size,
            "max_frequency_hz": self.sample_rate_hz / 2,
            "total_events_detected": len(self.detected_events),
            "event_breakdown": self._get_event_breakdown(),
            "machinery_monitored": len(self.machinery_baselines),
        }

    def _get_event_breakdown(self) -> Dict[str, int]:
        """Get count of events by signature type."""
        breakdown = {}
        for event in self.detected_events:
            sig_type = event.signature.value
            breakdown[sig_type] = breakdown.get(sig_type, 0) + 1
        return breakdown
