"""
Vibration Frequency Analyzer Plugin

Analyzes vibration patterns for bearing defects and structural health monitoring.
Performs FFT analysis across seismic, machinery, and bearing frequency bands.

Features:
- Bearing defect detection (BPFO, BPFI, BSF, FTF calculations)
- Structural health monitoring
- Frequency bands: seismic (0.1-10Hz), machinery (10-1000Hz), bearing (1-10kHz)
- Sample rate: 10 kHz minimum, configurable
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class BearingDefect(str, Enum):
    """Types of bearing defects that can be detected."""
    BPFO = "bpfo"  # Ball Pass Frequency Outer race
    BPFI = "bpfi"  # Ball Pass Frequency Inner race
    BSF = "bsf"    # Ball Spin Frequency
    FTF = "ftf"    # Fundamental Train Frequency (cage)
    NONE = "none"


class VibrationSeverity(str, Enum):
    """Vibration severity levels per ISO 10816."""
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    UNSATISFACTORY = "unsatisfactory"
    UNACCEPTABLE = "unacceptable"


@dataclass
class BearingGeometry:
    """Bearing geometry parameters for defect frequency calculation."""
    pitch_diameter_mm: float  # Bearing pitch diameter
    ball_diameter_mm: float   # Rolling element diameter
    num_balls: int            # Number of rolling elements
    contact_angle_deg: float  # Contact angle
    shaft_speed_rpm: float    # Shaft rotation speed


@dataclass
class VibrationEvent:
    """Represents a detected vibration event."""
    timestamp: float
    frequency_hz: float
    amplitude_g: float  # Acceleration in g's
    defect_type: BearingDefect
    severity: VibrationSeverity
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralHealth:
    """Structural health assessment based on vibration analysis."""
    structure_id: str
    health_score: float  # 0.0 (critical) to 1.0 (healthy)
    resonance_frequencies: List[float]
    damping_ratios: List[float]
    anomalies: List[str]
    timestamp: float


class VibrationFrequencyAnalyzer:
    """
    Vibration Frequency Analyzer for machinery diagnostics and structural monitoring.
    
    Detects bearing defects, analyzes machinery vibration patterns, and monitors
    structural health through frequency domain analysis.
    
    Attributes:
        sample_rate_hz: Sample rate for vibration data (default 10 kHz)
        fft_size: FFT window size
        sensitivity_g: Accelerometer sensitivity in g's
    """
    
    def __init__(
        self,
        sample_rate_hz: float = 10000.0,  # 10 kHz
        fft_size: int = 2048,
        sensitivity_g: float = 0.001  # 0.001g = 1 milli-g
    ):
        """
        Initialize Vibration Frequency Analyzer.
        
        Args:
            sample_rate_hz: Sample rate for vibration measurements
            fft_size: FFT window size
            sensitivity_g: Minimum detectable acceleration
        """
        self.sample_rate_hz = sample_rate_hz
        self.fft_size = fft_size
        self.sensitivity_g = sensitivity_g
        
        # Detected events
        self.detected_events: List[VibrationEvent] = []
        
        # Bearing databases
        self.bearing_geometries: Dict[str, BearingGeometry] = {}
        
        # Structural baselines
        self.structural_baselines: Dict[str, StructuralHealth] = {}
        
        logger.info(f"Vibration Analyzer initialized: {sample_rate_hz/1000:.1f} kHz sample rate, "
                   f"sensitivity: {sensitivity_g*1000:.3f} mg")
    
    def calculate_bearing_frequencies(
        self,
        geometry: BearingGeometry
    ) -> Dict[str, float]:
        """
        Calculate characteristic bearing defect frequencies.
        
        These frequencies help identify specific bearing faults:
        - BPFO: Outer race defect
        - BPFI: Inner race defect
        - BSF: Rolling element defect
        - FTF: Cage defect
        
        Args:
            geometry: Bearing geometry parameters
            
        Returns:
            Dictionary of defect frequencies
        """
        # Convert to radians
        contact_angle_rad = np.deg2rad(geometry.contact_angle_deg)
        
        # Calculate shaft frequency
        shaft_freq_hz = geometry.shaft_speed_rpm / 60.0
        
        # Diameter ratio
        d_ratio = geometry.ball_diameter_mm / geometry.pitch_diameter_mm
        
        # Fundamental Train Frequency (cage frequency)
        ftf = (shaft_freq_hz / 2) * (1 - d_ratio * np.cos(contact_angle_rad))
        
        # Ball Pass Frequency Outer race
        bpfo = (geometry.num_balls * shaft_freq_hz / 2) * (1 - d_ratio * np.cos(contact_angle_rad))
        
        # Ball Pass Frequency Inner race
        bpfi = (geometry.num_balls * shaft_freq_hz / 2) * (1 + d_ratio * np.cos(contact_angle_rad))
        
        # Ball Spin Frequency
        bsf = (geometry.pitch_diameter_mm / (2 * geometry.ball_diameter_mm)) * shaft_freq_hz * \
              (1 - (d_ratio * np.cos(contact_angle_rad)) ** 2)
        
        frequencies = {
            "shaft_freq": shaft_freq_hz,
            "ftf": ftf,
            "bpfo": bpfo,
            "bpfi": bpfi,
            "bsf": bsf
        }
        
        logger.debug(f"Bearing frequencies: BPFO={bpfo:.2f} Hz, BPFI={bpfi:.2f} Hz, "
                    f"BSF={bsf:.2f} Hz, FTF={ftf:.2f} Hz")
        
        return frequencies
    
    def analyze_vibration(
        self,
        vibration_data: Optional[np.ndarray] = None,
        bearing_id: Optional[str] = None,
        duration_seconds: float = 1.0
    ) -> List[VibrationEvent]:
        """
        Analyze vibration data for defects and anomalies.
        
        Args:
            vibration_data: Vibration measurements (if None, generates synthetic)
            bearing_id: Bearing identifier for defect analysis
            duration_seconds: Duration to analyze
            
        Returns:
            List of detected vibration events
        """
        # Generate or use provided data
        if vibration_data is None:
            vibration_data = self._simulate_vibration_data(duration_seconds)
        
        # Compute FFT
        freqs, power_spectrum = self._compute_fft(vibration_data)
        
        # Detect events
        events = []
        
        # If bearing ID provided, check for bearing defects
        if bearing_id and bearing_id in self.bearing_geometries:
            bearing_events = self._detect_bearing_defects(
                freqs, power_spectrum, self.bearing_geometries[bearing_id]
            )
            events.extend(bearing_events)
        
        # General vibration analysis
        general_events = self._analyze_general_vibration(freqs, power_spectrum)
        events.extend(general_events)
        
        self.detected_events.extend(events)
        
        return events
    
    def _simulate_vibration_data(self, duration: float) -> np.ndarray:
        """
        Generate synthetic vibration data for simulation.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Synthetic vibration samples
        """
        num_samples = int(self.sample_rate_hz * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Background noise
        vibration = np.random.normal(0, 0.01, num_samples)
        
        # Add some machinery vibration (fundamental + harmonics)
        fund_freq = 30  # 30 Hz (1800 RPM)
        vibration += 0.1 * np.sin(2 * np.pi * fund_freq * t)
        vibration += 0.05 * np.sin(2 * np.pi * 2 * fund_freq * t)
        vibration += 0.02 * np.sin(2 * np.pi * 3 * fund_freq * t)
        
        # Randomly add a defect signature
        if np.random.random() > 0.5:
            defect_freq = 120  # Hz
            vibration += 0.08 * np.sin(2 * np.pi * defect_freq * t)
        
        return vibration
    
    def _compute_fft(
        self,
        vibration_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of vibration data.
        
        Args:
            vibration_data: Vibration samples
            
        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        # Apply Hanning window
        window = np.hanning(len(vibration_data))
        windowed_data = vibration_data * window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed_data, n=self.fft_size)
        
        # Compute power spectrum
        power_spectrum = np.abs(fft_result) ** 2
        
        # Frequency bins
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate_hz)
        
        return freqs, power_spectrum
    
    def _detect_bearing_defects(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray,
        geometry: BearingGeometry,
        tolerance_hz: float = 2.0
    ) -> List[VibrationEvent]:
        """
        Detect bearing defects by matching spectrum peaks to characteristic frequencies.
        
        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            geometry: Bearing geometry
            tolerance_hz: Frequency matching tolerance
            
        Returns:
            List of detected bearing defect events
        """
        events = []
        timestamp = time.time()
        
        # Calculate bearing frequencies
        bearing_freqs = self.calculate_bearing_frequencies(geometry)
        
        # Find peaks in spectrum
        peaks = self._find_spectrum_peaks(freqs, power_spectrum)
        
        # Map defect types to frequencies
        defect_mapping = {
            BearingDefect.BPFO: bearing_freqs["bpfo"],
            BearingDefect.BPFI: bearing_freqs["bpfi"],
            BearingDefect.BSF: bearing_freqs["bsf"],
            BearingDefect.FTF: bearing_freqs["ftf"]
        }
        
        # Check each peak against bearing frequencies
        for peak_freq, peak_power in peaks:
            for defect_type, expected_freq in defect_mapping.items():
                if abs(peak_freq - expected_freq) <= tolerance_hz:
                    # Calculate amplitude in g's
                    amplitude_g = np.sqrt(peak_power) * self.sensitivity_g
                    
                    # Determine severity based on amplitude
                    severity = self._classify_vibration_severity(amplitude_g)
                    
                    # Calculate confidence based on frequency match
                    confidence = 1.0 - min(1.0, abs(peak_freq - expected_freq) / tolerance_hz)
                    
                    event = VibrationEvent(
                        timestamp=timestamp,
                        frequency_hz=peak_freq,
                        amplitude_g=amplitude_g,
                        defect_type=defect_type,
                        severity=severity,
                        confidence=confidence,
                        metadata={
                            "expected_freq": expected_freq,
                            "freq_error": abs(peak_freq - expected_freq),
                            "shaft_speed_rpm": geometry.shaft_speed_rpm
                        }
                    )
                    
                    events.append(event)
                    logger.warning(f"Bearing defect detected: {defect_type.value} at {peak_freq:.2f} Hz, "
                                 f"severity: {severity.value}")
        
        return events
    
    def _analyze_general_vibration(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray
    ) -> List[VibrationEvent]:
        """
        Analyze general vibration characteristics.
        
        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            
        Returns:
            List of detected vibration events
        """
        events = []
        timestamp = time.time()
        
        # Calculate overall vibration level
        total_power = np.sum(power_spectrum)
        rms_g = np.sqrt(total_power) * self.sensitivity_g
        
        # Analyze frequency bands
        bands = {
            "seismic": (0.1, 10),      # Seismic vibrations
            "machinery": (10, 1000),    # Machinery vibrations
            "bearing": (1000, 10000)    # Bearing vibrations
        }
        
        for band_name, (f_min, f_max) in bands.items():
            mask = (freqs >= f_min) & (freqs <= f_max)
            band_power = np.sum(power_spectrum[mask])
            
            if band_power > total_power * 0.1:  # If band has >10% of total energy
                band_rms_g = np.sqrt(band_power) * self.sensitivity_g
                severity = self._classify_vibration_severity(band_rms_g)
                
                # Find peak in band
                band_spectrum = power_spectrum[mask]
                if len(band_spectrum) > 0:
                    peak_idx = np.argmax(band_spectrum)
                    peak_freq = freqs[mask][peak_idx]
                    
                    event = VibrationEvent(
                        timestamp=timestamp,
                        frequency_hz=peak_freq,
                        amplitude_g=band_rms_g,
                        defect_type=BearingDefect.NONE,
                        severity=severity,
                        confidence=0.7,
                        metadata={
                            "band": band_name,
                            "band_power_ratio": float(band_power / total_power)
                        }
                    )
                    events.append(event)
        
        return events
    
    def _classify_vibration_severity(self, amplitude_g: float) -> VibrationSeverity:
        """
        Classify vibration severity based on ISO 10816 standards.
        
        Args:
            amplitude_g: Vibration amplitude in g's
            
        Returns:
            Vibration severity classification
        """
        # Convert to mm/s for ISO 10816 (approximate)
        # 1g ≈ 9.81 m/s² at 1 Hz, but frequency dependent
        # Using simplified thresholds
        
        if amplitude_g < 0.01:  # < 10 mg
            return VibrationSeverity.GOOD
        elif amplitude_g < 0.05:  # < 50 mg
            return VibrationSeverity.ACCEPTABLE
        elif amplitude_g < 0.1:  # < 100 mg
            return VibrationSeverity.UNSATISFACTORY
        else:
            return VibrationSeverity.UNACCEPTABLE
    
    def _find_spectrum_peaks(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray,
        threshold_factor: float = 3.0
    ) -> List[Tuple[float, float]]:
        """
        Find peaks in power spectrum.
        
        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            threshold_factor: Factor above mean to consider a peak
            
        Returns:
            List of (frequency, power) tuples
        """
        mean_power = np.mean(power_spectrum)
        threshold = mean_power * threshold_factor
        
        peaks = []
        
        for i in range(1, len(power_spectrum) - 1):
            if (power_spectrum[i] > threshold and
                power_spectrum[i] > power_spectrum[i-1] and
                power_spectrum[i] > power_spectrum[i+1]):
                peaks.append((freqs[i], power_spectrum[i]))
        
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:20]  # Top 20 peaks
    
    def monitor_structural_health(
        self,
        structure_id: str,
        vibration_data: Optional[np.ndarray] = None,
        duration_seconds: float = 10.0
    ) -> StructuralHealth:
        """
        Monitor structural health through vibration analysis.
        
        Detects resonance frequencies, damping, and structural anomalies.
        
        Args:
            structure_id: Unique identifier for structure
            vibration_data: Vibration measurements
            duration_seconds: Duration to analyze
            
        Returns:
            Structural health assessment
        """
        if vibration_data is None:
            vibration_data = self._simulate_vibration_data(duration_seconds)
        
        # Compute FFT
        freqs, power_spectrum = self._compute_fft(vibration_data)
        
        # Find resonance frequencies (major peaks)
        peaks = self._find_spectrum_peaks(freqs, power_spectrum, threshold_factor=5.0)
        resonance_frequencies = [freq for freq, _ in peaks[:5]]
        
        # Estimate damping ratios (simplified)
        damping_ratios = self._estimate_damping_ratios(freqs, power_spectrum, resonance_frequencies)
        
        # Detect anomalies
        anomalies = []
        health_score = 1.0
        
        # Check for unexpected high-frequency content
        high_freq_energy = np.sum(power_spectrum[freqs > 1000])
        total_energy = np.sum(power_spectrum)
        
        if high_freq_energy / total_energy > 0.3:
            anomalies.append("High frequency content - possible structural damage or cracking")
            health_score -= 0.3
        
        # Check for resonance shifts (if baseline exists)
        if structure_id in self.structural_baselines:
            baseline = self.structural_baselines[structure_id]
            if baseline.resonance_frequencies:
                # Compare first resonance frequency
                freq_shift = abs(resonance_frequencies[0] - baseline.resonance_frequencies[0])
                shift_percent = freq_shift / baseline.resonance_frequencies[0] * 100
                
                if shift_percent > 5:
                    anomalies.append(f"Resonance frequency shifted {shift_percent:.1f}% - possible stiffness change")
                    health_score -= 0.2
        
        # Check damping
        if damping_ratios and np.mean(damping_ratios) < 0.01:
            anomalies.append("Low damping - structure may be prone to resonance")
            health_score -= 0.1
        
        health_score = max(0.0, health_score)
        
        health = StructuralHealth(
            structure_id=structure_id,
            health_score=health_score,
            resonance_frequencies=resonance_frequencies,
            damping_ratios=damping_ratios,
            anomalies=anomalies,
            timestamp=time.time()
        )
        
        # Store/update baseline
        if structure_id not in self.structural_baselines:
            self.structural_baselines[structure_id] = health
        
        logger.info(f"Structural health for {structure_id}: score={health_score:.2f}, "
                   f"resonances={len(resonance_frequencies)}, anomalies={len(anomalies)}")
        
        return health
    
    def _estimate_damping_ratios(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray,
        resonance_frequencies: List[float]
    ) -> List[float]:
        """
        Estimate damping ratios at resonance frequencies.
        
        Uses half-power bandwidth method.
        
        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            resonance_frequencies: Resonance frequencies to analyze
            
        Returns:
            List of damping ratios
        """
        damping_ratios = []
        
        for res_freq in resonance_frequencies:
            # Find index of resonance
            res_idx = np.argmin(np.abs(freqs - res_freq))
            res_power = power_spectrum[res_idx]
            
            # Half-power points
            half_power = res_power / 2
            
            # Find bandwidth at half power
            left_idx = res_idx
            while left_idx > 0 and power_spectrum[left_idx] > half_power:
                left_idx -= 1
            
            right_idx = res_idx
            while right_idx < len(power_spectrum) - 1 and power_spectrum[right_idx] > half_power:
                right_idx += 1
            
            bandwidth = freqs[right_idx] - freqs[left_idx]
            
            # Damping ratio ≈ bandwidth / (2 * resonance_freq)
            damping = bandwidth / (2 * res_freq) if res_freq > 0 else 0
            damping_ratios.append(damping)
        
        return damping_ratios
    
    def register_bearing(
        self,
        bearing_id: str,
        geometry: BearingGeometry
    ):
        """
        Register a bearing for monitoring.
        
        Args:
            bearing_id: Unique identifier
            geometry: Bearing geometry parameters
        """
        self.bearing_geometries[bearing_id] = geometry
        logger.info(f"Registered bearing {bearing_id}: {geometry.num_balls} balls, "
                   f"{geometry.shaft_speed_rpm} RPM")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "fft_size": self.fft_size,
            "sensitivity_g": self.sensitivity_g,
            "total_events_detected": len(self.detected_events),
            "bearings_monitored": len(self.bearing_geometries),
            "structures_monitored": len(self.structural_baselines),
            "event_breakdown": self._get_event_breakdown()
        }
    
    def _get_event_breakdown(self) -> Dict[str, int]:
        """Get count of events by defect type."""
        breakdown = {}
        for event in self.detected_events:
            defect_type = event.defect_type.value
            breakdown[defect_type] = breakdown.get(defect_type, 0) + 1
        return breakdown
