"""
RF Spectrum Analyzer Plugin

Analyzes radio frequency spectrum from VLF to EHF bands (3 kHz - 300 GHz).
Detects various RF signals, jamming, and rogue transmitters.

Features:
- Frequency scanning across configurable bands (default: 20 MHz - 6 GHz)
- Signal classification (WiFi, cellular, IoT, drones, GPS)
- RF jamming detection via abnormal noise floor analysis
- Rogue transmitter detection for unauthorized signals
- Sensitivity: -120 dBm minimum
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Types of RF signals that can be detected."""
    WIFI = "wifi"
    CELLULAR = "cellular"
    IOT = "iot"
    DRONE = "drone"
    GPS = "gps"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    UNKNOWN = "unknown"
    ROGUE = "rogue"


class FrequencyBand(str, Enum):
    """Standard frequency bands."""
    VLF = "vlf"  # 3-30 kHz
    LF = "lf"    # 30-300 kHz
    MF = "mf"    # 300 kHz - 3 MHz
    HF = "hf"    # 3-30 MHz
    VHF = "vhf"  # 30-300 MHz
    UHF = "uhf"  # 300 MHz - 3 GHz
    SHF = "shf"  # 3-30 GHz
    EHF = "ehf"  # 30-300 GHz


@dataclass
class RFSignal:
    """Represents a detected RF signal."""
    frequency_hz: float
    power_dbm: float
    bandwidth_hz: float
    signal_type: SignalType
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RFThreat:
    """Represents a detected RF threat."""
    threat_type: str  # "jamming", "rogue_transmitter", "interference"
    severity: float  # 0.0 to 1.0
    frequency_hz: float
    description: str
    timestamp: float
    affected_signals: List[RFSignal] = field(default_factory=list)


class RFSpectrumAnalyzer:
    """
    RF Spectrum Analyzer for detecting and classifying radio frequency signals.
    
    Attributes:
        scan_range: Tuple of (min_freq_hz, max_freq_hz) for scanning
        sensitivity_dbm: Minimum detectable signal power in dBm
        sample_rate_hz: ADC sample rate for SDR
        fft_size: FFT window size for frequency analysis
    """
    
    def __init__(
        self,
        scan_range: Tuple[float, float] = (20e6, 6e9),  # 20 MHz to 6 GHz
        sensitivity_dbm: float = -120.0,
        sample_rate_hz: float = 20e6,  # 20 MHz sample rate
        fft_size: int = 2048
    ):
        """
        Initialize RF Spectrum Analyzer.
        
        Args:
            scan_range: Frequency range to scan (min_hz, max_hz)
            sensitivity_dbm: Minimum detectable power level
            sample_rate_hz: Sample rate for SDR
            fft_size: FFT window size
        """
        self.scan_range = scan_range
        self.sensitivity_dbm = sensitivity_dbm
        self.sample_rate_hz = sample_rate_hz
        self.fft_size = fft_size
        
        # Signal database for known signal types
        self.signal_database = self._initialize_signal_database()
        
        # Baseline noise floor for jamming detection
        self.baseline_noise_floor = {}  # freq_range -> baseline_dbm
        self.noise_floor_samples = 100  # Samples to establish baseline
        
        # Detected signals and threats
        self.detected_signals: List[RFSignal] = []
        self.detected_threats: List[RFThreat] = []
        
        # Authorization list for transmitters
        self.authorized_transmitters: List[Tuple[float, float]] = []  # (freq, tolerance)
        
        logger.info(f"RF Spectrum Analyzer initialized: {scan_range[0]/1e6:.1f}-{scan_range[1]/1e9:.1f} GHz, "
                   f"sensitivity: {sensitivity_dbm} dBm")
    
    def _initialize_signal_database(self) -> Dict[str, Tuple[float, float]]:
        """Initialize database of known signal frequency ranges."""
        return {
            "wifi_2.4ghz": (2.4e9, 2.5e9),
            "wifi_5ghz": (5.15e9, 5.85e9),
            "cellular_lte_700": (698e6, 798e6),
            "cellular_lte_850": (824e6, 894e6),
            "cellular_lte_1900": (1850e6, 1990e6),
            "cellular_5g": (24e9, 40e9),
            "bluetooth": (2.4e9, 2.4835e9),
            "zigbee": (2.4e9, 2.4835e9),
            "gps_l1": (1575.42e6, 1575.42e6 + 2e6),
            "gps_l2": (1227.6e6, 1227.6e6 + 2e6),
            "drone_control": (2.4e9, 5.8e9),  # Common drone frequencies
            "iot_868mhz": (863e6, 870e6),
            "iot_915mhz": (902e6, 928e6),
        }
    
    def scan_spectrum(
        self,
        duration_seconds: float = 1.0,
        adaptive_rate: bool = True,
        battery_level: float = 1.0
    ) -> List[RFSignal]:
        """
        Scan the RF spectrum and detect signals.
        
        Args:
            duration_seconds: Duration to scan
            adaptive_rate: Adjust scan rate based on battery level
            battery_level: Current battery level (0.0 to 1.0)
            
        Returns:
            List of detected RF signals
        """
        # Adaptive scanning based on battery level
        if adaptive_rate and battery_level < 0.3:
            # Reduce scan resolution when low on battery
            scan_points = max(10, int(100 * battery_level))
        else:
            scan_points = 100
        
        # Simulate spectrum scan (in real system, use SDR hardware)
        signals = self._simulate_spectrum_scan(scan_points, duration_seconds)
        
        # Classify detected signals
        classified_signals = []
        for signal in signals:
            signal_type = self._classify_signal(signal.frequency_hz, signal.bandwidth_hz)
            signal.signal_type = signal_type
            classified_signals.append(signal)
        
        self.detected_signals = classified_signals
        logger.debug(f"Scanned spectrum: detected {len(classified_signals)} signals")
        
        return classified_signals
    
    def _simulate_spectrum_scan(
        self,
        num_points: int,
        duration: float
    ) -> List[RFSignal]:
        """
        Simulate RF spectrum scan with synthetic data.
        In production, this would interface with SDR hardware.
        
        Args:
            num_points: Number of frequency points to scan
            duration: Scan duration
            
        Returns:
            List of detected signals
        """
        signals = []
        timestamp = time.time()
        
        # Generate synthetic signals at common frequencies
        common_freqs = [
            (2.437e9, 20e6, -50),   # WiFi channel 6
            (1575.42e6, 2e6, -130), # GPS L1
            (850e6, 10e6, -70),     # Cellular
        ]
        
        for freq, bw, power in common_freqs:
            if self.scan_range[0] <= freq <= self.scan_range[1]:
                # Add some noise to make it realistic
                noise = np.random.normal(0, 5)
                signal = RFSignal(
                    frequency_hz=freq + np.random.normal(0, bw * 0.1),
                    power_dbm=power + noise,
                    bandwidth_hz=bw,
                    signal_type=SignalType.UNKNOWN,
                    timestamp=timestamp,
                    confidence=0.9 if abs(noise) < 3 else 0.7
                )
                signals.append(signal)
        
        return signals
    
    def _classify_signal(self, frequency_hz: float, bandwidth_hz: float) -> SignalType:
        """
        Classify RF signal based on frequency and bandwidth.
        
        Args:
            frequency_hz: Signal center frequency
            bandwidth_hz: Signal bandwidth
            
        Returns:
            Classified signal type
        """
        # Check against known signal database
        if 2.4e9 <= frequency_hz <= 2.5e9 and bandwidth_hz > 10e6:
            return SignalType.WIFI
        elif 5.15e9 <= frequency_hz <= 5.85e9 and bandwidth_hz > 10e6:
            return SignalType.WIFI
        elif 698e6 <= frequency_hz <= 2.7e9 and bandwidth_hz > 5e6:
            return SignalType.CELLULAR
        elif abs(frequency_hz - 1575.42e6) < 5e6:
            return SignalType.GPS
        elif 2.4e9 <= frequency_hz <= 2.4835e9 and bandwidth_hz < 2e6:
            if bandwidth_hz < 1e6:
                return SignalType.BLUETOOTH
            else:
                return SignalType.IOT
        elif 863e6 <= frequency_hz <= 928e6:
            return SignalType.IOT
        
        return SignalType.UNKNOWN
    
    def detect_jamming(
        self,
        signals: Optional[List[RFSignal]] = None,
        threshold_db: float = 10.0
    ) -> Optional[RFThreat]:
        """
        Detect RF jamming by analyzing noise floor.
        
        Jamming is indicated by abnormally high noise floor across frequencies.
        
        Args:
            signals: Signals to analyze (uses self.detected_signals if None)
            threshold_db: Threshold above baseline to consider jamming
            
        Returns:
            RFThreat if jamming detected, None otherwise
        """
        if signals is None:
            signals = self.detected_signals
        
        if not signals:
            return None
        
        # Calculate average power across frequency bands
        power_by_band = {}
        for signal in signals:
            band = self._get_frequency_band(signal.frequency_hz)
            if band not in power_by_band:
                power_by_band[band] = []
            power_by_band[band].append(signal.power_dbm)
        
        # Check for abnormally high noise floor
        for band, powers in power_by_band.items():
            avg_power = np.mean(powers)
            
            # Get or establish baseline
            if band not in self.baseline_noise_floor:
                # First scan - establish baseline
                self.baseline_noise_floor[band] = avg_power
                continue
            
            baseline = self.baseline_noise_floor[band]
            
            # Check if current power exceeds baseline by threshold
            if avg_power > baseline + threshold_db:
                logger.warning(f"Jamming detected in {band} band: {avg_power:.1f} dBm "
                             f"(baseline: {baseline:.1f} dBm)")
                
                threat = RFThreat(
                    threat_type="jamming",
                    severity=min(1.0, (avg_power - baseline) / (2 * threshold_db)),
                    frequency_hz=np.mean([s.frequency_hz for s in signals if self._get_frequency_band(s.frequency_hz) == band]),
                    description=f"RF jamming detected in {band} band, {avg_power - baseline:.1f} dB above baseline",
                    timestamp=time.time(),
                    affected_signals=[s for s in signals if self._get_frequency_band(s.frequency_hz) == band]
                )
                
                self.detected_threats.append(threat)
                return threat
        
        return None
    
    def detect_rogue_transmitters(
        self,
        signals: Optional[List[RFSignal]] = None
    ) -> List[RFThreat]:
        """
        Detect unauthorized/rogue transmitters.
        
        Args:
            signals: Signals to analyze (uses self.detected_signals if None)
            
        Returns:
            List of detected rogue transmitter threats
        """
        if signals is None:
            signals = self.detected_signals
        
        threats = []
        
        for signal in signals:
            # Check if signal is authorized
            is_authorized = False
            for auth_freq, tolerance in self.authorized_transmitters:
                if abs(signal.frequency_hz - auth_freq) <= tolerance:
                    is_authorized = True
                    break
            
            # Check if signal is from known/common sources
            if signal.signal_type in [SignalType.WIFI, SignalType.CELLULAR, 
                                     SignalType.GPS, SignalType.BLUETOOTH]:
                is_authorized = True
            
            if not is_authorized and signal.power_dbm > self.sensitivity_dbm + 20:
                # Strong unidentified signal - potential rogue transmitter
                signal.signal_type = SignalType.ROGUE
                
                threat = RFThreat(
                    threat_type="rogue_transmitter",
                    severity=min(1.0, (signal.power_dbm - self.sensitivity_dbm) / 50.0),
                    frequency_hz=signal.frequency_hz,
                    description=f"Rogue transmitter detected at {signal.frequency_hz/1e9:.3f} GHz, {signal.power_dbm:.1f} dBm",
                    timestamp=time.time(),
                    affected_signals=[signal]
                )
                
                threats.append(threat)
                logger.warning(f"Rogue transmitter: {signal.frequency_hz/1e9:.3f} GHz, {signal.power_dbm:.1f} dBm")
        
        self.detected_threats.extend(threats)
        return threats
    
    def _get_frequency_band(self, frequency_hz: float) -> str:
        """Get frequency band name for a given frequency."""
        if frequency_hz < 30e3:
            return "VLF"
        elif frequency_hz < 300e3:
            return "LF"
        elif frequency_hz < 3e6:
            return "MF"
        elif frequency_hz < 30e6:
            return "HF"
        elif frequency_hz < 300e6:
            return "VHF"
        elif frequency_hz < 3e9:
            return "UHF"
        elif frequency_hz < 30e9:
            return "SHF"
        else:
            return "EHF"
    
    def add_authorized_transmitter(self, frequency_hz: float, tolerance_hz: float = 1e6):
        """
        Add an authorized transmitter to the whitelist.
        
        Args:
            frequency_hz: Authorized frequency
            tolerance_hz: Frequency tolerance
        """
        self.authorized_transmitters.append((frequency_hz, tolerance_hz))
        logger.info(f"Added authorized transmitter: {frequency_hz/1e9:.3f} GHz ± {tolerance_hz/1e6:.1f} MHz")
    
    def analyze_spectrum(
        self,
        duration_seconds: float = 1.0,
        battery_level: float = 1.0
    ) -> Dict[str, Any]:
        """
        Perform complete spectrum analysis including threat detection.
        
        Args:
            duration_seconds: Scan duration
            battery_level: Current battery level for adaptive scanning
            
        Returns:
            Dictionary with analysis results
        """
        # Scan spectrum
        signals = self.scan_spectrum(duration_seconds, adaptive_rate=True, battery_level=battery_level)
        
        # Detect threats
        jamming_threat = self.detect_jamming(signals)
        rogue_threats = self.detect_rogue_transmitters(signals)
        
        # Compile results
        results = {
            "timestamp": time.time(),
            "scan_duration": duration_seconds,
            "signals_detected": len(signals),
            "signal_breakdown": self._get_signal_breakdown(signals),
            "threats": {
                "jamming": jamming_threat.threat_type if jamming_threat else None,
                "rogue_transmitters": len(rogue_threats),
                "total_threats": (1 if jamming_threat else 0) + len(rogue_threats)
            },
            "spectrum_utilization": self._calculate_spectrum_utilization(signals)
        }
        
        return results
    
    def _get_signal_breakdown(self, signals: List[RFSignal]) -> Dict[str, int]:
        """Get count of signals by type."""
        breakdown = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            breakdown[signal_type] = breakdown.get(signal_type, 0) + 1
        return breakdown
    
    def _calculate_spectrum_utilization(self, signals: List[RFSignal]) -> float:
        """Calculate percentage of spectrum being used."""
        if not signals:
            return 0.0
        
        total_bandwidth = self.scan_range[1] - self.scan_range[0]
        used_bandwidth = sum(s.bandwidth_hz for s in signals)
        
        return min(100.0, (used_bandwidth / total_bandwidth) * 100.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "scan_range_ghz": (self.scan_range[0] / 1e9, self.scan_range[1] / 1e9),
            "sensitivity_dbm": self.sensitivity_dbm,
            "total_signals_detected": len(self.detected_signals),
            "total_threats_detected": len(self.detected_threats),
            "authorized_transmitters": len(self.authorized_transmitters),
            "baseline_noise_floors": {band: f"{power:.1f} dBm" for band, power in self.baseline_noise_floor.items()}
        }
