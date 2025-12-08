"""
Network Frequency Analyzer

Performs FFT analysis on network traffic patterns to detect periodic behaviors.
Detects C2 beaconing, DDoS patterns, port scanning, and covert timing channels.

Features:
- FFT on network traffic inter-arrival times
- C2 beaconing detection (~60 second periods)
- DDoS pattern recognition (sub-second periods)
- Port scanning detection (5-15 second periods)
- Covert timing channel detection
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

logger = logging.getLogger(__name__)


class TrafficPattern(str, Enum):
    """Types of network traffic patterns."""
    C2_BEACON = "c2_beacon"
    DDOS = "ddos"
    PORT_SCAN = "port_scan"
    COVERT_CHANNEL = "covert_channel"
    NORMAL = "normal"
    HEARTBEAT = "heartbeat"


class ThreatLevel(str, Enum):
    """Network threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NetworkEvent:
    """Represents a network event (packet, connection, etc.)."""
    timestamp: float
    source_ip: str
    dest_ip: str
    protocol: str
    size_bytes: int
    port: Optional[int] = None


@dataclass
class NetworkThreat:
    """Represents a detected network threat based on frequency analysis."""
    threat_type: TrafficPattern
    threat_level: ThreatLevel
    period_seconds: float
    confidence: float
    timestamp: float
    description: str
    affected_ips: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NetworkFrequencyAnalyzer:
    """
    Network Frequency Analyzer for detecting periodic patterns in network traffic.
    
    Uses FFT to analyze inter-arrival times and detect malicious patterns like
    C2 beaconing, DDoS attacks, and covert timing channels.
    
    Attributes:
        window_size: Number of events to analyze in each window
        min_period: Minimum detectable period in seconds
        max_period: Maximum detectable period in seconds
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        min_period: float = 0.1,  # 100ms
        max_period: float = 300.0  # 5 minutes
    ):
        """
        Initialize Network Frequency Analyzer.
        
        Args:
            window_size: Number of events in analysis window
            min_period: Minimum period to detect (seconds)
            max_period: Maximum period to detect (seconds)
        """
        self.window_size = window_size
        self.min_period = min_period
        self.max_period = max_period
        
        # Event buffers
        self.event_buffer: deque = deque(maxlen=window_size)
        self.ip_event_buffers: Dict[str, deque] = {}  # Per-IP tracking
        
        # Detected threats
        self.detected_threats: List[NetworkThreat] = []
        
        # Known beaconing baselines
        self.known_beacons: Dict[str, float] = {}  # ip -> period
        
        logger.info(f"Network Frequency Analyzer initialized: window={window_size}, "
                   f"period range={min_period}-{max_period}s")
    
    def add_network_event(
        self,
        event: NetworkEvent
    ):
        """
        Add a network event to the analyzer.
        
        Args:
            event: Network event to add
        """
        self.event_buffer.append(event)
        
        # Also track per-IP
        if event.source_ip not in self.ip_event_buffers:
            self.ip_event_buffers[event.source_ip] = deque(maxlen=self.window_size)
        self.ip_event_buffers[event.source_ip].append(event)
    
    def analyze_traffic_periodicity(
        self,
        events: Optional[List[NetworkEvent]] = None,
        source_ip: Optional[str] = None
    ) -> List[NetworkThreat]:
        """
        Analyze traffic for periodic patterns.
        
        Args:
            events: Events to analyze (uses buffer if None)
            source_ip: Analyze specific IP (if None, analyzes all traffic)
            
        Returns:
            List of detected network threats
        """
        threats = []
        
        # Determine which events to analyze
        if events is not None:
            event_list = events
        elif source_ip is not None and source_ip in self.ip_event_buffers:
            event_list = list(self.ip_event_buffers[source_ip])
        else:
            event_list = list(self.event_buffer)
        
        if len(event_list) < 10:
            return threats
        
        # Extract timestamps
        timestamps = np.array([e.timestamp for e in event_list])
        
        # Calculate inter-arrival times
        inter_arrival_times = np.diff(timestamps)
        
        if len(inter_arrival_times) < 5:
            return threats
        
        # Perform FFT on inter-arrival times
        freqs, power_spectrum = self._compute_fft_interarrival(inter_arrival_times)
        
        # Detect periodic patterns
        peaks = self._find_periodic_peaks(freqs, power_spectrum)
        
        for peak_freq, peak_power, period in peaks:
            # Classify the pattern based on period
            pattern_type, threat_level = self._classify_traffic_pattern(period, event_list)
            
            # Calculate confidence based on peak strength
            mean_power = np.mean(power_spectrum)
            confidence = min(1.0, (peak_power / mean_power) / 10.0)
            
            if confidence > 0.3:  # Only report if reasonably confident
                threat = NetworkThreat(
                    threat_type=pattern_type,
                    threat_level=threat_level,
                    period_seconds=period,
                    confidence=confidence,
                    timestamp=time.time(),
                    description=f"{pattern_type.value} detected with {period:.2f}s period",
                    affected_ips=[source_ip] if source_ip else list(set(e.source_ip for e in event_list)),
                    metadata={
                        "peak_frequency_hz": peak_freq,
                        "num_events": len(event_list),
                        "power_ratio": float(peak_power / mean_power)
                    }
                )
                
                threats.append(threat)
                
                if threat_level not in [ThreatLevel.NONE, ThreatLevel.LOW]:
                    logger.warning(f"Network threat: {pattern_type.value} from "
                                 f"{source_ip or 'multiple IPs'}, period={period:.2f}s")
        
        self.detected_threats.extend(threats)
        return threats
    
    def _compute_fft_interarrival(
        self,
        inter_arrival_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of inter-arrival times.
        
        Args:
            inter_arrival_times: Array of time differences between events
            
        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        # Remove outliers (cap at 99th percentile)
        percentile_99 = np.percentile(inter_arrival_times, 99)
        clipped_times = np.clip(inter_arrival_times, 0, percentile_99)
        
        # Note: For non-uniform time intervals, ideally we would resample to uniform intervals
        # using interpolation. For this implementation, we use the times directly with the
        # mean inter-arrival time as the effective sample rate. This provides reasonable
        # results for detection purposes while keeping the implementation simple.
        # Production systems may want to implement proper resampling for higher accuracy.
        n_fft = max(256, 2 ** int(np.ceil(np.log2(len(clipped_times)))))
        
        # Apply window
        window = np.hanning(len(clipped_times))
        windowed = clipped_times * window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed, n=n_fft)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Frequency bins - need to account for mean inter-arrival time
        mean_dt = np.mean(inter_arrival_times)
        sample_rate = 1.0 / mean_dt if mean_dt > 0 else 1.0
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        
        return freqs, power_spectrum
    
    def _find_periodic_peaks(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray,
        threshold_factor: float = 5.0
    ) -> List[Tuple[float, float, float]]:
        """
        Find periodic peaks in frequency spectrum.
        
        Args:
            freqs: Frequency bins
            power_spectrum: Power spectrum
            threshold_factor: Factor above mean to consider a peak
            
        Returns:
            List of (frequency_hz, power, period_seconds) tuples
        """
        mean_power = np.mean(power_spectrum)
        threshold = mean_power * threshold_factor
        
        peaks = []
        
        for i in range(1, len(power_spectrum) - 1):
            if (power_spectrum[i] > threshold and
                power_spectrum[i] > power_spectrum[i-1] and
                power_spectrum[i] > power_spectrum[i+1]):
                
                freq = freqs[i]
                if freq > 0:  # Avoid DC component
                    period = 1.0 / freq
                    
                    # Filter by period range
                    if self.min_period <= period <= self.max_period:
                        peaks.append((freq, power_spectrum[i], period))
        
        # Sort by power
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        return peaks[:10]  # Top 10 peaks
    
    def _classify_traffic_pattern(
        self,
        period: float,
        events: List[NetworkEvent]
    ) -> Tuple[TrafficPattern, ThreatLevel]:
        """
        Classify traffic pattern based on period and characteristics.
        
        Args:
            period: Detected period in seconds
            events: Events being analyzed
            
        Returns:
            Tuple of (pattern_type, threat_level)
        """
        # C2 Beaconing: typically 30-120 seconds
        if 30 <= period <= 120:
            # Check if it's from limited sources
            unique_sources = len(set(e.source_ip for e in events))
            if unique_sources < 5:
                return TrafficPattern.C2_BEACON, ThreatLevel.CRITICAL
            else:
                return TrafficPattern.HEARTBEAT, ThreatLevel.LOW
        
        # Port Scanning: typically 5-15 seconds per port
        elif 5 <= period <= 15:
            # Check for multiple destination ports
            unique_ports = len(set(e.port for e in events if e.port))
            if unique_ports > 10:
                return TrafficPattern.PORT_SCAN, ThreatLevel.HIGH
            else:
                return TrafficPattern.HEARTBEAT, ThreatLevel.NONE
        
        # DDoS: sub-second to few seconds period
        elif 0.1 <= period <= 5:
            # Check for high volume from multiple sources
            unique_sources = len(set(e.source_ip for e in events))
            if unique_sources > 10:
                return TrafficPattern.DDOS, ThreatLevel.CRITICAL
            else:
                return TrafficPattern.NORMAL, ThreatLevel.NONE
        
        # Covert timing channel: very precise timing (ms-level)
        elif 0.05 <= period <= 0.5:
            # Check for unusual precision in timing
            timestamps = [e.timestamp for e in events]
            if len(timestamps) > 2:
                inter_times = np.diff(timestamps)
                std_dev = np.std(inter_times)
                mean_time = np.mean(inter_times)
                
                # Low variance indicates intentional timing
                if std_dev / mean_time < 0.1:
                    return TrafficPattern.COVERT_CHANNEL, ThreatLevel.HIGH
        
        return TrafficPattern.NORMAL, ThreatLevel.NONE
    
    def detect_c2_beaconing(
        self,
        source_ip: Optional[str] = None,
        min_beacons: int = 5
    ) -> List[NetworkThreat]:
        """
        Specifically detect C2 beaconing patterns.
        
        C2 beacons are characterized by:
        - Regular periodic connections (30-120s typical)
        - Small data transfers
        - Consistent destination
        
        Args:
            source_ip: Specific IP to check (if None, checks all)
            min_beacons: Minimum number of beacons to confirm pattern
            
        Returns:
            List of C2 beacon threats
        """
        threats = []
        
        # Get IPs to check
        if source_ip:
            ips_to_check = [source_ip] if source_ip in self.ip_event_buffers else []
        else:
            ips_to_check = list(self.ip_event_buffers.keys())
        
        for ip in ips_to_check:
            events = list(self.ip_event_buffers[ip])
            
            if len(events) < min_beacons:
                continue
            
            # Check for regular intervals
            timestamps = np.array([e.timestamp for e in events])
            intervals = np.diff(timestamps)
            
            if len(intervals) < min_beacons - 1:
                continue
            
            # Calculate interval statistics
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # C2 beacons have low variance in timing
            if 30 <= mean_interval <= 120 and std_interval / mean_interval < 0.2:
                # Check for consistent destination
                dest_ips = [e.dest_ip for e in events]
                unique_dests = len(set(dest_ips))
                
                if unique_dests <= 3:  # Connecting to same few hosts
                    threat = NetworkThreat(
                        threat_type=TrafficPattern.C2_BEACON,
                        threat_level=ThreatLevel.CRITICAL,
                        period_seconds=mean_interval,
                        confidence=1.0 - std_interval / mean_interval,
                        timestamp=time.time(),
                        description=f"C2 beacon from {ip} every {mean_interval:.1f}s",
                        affected_ips=[ip],
                        metadata={
                            "beacon_count": len(events),
                            "std_deviation": float(std_interval),
                            "destinations": list(set(dest_ips))
                        }
                    )
                    
                    threats.append(threat)
                    self.known_beacons[ip] = mean_interval
                    
                    logger.critical(f"C2 beacon detected from {ip}: {mean_interval:.1f}s period, "
                                  f"{len(events)} beacons")
        
        self.detected_threats.extend(threats)
        return threats
    
    def detect_port_scanning(
        self,
        source_ip: Optional[str] = None,
        min_ports: int = 20
    ) -> List[NetworkThreat]:
        """
        Detect port scanning activity.
        
        Port scans are characterized by:
        - Sequential or random port access
        - Short duration per port
        - Many different ports accessed
        
        Args:
            source_ip: Specific IP to check
            min_ports: Minimum ports accessed to consider scanning
            
        Returns:
            List of port scan threats
        """
        threats = []
        
        # Get IPs to check
        if source_ip:
            ips_to_check = [source_ip] if source_ip in self.ip_event_buffers else []
        else:
            ips_to_check = list(self.ip_event_buffers.keys())
        
        for ip in ips_to_check:
            events = list(self.ip_event_buffers[ip])
            
            # Count unique destination ports
            ports_accessed = [e.port for e in events if e.port]
            unique_ports = set(ports_accessed)
            
            if len(unique_ports) >= min_ports:
                # Calculate scan rate
                if len(events) > 1:
                    duration = events[-1].timestamp - events[0].timestamp
                    scan_rate = len(unique_ports) / duration if duration > 0 else 0
                    
                    # Determine threat level based on scan rate
                    if scan_rate > 10:  # > 10 ports/sec
                        threat_level = ThreatLevel.HIGH
                    elif scan_rate > 1:  # > 1 port/sec
                        threat_level = ThreatLevel.MEDIUM
                    else:
                        threat_level = ThreatLevel.LOW
                    
                    threat = NetworkThreat(
                        threat_type=TrafficPattern.PORT_SCAN,
                        threat_level=threat_level,
                        period_seconds=1.0 / scan_rate if scan_rate > 0 else 0,
                        confidence=min(1.0, len(unique_ports) / 100.0),
                        timestamp=time.time(),
                        description=f"Port scan from {ip}: {len(unique_ports)} ports in {duration:.1f}s",
                        affected_ips=[ip],
                        metadata={
                            "ports_scanned": len(unique_ports),
                            "scan_rate_per_sec": float(scan_rate),
                            "duration_seconds": float(duration)
                        }
                    )
                    
                    threats.append(threat)
                    logger.warning(f"Port scan detected from {ip}: {len(unique_ports)} ports, "
                                 f"rate={scan_rate:.1f} ports/sec")
        
        self.detected_threats.extend(threats)
        return threats
    
    def detect_covert_timing_channel(
        self,
        events: Optional[List[NetworkEvent]] = None
    ) -> List[NetworkThreat]:
        """
        Detect covert timing channels.
        
        Covert channels use precise timing to encode information.
        
        Args:
            events: Events to analyze
            
        Returns:
            List of covert channel threats
        """
        threats = []
        
        if events is None:
            events = list(self.event_buffer)
        
        if len(events) < 20:
            return threats
        
        # Analyze timing precision
        timestamps = np.array([e.timestamp for e in events])
        intervals = np.diff(timestamps)
        
        # Look for suspiciously precise timing
        # Natural traffic has more variance
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if std_interval > 0:
            coefficient_of_variation = std_interval / mean_interval
            
            # Very low CoV indicates artificial timing
            if coefficient_of_variation < 0.05 and 0.05 <= mean_interval <= 0.5:
                threat = NetworkThreat(
                    threat_type=TrafficPattern.COVERT_CHANNEL,
                    threat_level=ThreatLevel.HIGH,
                    period_seconds=mean_interval,
                    confidence=1.0 - coefficient_of_variation,
                    timestamp=time.time(),
                    description=f"Covert timing channel: {mean_interval*1000:.2f}ms intervals, CoV={coefficient_of_variation:.4f}",
                    affected_ips=list(set(e.source_ip for e in events)),
                    metadata={
                        "mean_interval_ms": float(mean_interval * 1000),
                        "coefficient_of_variation": float(coefficient_of_variation),
                        "num_packets": len(events)
                    }
                )
                
                threats.append(threat)
                logger.warning(f"Covert timing channel detected: {mean_interval*1000:.2f}ms intervals")
        
        self.detected_threats.extend(threats)
        return threats
    
    def analyze_all(
        self,
        include_per_ip: bool = True
    ) -> Dict[str, List[NetworkThreat]]:
        """
        Run all detection methods.
        
        Args:
            include_per_ip: Also analyze per-IP traffic
            
        Returns:
            Dictionary of threats by detection method
        """
        results = {
            "periodicity": self.analyze_traffic_periodicity(),
            "c2_beaconing": self.detect_c2_beaconing(),
            "port_scanning": self.detect_port_scanning(),
            "covert_channels": self.detect_covert_timing_channel()
        }
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "window_size": self.window_size,
            "events_buffered": len(self.event_buffer),
            "ips_tracked": len(self.ip_event_buffers),
            "total_threats_detected": len(self.detected_threats),
            "known_beacons": len(self.known_beacons),
            "threat_breakdown": self._get_threat_breakdown()
        }
    
    def _get_threat_breakdown(self) -> Dict[str, int]:
        """Get count of threats by type."""
        breakdown = {}
        for threat in self.detected_threats:
            threat_type = threat.threat_type.value
            breakdown[threat_type] = breakdown.get(threat_type, 0) + 1
        return breakdown
