"""
Behavioral Frequency Analyzer

Analyzes malware behavior patterns through frequency domain analysis.
Detects cryptominers, ransomware, and other malicious process behaviors.

Features:
- Cryptominer detection via CPU/GPU usage FFT (5-10 second cycles)
- Ransomware detection via file access rate analysis
- Process behavior pattern analysis
- Resource usage frequency signatures
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

logger = logging.getLogger(__name__)


class MalwareType(str, Enum):
    """Types of malware behaviors."""

    CRYPTOMINER = "cryptominer"
    RANSOMWARE = "ransomware"
    KEYLOGGER = "keylogger"
    TROJAN = "trojan"
    ROOTKIT = "rootkit"
    WORM = "worm"
    BENIGN = "benign"


class BehaviorSeverity(str, Enum):
    """Behavior threat severity."""

    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"


@dataclass
class ProcessBehavior:
    """Represents process behavior metrics."""

    process_id: int
    process_name: str
    timestamp: float
    cpu_percent: float
    memory_mb: float
    disk_io_bytes: int
    network_io_bytes: int
    file_access_count: int


@dataclass
class BehaviorThreat:
    """Represents a detected behavioral threat."""

    malware_type: MalwareType
    severity: BehaviorSeverity
    process_id: int
    process_name: str
    confidence: float
    timestamp: float
    description: str
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BehavioralFrequencyAnalyzer:
    """
    Behavioral Frequency Analyzer for malware detection through periodic patterns.

    Analyzes CPU, memory, disk, and network usage patterns to identify malicious
    behavior based on characteristic frequencies.

    Attributes:
        window_size: Number of samples to analyze
        sample_interval: Time between samples in seconds
    """

    def __init__(
        self,
        window_size: int = 300,  # 5 minutes at 1Hz sampling
        sample_interval: float = 1.0,
    ):
        """
        Initialize Behavioral Frequency Analyzer.

        Args:
            window_size: Number of behavior samples in analysis window
            sample_interval: Sampling interval in seconds
        """
        self.window_size = window_size
        self.sample_interval = sample_interval

        # Process behavior buffers
        self.process_behaviors: Dict[int, deque] = (
            {}
        )  # pid -> deque of behaviors

        # Detected threats
        self.detected_threats: List[BehaviorThreat] = []

        # Process baselines
        self.process_baselines: Dict[int, Dict[str, float]] = {}

        # Cryptominer signatures (typical mining cycle frequencies)
        self.cryptominer_freq_range = (0.1, 0.2)  # 5-10 second cycles

        # Ransomware thresholds
        self.ransomware_file_rate_threshold = 10.0  # files/sec

        logger.info(
            f"Behavioral Frequency Analyzer initialized: window={window_size}, "
            f"interval={sample_interval}s"
        )

    def add_process_behavior(self, behavior: ProcessBehavior):
        """
        Add process behavior sample.

        Args:
            behavior: Process behavior metrics
        """
        pid = behavior.process_id

        if pid not in self.process_behaviors:
            self.process_behaviors[pid] = deque(maxlen=self.window_size)

        self.process_behaviors[pid].append(behavior)

    def analyze_process(
        self,
        process_id: int,
        behaviors: Optional[List[ProcessBehavior]] = None,
    ) -> List[BehaviorThreat]:
        """
        Analyze a process for malicious behavior patterns.

        Args:
            process_id: Process ID to analyze
            behaviors: Behavior samples (uses buffer if None)

        Returns:
            List of detected behavioral threats
        """
        threats = []

        # Get behavior samples
        if behaviors is not None:
            behavior_list = behaviors
        elif process_id in self.process_behaviors:
            behavior_list = list(self.process_behaviors[process_id])
        else:
            return threats

        if len(behavior_list) < 10:
            return threats

        process_name = behavior_list[0].process_name

        # Check for cryptominer
        cryptominer_threat = self._detect_cryptominer(
            process_id, process_name, behavior_list
        )
        if cryptominer_threat:
            threats.append(cryptominer_threat)

        # Check for ransomware
        ransomware_threat = self._detect_ransomware(
            process_id, process_name, behavior_list
        )
        if ransomware_threat:
            threats.append(ransomware_threat)

        # Check for resource abuse
        resource_threat = self._detect_resource_abuse(
            process_id, process_name, behavior_list
        )
        if resource_threat:
            threats.append(resource_threat)

        self.detected_threats.extend(threats)
        return threats

    def _detect_cryptominer(
        self,
        process_id: int,
        process_name: str,
        behaviors: List[ProcessBehavior],
    ) -> Optional[BehaviorThreat]:
        """
        Detect cryptomining activity.

        Cryptominers show characteristic patterns:
        - High sustained CPU usage (>80%)
        - Periodic GPU usage cycles (5-10 seconds)
        - Network activity for pool communication

        Args:
            process_id: Process ID
            process_name: Process name
            behaviors: Behavior samples

        Returns:
            Threat if cryptominer detected, None otherwise
        """
        if len(behaviors) < 30:  # Need enough samples
            return None

        # Extract CPU usage time series
        cpu_usage = np.array([b.cpu_percent for b in behaviors])
        timestamps = np.array([b.timestamp for b in behaviors])

        # Check for high sustained CPU usage
        mean_cpu = np.mean(cpu_usage)

        if mean_cpu < 70:  # Not high enough for mining
            return None

        # Perform FFT on CPU usage
        freqs, power_spectrum = self._compute_fft_timeseries(
            cpu_usage, timestamps
        )

        # Look for peaks in cryptominer frequency range
        miner_freq_min, miner_freq_max = self.cryptominer_freq_range
        miner_mask = (freqs >= miner_freq_min) & (freqs <= miner_freq_max)

        if not np.any(miner_mask):
            return None

        miner_power = power_spectrum[miner_mask]

        # Check if there's significant power in mining frequency range
        total_power = np.sum(power_spectrum)
        miner_power_ratio = (
            np.sum(miner_power) / total_power if total_power > 0 else 0
        )

        if miner_power_ratio > 0.3:  # Significant periodic component
            # Find dominant frequency
            peak_idx = np.argmax(miner_power)
            peak_freq = freqs[miner_mask][peak_idx]
            period = 1.0 / peak_freq if peak_freq > 0 else 0

            indicators = [
                f"High CPU usage: {mean_cpu:.1f}%",
                f"Periodic behavior: {period:.1f}s cycles",
                f"Mining frequency signature detected",
            ]

            # Check for network activity (pool communication)
            network_activity = sum(b.network_io_bytes for b in behaviors)
            if network_activity > 0:
                indicators.append(
                    f"Network activity: {network_activity / 1024:.1f} KB"
                )

            threat = BehaviorThreat(
                malware_type=MalwareType.CRYPTOMINER,
                severity=BehaviorSeverity.MALICIOUS,
                process_id=process_id,
                process_name=process_name,
                confidence=min(1.0, miner_power_ratio * 1.5),
                timestamp=time.time(),
                description=f"Cryptominer detected: {mean_cpu:.1f}% CPU with {period:.1f}s cycles",
                indicators=indicators,
                metadata={
                    "mean_cpu_percent": float(mean_cpu),
                    "period_seconds": float(period),
                    "power_ratio": float(miner_power_ratio),
                },
            )

            logger.warning(
                f"Cryptominer detected: PID {process_id} ({process_name})"
            )
            return threat

        return None

    def _detect_ransomware(
        self,
        process_id: int,
        process_name: str,
        behaviors: List[ProcessBehavior],
    ) -> Optional[BehaviorThreat]:
        """
        Detect ransomware activity.

        Ransomware shows characteristic patterns:
        - High file access rate (>10 files/sec)
        - High disk I/O
        - Rapid encryption activity

        Args:
            process_id: Process ID
            process_name: Process name
            behaviors: Behavior samples

        Returns:
            Threat if ransomware detected, None otherwise
        """
        if len(behaviors) < 5:
            return None

        # Calculate file access rate
        duration = behaviors[-1].timestamp - behaviors[0].timestamp
        if duration <= 0:
            return None

        total_file_accesses = sum(b.file_access_count for b in behaviors)
        file_access_rate = total_file_accesses / duration

        # Calculate disk I/O rate
        total_disk_io = sum(b.disk_io_bytes for b in behaviors)
        disk_io_rate_mbps = (total_disk_io / duration) / (1024 * 1024)

        indicators = []
        confidence = 0.0

        # Check file access rate
        if file_access_rate > self.ransomware_file_rate_threshold:
            indicators.append(
                f"High file access rate: {file_access_rate:.1f} files/sec"
            )
            confidence += 0.4

        # Check disk I/O
        if disk_io_rate_mbps > 10:  # > 10 MB/s
            indicators.append(f"High disk I/O: {disk_io_rate_mbps:.1f} MB/s")
            confidence += 0.3

        # Check for suspicious file access patterns
        file_accesses = [b.file_access_count for b in behaviors]
        if len(file_accesses) > 1:
            # Ransomware often shows sudden spike in file access
            max_access = max(file_accesses)
            mean_access = np.mean(file_accesses)

            if max_access > mean_access * 3:
                indicators.append("Spike in file access detected")
                confidence += 0.3

        if confidence > 0.6:
            threat = BehaviorThreat(
                malware_type=MalwareType.RANSOMWARE,
                severity=BehaviorSeverity.CRITICAL,
                process_id=process_id,
                process_name=process_name,
                confidence=min(1.0, confidence),
                timestamp=time.time(),
                description=f"Ransomware detected: {file_access_rate:.1f} files/sec access rate",
                indicators=indicators,
                metadata={
                    "file_access_rate": float(file_access_rate),
                    "disk_io_mbps": float(disk_io_rate_mbps),
                    "total_files_accessed": int(total_file_accesses),
                },
            )

            logger.critical(
                f"Ransomware detected: PID {process_id} ({process_name})"
            )
            return threat

        return None

    def _detect_resource_abuse(
        self,
        process_id: int,
        process_name: str,
        behaviors: List[ProcessBehavior],
    ) -> Optional[BehaviorThreat]:
        """
        Detect general resource abuse patterns.

        Args:
            process_id: Process ID
            process_name: Process name
            behaviors: Behavior samples

        Returns:
            Threat if resource abuse detected, None otherwise
        """
        if len(behaviors) < 10:
            return None

        # Calculate resource usage statistics
        cpu_usage = [b.cpu_percent for b in behaviors]
        memory_usage = [b.memory_mb for b in behaviors]

        mean_cpu = np.mean(cpu_usage)
        mean_memory = np.mean(memory_usage)

        # Check baseline if exists
        if process_id in self.process_baselines:
            baseline = self.process_baselines[process_id]
            cpu_increase = mean_cpu - baseline.get("cpu_percent", 0)
            mem_increase = mean_memory - baseline.get("memory_mb", 0)

            indicators = []
            confidence = 0.0

            # Significant increase from baseline
            if cpu_increase > 30:
                indicators.append(f"CPU usage increased {cpu_increase:.1f}%")
                confidence += 0.3

            if mem_increase > 100:  # > 100 MB increase
                indicators.append(
                    f"Memory usage increased {mem_increase:.1f} MB"
                )
                confidence += 0.2

            if confidence > 0.4:
                threat = BehaviorThreat(
                    malware_type=MalwareType.TROJAN,
                    severity=BehaviorSeverity.SUSPICIOUS,
                    process_id=process_id,
                    process_name=process_name,
                    confidence=confidence,
                    timestamp=time.time(),
                    description=f"Resource abuse: CPU +{cpu_increase:.1f}%, Memory +{mem_increase:.1f}MB",
                    indicators=indicators,
                    metadata={
                        "cpu_increase": float(cpu_increase),
                        "memory_increase_mb": float(mem_increase),
                    },
                )

                logger.warning(
                    f"Resource abuse detected: PID {process_id} ({process_name})"
                )
                return threat
        else:
            # Establish baseline
            self.process_baselines[process_id] = {
                "cpu_percent": mean_cpu,
                "memory_mb": mean_memory,
            }

        return None

    def _compute_fft_timeseries(
        self, values: np.ndarray, timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of a time series.

        Args:
            values: Value samples
            timestamps: Timestamps for samples

        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        # Calculate sample rate from timestamps
        if len(timestamps) > 1:
            dt = np.mean(np.diff(timestamps))
            sample_rate = 1.0 / dt if dt > 0 else 1.0
        else:
            sample_rate = 1.0 / self.sample_interval

        # Remove mean (DC component)
        values_centered = values - np.mean(values)

        # Apply window
        window = np.hanning(len(values_centered))
        windowed = values_centered * window

        # Compute FFT
        n_fft = max(256, 2 ** int(np.ceil(np.log2(len(windowed)))))
        fft_result = np.fft.rfft(windowed, n=n_fft)
        power_spectrum = np.abs(fft_result) ** 2

        # Frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        return freqs, power_spectrum

    def analyze_all_processes(self) -> Dict[int, List[BehaviorThreat]]:
        """
        Analyze all tracked processes.

        Returns:
            Dictionary mapping process ID to detected threats
        """
        results = {}

        for pid in self.process_behaviors.keys():
            threats = self.analyze_process(pid)
            if threats:
                results[pid] = threats

        return results

    def simulate_process_behavior(
        self,
        process_id: int,
        process_name: str,
        behavior_type: str = "normal",
        duration_seconds: float = 60.0,
    ) -> List[ProcessBehavior]:
        """
        Simulate process behavior for testing.

        Args:
            process_id: Process ID
            process_name: Process name
            behavior_type: Type of behavior ("normal", "cryptominer", "ransomware")
            duration_seconds: Duration to simulate

        Returns:
            List of simulated behaviors
        """
        behaviors = []
        num_samples = int(duration_seconds / self.sample_interval)

        for i in range(num_samples):
            timestamp = time.time() + i * self.sample_interval

            if behavior_type == "cryptominer":
                # High CPU with periodic variation
                base_cpu = 85
                cycle_freq = 0.15  # 6.7 second cycles
                cpu = base_cpu + 10 * np.sin(
                    2 * np.pi * cycle_freq * i * self.sample_interval
                )
                memory = 500 + np.random.normal(0, 10)
                disk_io = int(np.random.normal(1000, 100))
                network_io = int(
                    np.random.normal(5000, 500)
                )  # Pool communication
                file_access = 0

            elif behavior_type == "ransomware":
                # High file access and disk I/O
                cpu = 40 + np.random.normal(0, 5)
                memory = 200 + np.random.normal(0, 20)
                disk_io = int(
                    np.random.normal(50000000, 5000000)
                )  # High disk I/O
                network_io = int(np.random.normal(1000, 100))
                file_access = int(np.random.normal(20, 5))  # High file access

            else:  # normal
                cpu = 10 + np.random.normal(0, 3)
                memory = 100 + np.random.normal(0, 10)
                disk_io = int(np.random.normal(10000, 1000))
                network_io = int(np.random.normal(1000, 200))
                file_access = int(np.random.normal(0.5, 0.2))

            behavior = ProcessBehavior(
                process_id=process_id,
                process_name=process_name,
                timestamp=timestamp,
                cpu_percent=max(0, min(100, cpu)),
                memory_mb=max(0, memory),
                disk_io_bytes=max(0, disk_io),
                network_io_bytes=max(0, network_io),
                file_access_count=max(0, file_access),
            )

            behaviors.append(behavior)
            self.add_process_behavior(behavior)

        return behaviors

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "window_size": self.window_size,
            "sample_interval": self.sample_interval,
            "processes_tracked": len(self.process_behaviors),
            "total_threats_detected": len(self.detected_threats),
            "process_baselines": len(self.process_baselines),
            "threat_breakdown": self._get_threat_breakdown(),
        }

    def _get_threat_breakdown(self) -> Dict[str, int]:
        """Get count of threats by malware type."""
        breakdown = {}
        for threat in self.detected_threats:
            malware_type = threat.malware_type.value
            breakdown[malware_type] = breakdown.get(malware_type, 0) + 1
        return breakdown
