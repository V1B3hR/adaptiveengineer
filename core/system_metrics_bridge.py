"""
System Metrics Bridge - Map simulation to real-world systems

This module provides bidirectional translation between simulation metrics
and real system metrics (CPU, memory, network, battery). Enables deployment
of the adaptive agent system for actual cyber-defense and robotics applications.
"""

import logging
import time
import psutil
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger('system_metrics_bridge')


class MetricType(str, Enum):
    """Types of real-world metrics"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    BATTERY = "battery"
    DISK_IO = "disk_io"
    TEMPERATURE = "temperature"


@dataclass
class SystemMetrics:
    """Real system metrics snapshot"""
    timestamp: float
    cpu_percent: float  # 0-100
    memory_percent: float  # 0-100
    network_bytes_sent: int
    network_bytes_recv: int
    disk_read_bytes: int
    disk_write_bytes: int
    battery_percent: Optional[float] = None  # 0-100 if available
    temperature: Optional[float] = None  # Celsius if available


@dataclass
class SimulationState:
    """Simulation agent state"""
    node_id: int
    energy: float
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    communication_load: float
    trust_score: float


class SystemMetricsBridge:
    """
    Bridge between simulation and real-world metrics.
    
    Provides:
    - Metric collection from real systems
    - Translation to simulation units
    - Reverse mapping for action execution
    """
    
    def __init__(
        self,
        energy_scale: float = 10.0,
        position_to_network: bool = True
    ):
        """
        Initialize metrics bridge.
        
        Args:
            energy_scale: Maximum energy value in simulation
            position_to_network: Whether to map position to network topology
        """
        self.energy_scale = energy_scale
        self.position_to_network = position_to_network
        
        # Metric history for trend analysis
        self.metric_history: deque = deque(maxlen=1000)
        
        # Baseline metrics for normalization
        self.baseline_metrics: Optional[SystemMetrics] = None
        self._calibration_samples = []
        
        # Network counters (for delta calculation)
        self._last_network_counters: Optional[psutil._common.snetio] = None
        self._last_disk_counters: Optional[psutil._common.sdiskio] = None
        
        logger.info("SystemMetricsBridge initialized")
    
    def calibrate(self, samples: int = 10, interval: float = 1.0) -> None:
        """
        Calibrate baseline metrics.
        
        Collects multiple samples to establish normal operating levels.
        
        Args:
            samples: Number of calibration samples
            interval: Seconds between samples
        """
        logger.info(f"Calibrating baseline metrics ({samples} samples)...")
        
        self._calibration_samples = []
        
        for i in range(samples):
            metrics = self.collect_system_metrics()
            self._calibration_samples.append(metrics)
            
            if i < samples - 1:
                time.sleep(interval)
        
        # Calculate baseline as average
        if self._calibration_samples:
            self.baseline_metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=sum(m.cpu_percent for m in self._calibration_samples) / len(self._calibration_samples),
                memory_percent=sum(m.memory_percent for m in self._calibration_samples) / len(self._calibration_samples),
                network_bytes_sent=0,  # Delta-based, no baseline needed
                network_bytes_recv=0,
                disk_read_bytes=0,
                disk_write_bytes=0,
                battery_percent=(
                    sum(m.battery_percent for m in self._calibration_samples if m.battery_percent is not None) /
                    sum(1 for m in self._calibration_samples if m.battery_percent is not None)
                    if any(m.battery_percent is not None for m in self._calibration_samples) else None
                )
            )
            
            logger.info(f"Baseline established: CPU={self.baseline_metrics.cpu_percent:.1f}%, "
                       f"Memory={self.baseline_metrics.memory_percent:.1f}%")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system metrics.
        
        Returns:
            SystemMetrics snapshot
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        if self._last_network_counters:
            net_bytes_sent = net_io.bytes_sent - self._last_network_counters.bytes_sent
            net_bytes_recv = net_io.bytes_recv - self._last_network_counters.bytes_recv
        else:
            net_bytes_sent = 0
            net_bytes_recv = 0
        self._last_network_counters = net_io
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            if self._last_disk_counters:
                disk_read = disk_io.read_bytes - self._last_disk_counters.read_bytes
                disk_write = disk_io.write_bytes - self._last_disk_counters.write_bytes
            else:
                disk_read = 0
                disk_write = 0
            self._last_disk_counters = disk_io
        else:
            disk_read = 0
            disk_write = 0
        
        # Battery (if available)
        battery_percent = None
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = battery.percent
        except AttributeError:
            # Battery sensors not available on this platform
            pass
        
        # Temperature (if available)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature sensor
                for sensor_name, entries in temps.items():
                    if entries:
                        temperature = entries[0].current
                        break
        except AttributeError:
            # Temperature sensors not available on this platform
            pass
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            network_bytes_sent=net_bytes_sent,
            network_bytes_recv=net_bytes_recv,
            disk_read_bytes=disk_read,
            disk_write_bytes=disk_write,
            battery_percent=battery_percent,
            temperature=temperature
        )
        
        self.metric_history.append(metrics)
        
        return metrics
    
    def map_to_simulation(self, metrics: SystemMetrics) -> Dict[str, float]:
        """
        Map real system metrics to simulation parameters.
        
        Translates:
        - CPU/Memory/Battery → Energy
        - Network I/O → Communication load
        - Disk I/O → Processing activity
        
        Args:
            metrics: SystemMetrics to map
            
        Returns:
            Dictionary of simulation parameters
        """
        # Energy mapping: combine CPU, memory, and battery
        energy_from_cpu = 1.0 - (metrics.cpu_percent / 100.0)
        energy_from_memory = 1.0 - (metrics.memory_percent / 100.0)
        
        if metrics.battery_percent is not None:
            energy_from_battery = metrics.battery_percent / 100.0
            # Weighted combination
            energy_normalized = (
                0.3 * energy_from_cpu +
                0.3 * energy_from_memory +
                0.4 * energy_from_battery
            )
        else:
            # No battery, use CPU and memory
            energy_normalized = 0.5 * energy_from_cpu + 0.5 * energy_from_memory
        
        energy = energy_normalized * self.energy_scale
        
        # Communication load: network activity
        total_network = metrics.network_bytes_sent + metrics.network_bytes_recv
        # Normalize to 0-1 range (assume 1MB/s is high load)
        communication_load = min(1.0, total_network / (1024 * 1024))
        
        # Processing activity: disk I/O
        total_disk = metrics.disk_read_bytes + metrics.disk_write_bytes
        processing_activity = min(1.0, total_disk / (10 * 1024 * 1024))  # 10MB/s = high
        
        return {
            'energy': energy,
            'communication_load': communication_load,
            'processing_activity': processing_activity,
            'cpu_load': metrics.cpu_percent / 100.0,
            'memory_load': metrics.memory_percent / 100.0,
            'battery_level': metrics.battery_percent / 100.0 if metrics.battery_percent else 1.0
        }
    
    def map_from_simulation(
        self,
        sim_state: SimulationState,
        baseline: Optional[SystemMetrics] = None
    ) -> Dict[str, Any]:
        """
        Map simulation state to expected real-world metrics.
        
        Useful for prediction and anomaly detection.
        
        Args:
            sim_state: SimulationState to map
            baseline: Optional baseline metrics
            
        Returns:
            Dictionary of expected real metrics
        """
        if baseline is None:
            baseline = self.baseline_metrics
        
        # Energy → CPU/Memory usage
        energy_ratio = sim_state.energy / self.energy_scale
        
        # Lower energy means higher CPU/memory usage
        expected_cpu = (1.0 - energy_ratio) * 100.0
        expected_memory = (1.0 - energy_ratio) * 100.0
        
        # Communication load → Network bandwidth
        # Assume baseline + proportional increase
        expected_network_rate = sim_state.communication_load * 1024 * 1024  # Bytes/s
        
        return {
            'expected_cpu_percent': expected_cpu,
            'expected_memory_percent': expected_memory,
            'expected_network_rate': expected_network_rate,
            'expected_battery_drain': (1.0 - energy_ratio) * 10.0  # % per hour
        }
    
    def detect_anomaly(
        self,
        current_metrics: SystemMetrics,
        expected: Dict[str, Any],
        threshold: float = 0.3
    ) -> Tuple[bool, List[str]]:
        """
        Detect anomalies by comparing actual vs expected metrics.
        
        Args:
            current_metrics: Current system metrics
            expected: Expected metrics from simulation
            threshold: Anomaly detection threshold (0-1)
            
        Returns:
            Tuple of (is_anomaly, list of anomalous metrics)
        """
        anomalies = []
        
        # Check CPU
        cpu_diff = abs(current_metrics.cpu_percent - expected['expected_cpu_percent'])
        if cpu_diff > threshold * 100:
            anomalies.append(f"CPU: actual={current_metrics.cpu_percent:.1f}%, "
                           f"expected={expected['expected_cpu_percent']:.1f}%")
        
        # Check Memory
        mem_diff = abs(current_metrics.memory_percent - expected['expected_memory_percent'])
        if mem_diff > threshold * 100:
            anomalies.append(f"Memory: actual={current_metrics.memory_percent:.1f}%, "
                           f"expected={expected['expected_memory_percent']:.1f}%")
        
        # Check Network (if recent activity)
        current_network = current_metrics.network_bytes_sent + current_metrics.network_bytes_recv
        if current_network > 0:
            net_diff = abs(current_network - expected['expected_network_rate'])
            if net_diff > threshold * expected['expected_network_rate']:
                anomalies.append(f"Network: actual={current_network} B/s, "
                               f"expected={expected['expected_network_rate']:.0f} B/s")
        
        return len(anomalies) > 0, anomalies
    
    def apply_defensive_action(self, action: str, parameters: Optional[Dict] = None) -> bool:
        """
        Apply defensive countermeasure to real system.
        
        Translates simulation countermeasure to actual system action.
        
        Args:
            action: Action to apply
            parameters: Optional action parameters
            
        Returns:
            Success status
        """
        if parameters is None:
            parameters = {}
        
        logger.info(f"Applying defensive action: {action}")
        
        try:
            if action == "reduce_energy_consumption":
                # Lower process priority or throttle CPU
                self._throttle_processes()
                return True
                
            elif action == "switch_communication_channel":
                # In real system, would switch network interface or port
                logger.info("Would switch communication channel (simulated)")
                return True
                
            elif action == "apply_rate_limiting":
                # In real system, would configure firewall or QoS
                logger.info("Would apply rate limiting (simulated)")
                return True
                
            elif action == "isolate_process":
                # Kill or suspend suspicious process
                pid = parameters.get('pid')
                if pid:
                    self._isolate_process(pid)
                return True
                
            elif action == "alert_administrator":
                # Send alert (log, email, webhook, etc.)
                message = parameters.get('message', 'Security alert')
                logger.warning(f"ALERT: {message}")
                return True
                
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply action {action}: {e}")
            return False
    
    def _throttle_processes(self) -> None:
        """Lower priority of high CPU processes"""
        try:
            # Get high CPU processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 50:
                        # Lower priority (nice value)
                        p = psutil.Process(proc.info['pid'])
                        current_nice = p.nice()
                        if current_nice < 10:  # Don't over-throttle
                            p.nice(current_nice + 5)
                            logger.info(f"Throttled process {proc.info['name']} (PID {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.error(f"Failed to throttle processes: {e}")
    
    def _isolate_process(self, pid: int) -> None:
        """Isolate a suspicious process"""
        try:
            proc = psutil.Process(pid)
            logger.warning(f"Isolating process: {proc.name()} (PID {pid})")
            
            # In production, might suspend instead of kill
            # proc.suspend()
            
            # For now, just log
            logger.info(f"Would isolate process {pid} (simulated)")
            
        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} not found")
        except psutil.AccessDenied:
            logger.error(f"Access denied to process {pid}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        if not self.metric_history:
            return {
                'samples_collected': 0,
                'baseline_established': self.baseline_metrics is not None
            }
        
        recent_metrics = list(self.metric_history)[-10:]  # Last 10 samples
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        total_network = sum(
            m.network_bytes_sent + m.network_bytes_recv
            for m in recent_metrics
        )
        
        return {
            'samples_collected': len(self.metric_history),
            'baseline_established': self.baseline_metrics is not None,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'total_network_bytes': total_network,
            'battery_available': any(m.battery_percent is not None for m in recent_metrics)
        }
