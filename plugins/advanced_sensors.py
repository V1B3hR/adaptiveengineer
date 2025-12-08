"""
Advanced Sensor Suite Plugin

Comprehensive sensor suite for environmental monitoring and threat detection.
Includes physical sensors (humidity, light, acoustic array, motion, air quality, etc.)
and enhanced specifications for camera, LiDAR, and network monitoring.

Features:
- Environmental sensors (humidity, light, air quality, soil moisture)
- Acoustic array with direction finding and gunshot detection
- Motion detection (PIR)
- Enhanced vibration sensor
- Upgraded camera with emergency to forensics modes
- Automotive-grade LiDAR with adaptive scan rates
- Enhanced network monitoring with WiFi 6, 5G, 10G Ethernet
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class CameraResolution(str, Enum):
    """Camera resolution modes."""
    EMERGENCY = "emergency"    # 480p
    NORMAL = "normal"          # 1080p
    SURVEILLANCE = "surveillance"  # 4K
    FORENSICS = "forensics"    # 8K


class LiDARMode(str, Enum):
    """LiDAR scan rate modes."""
    IDLE = "idle"          # 5 Hz
    CRUISE = "cruise"      # 20 Hz
    EVASIVE = "evasive"    # 100 Hz
    EMERGENCY = "emergency"  # 200 Hz


@dataclass
class HumiditySensorReading:
    """Humidity sensor reading."""
    timestamp: float
    humidity_percent: float  # 0-100%
    temperature_celsius: float
    accuracy: float = 2.0  # ±2%


@dataclass
class LightSensorReading:
    """Light sensor reading."""
    timestamp: float
    lux: float  # 0.01-100k lux
    uv_index: float
    visible_light: float
    infrared: float


@dataclass
class AcousticArrayReading:
    """Acoustic array reading with direction finding."""
    timestamp: float
    direction_azimuth: float  # 0-360 degrees
    direction_elevation: float  # -90 to 90 degrees
    sound_level_db: float
    signature_detected: Optional[str] = None
    confidence: float = 0.0


@dataclass
class MotionDetection:
    """PIR motion sensor detection."""
    timestamp: float
    motion_detected: bool
    distance_meters: float
    confidence: float


@dataclass
class AirQualityReading:
    """Air quality sensor reading."""
    timestamp: float
    co2_ppm: float  # CO2 parts per million
    co_ppm: float   # CO parts per million
    voc_ppb: float  # Volatile Organic Compounds parts per billion
    pm2_5: float    # PM2.5 particulate matter μg/m³
    pm10: float     # PM10 particulate matter μg/m³
    smoke_detected: bool
    air_quality_index: int  # 0-500 AQI scale


@dataclass
class SoilMoistureReading:
    """Soil moisture sensor reading."""
    timestamp: float
    moisture_percent: float  # 0-100%
    conductivity: float  # mS/cm
    temperature_celsius: float
    depth_cm: float


@dataclass
class VibrationReading:
    """Enhanced vibration sensor reading."""
    timestamp: float
    frequency_hz: float
    amplitude_g: float  # 0.001g sensitivity
    frequency_range: str  # "0.1-10kHz"


@dataclass
class CameraCapture:
    """Camera capture metadata."""
    timestamp: float
    resolution_mode: CameraResolution
    resolution: Tuple[int, int]  # (width, height)
    frame_rate: int
    data_size_mb: float


@dataclass
class LiDARScan:
    """LiDAR scan data."""
    timestamp: float
    scan_mode: LiDARMode
    scan_rate_hz: int
    range_meters: float
    point_count: int
    objects_detected: List[Dict[str, Any]]


@dataclass
class NetworkMetrics:
    """Enhanced network monitoring metrics."""
    timestamp: float
    interface: str
    bandwidth_gbps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_p999: float
    packet_loss_percent: float


class AdvancedSensorSuite:
    """
    Advanced Sensor Suite for comprehensive environmental monitoring.
    
    Provides high-fidelity sensor data for security, robotics, and IoT applications.
    """
    
    def __init__(self):
        """Initialize Advanced Sensor Suite."""
        # Sensor enable flags
        self.humidity_enabled = True
        self.light_enabled = True
        self.acoustic_array_enabled = True
        self.motion_enabled = True
        self.air_quality_enabled = True
        self.soil_moisture_enabled = False  # For agricultural applications
        self.vibration_enabled = True
        self.camera_enabled = True
        self.lidar_enabled = True
        self.network_monitor_enabled = True
        
        # Sensor readings
        self.humidity_readings: List[HumiditySensorReading] = []
        self.light_readings: List[LightSensorReading] = []
        self.acoustic_readings: List[AcousticArrayReading] = []
        self.motion_detections: List[MotionDetection] = []
        self.air_quality_readings: List[AirQualityReading] = []
        self.soil_moisture_readings: List[SoilMoistureReading] = []
        self.vibration_readings: List[VibrationReading] = []
        self.camera_captures: List[CameraCapture] = []
        self.lidar_scans: List[LiDARScan] = []
        self.network_metrics: List[NetworkMetrics] = []
        
        # Camera settings
        self.camera_resolution_mode = CameraResolution.NORMAL
        
        # LiDAR settings
        self.lidar_mode = LiDARMode.CRUISE
        
        logger.info("Advanced Sensor Suite initialized")
    
    def read_humidity(self) -> HumiditySensorReading:
        """
        Read humidity sensor.
        
        Sensor specs: 0-100% RH, ±2% accuracy
        
        Returns:
            Humidity sensor reading
        """
        # Simulate reading (in production, interface with actual sensor)
        reading = HumiditySensorReading(
            timestamp=time.time(),
            humidity_percent=np.random.uniform(30, 70),
            temperature_celsius=np.random.uniform(15, 30),
            accuracy=2.0
        )
        
        self.humidity_readings.append(reading)
        return reading
    
    def read_light_sensor(self) -> LightSensorReading:
        """
        Read light sensor.
        
        Sensor specs: 0.01-100k lux, UV/visible/IR
        
        Returns:
            Light sensor reading
        """
        # Simulate reading
        reading = LightSensorReading(
            timestamp=time.time(),
            lux=np.random.uniform(10, 10000),
            uv_index=np.random.uniform(0, 11),
            visible_light=np.random.uniform(0, 100),
            infrared=np.random.uniform(0, 50)
        )
        
        self.light_readings.append(reading)
        return reading
    
    def read_acoustic_array(self) -> AcousticArrayReading:
        """
        Read acoustic array for direction finding.
        
        Array specs: 4+ microphones, gunshot detection, direction finding
        
        Returns:
            Acoustic array reading
        """
        # Simulate reading
        signatures = ["gunshot", "glass_break", "alarm", None]
        detected = np.random.choice(signatures, p=[0.01, 0.02, 0.02, 0.95])
        
        reading = AcousticArrayReading(
            timestamp=time.time(),
            direction_azimuth=np.random.uniform(0, 360),
            direction_elevation=np.random.uniform(-45, 45),
            sound_level_db=np.random.uniform(40, 90),
            signature_detected=detected,
            confidence=0.9 if detected else 0.0
        )
        
        self.acoustic_readings.append(reading)
        
        if detected:
            logger.warning(f"Acoustic signature detected: {detected} at "
                         f"{reading.direction_azimuth:.0f}° azimuth")
        
        return reading
    
    def read_motion_sensor(self) -> MotionDetection:
        """
        Read PIR motion sensor.
        
        Sensor specs: 5-10m range, PIR detection
        
        Returns:
            Motion detection result
        """
        # Simulate reading
        motion = np.random.random() > 0.9  # 10% chance of motion
        
        reading = MotionDetection(
            timestamp=time.time(),
            motion_detected=motion,
            distance_meters=np.random.uniform(1, 10) if motion else 0,
            confidence=0.95 if motion else 0.0
        )
        
        self.motion_detections.append(reading)
        
        if motion:
            logger.info(f"Motion detected at {reading.distance_meters:.1f}m")
        
        return reading
    
    def read_air_quality(self) -> AirQualityReading:
        """
        Read air quality sensors.
        
        Sensors: CO2, CO, VOC, smoke, PM2.5/PM10
        
        Returns:
            Air quality reading
        """
        # Simulate reading
        co2 = np.random.uniform(400, 1000)  # Normal: 400-1000 ppm
        co = np.random.uniform(0, 9)  # Normal: <9 ppm
        voc = np.random.uniform(0, 500)  # Normal: <500 ppb
        pm2_5 = np.random.uniform(0, 35)  # Good: 0-12, Moderate: 12.1-35.4
        pm10 = np.random.uniform(0, 50)  # Good: 0-54
        smoke = np.random.random() > 0.99  # 1% chance
        
        # Calculate AQI (simplified)
        aqi = int(max(
            pm2_5 / 12 * 50,  # PM2.5 contribution
            pm10 / 54 * 50,   # PM10 contribution
            co2 / 1000 * 50   # CO2 contribution
        ))
        
        reading = AirQualityReading(
            timestamp=time.time(),
            co2_ppm=co2,
            co_ppm=co,
            voc_ppb=voc,
            pm2_5=pm2_5,
            pm10=pm10,
            smoke_detected=smoke,
            air_quality_index=aqi
        )
        
        self.air_quality_readings.append(reading)
        
        if smoke:
            logger.critical("Smoke detected!")
        elif aqi > 150:
            logger.warning(f"Poor air quality: AQI {aqi}")
        
        return reading
    
    def read_soil_moisture(self) -> SoilMoistureReading:
        """
        Read soil moisture sensor.
        
        Sensor specs: 0-10cm depth, conductivity measurement
        
        Returns:
            Soil moisture reading
        """
        # Simulate reading
        reading = SoilMoistureReading(
            timestamp=time.time(),
            moisture_percent=np.random.uniform(10, 60),
            conductivity=np.random.uniform(0.1, 2.0),
            temperature_celsius=np.random.uniform(10, 25),
            depth_cm=5.0
        )
        
        self.soil_moisture_readings.append(reading)
        return reading
    
    def read_vibration_sensor(self) -> VibrationReading:
        """
        Read enhanced vibration sensor.
        
        Sensor specs: 0.1-10kHz, 0.001g sensitivity
        
        Returns:
            Vibration reading
        """
        # Simulate reading
        reading = VibrationReading(
            timestamp=time.time(),
            frequency_hz=np.random.uniform(0.1, 10000),
            amplitude_g=np.random.uniform(0.001, 0.1),
            frequency_range="0.1-10kHz"
        )
        
        self.vibration_readings.append(reading)
        return reading
    
    def capture_camera(
        self,
        resolution_mode: Optional[CameraResolution] = None
    ) -> CameraCapture:
        """
        Capture camera image/video.
        
        Camera modes:
        - Emergency: 480p
        - Normal: 1080p
        - Surveillance: 4K
        - Forensics: 8K
        
        Args:
            resolution_mode: Camera resolution mode
            
        Returns:
            Camera capture metadata
        """
        if resolution_mode:
            self.camera_resolution_mode = resolution_mode
        
        # Resolution and frame rate mapping
        resolution_map = {
            CameraResolution.EMERGENCY: ((640, 480), 60, 0.5),
            CameraResolution.NORMAL: ((1920, 1080), 30, 2.0),
            CameraResolution.SURVEILLANCE: ((3840, 2160), 30, 8.0),
            CameraResolution.FORENSICS: ((7680, 4320), 24, 30.0)
        }
        
        resolution, fps, size_mb = resolution_map[self.camera_resolution_mode]
        
        capture = CameraCapture(
            timestamp=time.time(),
            resolution_mode=self.camera_resolution_mode,
            resolution=resolution,
            frame_rate=fps,
            data_size_mb=size_mb
        )
        
        self.camera_captures.append(capture)
        
        logger.debug(f"Camera capture: {self.camera_resolution_mode.value} "
                    f"{resolution[0]}x{resolution[1]} @ {fps}fps")
        
        return capture
    
    def scan_lidar(
        self,
        scan_mode: Optional[LiDARMode] = None
    ) -> LiDARScan:
        """
        Perform LiDAR scan.
        
        LiDAR specs: 200m range, automotive grade
        Scan rates:
        - Idle: 5 Hz
        - Cruise: 20 Hz
        - Evasive: 100 Hz
        - Emergency: 200 Hz
        
        Args:
            scan_mode: LiDAR scan mode
            
        Returns:
            LiDAR scan data
        """
        if scan_mode:
            self.lidar_mode = scan_mode
        
        # Scan rate mapping
        scan_rate_map = {
            LiDARMode.IDLE: 5,
            LiDARMode.CRUISE: 20,
            LiDARMode.EVASIVE: 100,
            LiDARMode.EMERGENCY: 200
        }
        
        scan_rate = scan_rate_map[self.lidar_mode]
        
        # Simulate object detection
        num_objects = np.random.randint(0, 10)
        objects = []
        for i in range(num_objects):
            objects.append({
                "id": i,
                "distance_m": float(np.random.uniform(1, 200)),
                "azimuth_deg": float(np.random.uniform(0, 360)),
                "velocity_mps": float(np.random.uniform(-20, 20)),
                "type": np.random.choice(["vehicle", "pedestrian", "obstacle"])
            })
        
        scan = LiDARScan(
            timestamp=time.time(),
            scan_mode=self.lidar_mode,
            scan_rate_hz=scan_rate,
            range_meters=200.0,
            point_count=int(100000 * (scan_rate / 200)),  # More points at higher rates
            objects_detected=objects
        )
        
        self.lidar_scans.append(scan)
        
        if objects:
            logger.debug(f"LiDAR: {len(objects)} objects detected at {scan_rate}Hz")
        
        return scan
    
    def monitor_network(
        self,
        interface: str = "eth0"
    ) -> NetworkMetrics:
        """
        Monitor network performance.
        
        Network specs:
        - WiFi 6: 9.6 Gbps
        - 5G: 20 Gbps
        - 10G Ethernet: 10 Gbps
        Latency tracking: P50, P95, P99, P99.9
        
        Args:
            interface: Network interface to monitor
            
        Returns:
            Network metrics
        """
        # Determine bandwidth based on interface type
        if "wifi" in interface.lower() or "wlan" in interface.lower():
            max_bandwidth = 9.6  # WiFi 6
            interface_name = "WiFi6"
        elif "5g" in interface.lower():
            max_bandwidth = 20.0  # 5G
            interface_name = "5G"
        elif "eth" in interface.lower():
            max_bandwidth = 10.0  # 10G Ethernet
            interface_name = "10GEth"
        else:
            max_bandwidth = 1.0  # Fallback
            interface_name = interface
        
        # Simulate metrics
        bandwidth = np.random.uniform(0.1, max_bandwidth)
        
        # Generate latency distribution (log-normal)
        base_latency = np.random.lognormal(0, 0.5)
        
        metrics = NetworkMetrics(
            timestamp=time.time(),
            interface=interface_name,
            bandwidth_gbps=bandwidth,
            latency_p50=base_latency,
            latency_p95=base_latency * 2,
            latency_p99=base_latency * 5,
            latency_p999=base_latency * 10,
            packet_loss_percent=np.random.uniform(0, 0.1)
        )
        
        self.network_metrics.append(metrics)
        
        if metrics.latency_p99 > 100:
            logger.warning(f"High network latency: P99={metrics.latency_p99:.1f}ms")
        
        return metrics
    
    def read_all_sensors(self) -> Dict[str, Any]:
        """
        Read all enabled sensors.
        
        Returns:
            Dictionary with all sensor readings
        """
        readings = {
            "timestamp": time.time()
        }
        
        if self.humidity_enabled:
            readings["humidity"] = self.read_humidity()
        
        if self.light_enabled:
            readings["light"] = self.read_light_sensor()
        
        if self.acoustic_array_enabled:
            readings["acoustic"] = self.read_acoustic_array()
        
        if self.motion_enabled:
            readings["motion"] = self.read_motion_sensor()
        
        if self.air_quality_enabled:
            readings["air_quality"] = self.read_air_quality()
        
        if self.soil_moisture_enabled:
            readings["soil_moisture"] = self.read_soil_moisture()
        
        if self.vibration_enabled:
            readings["vibration"] = self.read_vibration_sensor()
        
        if self.camera_enabled:
            readings["camera"] = self.capture_camera()
        
        if self.lidar_enabled:
            readings["lidar"] = self.scan_lidar()
        
        if self.network_monitor_enabled:
            readings["network"] = self.monitor_network()
        
        return readings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor suite statistics."""
        return {
            "humidity_readings": len(self.humidity_readings),
            "light_readings": len(self.light_readings),
            "acoustic_readings": len(self.acoustic_readings),
            "motion_detections": len(self.motion_detections),
            "air_quality_readings": len(self.air_quality_readings),
            "soil_moisture_readings": len(self.soil_moisture_readings),
            "vibration_readings": len(self.vibration_readings),
            "camera_captures": len(self.camera_captures),
            "lidar_scans": len(self.lidar_scans),
            "network_metrics": len(self.network_metrics),
            "enabled_sensors": {
                "humidity": self.humidity_enabled,
                "light": self.light_enabled,
                "acoustic_array": self.acoustic_array_enabled,
                "motion": self.motion_enabled,
                "air_quality": self.air_quality_enabled,
                "soil_moisture": self.soil_moisture_enabled,
                "vibration": self.vibration_enabled,
                "camera": self.camera_enabled,
                "lidar": self.lidar_enabled,
                "network": self.network_monitor_enabled
            }
        }
