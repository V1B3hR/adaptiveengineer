# Advanced Sensor Suite

## Overview

The Advanced Sensor Suite provides comprehensive environmental monitoring and threat detection capabilities. It includes physical sensors (humidity, light, acoustic array, motion, air quality, soil moisture, vibration) and enhanced specifications for camera, LiDAR, and network monitoring.

## Sensor Specifications

### Environmental Sensors

#### Humidity Sensor
- **Range**: 0-100% RH (Relative Humidity)
- **Accuracy**: ±2%
- **Temperature**: Included (Celsius)
- **Use Cases**: Environmental monitoring, HVAC optimization, data center management

```python
from plugins.advanced_sensors import AdvancedSensorSuite

suite = AdvancedSensorSuite()
reading = suite.read_humidity()
print(f"Humidity: {reading.humidity_percent}%")
print(f"Temperature: {reading.temperature_celsius}°C")
```

#### Light Sensor
- **Range**: 0.01-100k lux
- **Measurements**: UV index, visible light, infrared
- **Use Cases**: Daylight harvesting, security (detecting intrusions), agriculture

```python
reading = suite.read_light_sensor()
print(f"Lux: {reading.lux}")
print(f"UV Index: {reading.uv_index}")
```

#### Air Quality Sensor
- **CO2**: Parts per million (ppm)
- **CO**: Parts per million (ppm)
- **VOC**: Parts per billion (ppb)
- **Particulate Matter**: PM2.5 and PM10 (μg/m³)
- **Smoke Detection**: Boolean
- **AQI**: Air Quality Index (0-500 scale)

```python
reading = suite.read_air_quality()
print(f"CO2: {reading.co2_ppm} ppm")
print(f"AQI: {reading.air_quality_index}")
if reading.smoke_detected:
    print("⚠️ SMOKE DETECTED!")
```

#### Soil Moisture Sensor
- **Range**: 0-100% moisture
- **Depth**: 0-10cm
- **Conductivity**: mS/cm
- **Temperature**: Celsius
- **Use Cases**: Agriculture, landscaping, environmental monitoring

```python
reading = suite.read_soil_moisture()
print(f"Moisture: {reading.moisture_percent}%")
print(f"Conductivity: {reading.conductivity} mS/cm")
```

### Acoustic Array

- **Configuration**: 4+ microphones for direction finding
- **Direction Finding**: Azimuth (0-360°) and Elevation (-90 to 90°)
- **Sound Level**: Decibels (dB)
- **Signature Detection**: Gunshots, glass break, alarms
- **Use Cases**: Security, gunshot detection, acoustic surveillance

```python
reading = suite.read_acoustic_array()
print(f"Direction: {reading.direction_azimuth}° azimuth")
print(f"Sound level: {reading.sound_level_db} dB")
if reading.signature_detected:
    print(f"Detected: {reading.signature_detected}")
```

### Motion Sensor (PIR)

- **Type**: Passive Infrared (PIR)
- **Range**: 5-10 meters
- **Detection**: Boolean motion detected
- **Distance**: Estimated distance in meters
- **Confidence**: 0.0-1.0
- **Use Cases**: Intrusion detection, occupancy sensing, automation

```python
reading = suite.read_motion_sensor()
if reading.motion_detected:
    print(f"Motion at {reading.distance_meters}m")
```

### Enhanced Vibration Sensor

- **Frequency Range**: 0.1-10 kHz
- **Sensitivity**: 0.001g (1 milli-g)
- **Measurements**: Frequency, amplitude
- **Use Cases**: Machinery diagnostics, structural health, seismic monitoring

```python
reading = suite.read_vibration_sensor()
print(f"Frequency: {reading.frequency_hz} Hz")
print(f"Amplitude: {reading.amplitude_g} g")
```

## Imaging and Ranging

### Camera System

Enhanced camera with multiple resolution modes for different operational scenarios:

#### Resolution Modes

| Mode | Resolution | FPS | Use Case | Data Size |
|------|-----------|-----|----------|-----------|
| **Emergency** | 480p (640×480) | 60 | Low-bandwidth, fast response | 0.5 MB/frame |
| **Normal** | 1080p (1920×1080) | 30 | Standard operations | 2.0 MB/frame |
| **Surveillance** | 4K (3840×2160) | 30 | High-quality monitoring | 8.0 MB/frame |
| **Forensics** | 8K (7680×4320) | 24 | Evidence collection | 30.0 MB/frame |

```python
from plugins.advanced_sensors import CameraResolution

# Emergency mode - low bandwidth
capture = suite.capture_camera(CameraResolution.EMERGENCY)

# Surveillance mode - high quality
capture = suite.capture_camera(CameraResolution.SURVEILLANCE)

# Forensics mode - maximum detail
capture = suite.capture_camera(CameraResolution.FORENSICS)

print(f"Resolution: {capture.resolution}")
print(f"Frame rate: {capture.frame_rate} fps")
```

### LiDAR System

Automotive-grade LiDAR with adaptive scan rates:

#### Specifications
- **Range**: 200 meters
- **Technology**: Automotive grade
- **Point Cloud**: Variable density based on scan rate

#### Scan Modes

| Mode | Scan Rate | Use Case | Point Count |
|------|-----------|----------|-------------|
| **Idle** | 5 Hz | Stationary monitoring | 25,000 pts |
| **Cruise** | 20 Hz | Normal navigation | 100,000 pts |
| **Evasive** | 100 Hz | High-speed maneuvering | 500,000 pts |
| **Emergency** | 200 Hz | Critical situations | 1,000,000 pts |

```python
from plugins.advanced_sensors import LiDARMode

# Cruise mode for normal operations
scan = suite.scan_lidar(LiDARMode.CRUISE)

# Emergency mode for critical response
scan = suite.scan_lidar(LiDARMode.EMERGENCY)

print(f"Scan rate: {scan.scan_rate_hz} Hz")
print(f"Objects detected: {len(scan.objects_detected)}")

for obj in scan.objects_detected:
    print(f"  - {obj['type']} at {obj['distance_m']}m")
```

## Network Monitoring

Enhanced network monitoring with support for modern high-speed networks:

### Supported Interfaces

| Interface | Bandwidth | Description |
|-----------|-----------|-------------|
| **WiFi 6** | 9.6 Gbps | Next-gen wireless |
| **5G** | 20 Gbps | Cellular networking |
| **10G Ethernet** | 10 Gbps | High-speed wired |

### Latency Tracking

Monitors latency at multiple percentiles for comprehensive performance analysis:

- **P50**: Median latency
- **P95**: 95th percentile
- **P99**: 99th percentile  
- **P99.9**: 99.9th percentile

```python
metrics = suite.monitor_network(interface="eth0")

print(f"Bandwidth: {metrics.bandwidth_gbps} Gbps")
print(f"Latency P50: {metrics.latency_p50}ms")
print(f"Latency P99: {metrics.latency_p99}ms")
print(f"Packet loss: {metrics.packet_loss_percent}%")
```

## Complete Sensor Sweep

Read all enabled sensors at once:

```python
suite = AdvancedSensorSuite()

# Read all sensors
readings = suite.read_all_sensors()

print(f"Humidity: {readings['humidity'].humidity_percent}%")
print(f"Light: {readings['light'].lux} lux")
print(f"Motion: {readings['motion'].motion_detected}")
print(f"Air Quality: {readings['air_quality'].air_quality_index}")
print(f"Camera: {readings['camera'].resolution}")
print(f"LiDAR: {len(readings['lidar'].objects_detected)} objects")
print(f"Network: {readings['network'].bandwidth_gbps} Gbps")
```

## Sensor Enable/Disable

Control which sensors are active:

```python
suite = AdvancedSensorSuite()

# Disable sensors not needed
suite.humidity_enabled = False
suite.soil_moisture_enabled = False

# Enable only required sensors
suite.camera_enabled = True
suite.lidar_enabled = True
suite.network_monitor_enabled = True

# Read only enabled sensors
readings = suite.read_all_sensors()
```

## Use Cases

### Security Operations
- **Acoustic Array**: Gunshot detection
- **Motion Sensor**: Perimeter intrusion
- **Camera**: Visual verification
- **LiDAR**: Object tracking
- **Network**: Intrusion detection

### Industrial Monitoring
- **Vibration**: Machinery health
- **Humidity/Temperature**: Environmental control
- **Air Quality**: Safety monitoring
- **Acoustic**: Equipment diagnostics

### Autonomous Vehicles
- **LiDAR**: Obstacle avoidance
- **Camera**: Visual navigation
- **Network**: V2V communication
- **Motion**: Occupancy detection

### Agriculture
- **Soil Moisture**: Irrigation optimization
- **Light**: Growth optimization
- **Humidity/Temperature**: Climate control
- **Air Quality**: Greenhouse management

### Smart Buildings
- **Occupancy**: Motion sensors
- **HVAC**: Humidity, temperature, CO2
- **Security**: Camera, acoustic, motion
- **Network**: Building automation

## Integration Examples

### Drone Surveillance

```python
# Configure for aerial surveillance
suite.camera_enabled = True
suite.lidar_enabled = True
suite.acoustic_array_enabled = True

# Set appropriate modes
suite.capture_camera(CameraResolution.SURVEILLANCE)
suite.scan_lidar(LiDARMode.EVASIVE)

# Continuous monitoring
while flying:
    readings = suite.read_all_sensors()
    
    # Check for threats
    if readings['acoustic'].signature_detected == "gunshot":
        # Record location
        position = drone.position
        # Alert authorities
        send_alert(position, readings['acoustic'])
```

### Industrial Machinery Monitoring

```python
# Configure for machinery monitoring
suite.vibration_enabled = True
suite.acoustic_array_enabled = True
suite.air_quality_enabled = True

# Continuous monitoring
while operating:
    vibration = suite.read_vibration_sensor()
    air = suite.read_air_quality()
    
    # Check for anomalies
    if vibration.amplitude_g > 0.1:
        log_warning("Excessive vibration detected")
    
    if air.smoke_detected:
        log_critical("Smoke detected - shutdown machinery")
        emergency_shutdown()
```

### Smart Building Automation

```python
# Configure for building automation
suite.humidity_enabled = True
suite.light_enabled = True
suite.motion_enabled = True
suite.air_quality_enabled = True

# Optimization loop
while True:
    humidity = suite.read_humidity()
    light = suite.read_light_sensor()
    motion = suite.read_motion_sensor()
    air = suite.read_air_quality()
    
    # HVAC control
    if humidity.humidity_percent > 60:
        activate_dehumidifier()
    
    # Lighting control
    if light.lux < 300 and motion.motion_detected:
        turn_on_lights()
    
    # Ventilation control
    if air.co2_ppm > 1000:
        increase_ventilation()
```

## Performance

- **Sensor Read Time**: <10ms per sensor
- **All Sensors Sweep**: <100ms
- **Camera Capture**: 33ms @ 30fps, 17ms @ 60fps
- **LiDAR Scan**: 5-200ms depending on mode
- **Network Metrics**: Real-time

## Hardware Recommendations

### Environmental Sensors
- **Humidity**: DHT22, SHT31, BME280
- **Light**: BH1750, TSL2591, VEML7700
- **Air Quality**: BME680, CCS811, PMS5003

### Acoustic
- **Microphones**: MEMS microphones (ICS-43434, SPH0645)
- **Array**: 4-8 microphones in geometric pattern

### Motion
- **PIR**: HC-SR501, AM312

### Vibration
- **Accelerometer**: ADXL345, MPU6050, LIS3DH
- **High-End**: PCB Piezotronics ICP sensors

### Camera
- **Embedded**: Raspberry Pi Camera, Intel RealSense
- **Industrial**: Basler, FLIR

### LiDAR
- **Automotive**: Velodyne, Ouster, Livox
- **Low-Cost**: RPLidar, YDLiDAR

## Statistics and Monitoring

Track sensor performance:

```python
stats = suite.get_statistics()

print(f"Humidity readings: {stats['humidity_readings']}")
print(f"Motion detections: {stats['motion_detections']}")
print(f"Camera captures: {stats['camera_captures']}")
print(f"LiDAR scans: {stats['lidar_scans']}")

# Check which sensors are enabled
enabled = stats['enabled_sensors']
for sensor, status in enabled.items():
    print(f"{sensor}: {'ON' if status else 'OFF'}")
```

## Troubleshooting

### Sensor Not Reading
- Check enable flag
- Verify hardware connection
- Confirm proper initialization

### High Noise in Readings
- Check sensor calibration
- Shield from interference
- Average multiple readings

### Performance Issues
- Disable unused sensors
- Reduce sample rates
- Use selective reading

## Future Enhancements

- Sensor fusion algorithms
- Automatic calibration
- Cloud-based analytics
- Machine learning integration
- Predictive maintenance

## License

See main project LICENSE file.
