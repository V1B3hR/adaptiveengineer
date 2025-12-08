# Frequency Intelligence System

## Overview

The Frequency Intelligence System is a comprehensive threat detection and analysis platform that operates across both physical (RF, acoustic, vibration) and cyber (network, malware, biometric) domains. It uses Fast Fourier Transform (FFT) analysis to detect periodic patterns and correlate threats across multiple domains for advanced threat detection.

## Architecture

### Core Components

1. **RF Spectrum Analyzer** (`plugins/rf_spectrum_analyzer.py`)
   - Frequency scanning: VLF to EHF bands (3 kHz - 300 GHz)
   - Default scan range: 20 MHz - 6 GHz (configurable)
   - Sensitivity: -120 dBm minimum
   - Signal classification: WiFi, cellular, IoT, drones, GPS
   - Threat detection: RF jamming, rogue transmitters

2. **Acoustic Frequency Analyzer** (`plugins/acoustic_frequency.py`)
   - FFT analysis across frequency ranges:
     - Infrasound: 0-20 Hz
     - Audible: 20-20 kHz
     - Ultrasound: 20-100 kHz
   - Sample rate: 44.1 kHz (CD quality), configurable to 192 kHz
   - Signature recognition: gunshots, glass break, machinery, drones, alarms
   - Machinery diagnostics via harmonics analysis
   - Covert ultrasonic communication detection

3. **Vibration Frequency Analyzer** (`plugins/vibration_frequency.py`)
   - Bearing defect detection:
     - BPFO (Ball Pass Frequency Outer race)
     - BPFI (Ball Pass Frequency Inner race)
     - BSF (Ball Spin Frequency)
     - FTF (Fundamental Train Frequency - cage)
   - Structural health monitoring
   - Frequency bands:
     - Seismic: 0.1-10 Hz
     - Machinery: 10-1000 Hz
     - Bearing: 1-10 kHz
   - Sample rate: 10 kHz minimum
   - Sensitivity: 0.001g (1 milli-g)

4. **Network Frequency Analyzer** (`core/network_frequency_analyzer.py`)
   - FFT on network traffic inter-arrival times
   - Detects periodic patterns:
     - C2 beaconing: ~60 second periods
     - DDoS: sub-second periods
     - Port scanning: 5-15 second periods
   - Covert timing channel detection
   - Traffic pattern classification

5. **Behavioral Frequency Analyzer** (`core/behavioral_frequency.py`)
   - Process behavior pattern analysis via FFT
   - Cryptominer detection:
     - CPU usage FFT
     - 5-10 second GPU mining cycles
     - Periodic resource consumption patterns
   - Ransomware detection:
     - File access rate >10 files/sec
     - Disk I/O spike detection
   - Resource usage frequency signatures

6. **Keystroke Frequency Analyzer** (`core/keystroke_frequency.py`)
   - Behavioral biometrics via typing rhythm
   - Measurements:
     - Dwell time: key hold duration
     - Flight time: inter-key timing
   - FFT-based rhythm signature (20 coefficients)
   - Impersonation detection via baseline comparison
   - Continuous authentication capability

7. **Unified Frequency Intelligence** (`core/unified_frequency_intelligence.py`)
   - Orchestrates all frequency analyzers
   - Cross-domain threat correlation:
     - Physical + Cyber Attack: RF jamming + network disruption
     - APT Campaign: Ultrasonic + C2 beacon
     - Insider Threat: Keystroke anomaly + data exfiltration
     - Coordinated Attack: Multiple domains simultaneously
     - Reconnaissance: Port scan + acoustic monitoring
   - Probabilistic threat scoring
   - Actionable countermeasure recommendations

## Usage

### Basic Setup

```python
from core.unified_frequency_intelligence import UnifiedFrequencyIntelligence

# Initialize with all domains enabled
unified = UnifiedFrequencyIntelligence(
    enable_rf=True,
    enable_acoustic=True,
    enable_vibration=True,
    enable_network=True,
    enable_behavioral=True,
    enable_keystroke=True
)

# Analyze environment
environment = unified.analyze_all_frequencies(
    battery_level=1.0,
    duration_seconds=2.0
)

print(f"Threat score: {environment.overall_threat_score}")
print(f"Active correlations: {environment.active_correlations}")
```

### Integration with AliveLoopNode

```python
from adaptiveengineer import AliveLoopNode

# Create node
node = AliveLoopNode(
    position=(0, 0),
    velocity=(0, 0),
    initial_energy=10.0,
    node_id=1
)

# Enable frequency intelligence
node.enable_frequency_intelligence(
    enable_rf=True,
    enable_acoustic=True,
    enable_network=True
)

# Analyze environment
results = node.analyze_frequency_environment(duration_seconds=1.0)

# Adapt to threats
actions = node.adapt_to_frequency_threats()
```

### Individual Analyzer Usage

#### RF Spectrum Analysis

```python
from plugins.rf_spectrum_analyzer import RFSpectrumAnalyzer

analyzer = RFSpectrumAnalyzer(
    scan_range=(20e6, 6e9),  # 20 MHz to 6 GHz
    sensitivity_dbm=-120.0
)

# Add authorized transmitters
analyzer.add_authorized_transmitter(2.437e9, 100e6)  # WiFi

# Scan spectrum
signals = analyzer.scan_spectrum(duration_seconds=1.0, battery_level=1.0)

# Detect threats
jamming = analyzer.detect_jamming()
rogues = analyzer.detect_rogue_transmitters()
```

#### Acoustic Analysis

```python
from plugins.acoustic_frequency import AcousticFrequencyAnalyzer

analyzer = AcousticFrequencyAnalyzer(sample_rate_hz=44100)

# Analyze audio
events = analyzer.analyze_audio_sample(duration_seconds=2.0)

# Machinery diagnostics
diagnostic = analyzer.diagnose_machinery(
    machine_id="motor_001",
    duration_seconds=5.0
)

print(f"Health score: {diagnostic.health_score}")
```

#### Vibration Analysis

```python
from plugins.vibration_frequency import (
    VibrationFrequencyAnalyzer, 
    BearingGeometry
)

analyzer = VibrationFrequencyAnalyzer(sample_rate_hz=10000)

# Register bearing
bearing = BearingGeometry(
    pitch_diameter_mm=100,
    ball_diameter_mm=12,
    num_balls=9,
    contact_angle_deg=15,
    shaft_speed_rpm=1800
)

analyzer.register_bearing("bearing_001", bearing)

# Analyze vibration
events = analyzer.analyze_vibration(bearing_id="bearing_001")

# Structural health
health = analyzer.monitor_structural_health("bridge_001")
```

#### Network Analysis

```python
from core.network_frequency_analyzer import NetworkFrequencyAnalyzer, NetworkEvent

analyzer = NetworkFrequencyAnalyzer()

# Add network events
event = NetworkEvent(
    timestamp=time.time(),
    source_ip="192.168.1.100",
    dest_ip="8.8.8.8",
    protocol="TCP",
    size_bytes=1500,
    port=443
)
analyzer.add_network_event(event)

# Detect threats
c2_threats = analyzer.detect_c2_beaconing()
port_scans = analyzer.detect_port_scanning()
covert_channels = analyzer.detect_covert_timing_channel()
```

#### Behavioral Analysis

```python
from core.behavioral_frequency import BehavioralFrequencyAnalyzer, ProcessBehavior

analyzer = BehavioralFrequencyAnalyzer()

# Add process behavior samples
behavior = ProcessBehavior(
    process_id=1000,
    process_name="app.exe",
    timestamp=time.time(),
    cpu_percent=50.0,
    memory_mb=200.0,
    disk_io_bytes=1000000,
    network_io_bytes=5000,
    file_access_count=5
)
analyzer.add_process_behavior(behavior)

# Analyze process
threats = analyzer.analyze_process(1000)
```

#### Keystroke Analysis

```python
from core.keystroke_frequency import KeystrokeFrequencyAnalyzer, KeystrokeEvent

analyzer = KeystrokeFrequencyAnalyzer()

# Add keystroke events
event = KeystrokeEvent(
    key="a",
    press_time=time.time(),
    release_time=time.time() + 0.15,
    user_id="alice"
)
analyzer.add_keystroke(event)

# Establish baseline (requires 100+ keystrokes)
signature = analyzer.establish_baseline("alice")

# Authenticate user
result = analyzer.authenticate("alice")
print(f"Status: {result.status}, Confidence: {result.confidence}")
```

## Cross-Domain Threat Correlation

The Unified Frequency Intelligence system automatically correlates threats across domains:

### Physical + Cyber Attack
- **Indicators**: RF jamming + Network disruption
- **Severity**: Critical
- **Countermeasures**:
  - Switch to backup RF channels
  - Enable network redundancy protocols
  - Activate physical security measures
  - Isolate critical systems

### APT Campaign
- **Indicators**: Ultrasonic communication + C2 beaconing
- **Severity**: Critical
- **Countermeasures**:
  - Block ultrasonic frequencies (>18kHz)
  - Isolate systems with C2 beaconing
  - Enable advanced packet inspection
  - Conduct full malware scan

### Insider Threat
- **Indicators**: Keystroke anomaly + Malicious behavior
- **Severity**: High
- **Countermeasures**:
  - Lock user account immediately
  - Require multi-factor re-authentication
  - Review recent file access logs
  - Monitor data exfiltration attempts

### Coordinated Attack
- **Indicators**: Threats in 3+ domains simultaneously
- **Severity**: Critical
- **Countermeasures**:
  - Activate emergency defense protocols
  - Isolate all critical systems
  - Enable all redundancy measures
  - Alert all security teams

## Performance Characteristics

- **FFT Operations**: <100ms typical
- **RF Scan**: 1-5 seconds (adaptive based on battery)
- **Acoustic Analysis**: <50ms per sample
- **Vibration Analysis**: <100ms per sample
- **Network Analysis**: Real-time (event-based)
- **Behavioral Analysis**: 1-60 seconds (window-based)
- **Keystroke Analysis**: <10ms per authentication

## Energy-Aware Operation

All analyzers support adaptive operation based on battery level:

```python
# High battery: Full resolution
environment = unified.analyze_all_frequencies(
    battery_level=1.0,
    duration_seconds=2.0
)

# Low battery: Reduced resolution
environment = unified.analyze_all_frequencies(
    battery_level=0.2,
    duration_seconds=0.5
)
```

## Hardware Integration

### RF Spectrum
- **Hardware**: Software Defined Radio (SDR)
- **Recommended**: HackRF One, RTL-SDR, BladeRF
- **Interface**: SoapySDR, GNU Radio

### Acoustic
- **Hardware**: High-quality microphone or acoustic array
- **Recommended**: 4+ microphone array for direction finding
- **Sample Rate**: 44.1 kHz minimum, 192 kHz for ultrasonic

### Vibration
- **Hardware**: Accelerometer or piezoelectric sensor
- **Recommended**: MEMS accelerometer, ICP sensors
- **Sample Rate**: 10 kHz minimum
- **Sensitivity**: 0.001g or better

### Network
- **Hardware**: Network tap or SPAN port
- **Interface**: libpcap, Scapy

### Behavioral
- **Hardware**: Host-based agent
- **Interface**: OS APIs (psutil, WMI, /proc)

### Keystroke
- **Hardware**: Keyboard event capture
- **Interface**: OS event hooks (Windows Hooks, X11, evdev)

## Examples

See the `example/` directory for complete demonstrations:

- `example_frequency_defense.py`: Comprehensive showcase of all analyzers
- `example_3d_drone_swarm.py`: 3D spatial navigation with frequency sensors
- `example_cyber_sensors.py`: Cyber-defense focused demonstration

## Testing

Run tests with:

```bash
python3 -m pytest tests/test_*frequency*.py -v
```

Target coverage: >85%

## Security Considerations

1. **Sensor Data Privacy**: Acoustic and keystroke data may contain sensitive information
2. **False Positives**: Tune thresholds based on your environment
3. **Resource Usage**: FFT operations are CPU-intensive
4. **Network Monitoring**: Ensure compliance with network monitoring policies
5. **Access Control**: Restrict access to frequency intelligence data

## Troubleshooting

### High False Positive Rate
- Establish longer baseline periods
- Adjust detection thresholds
- Add known-good signals to whitelists

### Performance Issues
- Reduce FFT window sizes
- Decrease sample rates
- Enable adaptive analysis
- Use battery-aware settings

### Integration Issues
- Check hardware compatibility
- Verify sample rates
- Ensure proper permissions
- Review API documentation

## Future Enhancements

- Machine learning-based signature recognition
- Distributed frequency analysis across node swarms
- Real-time hardware acceleration (GPU/FPGA)
- Advanced signal processing (wavelets, STFT)
- Cloud-based threat intelligence sharing

## References

- FFT algorithms: Cooley-Tukey, numpy.fft
- RF spectrum analysis: GNU Radio, SDR
- Bearing defect frequencies: ISO 15243
- Keystroke dynamics: CMU Keystroke Dynamics Benchmark
- Network analysis: MITRE ATT&CK Framework

## License

See main project LICENSE file.

## Contributors

Part of the Adaptive Engineer project.
