#!/usr/bin/env python3
"""
Example: 3D Drone Swarm with Frequency Sensors

Demonstrates 3D spatial navigation with frequency-based threat detection.
Shows how drones can use frequency intelligence for situational awareness.

Usage:
    python3 example/example_3d_drone_swarm.py
"""

import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptiveengineer import AliveLoopNode
from plugins.rf_spectrum_analyzer import RFSpectrumAnalyzer
from plugins.acoustic_frequency import AcousticFrequencyAnalyzer
from plugins.advanced_sensors import AdvancedSensorSuite, LiDARMode, CameraResolution

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """3D Drone Swarm Demonstration."""
    logger.info("\n" + "="*80)
    logger.info("3D DRONE SWARM WITH FREQUENCY SENSORS")
    logger.info("="*80)
    
    # Create drones in 3D space
    drones = []
    for i in range(3):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(0, 50)  # Altitude
        
        drone = AliveLoopNode(
            position=(x, y, z),
            velocity=(0, 0, 0),
            initial_energy=10.0,
            node_id=i+1,
            spatial_dims=3
        )
        drones.append(drone)
        logger.info(f"   Drone {i+1} positioned at ({x:.1f}, {y:.1f}, {z:.1f}m altitude)")
    
    # Equip each drone with sensors
    logger.info("\n   Equipping drones with sensors...")
    for i, drone in enumerate(drones):
        # RF sensors for communication and threat detection
        rf_sensor = RFSpectrumAnalyzer()
        
        # Acoustic sensors for direction finding
        acoustic_sensor = AcousticFrequencyAnalyzer()
        
        # Advanced sensor suite
        sensor_suite = AdvancedSensorSuite()
        
        drone.rf_sensor = rf_sensor
        drone.acoustic_sensor = acoustic_sensor
        drone.sensor_suite = sensor_suite
        
        logger.info(f"      Drone {i+1}: RF + Acoustic + LiDAR + Camera")
    
    # Simulate drone operations
    logger.info("\n   Simulating drone swarm operations...")
    
    for i, drone in enumerate(drones):
        logger.info(f"\n   === Drone {i+1} Sensor Sweep ===")
        
        # RF scan for threats
        rf_results = drone.rf_sensor.analyze_spectrum(battery_level=drone.energy/10.0)
        logger.info(f"      RF: {rf_results['signals_detected']} signals, "
                   f"{rf_results['threats']['total_threats']} threats")
        
        # Acoustic direction finding
        acoustic_reading = drone.sensor_suite.read_acoustic_array()
        if acoustic_reading.signature_detected:
            logger.info(f"      Acoustic: {acoustic_reading.signature_detected} detected at "
                       f"{acoustic_reading.direction_azimuth:.0f}° azimuth")
        
        # LiDAR scan (emergency mode for high-speed response)
        lidar_scan = drone.sensor_suite.scan_lidar(LiDARMode.EMERGENCY)
        logger.info(f"      LiDAR: {len(lidar_scan.objects_detected)} objects at {lidar_scan.scan_rate_hz}Hz")
        
        # Camera capture (surveillance mode)
        camera = drone.sensor_suite.capture_camera(CameraResolution.SURVEILLANCE)
        logger.info(f"      Camera: {camera.resolution[0]}x{camera.resolution[1]} @ {camera.frame_rate}fps")
    
    # Swarm coordination
    logger.info("\n   Swarm coordination and threat sharing...")
    
    threats_detected = []
    for drone in drones:
        if drone.rf_sensor.detected_threats:
            threats_detected.extend(drone.rf_sensor.detected_threats)
    
    if threats_detected:
        logger.info(f"      ⚠️  Swarm detected {len(threats_detected)} threats")
        logger.info("      Broadcasting to all drones for coordinated response")
    else:
        logger.info("      ✓ No threats detected - swarm operating normally")
    
    logger.info("\n" + "="*80)
    logger.info("3D DRONE SWARM DEMONSTRATION COMPLETE")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
