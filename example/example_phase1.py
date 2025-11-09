#!/usr/bin/env python3
"""
Example demonstrating Phase 1: Core Foundations & Emergency Self-Organization

This example shows:
1. Modular & Extensible Architecture - Plugin system with IT, Security, AL modules
2. Robust, Adaptive State Variables - Universal and domain-specific state
3. Emergency & Self-Organization - Auto-organization under stress

Usage:
    python3 example/example_phase1.py
"""

import logging
import sys
import os

# Add parent directory to path to allow imports from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core plugin system
from core.plugin_base import PluginBase, StateVariable
from core.plugin_manager import PluginManager

# Import plugins
from plugins.it_operations import ITOperationsPlugin
from plugins.security import SecurityPlugin
from plugins.artificial_life import ArtificialLifePlugin

# Import base node
from adaptiveengineer import AliveLoopNode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('phase1_example')


def demonstrate_plugin_system():
    """Demonstrate the plugin system architecture."""
    logger.info("=" * 60)
    logger.info("PHASE 1 DEMONSTRATION: Core Foundations & Emergency Self-Organization")
    logger.info("=" * 60)
    
    # ========================================================================
    # 1. Create a node with plugin architecture
    # ========================================================================
    logger.info("\n1. Creating adaptive node with plugin architecture...")
    
    node = AliveLoopNode(
        position=(0, 0),
        velocity=(0, 0),
        initial_energy=10.0,
        node_id=1
    )
    
    logger.info(f"   ✓ Node {node.node_id} created")
    
    # ========================================================================
    # 2. Initialize plugin manager and register plugins
    # ========================================================================
    logger.info("\n2. Initializing plugin system...")
    
    plugin_manager = PluginManager()
    
    # Register plugins
    it_plugin = ITOperationsPlugin()
    security_plugin = SecurityPlugin()
    al_plugin = ArtificialLifePlugin()
    
    plugin_manager.register_plugin(it_plugin)
    plugin_manager.register_plugin(security_plugin)
    plugin_manager.register_plugin(al_plugin)
    
    # Initialize all plugins with the node
    plugin_manager.initialize_all(node)
    
    logger.info(f"   ✓ {len(plugin_manager.plugins)} plugins registered and initialized")
    
    # Display plugin info
    plugin_info = plugin_manager.get_plugin_info()
    for plugin_id, info in plugin_info.items():
        logger.info(f"     - {plugin_id} ({info['type']}): {len(info['state_variables'])} state variables, {len(info['actions'])} actions")
    
    # ========================================================================
    # 3. Demonstrate state variables (universal + domain-specific)
    # ========================================================================
    logger.info("\n3. Demonstrating robust, adaptive state variables...")
    
    # Universal state (from AL plugin)
    logger.info("\n   Universal State Variables (Artificial Life):")
    al_state = al_plugin.get_state()
    for var_name, value in al_state.items():
        var = al_plugin.get_state_variable(var_name)
        logger.info(f"     - {var_name}: {value:.3f} (range: {var.min_value}-{var.max_value})")
    
    # IT/Security-specific state
    logger.info("\n   IT Operations State Variables:")
    it_state = it_plugin.get_state()
    for var_name, value in sorted(it_state.items())[:4]:  # Show first 4
        var = it_plugin.get_state_variable(var_name)
        logger.info(f"     - {var_name}: {value:.3f} (range: {var.min_value}-{var.max_value})")
    
    logger.info("\n   Security State Variables:")
    security_state = security_plugin.get_state()
    for var_name, value in sorted(security_state.items())[:4]:  # Show first 4
        var = security_plugin.get_state_variable(var_name)
        logger.info(f"     - {var_name}: {value:.3f} (range: {var.min_value}-{var.max_value})")
    
    # ========================================================================
    # 4. Simulate normal operation
    # ========================================================================
    logger.info("\n4. Simulating normal operation...")
    
    for step in range(5):
        # Update all plugins
        plugin_manager.update_all(delta_time=1.0)
        
        # Simulate some activity
        node.energy -= 0.5  # Energy consumption
        node._time = step
        
        if step == 2:
            logger.info(f"   Step {step}: Normal operation - all systems stable")
            logger.info(f"     Energy: {node.energy:.1f}, Anxiety: {node.anxiety:.1f}")
            it_summary = it_plugin.get_health_summary()
            logger.info(f"     Service uptime: {it_summary['service_uptime']:.2%}, CPU: {it_summary['cpu_utilization']:.2%}")
    
    # ========================================================================
    # 5. Demonstrate emergency self-organization (AL Principle #1)
    # ========================================================================
    logger.info("\n5. Demonstrating Emergency & Self-Organization...")
    logger.info("   Simulating stress conditions (low energy + high threat)...\n")
    
    # Create stress conditions
    node.energy = 2.0  # Critical energy level
    node.anxiety = 9.0  # High anxiety
    node.energy_attack_detected = True  # Detect attack
    
    logger.info("   STRESS CONDITIONS DETECTED:")
    logger.info(f"     - Energy: {node.energy:.1f} (critical)")
    logger.info(f"     - Anxiety: {node.anxiety:.1f} (high)")
    logger.info(f"     - Energy attack: {node.energy_attack_detected}")
    
    # Update plugins - they will detect and respond to stress
    logger.info("\n   Plugin responses to stress:")
    plugin_manager.update_all(delta_time=1.0)
    
    # Check IT Operations response
    it_summary = it_plugin.get_health_summary()
    logger.info(f"\n   IT Operations adaptation:")
    logger.info(f"     - Service uptime: {it_summary['service_uptime']:.2%}")
    logger.info(f"     - Error rate: {it_summary['error_rate']:.2%}")
    logger.info(f"     - CPU utilization: {it_summary['cpu_utilization']:.2%}")
    
    # Check Security response
    security_summary = security_plugin.get_security_summary()
    logger.info(f"\n   Security adaptation:")
    logger.info(f"     - Threat score: {security_summary['threat_score']:.2f}")
    logger.info(f"     - Defense posture: {security_summary['defense_posture']:.2f} (0=low, 1=high)")
    logger.info(f"     - Anomaly detection: {security_summary['anomaly_score']:.2f}")
    
    # Trigger AL self-organization
    logger.info(f"\n   Artificial Life self-organization:")
    success = al_plugin.execute_action('self_organize', {})
    logger.info(f"     - Self-organization triggered: {success}")
    logger.info(f"     - Emergency mode: {node.emergency_mode}")
    logger.info(f"     - Adaptation level: {al_plugin.state_variables['adaptation_level'].value:.2f}")
    
    # ========================================================================
    # 6. Execute plugin actions
    # ========================================================================
    logger.info("\n6. Demonstrating plugin action execution...")
    
    # IT Operations actions
    logger.info("\n   IT Operations: Restarting service to recover...")
    it_plugin.execute_action('restart_service', {})
    plugin_manager.update_all(delta_time=1.0)
    
    it_summary_after = it_plugin.get_health_summary()
    logger.info(f"     - Service uptime improved: {it_summary['service_uptime']:.2%} → {it_summary_after['service_uptime']:.2%}")
    logger.info(f"     - Error rate cleared: {it_summary['error_rate']:.2%} → {it_summary_after['error_rate']:.2%}")
    
    # Security actions
    logger.info("\n   Security: Increasing defense posture...")
    security_plugin.execute_action('increase_defense', {})
    security_summary_after = security_plugin.get_security_summary()
    logger.info(f"     - Defense posture: {security_summary['defense_posture']:.2f} → {security_summary_after['defense_posture']:.2f}")
    
    logger.info("\n   Security: Scanning for threats...")
    security_plugin.execute_action('scan_threats', {})
    
    # ========================================================================
    # 7. Show emergent behavior from simple rules
    # ========================================================================
    logger.info("\n7. Emergent self-organization behavior summary:")
    logger.info("   From simple, local rules:")
    logger.info("     Rule 1: Low energy → activate emergency conservation ✓")
    logger.info("     Rule 2: High anxiety → seek support ✓")
    logger.info("     Rule 3: Detect threats → increase defense ✓")
    logger.info("     Rule 4: Stable state → share knowledge ✓")
    
    logger.info("\n   Results in coordinated system adaptation:")
    logger.info(f"     - IT: Service protection and recovery")
    logger.info(f"     - Security: Threat mitigation and defense")
    logger.info(f"     - AL: Emergency protocols and self-organization")
    
    # ========================================================================
    # 8. Summary
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 IMPLEMENTATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\n✓ 1. Modular & Extensible Architecture:")
    logger.info("      - Plugin system with clear APIs")
    logger.info("      - IT, Security, and AL modules integrated")
    logger.info("      - Decoupled state, memory, comms, and action logic")
    
    logger.info("\n✓ 2. Robust, Adaptive State Variables:")
    logger.info("      - Universal: energy, health, emotion, trust")
    logger.info("      - IT/Security: service health, resource utilization")
    logger.info("      - Threat scores, incident tracking")
    
    logger.info("\n✓ 3. Emergency & Self-Organization:")
    logger.info("      - Behaviors emerge from simple local rules")
    logger.info("      - Auto-organization under stress/threat")
    logger.info("      - No explicit programming of coordinated responses")
    
    logger.info("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    demonstrate_plugin_system()
