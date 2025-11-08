# Adaptive Engineer

An advanced, adaptive agent system for multi-agent simulations with support for **IT Operations**, **Cybersecurity**, and **Artificial Life** domains.

## Overview

The **adaptiveengineer** module provides:

- **AliveLoopNode**: Socially-aware, energetically-limited agents with rich signal mechanisms
- **Plugin Architecture**: Modular system for IT, Security, AL, and Communication domains
- **Adaptive State Management**: Universal (energy, health, emotion, trust) and domain-specific state variables
- **Self-Organization**: Emergency behaviors emerging from simple, local rules
- **Secure Messaging**: Rich message types with idempotency, tracing, and replay support
- **Event Integrations**: Real-time streaming to SIEM, AIOps, and monitoring tools
- **Incident Memory**: Pattern recognition and learning from historical events
- **Evolutionary Learning**: Genetic algorithms for strategy optimization
- **Adaptive Learning**: Auto-tuning thresholds and learning normal behavior
- **Byzantine-Resilient Consensus**: Fault-tolerant distributed decision making

## Features

### Core Capabilities
- Complex internal states: energy, memory with privacy controls, emotional health (anxiety, joy, grief, curiosity, resilience)
- Trust network management and Byzantine-resilient consensus
- Proactive interventions for anxiety, grief, hope, and calm
- Energy attack detection and adaptive defense
- Social learning and cooperative behaviors

### Plugin System (Phase 1 & 2)
- **IT Operations Plugin**: Service health monitoring, resource utilization, incident tracking
- **Security Plugin**: Threat detection, anomaly scoring, defense posture management
- **Artificial Life Plugin**: Wraps existing AL behaviors in extensible plugin interface
- **Communication Plugin**: Secure messaging, event streaming, incident memory with pattern recognition

### Learning & Consensus (Phase 3)
- **Evolution Engine**: Genetic algorithms evolve detection, mitigation, and recovery strategies
- **Adaptive Learning**: Learn normal behavior patterns and auto-tune thresholds
- **Byzantine-Resilient Consensus**: Tolerate up to 33% malicious nodes in distributed voting
- **Trust Network**: Detect compromised nodes and discount unreliable input

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Example

```python
from adaptiveengineer import AliveLoopNode
from core.plugin_manager import PluginManager
from plugins.it_operations import ITOperationsPlugin
from plugins.security import SecurityPlugin
from plugins.communication import CommunicationPlugin

# Create node
node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)

# Setup plugins
manager = PluginManager()
manager.register_plugin(ITOperationsPlugin())
manager.register_plugin(SecurityPlugin())
manager.register_plugin(CommunicationPlugin())
manager.initialize_all(node)

# Update state
manager.update_all(delta_time=1.0)
```

### Run Demonstrations

**Phase 1: Core Foundations & Emergency Self-Organization**
```bash
python3 example_phase1.py
```

Demonstrates:
1. Plugin system architecture
2. Universal and domain-specific state variables
3. Emergency self-organization under stress
4. Coordinated adaptation across IT, Security, and AL domains

**Phase 2: Communication, Security, and Incident Memory**
```bash
python3 example_phase2.py
```

Demonstrates:
1. Secure, structured messaging with rich message types
2. Event-driven integrations (SIEM, AIOps, webhooks)
3. Incident memory with pattern recognition
4. Privacy controls and compliance features

**Phase 3: Learning, Evolution, Trust, and Consensus**
```bash
python3 example_phase3.py
```

Demonstrates:
1. Evolutionary learning with genetic algorithms
2. Adaptive learning and auto-tuning thresholds
3. Byzantine fault detection and resilience
4. Distributed consensus in adversarial environments

## Documentation

- [Phase 1 Implementation](PHASE1_IMPLEMENTATION.md) - Plugin system and self-organization
- [Phase 2 Implementation](PHASE2_IMPLEMENTATION.md) - Communication, security, and incident memory
- [Phase 3 Implementation](PHASE3_IMPLEMENTATION.md) - Learning, evolution, trust, and consensus
- [Roadmap](roadmap.md) - Full development roadmap with all planned phases

## Architecture

The system emphasizes:
- **Modularity**: Clean plugin interfaces with decoupled state, memory, comms, and actions
- **Robustness**: Fault-tolerant design with emergency protocols
- **Ethics**: Auditable decisions respecting privacy and human authority
- **Social Dynamics**: Trust networks, consensus mechanisms, emotional support

Suitable for research and engineering of adaptive, fault-tolerant, social AI collectives.
