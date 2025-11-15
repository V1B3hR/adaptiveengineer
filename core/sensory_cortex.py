"""
Digital Sensory Cortex - Human-like Sensing for Agent Collective

This module implements specialized sensor agents that mimic human senses:
- Sight: Pattern & topology analysis
- Hearing: Signal & broadcast monitoring
- Smell: Ambient & pheromone detection
- Taste: Data & packet inspection
- Touch: Direct probe & health checks

Each sensory modality provides unique perspectives on the Living Graph environment.
"""

import logging
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SenseType(str, Enum):
    """Types of sensory modalities."""
    SIGHT = "sight"          # Pattern & topology analysis
    HEARING = "hearing"      # Signal & broadcast monitoring
    SMELL = "smell"          # Ambient & pheromone detection
    TASTE = "taste"          # Data & packet inspection
    TOUCH = "touch"          # Direct probe & health checks


class AnomalyLevel(str, Enum):
    """Severity levels for detected anomalies."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SensoryInput:
    """
    A single sensory input detected by a sensor agent.
    
    Attributes:
        sense_type: Type of sensory input
        timestamp: When input was detected
        source_node: Node where input originated
        confidence: Confidence level (0.0-1.0)
        data: Sensor-specific data
        anomaly_level: Detected anomaly severity
        signature: Unique signature/hash of the input
    """
    sense_type: SenseType
    timestamp: float
    source_node: str
    confidence: float
    data: Dict[str, Any] = field(default_factory=dict)
    anomaly_level: AnomalyLevel = AnomalyLevel.NONE
    signature: Optional[str] = None
    
    def __post_init__(self):
        """Generate signature if not provided."""
        if self.signature is None:
            data_str = f"{self.sense_type}-{self.source_node}-{self.data}"
            self.signature = hashlib.md5(data_str.encode()).hexdigest()[:16]


class BaseSensorAgent(ABC):
    """
    Abstract base class for all sensor agents.
    
    Sensor agents are specialized to detect specific types of anomalies
    and environmental conditions in the Living Graph.
    """
    
    def __init__(self, sensor_id: str, sense_type: SenseType):
        """
        Initialize base sensor agent.
        
        Args:
            sensor_id: Unique identifier for this sensor
            sense_type: Type of sensory modality
        """
        self.sensor_id = sensor_id
        self.sense_type = sense_type
        self.inputs_detected: List[SensoryInput] = []
        self.recent_signatures: deque = deque(maxlen=100)
        self.detection_count = 0
        self.false_positive_count = 0
        
    @abstractmethod
    def sense(self, environment: Any, target_node: str) -> Optional[SensoryInput]:
        """
        Perform sensory detection on a target node.
        
        Args:
            environment: The Living Graph environment
            target_node: Node to sense
            
        Returns:
            SensoryInput if something detected, None otherwise
        """
        pass
    
    def record_detection(self, sensory_input: SensoryInput) -> None:
        """Record a sensory detection."""
        self.inputs_detected.append(sensory_input)
        self.recent_signatures.append(sensory_input.signature)
        self.detection_count += 1
        
    def has_seen_recently(self, signature: str) -> bool:
        """Check if signature was recently detected."""
        return signature in self.recent_signatures
    
    def get_recent_inputs(self, limit: int = 10) -> List[SensoryInput]:
        """Get most recent sensory inputs."""
        return self.inputs_detected[-limit:]


class SightAgent(BaseSensorAgent):
    """
    Pattern & Topology Agents - "Sight"
    
    These agents analyze graph topology and traffic flows.
    They "see" bottlenecks, unusual communication patterns between nodes,
    and structural anomalies.
    
    Question: "Does the system look right?"
    """
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SenseType.SIGHT)
        self.baseline_patterns: Dict[str, Dict[str, Any]] = {}
        
    def sense(self, environment: Any, target_node: str) -> Optional[SensoryInput]:
        """
        Analyze topology and traffic patterns around a node.
        
        Detects:
        - Unusual degree centrality
        - Traffic bottlenecks
        - Structural anomalies
        - Communication pattern changes
        """
        if not hasattr(environment, 'get_node') or not hasattr(environment, 'get_edges'):
            return None
            
        node = environment.get_node(target_node)
        if node is None:
            return None
            
        # Get edges connected to this node
        edges = environment.get_edges(target_node)
        
        # Analyze pattern
        degree = len(edges)
        avg_load = sum(e.get('attributes', {}).get('load', 0.0) for e in edges) / max(1, degree)
        
        # Check for anomalies
        anomaly_level = AnomalyLevel.NONE
        confidence = 0.5
        data = {
            'degree': degree,
            'avg_load': avg_load,
            'pattern_type': 'topology'
        }
        
        # Detect bottleneck
        if avg_load > 0.8:
            anomaly_level = AnomalyLevel.HIGH
            confidence = 0.85
            data['anomaly_type'] = 'bottleneck'
            
        # Detect unusual degree
        elif degree > 20:  # Hub node
            anomaly_level = AnomalyLevel.MEDIUM
            confidence = 0.70
            data['anomaly_type'] = 'high_degree'
            
        if anomaly_level != AnomalyLevel.NONE:
            return SensoryInput(
                sense_type=self.sense_type,
                timestamp=time.time(),
                source_node=target_node,
                confidence=confidence,
                data=data,
                anomaly_level=anomaly_level
            )
            
        return None


class HearingAgent(BaseSensorAgent):
    """
    Signal & Broadcast Agents - "Hearing"
    
    These agents are specialized listeners. They monitor the communication
    bus (Signals & Gossip) for specific distress calls, threat signatures
    being broadcast, or changes in the "mood" of the collective.
    
    Question: "What are the other agents saying?"
    """
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SenseType.HEARING)
        self.signal_history: deque = deque(maxlen=1000)
        
    def sense(self, environment: Any, target_node: str) -> Optional[SensoryInput]:
        """
        Monitor communication signals and broadcasts.
        
        Detects:
        - Distress calls
        - Threat signature broadcasts
        - Collective mood changes
        - Unusual signal patterns
        """
        # Listen for signals on the node
        if not hasattr(environment, 'get_node_signals'):
            return None
            
        signals = environment.get_node_signals(target_node)
        if not signals:
            return None
            
        # Analyze signals
        distress_count = sum(1 for s in signals if s.get('priority', 0) > 0.8)
        threat_count = sum(1 for s in signals if 'threat' in s.get('type', ''))
        
        anomaly_level = AnomalyLevel.NONE
        confidence = 0.6
        data = {
            'signal_count': len(signals),
            'distress_count': distress_count,
            'threat_count': threat_count,
            'pattern_type': 'communication'
        }
        
        # Detect distress pattern
        if distress_count > 3:
            anomaly_level = AnomalyLevel.CRITICAL
            confidence = 0.90
            data['anomaly_type'] = 'distress_pattern'
            
        # Detect threat broadcasts
        elif threat_count > 1:
            anomaly_level = AnomalyLevel.HIGH
            confidence = 0.85
            data['anomaly_type'] = 'threat_broadcast'
            
        if anomaly_level != AnomalyLevel.NONE:
            return SensoryInput(
                sense_type=self.sense_type,
                timestamp=time.time(),
                source_node=target_node,
                confidence=confidence,
                data=data,
                anomaly_level=anomaly_level
            )
            
        return None


class SmellAgent(BaseSensorAgent):
    """
    Ambient & Pheromone Agents - "Smell"
    
    These agents are the evolution of the pheromone system. They "smell"
    the ambient environment, detecting the faint, decaying traces of past
    events, malicious code signatures, or the subtle scent of a struggling process.
    
    Question: "What happened here recently?"
    """
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SenseType.SMELL)
        self.known_threat_signatures: Set[str] = set()
        
    def sense(self, environment: Any, target_node: str) -> Optional[SensoryInput]:
        """
        Detect ambient pheromones and historical traces.
        
        Detects:
        - Decaying threat pheromones
        - Malicious code signatures
        - Process struggle indicators
        - Historical event traces
        """
        if not hasattr(environment, 'get_node_pheromones'):
            return None
            
        pheromones = environment.get_node_pheromones(target_node)
        if not pheromones:
            return None
            
        # Analyze pheromone field
        threat_pheromones = [p for p in pheromones if p.get('type') == 'threat']
        strongest_threat = max(threat_pheromones, key=lambda p: p.get('strength', 0), default=None)
        
        anomaly_level = AnomalyLevel.NONE
        confidence = 0.7
        data = {
            'pheromone_count': len(pheromones),
            'threat_count': len(threat_pheromones),
            'pattern_type': 'ambient'
        }
        
        if strongest_threat:
            strength = strongest_threat.get('strength', 0)
            signature = strongest_threat.get('signature', 'unknown')
            
            if strength > 0.7:
                anomaly_level = AnomalyLevel.HIGH
                confidence = 0.85
            elif strength > 0.4:
                anomaly_level = AnomalyLevel.MEDIUM
                confidence = 0.75
                
            if anomaly_level != AnomalyLevel.NONE:
                data['anomaly_type'] = 'threat_trace'
                data['threat_signature'] = signature
                data['threat_strength'] = strength
                
                # Track known threats
                self.known_threat_signatures.add(signature)
                
                return SensoryInput(
                    sense_type=self.sense_type,
                    timestamp=time.time(),
                    source_node=target_node,
                    confidence=confidence,
                    data=data,
                    anomaly_level=anomaly_level
                )
                
        return None


class TasteAgent(BaseSensorAgent):
    """
    Data & Packet Inspection Agents - "Taste"
    
    These are highly specialized, resource-intensive agents that can perform
    deep analysis on the "substance" of the system. They can "taste" data
    payloads, sample log files, or analyze file hashes to find specific
    indicators of compromise.
    
    Question: "Is the content of this node toxic?"
    """
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SenseType.TASTE)
        self.known_malicious_hashes: Set[str] = set()
        self.inspection_cost = 0.5  # High resource cost
        
    def sense(self, environment: Any, target_node: str) -> Optional[SensoryInput]:
        """
        Perform deep inspection of node content.
        
        Detects:
        - Malicious data payloads
        - Suspicious log patterns
        - Known bad file hashes
        - Indicators of compromise
        """
        if not hasattr(environment, 'get_node'):
            return None
            
        node = environment.get_node(target_node)
        if node is None:
            return None
            
        # Get node attributes for inspection
        attributes = node.get('attributes', {})
        
        # Deep inspection (expensive)
        anomaly_level = AnomalyLevel.NONE
        confidence = 0.8
        data = {
            'inspection_type': 'deep',
            'pattern_type': 'content'
        }
        
        # Check for toxic indicators
        threat_score = attributes.get('threat_score', 0.0)
        malicious_processes = attributes.get('malicious_processes', 0)
        
        if threat_score > 0.9 or malicious_processes > 0:
            anomaly_level = AnomalyLevel.CRITICAL
            confidence = 0.95
            data['anomaly_type'] = 'toxic_content'
            data['threat_score'] = threat_score
            data['malicious_processes'] = malicious_processes
            
        elif threat_score > 0.7:
            anomaly_level = AnomalyLevel.HIGH
            confidence = 0.85
            data['anomaly_type'] = 'suspicious_content'
            data['threat_score'] = threat_score
            
        if anomaly_level != AnomalyLevel.NONE:
            return SensoryInput(
                sense_type=self.sense_type,
                timestamp=time.time(),
                source_node=target_node,
                confidence=confidence,
                data=data,
                anomaly_level=anomaly_level
            )
            
        return None


class TouchAgent(BaseSensorAgent):
    """
    Direct Probe & Health Agents - "Touch"
    
    These agents are the system's "nerve endings." They perform direct,
    active health checks on specific nodes and edges, sensing CPU load,
    memory usage, and latency in real-time.
    
    Question: "How does this node feel right now?"
    """
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SenseType.TOUCH)
        self.health_baselines: Dict[str, Dict[str, float]] = {}
        
    def sense(self, environment: Any, target_node: str) -> Optional[SensoryInput]:
        """
        Perform direct health check on a node.
        
        Detects:
        - High CPU load
        - Memory exhaustion
        - Network latency
        - Resource contention
        """
        if not hasattr(environment, 'get_node'):
            return None
            
        node = environment.get_node(target_node)
        if node is None:
            return None
            
        # Direct health probe
        attributes = node.get('attributes', {})
        cpu_load = attributes.get('cpu_load', 0.0)
        memory_usage = attributes.get('memory_usage', 0.0)
        health_status = attributes.get('health_status', 'OK')
        
        anomaly_level = AnomalyLevel.NONE
        confidence = 0.9  # High confidence - direct measurement
        data = {
            'cpu_load': cpu_load,
            'memory_usage': memory_usage,
            'health_status': health_status,
            'pattern_type': 'health'
        }
        
        # Detect critical health issues
        if health_status == 'FAILED':
            anomaly_level = AnomalyLevel.CRITICAL
            confidence = 1.0
            data['anomaly_type'] = 'node_failure'
            
        elif cpu_load > 0.9 or memory_usage > 0.9:
            anomaly_level = AnomalyLevel.HIGH
            confidence = 0.95
            data['anomaly_type'] = 'resource_exhaustion'
            
        elif cpu_load > 0.7 or memory_usage > 0.7:
            anomaly_level = AnomalyLevel.MEDIUM
            confidence = 0.85
            data['anomaly_type'] = 'resource_pressure'
            
        if anomaly_level != AnomalyLevel.NONE:
            return SensoryInput(
                sense_type=self.sense_type,
                timestamp=time.time(),
                source_node=target_node,
                confidence=confidence,
                data=data,
                anomaly_level=anomaly_level
            )
            
        return None


class SensoryCortex:
    """
    Digital Sensory Cortex - Coordinates all sensor agents.
    
    This class manages the deployment and coordination of specialized
    sensor agents across the Living Graph, providing a unified interface
    for multi-modal sensory input.
    """
    
    def __init__(self):
        """Initialize the sensory cortex."""
        self.sight_agents: List[SightAgent] = []
        self.hearing_agents: List[HearingAgent] = []
        self.smell_agents: List[SmellAgent] = []
        self.taste_agents: List[TasteAgent] = []
        self.touch_agents: List[TouchAgent] = []
        
        self.all_inputs: List[SensoryInput] = []
        self.anomaly_count_by_level: Dict[AnomalyLevel, int] = defaultdict(int)
        
    def deploy_sensors(self, 
                       sight_count: int = 2,
                       hearing_count: int = 2,
                       smell_count: int = 2,
                       taste_count: int = 1,
                       touch_count: int = 3) -> None:
        """
        Deploy sensor agents of each type.
        
        Args:
            sight_count: Number of sight agents to deploy
            hearing_count: Number of hearing agents
            smell_count: Number of smell agents
            taste_count: Number of taste agents (expensive)
            touch_count: Number of touch agents
        """
        # Deploy sight agents
        for i in range(sight_count):
            self.sight_agents.append(SightAgent(f"sight_{i}"))
            
        # Deploy hearing agents
        for i in range(hearing_count):
            self.hearing_agents.append(HearingAgent(f"hearing_{i}"))
            
        # Deploy smell agents
        for i in range(smell_count):
            self.smell_agents.append(SmellAgent(f"smell_{i}"))
            
        # Deploy taste agents (fewer due to cost)
        for i in range(taste_count):
            self.taste_agents.append(TasteAgent(f"taste_{i}"))
            
        # Deploy touch agents
        for i in range(touch_count):
            self.touch_agents.append(TouchAgent(f"touch_{i}"))
            
        logger.info(f"Deployed {self.get_total_sensor_count()} sensor agents")
        
    def sense_node(self, environment: Any, target_node: str) -> List[SensoryInput]:
        """
        Perform multi-modal sensing on a target node.
        
        Args:
            environment: The Living Graph environment
            target_node: Node to sense
            
        Returns:
            List of detected sensory inputs from all modalities
        """
        inputs = []
        
        # Sight sensing
        for agent in self.sight_agents:
            result = agent.sense(environment, target_node)
            if result:
                agent.record_detection(result)
                inputs.append(result)
                
        # Hearing sensing
        for agent in self.hearing_agents:
            result = agent.sense(environment, target_node)
            if result:
                agent.record_detection(result)
                inputs.append(result)
                
        # Smell sensing
        for agent in self.smell_agents:
            result = agent.sense(environment, target_node)
            if result:
                agent.record_detection(result)
                inputs.append(result)
                
        # Taste sensing (selective due to cost)
        if self.taste_agents and len(inputs) > 0:  # Only taste if other senses detected something
            for agent in self.taste_agents:
                result = agent.sense(environment, target_node)
                if result:
                    agent.record_detection(result)
                    inputs.append(result)
                    
        # Touch sensing
        for agent in self.touch_agents:
            result = agent.sense(environment, target_node)
            if result:
                agent.record_detection(result)
                inputs.append(result)
                
        # Record all inputs
        self.all_inputs.extend(inputs)
        for inp in inputs:
            self.anomaly_count_by_level[inp.anomaly_level] += 1
            
        return inputs
    
    def get_total_sensor_count(self) -> int:
        """Get total number of deployed sensors."""
        return (len(self.sight_agents) + len(self.hearing_agents) +
                len(self.smell_agents) + len(self.taste_agents) +
                len(self.touch_agents))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about sensory activity."""
        total_detections = sum(
            agent.detection_count 
            for agents in [self.sight_agents, self.hearing_agents, 
                          self.smell_agents, self.taste_agents, self.touch_agents]
            for agent in agents
        )
        
        return {
            'total_sensors': self.get_total_sensor_count(),
            'total_detections': total_detections,
            'sight_detections': sum(a.detection_count for a in self.sight_agents),
            'hearing_detections': sum(a.detection_count for a in self.hearing_agents),
            'smell_detections': sum(a.detection_count for a in self.smell_agents),
            'taste_detections': sum(a.detection_count for a in self.taste_agents),
            'touch_detections': sum(a.detection_count for a in self.touch_agents),
            'anomalies_by_level': dict(self.anomaly_count_by_level),
            'recent_inputs_count': len(self.all_inputs[-100:])
        }
