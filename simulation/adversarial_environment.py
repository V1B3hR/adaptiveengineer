"""
Adversarial Co-Evolution Environment - Red Team vs Blue Team

This module simulates an evolutionary arms race between attackers (red team)
and defenders (blue team). Both sides evolve strategies through competitive
pressure, leading to increasingly sophisticated attack and defense patterns.
"""

import logging
import time
import math
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from core.threat_patterns import ThreatPattern, ThreatLibrary

logger = logging.getLogger('adversarial_environment')


class AttackType(str, Enum):
    """Types of attacks in the adversarial environment"""
    ENERGY_DRAIN = "energy_drain"  # DDoS-like attacks
    COMMUNICATION_JAMMING = "communication_jamming"  # Signal interference
    TRUST_POISONING = "trust_poisoning"  # Compromised node behavior
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Computational overload
    COORDINATED_MULTI_VECTOR = "coordinated_multi_vector"  # Combined attacks


class AgentRole(str, Enum):
    """Agent roles in adversarial environment"""
    ATTACKER = "attacker"
    DEFENDER = "defender"


@dataclass
class AttackEvent:
    """Record of an attack attempt"""
    event_id: str
    attacker_id: int
    target_id: int
    attack_type: AttackType
    timestamp: float
    success: bool
    energy_drained: float
    detection_time: Optional[float] = None
    countermeasure_applied: Optional[str] = None
    

@dataclass
class DefenseEvent:
    """Record of a defensive action"""
    event_id: str
    defender_id: int
    threat_pattern_id: str
    timestamp: float
    detection_success: bool
    mitigation_success: bool
    response_time: float
    countermeasure: str


class AttackerAgent:
    """
    Attacker agent that evolves attack strategies based on success.
    
    Learns from failed attacks and adapts patterns to evade detection.
    Represents the adversarial intelligence in the red team.
    """
    
    def __init__(self, agent_id: int, initial_energy: float = 10.0):
        """
        Initialize attacker agent.
        
        Args:
            agent_id: Unique identifier
            initial_energy: Starting energy for attacks
        """
        self.agent_id = agent_id
        self.energy = initial_energy
        self.max_energy = initial_energy
        
        # Attack capabilities
        self.attack_strategies: Dict[AttackType, ThreatPattern] = {}
        self._initialize_attack_strategies()
        
        # Learning and evolution
        self.successful_attacks = 0
        self.failed_attacks = 0
        self.total_energy_drained = 0.0
        self.times_detected = 0
        self.evolution_score = 0.0
        
        # Attack history
        self.attack_history: deque = deque(maxlen=100)
        
        logger.info(f"AttackerAgent {agent_id} initialized with {len(self.attack_strategies)} strategies")
    
    def _initialize_attack_strategies(self) -> None:
        """Initialize basic attack strategies"""
        # Energy drain attack
        self.attack_strategies[AttackType.ENERGY_DRAIN] = ThreatPattern(
            signature=[0.8, 0.2, 0.9, 0.1, 0.7],  # High energy consumption pattern
            severity=0.6,
            mutation_rate=0.2,
            attack_type=AttackType.ENERGY_DRAIN.value
        )
        
        # Communication jamming
        self.attack_strategies[AttackType.COMMUNICATION_JAMMING] = ThreatPattern(
            signature=[0.3, 0.9, 0.2, 0.8, 0.4],  # High communication disruption
            severity=0.5,
            mutation_rate=0.15,
            attack_type=AttackType.COMMUNICATION_JAMMING.value
        )
        
        # Trust poisoning
        self.attack_strategies[AttackType.TRUST_POISONING] = ThreatPattern(
            signature=[0.5, 0.4, 0.6, 0.7, 0.8],  # Subtle trust manipulation
            severity=0.7,
            mutation_rate=0.1,
            attack_type=AttackType.TRUST_POISONING.value
        )
        
        # Resource exhaustion
        self.attack_strategies[AttackType.RESOURCE_EXHAUSTION] = ThreatPattern(
            signature=[0.9, 0.5, 0.8, 0.3, 0.9],  # High resource usage
            severity=0.6,
            mutation_rate=0.2,
            attack_type=AttackType.RESOURCE_EXHAUSTION.value
        )
    
    def select_attack_type(self) -> AttackType:
        """
        Select attack type based on past success.
        
        Prefers attacks that have been successful and evaded detection.
        Occasionally explores new attack types.
        
        Returns:
            Selected AttackType
        """
        # Exploration vs exploitation
        if random.random() < 0.2:  # 20% exploration
            return random.choice(list(AttackType))
        
        # Select based on success rate
        success_rates = {}
        for attack_type, pattern in self.attack_strategies.items():
            total = pattern.detection_count + 1  # Avoid division by zero
            success = pattern.failed_mitigations + 1
            success_rates[attack_type] = success / total
        
        # Weighted random selection
        total_weight = sum(success_rates.values())
        if total_weight == 0:
            return random.choice(list(self.attack_strategies.keys()))
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        for attack_type, weight in success_rates.items():
            cumulative += weight
            if r <= cumulative:
                return attack_type
        
        return list(self.attack_strategies.keys())[0]
    
    def execute_attack(
        self,
        target_id: int,
        attack_type: Optional[AttackType] = None
    ) -> Tuple[ThreatPattern, AttackEvent]:
        """
        Execute an attack against a target.
        
        Args:
            target_id: ID of target agent
            attack_type: Optional specific attack type
            
        Returns:
            Tuple of (ThreatPattern used, AttackEvent record)
        """
        if attack_type is None:
            attack_type = self.select_attack_type()
        
        if attack_type not in self.attack_strategies:
            attack_type = list(self.attack_strategies.keys())[0]
        
        pattern = self.attack_strategies[attack_type]
        
        # Calculate attack effectiveness based on pattern severity
        energy_cost = 0.5 * pattern.severity
        if self.energy < energy_cost:
            # Not enough energy to attack
            event = AttackEvent(
                event_id=str(uuid.uuid4()),
                attacker_id=self.agent_id,
                target_id=target_id,
                attack_type=attack_type,
                timestamp=time.time(),
                success=False,
                energy_drained=0.0
            )
            return pattern, event
        
        # Consume energy
        self.energy -= energy_cost
        
        # Calculate damage (energy drained from target)
        base_damage = pattern.severity * 2.0
        variance = random.gauss(1.0, 0.2)
        energy_drained = max(0.0, base_damage * variance)
        
        # Determine success (higher severity = higher success chance)
        success_chance = 0.5 + 0.3 * pattern.severity
        success = random.random() < success_chance
        
        if success:
            self.successful_attacks += 1
            self.total_energy_drained += energy_drained
        else:
            self.failed_attacks += 1
        
        event = AttackEvent(
            event_id=str(uuid.uuid4()),
            attacker_id=self.agent_id,
            target_id=target_id,
            attack_type=attack_type,
            timestamp=time.time(),
            success=success,
            energy_drained=energy_drained if success else 0.0
        )
        
        self.attack_history.append(event)
        
        return pattern, event
    
    def learn_from_detection(self, attack_type: AttackType, detected: bool) -> None:
        """
        Learn from attack outcome - adapt strategies if detected.
        
        Args:
            attack_type: Type of attack that was attempted
            detected: Whether the attack was detected
        """
        if detected:
            self.times_detected += 1
            
            # Increase mutation rate to evade detection
            if attack_type in self.attack_strategies:
                pattern = self.attack_strategies[attack_type]
                pattern.mutation_rate = min(0.5, pattern.mutation_rate * 1.2)
    
    def evolve_strategies(self) -> None:
        """
        Evolve attack strategies based on accumulated experience.
        
        Mutates patterns that have been detected to evade defenses.
        """
        for attack_type, pattern in self.attack_strategies.items():
            # Evolve if detection rate is high
            detection_rate = (
                self.times_detected / max(1, self.successful_attacks + self.failed_attacks)
            )
            
            if detection_rate > 0.5:
                # Mutate to evade detection
                mutated = pattern.mutate()
                self.attack_strategies[attack_type] = mutated
                logger.info(f"AttackerAgent {self.agent_id} evolved {attack_type.value} strategy")
    
    def recharge_energy(self, amount: float) -> None:
        """Recharge attacker energy"""
        self.energy = min(self.max_energy, self.energy + amount)
    
    def get_fitness(self) -> float:
        """
        Calculate attacker fitness for evolutionary selection.
        
        Fitness based on:
        - Energy drained from targets
        - Success rate
        - Detection evasion
        
        Returns:
            Fitness score
        """
        if self.successful_attacks + self.failed_attacks == 0:
            return 0.0
        
        success_rate = self.successful_attacks / (self.successful_attacks + self.failed_attacks)
        detection_rate = self.times_detected / (self.successful_attacks + self.failed_attacks)
        evasion_score = 1.0 - detection_rate
        
        fitness = (
            0.4 * self.total_energy_drained +
            0.3 * success_rate +
            0.3 * evasion_score
        )
        
        self.evolution_score = fitness
        return fitness


class AdversarialEnvironment:
    """
    Manages adversarial co-evolution between attackers and defenders.
    
    Simulates the arms race where both sides evolve strategies through
    competitive pressure. Tracks fitness metrics and evolutionary progress.
    """
    
    def __init__(
        self,
        num_attackers: int = 5,
        num_defenders: int = 10,
        evolution_interval: int = 50
    ):
        """
        Initialize adversarial environment.
        
        Args:
            num_attackers: Number of attacker agents
            num_defenders: Number of defender agents (AliveLoopNodes)
            evolution_interval: Steps between evolution phases
        """
        self.num_attackers = num_attackers
        self.num_defenders = num_defenders
        self.evolution_interval = evolution_interval
        
        # Agent collections
        self.attackers: Dict[int, AttackerAgent] = {}
        self.defender_ids: Set[int] = set()  # Defender nodes managed externally
        
        # Initialize attackers
        for i in range(num_attackers):
            self.attackers[i] = AttackerAgent(i)
        
        # Threat intelligence
        self.threat_library = ThreatLibrary()
        
        # Event tracking
        self.attack_events: deque = deque(maxlen=1000)
        self.defense_events: deque = deque(maxlen=1000)
        
        # Evolution tracking
        self.current_generation = 0
        self.steps = 0
        
        # Metrics
        self.total_attacks = 0
        self.total_successful_attacks = 0
        self.total_detections = 0
        self.total_mitigations = 0
        
        logger.info(f"AdversarialEnvironment initialized: "
                   f"{num_attackers} attackers, {num_defenders} defenders")
    
    def register_defender(self, defender_id: int) -> None:
        """Register a defender node with the environment"""
        self.defender_ids.add(defender_id)
    
    def simulate_attack_wave(
        self,
        defender_nodes: Dict[int, Any]
    ) -> List[Tuple[ThreatPattern, AttackEvent]]:
        """
        Simulate a wave of attacks against defender nodes.
        
        Args:
            defender_nodes: Dictionary of defender AliveLoopNode objects
            
        Returns:
            List of (ThreatPattern, AttackEvent) tuples
        """
        attacks = []
        
        for attacker in self.attackers.values():
            if attacker.energy < 0.3:
                # Recharge if energy too low
                attacker.recharge_energy(2.0)
                continue
            
            # Select random target
            if not defender_nodes:
                continue
            
            target_id = random.choice(list(defender_nodes.keys()))
            
            # Execute attack
            pattern, event = attacker.execute_attack(target_id)
            attacks.append((pattern, event))
            
            # Register pattern in threat library
            self.threat_library.add_pattern(pattern)
            
            # Track event
            self.attack_events.append(event)
            self.total_attacks += 1
            if event.success:
                self.total_successful_attacks += 1
        
        self.steps += 1
        
        # Check for evolution
        if self.steps % self.evolution_interval == 0:
            self.evolve_population()
        
        return attacks
    
    def record_detection(
        self,
        attacker_id: int,
        attack_type: AttackType,
        defender_id: int,
        detection_time: float
    ) -> None:
        """
        Record that an attack was detected by a defender.
        
        Args:
            attacker_id: ID of attacker
            attack_type: Type of attack detected
            defender_id: ID of defender who detected
            detection_time: Time taken to detect
        """
        self.total_detections += 1
        
        # Inform attacker of detection
        if attacker_id in self.attackers:
            self.attackers[attacker_id].learn_from_detection(attack_type, True)
    
    def record_mitigation(
        self,
        defender_id: int,
        pattern_id: str,
        success: bool,
        countermeasure: str,
        response_time: float
    ) -> None:
        """
        Record a defensive mitigation attempt.
        
        Args:
            defender_id: ID of defender
            pattern_id: ID of threat pattern
            success: Whether mitigation was successful
            countermeasure: Description of countermeasure used
            response_time: Time taken to respond
        """
        event = DefenseEvent(
            event_id=str(uuid.uuid4()),
            defender_id=defender_id,
            threat_pattern_id=pattern_id,
            timestamp=time.time(),
            detection_success=True,
            mitigation_success=success,
            response_time=response_time,
            countermeasure=countermeasure
        )
        
        self.defense_events.append(event)
        
        if success:
            self.total_mitigations += 1
    
    def evolve_population(self) -> None:
        """
        Evolve both attacker and defender populations.
        
        Attackers with high fitness reproduce and mutate strategies.
        This simulates the evolutionary arms race.
        """
        self.current_generation += 1
        
        # Evolve attacker strategies
        for attacker in self.attackers.values():
            attacker.evolve_strategies()
        
        # Evolve threat patterns in library
        self.threat_library.evolve_patterns(count=3)
        
        logger.info(f"Evolution completed - Generation {self.current_generation}")
    
    def get_attacker_fitness_scores(self) -> Dict[int, float]:
        """Get fitness scores for all attackers"""
        return {
            aid: attacker.get_fitness()
            for aid, attacker in self.attackers.items()
        }
    
    def get_defender_metrics(
        self,
        defender_nodes: Dict[int, Any]
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate fitness metrics for defenders.
        
        Args:
            defender_nodes: Dictionary of defender AliveLoopNode objects
            
        Returns:
            Dictionary of defender_id -> metrics
        """
        metrics = {}
        
        for defender_id, node in defender_nodes.items():
            # Calculate survival time (based on energy)
            survival = node.energy / node.max_energy if hasattr(node, 'max_energy') else 1.0
            
            # Count detections by this defender
            detections = sum(
                1 for event in self.defense_events
                if event.defender_id == defender_id and event.detection_success
            )
            
            # Count successful mitigations
            mitigations = sum(
                1 for event in self.defense_events
                if event.defender_id == defender_id and event.mitigation_success
            )
            
            # Calculate detection accuracy
            total_defense_events = sum(
                1 for event in self.defense_events
                if event.defender_id == defender_id
            )
            detection_accuracy = (
                detections / total_defense_events
                if total_defense_events > 0 else 0.0
            )
            
            metrics[defender_id] = {
                'survival': survival,
                'detections': detections,
                'mitigations': mitigations,
                'detection_accuracy': detection_accuracy,
                'energy': node.energy
            }
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive environment statistics"""
        attacker_stats = {
            'total_attackers': len(self.attackers),
            'avg_fitness': sum(a.get_fitness() for a in self.attackers.values()) / len(self.attackers)
                          if self.attackers else 0.0,
            'total_energy_drained': sum(a.total_energy_drained for a in self.attackers.values()),
            'avg_success_rate': (
                sum(
                    a.successful_attacks / max(1, a.successful_attacks + a.failed_attacks)
                    for a in self.attackers.values()
                ) / len(self.attackers)
                if self.attackers else 0.0
            )
        }
        
        threat_stats = self.threat_library.get_statistics()
        
        return {
            'generation': self.current_generation,
            'steps': self.steps,
            'total_attacks': self.total_attacks,
            'successful_attacks': self.total_successful_attacks,
            'attack_success_rate': (
                self.total_successful_attacks / self.total_attacks
                if self.total_attacks > 0 else 0.0
            ),
            'total_detections': self.total_detections,
            'total_mitigations': self.total_mitigations,
            'detection_rate': (
                self.total_detections / self.total_attacks
                if self.total_attacks > 0 else 0.0
            ),
            'mitigation_rate': (
                self.total_mitigations / self.total_detections
                if self.total_detections > 0 else 0.0
            ),
            'attacker_stats': attacker_stats,
            'threat_library': threat_stats
        }
    
    def visualize_arms_race(self) -> str:
        """Generate text visualization of the evolutionary arms race"""
        stats = self.get_statistics()
        
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"ADVERSARIAL ARMS RACE - Generation {self.current_generation}")
        lines.append(f"{'='*70}")
        
        lines.append(f"\n🎯 Attack Statistics:")
        lines.append(f"  Total Attacks: {stats['total_attacks']}")
        lines.append(f"  Success Rate: {stats['attack_success_rate']*100:.1f}%")
        lines.append(f"  Energy Drained: {stats['attacker_stats']['total_energy_drained']:.2f}")
        
        lines.append(f"\n🛡️  Defense Statistics:")
        lines.append(f"  Detections: {stats['total_detections']}")
        lines.append(f"  Detection Rate: {stats['detection_rate']*100:.1f}%")
        lines.append(f"  Mitigations: {stats['total_mitigations']}")
        lines.append(f"  Mitigation Rate: {stats['mitigation_rate']*100:.1f}%")
        
        lines.append(f"\n🧬 Evolution:")
        lines.append(f"  Threat Patterns: {stats['threat_library']['total_patterns']}")
        lines.append(f"  Evolution Generations: {stats['threat_library']['evolution_generations']}")
        lines.append(f"  Avg Attacker Fitness: {stats['attacker_stats']['avg_fitness']:.3f}")
        
        lines.append(f"\n{'='*70}\n")
        
        return '\n'.join(lines)
