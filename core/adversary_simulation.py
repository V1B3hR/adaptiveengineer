"""
Adversary Simulation - Evolving threat models

This module simulates threats (malware, attackers) as evolving, adaptive entities:
- Adaptive adversary behaviors
- Evolving attack strategies
- Learning from defense responses
- Realistic threat progression
"""

import logging
import time
import random
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger('adversary_simulation')


class AdversaryType(str, Enum):
    """Types of adversaries"""
    MALWARE = "malware"
    APT = "apt"  # Advanced Persistent Threat
    SCRIPT_KIDDIE = "script_kiddie"
    INSIDER = "insider"
    BOTNET = "botnet"
    RANSOMWARE = "ransomware"


class AttackVector(str, Enum):
    """Attack vectors"""
    PHISHING = "phishing"
    EXPLOIT = "exploit"
    BRUTE_FORCE = "brute_force"
    SOCIAL_ENGINEERING = "social_engineering"
    ZERO_DAY = "zero_day"
    SUPPLY_CHAIN = "supply_chain"
    CREDENTIAL_THEFT = "credential_theft"


class AttackPhase(str, Enum):
    """Attack kill chain phases"""
    RECONNAISSANCE = "reconnaissance"
    WEAPONIZATION = "weaponization"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    INSTALLATION = "installation"
    COMMAND_CONTROL = "command_control"
    ACTIONS_OBJECTIVES = "actions_objectives"


@dataclass
class AttackCapability:
    """Individual attack capability"""
    name: str
    vector: AttackVector
    success_rate: float  # 0.0 to 1.0
    detection_probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    sophistication: float  # 0.0 to 1.0


@dataclass
class AttackAttempt:
    """Record of an attack attempt"""
    attempt_id: str
    timestamp: float
    adversary_id: str
    capability: AttackCapability
    phase: AttackPhase
    target: str
    success: bool
    detected: bool
    blocked: bool


@dataclass
class AdversaryGenome:
    """Genetic representation of adversary strategy"""
    capabilities: List[AttackCapability]
    aggression: float  # 0.0 to 1.0 (attack frequency)
    stealth: float  # 0.0 to 1.0 (evasion capability)
    persistence: float  # 0.0 to 1.0 (resilience to defense)
    adaptability: float  # 0.0 to 1.0 (learning rate)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'AdversaryGenome':
        """Mutate adversary genome to evolve"""
        import copy
        mutated = copy.deepcopy(self)
        
        # Mutate attributes
        if random.random() < mutation_rate:
            mutated.aggression = max(0.0, min(1.0, 
                mutated.aggression + random.gauss(0, 0.1)))
        
        if random.random() < mutation_rate:
            mutated.stealth = max(0.0, min(1.0, 
                mutated.stealth + random.gauss(0, 0.1)))
        
        if random.random() < mutation_rate:
            mutated.persistence = max(0.0, min(1.0, 
                mutated.persistence + random.gauss(0, 0.1)))
        
        if random.random() < mutation_rate:
            mutated.adaptability = max(0.0, min(1.0, 
                mutated.adaptability + random.gauss(0, 0.1)))
        
        # Mutate capabilities
        for capability in mutated.capabilities:
            if random.random() < mutation_rate:
                capability.success_rate = max(0.0, min(1.0,
                    capability.success_rate + random.gauss(0, 0.05)))
            
            if random.random() < mutation_rate:
                capability.detection_probability = max(0.0, min(1.0,
                    capability.detection_probability + random.gauss(0, 0.05)))
        
        return mutated


class EvolvingAdversary:
    """
    Adaptive adversary that evolves based on defense responses.
    
    Like biological evolution:
    - Learns from failures
    - Adapts tactics to evade defenses
    - Evolves more sophisticated capabilities
    """
    
    def __init__(
        self,
        adversary_id: str,
        adversary_type: AdversaryType,
        initial_sophistication: float = 0.5
    ):
        """
        Initialize evolving adversary.
        
        Args:
            adversary_id: Unique identifier
            adversary_type: Type of adversary
            initial_sophistication: Initial skill level (0.0-1.0)
        """
        self.adversary_id = adversary_id
        self.adversary_type = adversary_type
        self.sophistication = initial_sophistication
        
        # Initialize genome
        self.genome = self._initialize_genome(initial_sophistication)
        
        # State tracking
        self.current_phase = AttackPhase.RECONNAISSANCE
        self.active = True
        self.attacks_attempted = 0
        self.attacks_successful = 0
        self.times_detected = 0
        self.times_blocked = 0
        
        # Learning history
        self.attack_history: deque = deque(maxlen=100)
        self.learned_defenses: List[str] = []
        
        # Evolution tracking
        self.generation = 1
        self.fitness = 0.0
        
        logger.info(f"Initialized {adversary_type} adversary {adversary_id} "
                   f"(sophistication={initial_sophistication:.2f})")
    
    def _initialize_genome(self, sophistication: float) -> AdversaryGenome:
        """Initialize adversary genome based on type and sophistication"""
        
        # Base capabilities vary by adversary type
        if self.adversary_type == AdversaryType.APT:
            capabilities = [
                AttackCapability("spear_phishing", AttackVector.PHISHING, 0.7, 0.3, 0.8, 0.9),
                AttackCapability("zero_day_exploit", AttackVector.ZERO_DAY, 0.6, 0.2, 0.95, 0.95),
                AttackCapability("supply_chain_compromise", AttackVector.SUPPLY_CHAIN, 0.5, 0.1, 0.9, 0.9),
            ]
            aggression = 0.3  # APTs are patient
            stealth = 0.9  # Very stealthy
            persistence = 0.95  # Highly persistent
            adaptability = 0.8  # Highly adaptive
        
        elif self.adversary_type == AdversaryType.RANSOMWARE:
            capabilities = [
                AttackCapability("phishing_campaign", AttackVector.PHISHING, 0.6, 0.5, 0.7, 0.6),
                AttackCapability("exploit_kit", AttackVector.EXPLOIT, 0.7, 0.6, 0.8, 0.7),
                AttackCapability("credential_theft", AttackVector.CREDENTIAL_THEFT, 0.5, 0.4, 0.6, 0.5),
            ]
            aggression = 0.8  # Aggressive
            stealth = 0.4  # Less stealthy
            persistence = 0.6  # Moderate persistence
            adaptability = 0.5  # Moderate adaptability
        
        elif self.adversary_type == AdversaryType.SCRIPT_KIDDIE:
            capabilities = [
                AttackCapability("brute_force", AttackVector.BRUTE_FORCE, 0.4, 0.8, 0.3, 0.2),
                AttackCapability("known_exploit", AttackVector.EXPLOIT, 0.5, 0.7, 0.4, 0.3),
            ]
            aggression = 0.7  # Aggressive but unskilled
            stealth = 0.2  # Not stealthy
            persistence = 0.3  # Low persistence
            adaptability = 0.3  # Low adaptability
        
        elif self.adversary_type == AdversaryType.INSIDER:
            capabilities = [
                AttackCapability("privilege_abuse", AttackVector.CREDENTIAL_THEFT, 0.8, 0.4, 0.7, 0.6),
                AttackCapability("data_exfiltration", AttackVector.SOCIAL_ENGINEERING, 0.7, 0.5, 0.8, 0.5),
            ]
            aggression = 0.4  # Careful
            stealth = 0.7  # Stealthy
            persistence = 0.5  # Moderate persistence
            adaptability = 0.6  # Moderate adaptability
        
        else:  # MALWARE, BOTNET
            capabilities = [
                AttackCapability("automated_scan", AttackVector.EXPLOIT, 0.6, 0.6, 0.5, 0.4),
                AttackCapability("mass_attack", AttackVector.BRUTE_FORCE, 0.5, 0.7, 0.4, 0.3),
            ]
            aggression = 0.9  # Very aggressive
            stealth = 0.3  # Not very stealthy
            persistence = 0.4  # Low persistence
            adaptability = 0.4  # Low adaptability
        
        # Scale by sophistication
        for cap in capabilities:
            cap.success_rate *= sophistication
            cap.detection_probability *= (1.0 - sophistication * 0.5)
            cap.sophistication = sophistication
        
        return AdversaryGenome(
            capabilities=capabilities,
            aggression=aggression,
            stealth=stealth,
            persistence=persistence,
            adaptability=adaptability
        )
    
    def attempt_attack(
        self,
        target: str,
        defense_level: float = 0.5
    ) -> AttackAttempt:
        """
        Attempt an attack against target.
        
        Args:
            target: Target of attack
            defense_level: Current defense effectiveness (0.0-1.0)
        
        Returns:
            AttackAttempt record
        """
        # Select capability based on current phase and learned defenses
        capability = self._select_capability(defense_level)
        
        attempt_id = f"attack_{self.adversary_id}_{int(time.time()*1000)}"
        
        # Calculate success probability
        # Success depends on capability vs defense and stealth
        base_success = capability.success_rate
        defense_factor = (1.0 - defense_level) * 0.5
        stealth_bonus = self.genome.stealth * 0.2
        
        success_prob = base_success * (1.0 + defense_factor + stealth_bonus)
        success_prob = max(0.0, min(1.0, success_prob))
        
        # Calculate detection probability
        # Detection depends on stealth and defense level
        base_detection = capability.detection_probability
        defense_bonus = defense_level * 0.3
        stealth_reduction = self.genome.stealth * 0.4
        
        detection_prob = base_detection * (1.0 + defense_bonus - stealth_reduction)
        detection_prob = max(0.0, min(1.0, detection_prob))
        
        # Execute attack
        success = random.random() < success_prob
        detected = random.random() < detection_prob
        blocked = detected and (random.random() < defense_level)
        
        # If blocked, attack fails
        if blocked:
            success = False
        
        attempt = AttackAttempt(
            attempt_id=attempt_id,
            timestamp=time.time(),
            adversary_id=self.adversary_id,
            capability=capability,
            phase=self.current_phase,
            target=target,
            success=success,
            detected=detected,
            blocked=blocked
        )
        
        # Update metrics
        self.attacks_attempted += 1
        if success:
            self.attacks_successful += 1
        if detected:
            self.times_detected += 1
        if blocked:
            self.times_blocked += 1
        
        # Record in history
        self.attack_history.append(attempt)
        
        # Learn from attempt
        self._learn_from_attempt(attempt, defense_level)
        
        # Progress through attack phases if successful
        if success:
            self._progress_attack_phase()
        
        logger.info(f"Adversary {self.adversary_id} attacked {target}: "
                   f"success={success}, detected={detected}, blocked={blocked}")
        
        return attempt
    
    def _select_capability(self, defense_level: float) -> AttackCapability:
        """
        Select attack capability based on learned defenses.
        
        Adaptively choose capabilities that have worked before.
        """
        # If highly adaptive, prefer capabilities that weren't blocked recently
        if self.genome.adaptability > 0.6:
            recent_blocked = {
                a.capability.name for a in list(self.attack_history)[-10:]
                if a.blocked
            }
            
            # Filter out recently blocked capabilities
            available = [
                c for c in self.genome.capabilities
                if c.name not in recent_blocked
            ]
            
            if available:
                # Choose based on success rate
                weights = [c.success_rate for c in available]
                return random.choices(available, weights=weights)[0]
        
        # Default: choose randomly weighted by success rate
        weights = [c.success_rate for c in self.genome.capabilities]
        return random.choices(self.genome.capabilities, weights=weights)[0]
    
    def _learn_from_attempt(self, attempt: AttackAttempt, defense_level: float):
        """
        Learn from attack attempt to improve future attacks.
        
        Evolves tactics based on what works and what doesn't.
        """
        learning_rate = self.genome.adaptability * 0.1
        
        if attempt.success:
            # Reinforce successful capability
            attempt.capability.success_rate = min(1.0,
                attempt.capability.success_rate + learning_rate)
        
        if attempt.blocked:
            # Reduce reliance on blocked capability
            attempt.capability.success_rate = max(0.1,
                attempt.capability.success_rate - learning_rate)
            
            # Learn defense signature
            defense_sig = f"{attempt.capability.vector}:{defense_level:.1f}"
            if defense_sig not in self.learned_defenses:
                self.learned_defenses.append(defense_sig)
                logger.debug(f"Adversary {self.adversary_id} learned defense: {defense_sig}")
        
        if attempt.detected and not attempt.blocked:
            # Improve stealth if detected but not blocked
            self.genome.stealth = min(1.0,
                self.genome.stealth + learning_rate * 0.5)
    
    def _progress_attack_phase(self):
        """Progress through attack kill chain"""
        phase_order = [
            AttackPhase.RECONNAISSANCE,
            AttackPhase.WEAPONIZATION,
            AttackPhase.DELIVERY,
            AttackPhase.EXPLOITATION,
            AttackPhase.INSTALLATION,
            AttackPhase.COMMAND_CONTROL,
            AttackPhase.ACTIONS_OBJECTIVES
        ]
        
        current_idx = phase_order.index(self.current_phase)
        if current_idx < len(phase_order) - 1:
            self.current_phase = phase_order[current_idx + 1]
            logger.info(f"Adversary {self.adversary_id} progressed to {self.current_phase}")
    
    def evolve(self, mutation_rate: float = 0.1) -> 'EvolvingAdversary':
        """
        Evolve adversary to create more sophisticated variant.
        
        Args:
            mutation_rate: Rate of mutation (0.0-1.0)
        
        Returns:
            New evolved adversary
        """
        # Create offspring with mutated genome
        new_id = f"{self.adversary_id}_gen{self.generation + 1}"
        
        offspring = EvolvingAdversary(
            adversary_id=new_id,
            adversary_type=self.adversary_type,
            initial_sophistication=self.sophistication
        )
        
        # Inherit and mutate genome
        offspring.genome = self.genome.mutate(mutation_rate)
        offspring.generation = self.generation + 1
        
        # Increase sophistication slightly
        offspring.sophistication = min(1.0, self.sophistication + 0.05)
        
        # Inherit learned defenses
        offspring.learned_defenses = self.learned_defenses.copy()
        
        logger.info(f"Adversary {self.adversary_id} evolved to {new_id} "
                   f"(gen={offspring.generation}, soph={offspring.sophistication:.2f})")
        
        return offspring
    
    def calculate_fitness(self) -> float:
        """
        Calculate fitness score for evolution.
        
        Fitness based on:
        - Attack success rate
        - Evasion (avoiding detection)
        - Persistence (not being blocked)
        """
        if self.attacks_attempted == 0:
            return 0.0
        
        success_rate = self.attacks_successful / self.attacks_attempted
        evasion_rate = 1.0 - (self.times_detected / self.attacks_attempted)
        persistence_rate = 1.0 - (self.times_blocked / self.attacks_attempted)
        
        # Weighted fitness
        fitness = (
            success_rate * 0.4 +
            evasion_rate * 0.3 +
            persistence_rate * 0.3
        )
        
        self.fitness = fitness
        return fitness
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adversary metrics"""
        success_rate = (
            self.attacks_successful / self.attacks_attempted
            if self.attacks_attempted > 0 else 0.0
        )
        
        detection_rate = (
            self.times_detected / self.attacks_attempted
            if self.attacks_attempted > 0 else 0.0
        )
        
        return {
            'adversary_id': self.adversary_id,
            'type': self.adversary_type,
            'sophistication': self.sophistication,
            'generation': self.generation,
            'fitness': self.fitness,
            'current_phase': self.current_phase,
            'active': self.active,
            'attacks_attempted': self.attacks_attempted,
            'attacks_successful': self.attacks_successful,
            'success_rate': success_rate,
            'times_detected': self.times_detected,
            'times_blocked': self.times_blocked,
            'detection_rate': detection_rate,
            'learned_defenses': len(self.learned_defenses),
            'genome': {
                'aggression': self.genome.aggression,
                'stealth': self.genome.stealth,
                'persistence': self.genome.persistence,
                'adaptability': self.genome.adaptability,
                'capabilities': len(self.genome.capabilities)
            }
        }


class AdversarySimulation:
    """
    Simulation environment for evolving adversaries.
    
    Manages population of adversaries that evolve over time.
    """
    
    def __init__(self, population_size: int = 5):
        """
        Initialize adversary simulation.
        
        Args:
            population_size: Number of adversaries to simulate
        """
        self.population_size = population_size
        self.adversaries: Dict[str, EvolvingAdversary] = {}
        self.generation = 1
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"Adversary simulation initialized with {population_size} adversaries")
    
    def _initialize_population(self):
        """Initialize diverse population of adversaries"""
        adversary_types = list(AdversaryType)
        
        for i in range(self.population_size):
            adv_type = adversary_types[i % len(adversary_types)]
            sophistication = random.uniform(0.3, 0.7)
            
            adversary_id = f"adv_{i}"
            adversary = EvolvingAdversary(adversary_id, adv_type, sophistication)
            self.adversaries[adversary_id] = adversary
    
    def simulate_attacks(
        self,
        targets: List[str],
        defense_level: float,
        rounds: int = 10
    ) -> List[AttackAttempt]:
        """
        Simulate multiple rounds of attacks.
        
        Args:
            targets: List of target systems
            defense_level: Defense effectiveness (0.0-1.0)
            rounds: Number of attack rounds
        
        Returns:
            List of all attack attempts
        """
        all_attempts = []
        
        for round_num in range(rounds):
            for adversary in list(self.adversaries.values()):
                if not adversary.active:
                    continue
                
                # Each adversary attacks based on aggression level
                if random.random() < adversary.genome.aggression:
                    target = random.choice(targets)
                    attempt = adversary.attempt_attack(target, defense_level)
                    all_attempts.append(attempt)
        
        return all_attempts
    
    def evolve_population(self, selection_rate: float = 0.5):
        """
        Evolve adversary population based on fitness.
        
        Args:
            selection_rate: Fraction of population to select for reproduction
        """
        # Calculate fitness for all adversaries
        for adversary in self.adversaries.values():
            adversary.calculate_fitness()
        
        # Sort by fitness
        sorted_adversaries = sorted(
            self.adversaries.values(),
            key=lambda a: a.fitness,
            reverse=True
        )
        
        # Select top performers
        selection_count = max(1, int(len(sorted_adversaries) * selection_rate))
        selected = sorted_adversaries[:selection_count]
        
        # Create new population through evolution
        new_population = {}
        
        # Keep top performers (elitism)
        for adversary in selected:
            new_population[adversary.adversary_id] = adversary
        
        # Fill rest with evolved offspring
        offspring_counter = 0
        while len(new_population) < self.population_size:
            parent = random.choice(selected)
            offspring = parent.evolve(mutation_rate=0.1)
            # Ensure unique ID
            unique_id = f"{offspring.adversary_id}_{offspring_counter}"
            offspring.adversary_id = unique_id
            offspring_counter += 1
            new_population[unique_id] = offspring
        
        self.adversaries = new_population
        self.generation += 1
        
        logger.info(f"Evolved adversary population to generation {self.generation}")
    
    def get_simulation_metrics(self) -> Dict[str, Any]:
        """Get overall simulation metrics"""
        total_attacks = sum(a.attacks_attempted for a in self.adversaries.values())
        total_successful = sum(a.attacks_successful for a in self.adversaries.values())
        
        avg_sophistication = sum(a.sophistication for a in self.adversaries.values()) / len(self.adversaries)
        avg_fitness = sum(a.fitness for a in self.adversaries.values()) / len(self.adversaries)
        
        return {
            'generation': self.generation,
            'population_size': len(self.adversaries),
            'total_attacks': total_attacks,
            'total_successful': total_successful,
            'avg_sophistication': avg_sophistication,
            'avg_fitness': avg_fitness,
            'adversaries': [a.get_metrics() for a in self.adversaries.values()]
        }
