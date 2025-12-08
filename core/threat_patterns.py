"""
Threat Pattern Genome System - Intelligent threat detection and evolution

This module implements a genome-based threat pattern system that learns and adapts
through adversarial co-evolution. Threats are represented as patterns with behavioral
signatures that can mutate and evolve, while defensive agents learn countermeasures.
"""

import logging
import time
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import random

logger = logging.getLogger('threat_patterns')


@dataclass
class ThreatPattern:
    """
    Represents a threat pattern with behavioral signature and learned defenses.
    
    The signature captures behavioral indicators like energy drain patterns,
    communication anomalies, and attack vectors. Countermeasures are learned
    through experience and tracked for effectiveness.
    """
    signature: List[float]  # Behavioral indicators (normalized 0-1)
    severity: float  # Threat level 0.0 to 1.0
    mutation_rate: float  # Evolution speed
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    countermeasures: List[str] = field(default_factory=list)
    effectiveness_history: Dict[str, List[float]] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    detection_count: int = 0
    successful_mitigations: int = 0
    failed_mitigations: int = 0
    attack_type: str = "unknown"  # energy_drain, jamming, trust_poisoning, etc.
    
    def __post_init__(self):
        """Validate threat pattern parameters"""
        self.severity = max(0.0, min(1.0, self.severity))
        self.mutation_rate = max(0.0, min(1.0, self.mutation_rate))
        
        # Ensure signature values are normalized
        self.signature = [max(0.0, min(1.0, v)) for v in self.signature]
    
    def update_last_seen(self) -> None:
        """Update last seen timestamp and increment detection count"""
        self.last_seen = time.time()
        self.detection_count += 1
    
    def add_countermeasure(self, countermeasure: str, effectiveness: float) -> None:
        """
        Add or update a countermeasure with its effectiveness.
        
        Args:
            countermeasure: Description of defensive strategy
            effectiveness: Success rate (0.0 to 1.0)
        """
        if countermeasure not in self.countermeasures:
            self.countermeasures.append(countermeasure)
        
        if countermeasure not in self.effectiveness_history:
            self.effectiveness_history[countermeasure] = []
        
        self.effectiveness_history[countermeasure].append(
            max(0.0, min(1.0, effectiveness))
        )
        
        # Track mitigation results
        if effectiveness > 0.6:
            self.successful_mitigations += 1
        else:
            self.failed_mitigations += 1
    
    def get_best_countermeasures(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Get most effective countermeasures based on historical performance.
        
        Args:
            top_n: Number of top countermeasures to return
            
        Returns:
            List of (countermeasure, avg_effectiveness) tuples
        """
        if not self.effectiveness_history:
            return []
        
        # Calculate average effectiveness for each countermeasure
        avg_effectiveness = []
        for cm, history in self.effectiveness_history.items():
            if history:
                avg = sum(history) / len(history)
                avg_effectiveness.append((cm, avg))
        
        # Sort by effectiveness
        avg_effectiveness.sort(key=lambda x: x[1], reverse=True)
        
        return avg_effectiveness[:top_n]
    
    def mutate(self) -> 'ThreatPattern':
        """
        Create a mutated version of this threat pattern.
        
        Simulates attacker evolution - patterns mutate to evade detection
        while maintaining core attack characteristics.
        
        Returns:
            New mutated ThreatPattern
        """
        # Mutate signature
        mutated_signature = []
        for value in self.signature:
            if random.random() < self.mutation_rate:
                # Add random noise scaled by mutation rate
                noise = random.gauss(0, self.mutation_rate * 0.2)
                mutated_value = value + noise
                mutated_signature.append(max(0.0, min(1.0, mutated_value)))
            else:
                mutated_signature.append(value)
        
        # Slightly increase severity if pattern is successful
        success_rate = (
            self.successful_mitigations / max(1, self.detection_count)
            if self.detection_count > 0 else 0.0
        )
        severity_adjustment = 0.05 if success_rate < 0.3 else -0.05
        new_severity = max(0.0, min(1.0, self.severity + severity_adjustment))
        
        # Create mutated pattern
        return ThreatPattern(
            signature=mutated_signature,
            severity=new_severity,
            mutation_rate=self.mutation_rate * random.uniform(0.9, 1.1),
            attack_type=self.attack_type,
            pattern_id=str(uuid.uuid4())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize threat pattern to dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'signature': self.signature,
            'severity': self.severity,
            'mutation_rate': self.mutation_rate,
            'countermeasures': self.countermeasures,
            'effectiveness_history': self.effectiveness_history,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'detection_count': self.detection_count,
            'successful_mitigations': self.successful_mitigations,
            'failed_mitigations': self.failed_mitigations,
            'attack_type': self.attack_type
        }


class ThreatLibrary:
    """
    Manages collection of known threat patterns with similarity search.
    
    Provides pattern matching, evolution simulation, and countermeasure
    recommendations based on historical threat data. Acts as the memory
    system for the immune-inspired defense.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize threat library.
        
        Args:
            similarity_threshold: Minimum similarity score to consider patterns related
        """
        self.patterns: Dict[str, ThreatPattern] = {}
        self.similarity_threshold = similarity_threshold
        self.pattern_lineage: Dict[str, List[str]] = {}  # Track pattern evolution
        
        # Metrics
        self.total_threats_detected = 0
        self.total_threats_mitigated = 0
        self.evolution_generations = 0
        
        logger.info(f"ThreatLibrary initialized with similarity threshold {similarity_threshold}")
    
    def add_pattern(self, pattern: ThreatPattern) -> None:
        """
        Register a new threat pattern in the library.
        
        Args:
            pattern: ThreatPattern to add
        """
        self.patterns[pattern.pattern_id] = pattern
        self.total_threats_detected += 1
        
        logger.info(f"Added threat pattern {pattern.pattern_id}: "
                   f"type={pattern.attack_type}, severity={pattern.severity:.2f}")
    
    def find_similar(
        self,
        pattern: ThreatPattern,
        threshold: Optional[float] = None
    ) -> List[Tuple[ThreatPattern, float]]:
        """
        Find similar threat patterns using behavioral signature matching.
        
        Uses Euclidean distance in signature space to identify related threats.
        This enables transfer learning - defenses for similar threats can be applied.
        
        Args:
            pattern: Pattern to match
            threshold: Optional custom similarity threshold
            
        Returns:
            List of (pattern, similarity_score) tuples, sorted by similarity
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        similar = []
        
        for stored_pattern in self.patterns.values():
            if stored_pattern.pattern_id == pattern.pattern_id:
                continue
            
            similarity = self._calculate_similarity(pattern, stored_pattern)
            
            if similarity >= threshold:
                similar.append((stored_pattern, similarity))
        
        # Sort by similarity (highest first)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar
    
    def _calculate_similarity(
        self,
        pattern1: ThreatPattern,
        pattern2: ThreatPattern
    ) -> float:
        """
        Calculate similarity between two threat patterns.
        
        Uses normalized Euclidean distance, considering:
        - Signature similarity (behavioral indicators)
        - Attack type match
        - Severity similarity
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Ensure signatures have same length
        min_len = min(len(pattern1.signature), len(pattern2.signature))
        if min_len == 0:
            return 0.0
        
        sig1 = pattern1.signature[:min_len]
        sig2 = pattern2.signature[:min_len]
        
        # Euclidean distance in signature space
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(sig1, sig2)))
        
        # Normalize by maximum possible distance
        max_distance = math.sqrt(min_len)
        normalized_distance = distance / max_distance if max_distance > 0 else 0.0
        
        # Convert distance to similarity
        signature_similarity = 1.0 - normalized_distance
        
        # Boost similarity if attack types match
        type_match = 1.0 if pattern1.attack_type == pattern2.attack_type else 0.7
        
        # Consider severity similarity
        severity_diff = abs(pattern1.severity - pattern2.severity)
        severity_similarity = 1.0 - severity_diff
        
        # Weighted combination
        similarity = (
            0.6 * signature_similarity +
            0.25 * type_match +
            0.15 * severity_similarity
        )
        
        return max(0.0, min(1.0, similarity))
    
    def evolve_patterns(self, count: int = 5) -> List[ThreatPattern]:
        """
        Simulate attacker evolution by mutating existing patterns.
        
        Selects most successful patterns and mutates them to create
        new variants. This simulates the adversarial arms race.
        
        Args:
            count: Number of new patterns to generate
            
        Returns:
            List of newly evolved patterns
        """
        if not self.patterns:
            return []
        
        # Select patterns for evolution (prefer successful ones)
        patterns_list = list(self.patterns.values())
        
        # Sort by success (low mitigation rate = successful attacker)
        patterns_list.sort(
            key=lambda p: (
                p.failed_mitigations / max(1, p.successful_mitigations + p.failed_mitigations)
            ),
            reverse=True
        )
        
        # Mutate top patterns
        evolved = []
        for i in range(min(count, len(patterns_list))):
            parent = patterns_list[i % len(patterns_list)]
            mutant = parent.mutate()
            
            # Track lineage
            if parent.pattern_id not in self.pattern_lineage:
                self.pattern_lineage[parent.pattern_id] = []
            self.pattern_lineage[parent.pattern_id].append(mutant.pattern_id)
            
            evolved.append(mutant)
            self.add_pattern(mutant)
        
        self.evolution_generations += 1
        
        logger.info(f"Evolved {len(evolved)} threat patterns (generation {self.evolution_generations})")
        
        return evolved
    
    def get_effective_countermeasures(
        self,
        pattern: ThreatPattern,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve effective countermeasures for a threat pattern.
        
        Combines direct countermeasures with those from similar patterns
        (transfer learning). This enables defense against novel threats
        based on similarity to known threats.
        
        Args:
            pattern: Threat pattern to defend against
            top_n: Number of countermeasures to return
            
        Returns:
            List of (countermeasure, confidence) tuples
        """
        countermeasures: Dict[str, List[float]] = {}
        
        # Get direct countermeasures from this pattern
        if pattern.pattern_id in self.patterns:
            stored = self.patterns[pattern.pattern_id]
            for cm, effectiveness in stored.effectiveness_history.items():
                if cm not in countermeasures:
                    countermeasures[cm] = []
                # Weight by 1.0 (direct match)
                countermeasures[cm].extend([e * 1.0 for e in effectiveness])
        
        # Get countermeasures from similar patterns (transfer learning)
        similar_patterns = self.find_similar(pattern)
        for similar, similarity in similar_patterns[:3]:  # Top 3 similar
            for cm, effectiveness in similar.effectiveness_history.items():
                if cm not in countermeasures:
                    countermeasures[cm] = []
                # Weight by similarity score
                countermeasures[cm].extend([e * similarity for e in effectiveness])
        
        # Calculate average confidence for each countermeasure
        avg_confidence = []
        for cm, scores in countermeasures.items():
            if scores:
                avg = sum(scores) / len(scores)
                avg_confidence.append((cm, avg))
        
        # Sort by confidence
        avg_confidence.sort(key=lambda x: x[1], reverse=True)
        
        return avg_confidence[:top_n]
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[ThreatPattern]:
        """Get a specific pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_type(self, attack_type: str) -> List[ThreatPattern]:
        """Get all patterns of a specific attack type"""
        return [
            p for p in self.patterns.values()
            if p.attack_type == attack_type
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics"""
        if not self.patterns:
            return {
                'total_patterns': 0,
                'total_threats_detected': self.total_threats_detected,
                'total_threats_mitigated': self.total_threats_mitigated,
                'mitigation_rate': 0.0,
                'evolution_generations': self.evolution_generations,
                'attack_types': {}
            }
        
        # Count attack types
        attack_types = {}
        total_detections = 0
        total_mitigations = 0
        
        for pattern in self.patterns.values():
            attack_type = pattern.attack_type
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
            total_detections += pattern.detection_count
            total_mitigations += pattern.successful_mitigations
        
        mitigation_rate = (
            total_mitigations / total_detections
            if total_detections > 0 else 0.0
        )
        
        return {
            'total_patterns': len(self.patterns),
            'total_threats_detected': total_detections,
            'total_threats_mitigated': total_mitigations,
            'mitigation_rate': mitigation_rate,
            'evolution_generations': self.evolution_generations,
            'attack_types': attack_types
        }
    
    def clear(self) -> None:
        """Clear all patterns from library"""
        self.patterns.clear()
        self.pattern_lineage.clear()
        logger.info("ThreatLibrary cleared")
