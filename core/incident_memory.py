"""
Incident and Pattern Memory System.

Provides:
- Persistent storage for significant events (alerts, remediations, failures)
- Pattern recognition for recurring issues and threats
- Privacy/retention controls for compliance (GDPR/SOC2)
- Support for online/continual learning
"""

import json
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from enum import Enum
import threading


logger = logging.getLogger('incident_memory')


class IncidentType(str, Enum):
    """Types of incidents to track."""
    ALERT = "alert"
    REMEDIATION = "remediation"
    FAILURE = "failure"
    ANOMALY = "anomaly"
    THREAT = "threat"
    RECOVERY = "recovery"
    DEGRADATION = "degradation"


class IncidentStatus(str, Enum):
    """Incident status lifecycle."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    RECURRING = "recurring"


class DataClassification(str, Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class Incident:
    """
    Record of a significant event in the system.
    
    Attributes:
        incident_id: Unique identifier
        incident_type: Type of incident
        status: Current incident status
        timestamp: When incident occurred
        source: Source node or component
        description: Human-readable description
        severity: Severity level (0.0-1.0)
        data: Additional incident data
        tags: Tags for categorization
        classification: Data classification level
        retention_days: Days to retain (for compliance)
        related_incidents: IDs of related incidents
    """
    incident_id: str
    incident_type: IncidentType
    status: IncidentStatus
    timestamp: float
    source: str
    description: str
    severity: float = 0.5
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    classification: DataClassification = DataClassification.INTERNAL
    retention_days: Optional[int] = None  # None = default retention
    related_incidents: List[str] = field(default_factory=list)
    resolution_time: Optional[float] = None
    resolution_notes: Optional[str] = None
    
    def __post_init__(self):
        """Normalize enum types."""
        if isinstance(self.incident_type, str):
            self.incident_type = IncidentType(self.incident_type)
        if isinstance(self.status, str):
            self.status = IncidentStatus(self.status)
        if isinstance(self.classification, str):
            self.classification = DataClassification(self.classification)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary."""
        data = asdict(self)
        data['incident_type'] = self.incident_type.value
        data['status'] = self.status.value
        data['classification'] = self.classification.value
        return data
    
    def to_json(self) -> str:
        """Convert incident to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Incident':
        """Create incident from dictionary."""
        return cls(**data)
    
    def should_expire(self, current_time: float, default_retention_days: int = 90) -> bool:
        """
        Check if incident should be expired based on retention policy.
        
        Args:
            current_time: Current timestamp
            default_retention_days: Default retention if not specified
            
        Returns:
            True if incident should be expired
        """
        retention = self.retention_days if self.retention_days is not None else default_retention_days
        age_days = (current_time - self.timestamp) / 86400  # Convert to days
        return age_days > retention
    
    def get_signature(self) -> str:
        """
        Get a signature for pattern matching.
        
        Returns:
            Hash signature based on incident characteristics
        """
        sig_data = {
            'incident_type': self.incident_type.value,
            'source': self.source,
            'tags': sorted(self.tags)
        }
        sig_str = json.dumps(sig_data, sort_keys=True)
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]


@dataclass
class Pattern:
    """
    Identified pattern in incident data.
    
    Attributes:
        pattern_id: Unique identifier
        signature: Pattern signature
        incident_type: Type of incidents in pattern
        occurrence_count: Number of times pattern occurred
        first_seen: Timestamp of first occurrence
        last_seen: Timestamp of last occurrence
        sources: Set of sources where pattern occurred
        tags: Common tags in pattern
        severity_avg: Average severity
        confidence: Confidence score (0.0-1.0)
        description: Pattern description
    """
    pattern_id: str
    signature: str
    incident_type: IncidentType
    occurrence_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    sources: Set[str] = field(default_factory=set)
    tags: List[str] = field(default_factory=list)
    severity_avg: float = 0.5
    confidence: float = 0.0
    description: str = ""
    
    def update_from_incident(self, incident: Incident) -> None:
        """
        Update pattern statistics from a new incident.
        
        Args:
            incident: Incident matching this pattern
        """
        self.occurrence_count += 1
        self.last_seen = incident.timestamp
        self.sources.add(incident.source)
        
        # Update average severity
        self.severity_avg = (self.severity_avg * (self.occurrence_count - 1) + incident.severity) / self.occurrence_count
        
        # Update confidence based on occurrence count
        # More occurrences = higher confidence
        self.confidence = min(1.0, self.occurrence_count / 10.0)
        
        # Merge tags
        for tag in incident.tags:
            if tag not in self.tags:
                self.tags.append(tag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'signature': self.signature,
            'incident_type': self.incident_type.value,
            'occurrence_count': self.occurrence_count,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'sources': list(self.sources),
            'tags': self.tags,
            'severity_avg': self.severity_avg,
            'confidence': self.confidence,
            'description': self.description
        }


class IncidentStore(ABC):
    """Abstract base class for incident storage."""
    
    @abstractmethod
    def store_incident(self, incident: Incident) -> bool:
        """Store an incident."""
        pass
    
    @abstractmethod
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Retrieve an incident by ID."""
        pass
    
    @abstractmethod
    def query_incidents(self, 
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       incident_types: Optional[List[IncidentType]] = None,
                       sources: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None,
                       limit: int = 100) -> List[Incident]:
        """Query incidents with filters."""
        pass
    
    @abstractmethod
    def delete_incident(self, incident_id: str) -> bool:
        """Delete an incident (for compliance)."""
        pass
    
    @abstractmethod
    def clear_expired(self, current_time: float) -> int:
        """Clear expired incidents based on retention policy."""
        pass


class InMemoryIncidentStore(IncidentStore):
    """
    In-memory incident storage implementation.
    
    Suitable for testing and single-process scenarios.
    """
    
    def __init__(self, default_retention_days: int = 90):
        """
        Initialize in-memory incident store.
        
        Args:
            default_retention_days: Default retention period
        """
        self.incidents: Dict[str, Incident] = {}
        self.default_retention_days = default_retention_days
        self._lock = threading.Lock()
        logger.info(f"InMemoryIncidentStore initialized (retention: {default_retention_days} days)")
    
    def store_incident(self, incident: Incident) -> bool:
        """Store an incident."""
        try:
            with self._lock:
                self.incidents[incident.incident_id] = incident
            logger.debug(f"Stored incident {incident.incident_id} ({incident.incident_type.value})")
            return True
        except Exception as e:
            logger.error(f"Error storing incident: {e}", exc_info=True)
            return False
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Retrieve an incident by ID."""
        with self._lock:
            return self.incidents.get(incident_id)
    
    def query_incidents(self,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       incident_types: Optional[List[IncidentType]] = None,
                       sources: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None,
                       limit: int = 100) -> List[Incident]:
        """Query incidents with filters."""
        with self._lock:
            incidents = list(self.incidents.values())
        
        # Apply filters
        if start_time is not None:
            incidents = [i for i in incidents if i.timestamp >= start_time]
        if end_time is not None:
            incidents = [i for i in incidents if i.timestamp <= end_time]
        if incident_types is not None:
            incidents = [i for i in incidents if i.incident_type in incident_types]
        if sources is not None:
            incidents = [i for i in incidents if i.source in sources]
        if tags is not None:
            incidents = [i for i in incidents if any(tag in i.tags for tag in tags)]
        
        # Sort by timestamp (newest first) and limit
        incidents.sort(key=lambda i: i.timestamp, reverse=True)
        return incidents[:limit]
    
    def delete_incident(self, incident_id: str) -> bool:
        """Delete an incident."""
        try:
            with self._lock:
                if incident_id in self.incidents:
                    del self.incidents[incident_id]
                    logger.info(f"Deleted incident {incident_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error deleting incident: {e}", exc_info=True)
            return False
    
    def clear_expired(self, current_time: float) -> int:
        """Clear expired incidents based on retention policy."""
        with self._lock:
            expired = [
                iid for iid, incident in self.incidents.items()
                if incident.should_expire(current_time, self.default_retention_days)
            ]
            
            for iid in expired:
                del self.incidents[iid]
        
        if expired:
            logger.info(f"Cleared {len(expired)} expired incidents")
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total = len(self.incidents)
            by_type = Counter(i.incident_type.value for i in self.incidents.values())
            by_status = Counter(i.status.value for i in self.incidents.values())
        
        return {
            'total_incidents': total,
            'by_type': dict(by_type),
            'by_status': dict(by_status)
        }


class PatternRecognizer:
    """
    Recognizes patterns in incident data.
    
    Identifies recurring issues, threats, and anomalies for learning.
    """
    
    def __init__(self, min_occurrences: int = 3):
        """
        Initialize pattern recognizer.
        
        Args:
            min_occurrences: Minimum occurrences to identify as pattern
        """
        self.min_occurrences = min_occurrences
        self.patterns: Dict[str, Pattern] = {}
        self._lock = threading.Lock()
        logger.info(f"PatternRecognizer initialized (min_occurrences: {min_occurrences})")
    
    def analyze_incident(self, incident: Incident) -> Optional[Pattern]:
        """
        Analyze an incident for patterns.
        
        Args:
            incident: Incident to analyze
            
        Returns:
            Pattern if incident matches existing pattern, None otherwise
        """
        signature = incident.get_signature()
        
        with self._lock:
            if signature in self.patterns:
                # Update existing pattern
                pattern = self.patterns[signature]
                pattern.update_from_incident(incident)
                logger.debug(f"Updated pattern {pattern.pattern_id} (count: {pattern.occurrence_count})")
                return pattern
            else:
                # Create new pattern
                pattern = Pattern(
                    pattern_id=f"pattern_{signature}",
                    signature=signature,
                    incident_type=incident.incident_type,
                    description=f"Pattern for {incident.incident_type.value} from {incident.source}"
                )
                pattern.update_from_incident(incident)
                self.patterns[signature] = pattern
                logger.debug(f"Created new pattern {pattern.pattern_id}")
                return None  # Not a pattern yet (first occurrence)
    
    def get_patterns(self, 
                     min_confidence: float = 0.0,
                     incident_type: Optional[IncidentType] = None) -> List[Pattern]:
        """
        Get identified patterns.
        
        Args:
            min_confidence: Minimum confidence threshold
            incident_type: Filter by incident type
            
        Returns:
            List of patterns matching criteria
        """
        with self._lock:
            patterns = list(self.patterns.values())
        
        # Filter by minimum occurrences
        patterns = [p for p in patterns if p.occurrence_count >= self.min_occurrences]
        
        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        # Filter by type
        if incident_type is not None:
            patterns = [p for p in patterns if p.incident_type == incident_type]
        
        # Sort by occurrence count
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
        return patterns
    
    def get_pattern_by_signature(self, signature: str) -> Optional[Pattern]:
        """Get pattern by signature."""
        with self._lock:
            return self.patterns.get(signature)
    
    def clear_old_patterns(self, age_threshold: float) -> int:
        """
        Clear patterns that haven't been seen recently.
        
        Args:
            age_threshold: Age threshold in seconds
            
        Returns:
            Number of patterns cleared
        """
        current_time = time.time()
        with self._lock:
            old_patterns = [
                sig for sig, pattern in self.patterns.items()
                if current_time - pattern.last_seen > age_threshold
            ]
            
            for sig in old_patterns:
                del self.patterns[sig]
        
        if old_patterns:
            logger.info(f"Cleared {len(old_patterns)} old patterns")
        return len(old_patterns)


class IncidentMemorySystem:
    """
    Complete incident and pattern memory system.
    
    Provides:
    - Incident storage with retention policies
    - Pattern recognition
    - Learning support
    - Compliance controls
    """
    
    def __init__(self, 
                 incident_store: Optional[IncidentStore] = None,
                 pattern_recognizer: Optional[PatternRecognizer] = None):
        """
        Initialize incident memory system.
        
        Args:
            incident_store: Incident storage backend (default: in-memory)
            pattern_recognizer: Pattern recognizer (default: new instance)
        """
        self.incident_store = incident_store or InMemoryIncidentStore()
        self.pattern_recognizer = pattern_recognizer or PatternRecognizer()
        self._lock = threading.Lock()
        self.total_incidents = 0
        self.total_patterns = 0
        logger.info("IncidentMemorySystem initialized")
    
    def record_incident(self, incident: Incident) -> Tuple[bool, Optional[Pattern]]:
        """
        Record an incident and check for patterns.
        
        Args:
            incident: Incident to record
            
        Returns:
            Tuple of (success, pattern) where pattern is not None if recurring
        """
        # Store incident
        success = self.incident_store.store_incident(incident)
        if not success:
            return False, None
        
        self.total_incidents += 1
        
        # Analyze for patterns
        pattern = self.pattern_recognizer.analyze_incident(incident)
        
        # If this is a recurring pattern, update incident status
        if pattern and pattern.occurrence_count >= 3:
            incident.status = IncidentStatus.RECURRING
            incident.related_incidents = [
                i.incident_id 
                for i in self.incident_store.query_incidents(limit=100)
                if i.get_signature() == pattern.signature and i.incident_id != incident.incident_id
            ][:10]  # Limit to 10 related incidents
            self.incident_store.store_incident(incident)  # Update with new status
            
            logger.warning(f"Recurring pattern detected: {pattern.pattern_id} "
                          f"(count: {pattern.occurrence_count})")
        
        return True, pattern
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Retrieve an incident by ID."""
        return self.incident_store.get_incident(incident_id)
    
    def query_incidents(self, **kwargs) -> List[Incident]:
        """Query incidents with filters."""
        return self.incident_store.query_incidents(**kwargs)
    
    def get_patterns(self, **kwargs) -> List[Pattern]:
        """Get identified patterns."""
        return self.pattern_recognizer.get_patterns(**kwargs)
    
    def get_learning_data(self, incident_type: Optional[IncidentType] = None) -> Dict[str, Any]:
        """
        Get data for continual learning.
        
        Args:
            incident_type: Filter by incident type
            
        Returns:
            Learning data including incidents and patterns
        """
        # Get recent incidents
        recent_incidents = self.incident_store.query_incidents(
            incident_types=[incident_type] if incident_type else None,
            limit=1000
        )
        
        # Get patterns
        patterns = self.pattern_recognizer.get_patterns(
            incident_type=incident_type
        )
        
        # Compile learning data
        return {
            'incident_count': len(recent_incidents),
            'pattern_count': len(patterns),
            'incidents': [i.to_dict() for i in recent_incidents],
            'patterns': [p.to_dict() for p in patterns],
            'timestamp': time.time()
        }
    
    def maintenance(self) -> Dict[str, int]:
        """
        Perform maintenance operations.
        
        - Clear expired incidents
        - Clear old patterns
        
        Returns:
            Statistics on maintenance operations
        """
        current_time = time.time()
        
        # Clear expired incidents
        expired_incidents = self.incident_store.clear_expired(current_time)
        
        # Clear old patterns (not seen in 30 days)
        old_patterns = self.pattern_recognizer.clear_old_patterns(30 * 86400)
        
        logger.info(f"Maintenance: cleared {expired_incidents} incidents, {old_patterns} patterns")
        
        return {
            'expired_incidents': expired_incidents,
            'old_patterns': old_patterns
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        if isinstance(self.incident_store, InMemoryIncidentStore):
            store_stats = self.incident_store.get_stats()
        else:
            store_stats = {}
        
        patterns = self.pattern_recognizer.get_patterns()
        
        return {
            'total_incidents_processed': self.total_incidents,
            'total_patterns_identified': len(patterns),
            'store_stats': store_stats,
            'top_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'type': p.incident_type.value,
                    'count': p.occurrence_count,
                    'confidence': p.confidence
                }
                for p in patterns[:10]
            ]
        }


# Convenience functions
def create_incident_from_alert(alert_data: Dict[str, Any], source: str) -> Incident:
    """Create an incident from an alert."""
    return Incident(
        incident_id=alert_data.get('alert_id', f"alert_{int(time.time())}"),
        incident_type=IncidentType.ALERT,
        status=IncidentStatus.OPEN,
        timestamp=alert_data.get('timestamp', time.time()),
        source=source,
        description=alert_data.get('description', 'Alert triggered'),
        severity=alert_data.get('severity', 0.5),
        data=alert_data,
        tags=alert_data.get('tags', [])
    )


def create_incident_from_failure(failure_data: Dict[str, Any], source: str) -> Incident:
    """Create an incident from a failure."""
    return Incident(
        incident_id=failure_data.get('failure_id', f"failure_{int(time.time())}"),
        incident_type=IncidentType.FAILURE,
        status=IncidentStatus.OPEN,
        timestamp=failure_data.get('timestamp', time.time()),
        source=source,
        description=failure_data.get('description', 'System failure'),
        severity=failure_data.get('severity', 0.8),
        data=failure_data,
        tags=failure_data.get('tags', ['failure'])
    )
