"""
Event-driven integrations for external IT/SOC tools.

Provides:
- Webhook/APM trigger system
- Real-time signal streaming for SIEM, AIOps, monitoring tools
- Event filtering and routing
"""

import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from collections import deque
from enum import Enum

from core.messaging import Message, MessageType


logger = logging.getLogger('event_integrations')


class EventSeverity(str, Enum):
    """Event severity levels for external integrations."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Event:
    """
    Event structure for external integrations.
    
    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        severity: Event severity level
        source: Event source (node_id, service, etc.)
        timestamp: Event timestamp
        data: Event data payload
        tags: Event tags for filtering
    """
    event_id: str
    event_type: str
    severity: EventSeverity
    source: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Normalize enum types."""
        if isinstance(self.severity, str):
            self.severity = EventSeverity(self.severity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp,
            'data': self.data,
            'tags': self.tags
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class EventFilter:
    """Filter events based on criteria."""
    
    def __init__(self, 
                 event_types: Optional[List[str]] = None,
                 severities: Optional[List[EventSeverity]] = None,
                 tags: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None):
        """
        Initialize event filter.
        
        Args:
            event_types: List of event types to match (None = all)
            severities: List of severities to match (None = all)
            tags: List of tags to match (None = all)
            sources: List of sources to match (None = all)
        """
        self.event_types = event_types
        self.severities = severities
        self.tags = tags
        self.sources = sources
    
    def matches(self, event: Event) -> bool:
        """
        Check if event matches filter criteria.
        
        Args:
            event: Event to check
            
        Returns:
            True if event matches all filter criteria
        """
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.severities and event.severity not in self.severities:
            return False
        
        if self.tags and not any(tag in event.tags for tag in self.tags):
            return False
        
        if self.sources and event.source not in self.sources:
            return False
        
        return True


class EventStream(ABC):
    """Abstract base class for event streaming."""
    
    @abstractmethod
    def stream_event(self, event: Event) -> bool:
        """
        Stream an event to external system.
        
        Args:
            event: Event to stream
            
        Returns:
            True if event was streamed successfully
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the event stream."""
        pass


class WebhookStream(EventStream):
    """
    Stream events to webhook endpoints.
    
    Simulates webhook calls for external systems (SIEM, monitoring, etc.)
    """
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize webhook stream.
        
        Args:
            url: Webhook URL
            headers: Optional HTTP headers
        """
        self.url = url
        self.headers = headers or {}
        self.event_count = 0
        logger.info(f"WebhookStream initialized for {url}")
    
    def stream_event(self, event: Event) -> bool:
        """
        Stream event to webhook.
        
        Args:
            event: Event to stream
            
        Returns:
            True if successful
        """
        try:
            # In production, this would make an actual HTTP POST
            # For now, we log the event as it would be sent
            logger.debug(f"Webhook POST to {self.url}: {event.to_json()}")
            self.event_count += 1
            return True
        except Exception as e:
            logger.error(f"Error streaming to webhook: {e}", exc_info=True)
            return False
    
    def close(self) -> None:
        """Close webhook stream."""
        logger.info(f"WebhookStream closed ({self.event_count} events sent)")


class SIEMStream(EventStream):
    """
    Stream events to SIEM (Security Information and Event Management) systems.
    
    Provides structured event streaming for security monitoring.
    """
    
    def __init__(self, siem_config: Dict[str, Any]):
        """
        Initialize SIEM stream.
        
        Args:
            siem_config: SIEM configuration (endpoint, API key, etc.)
        """
        self.siem_config = siem_config
        self.event_count = 0
        self.security_event_count = 0
        logger.info(f"SIEMStream initialized")
    
    def stream_event(self, event: Event) -> bool:
        """
        Stream event to SIEM.
        
        Args:
            event: Event to stream
            
        Returns:
            True if successful
        """
        try:
            # Convert to SIEM-specific format
            siem_event = self._format_for_siem(event)
            
            # In production, this would send to actual SIEM
            logger.debug(f"SIEM event: {json.dumps(siem_event)}")
            
            self.event_count += 1
            if event.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]:
                self.security_event_count += 1
            
            return True
        except Exception as e:
            logger.error(f"Error streaming to SIEM: {e}", exc_info=True)
            return False
    
    def _format_for_siem(self, event: Event) -> Dict[str, Any]:
        """Format event for SIEM ingestion."""
        return {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'severity': event.severity.value,
            'source': event.source,
            'timestamp': event.timestamp,
            'data': event.data,
            'tags': event.tags,
            'siem_metadata': {
                'ingestion_time': time.time(),
                'source_system': 'adaptive_engineer'
            }
        }
    
    def close(self) -> None:
        """Close SIEM stream."""
        logger.info(f"SIEMStream closed ({self.event_count} events, {self.security_event_count} security events)")


class AIOpsStream(EventStream):
    """
    Stream events to AIOps platforms.
    
    Provides operational event streaming for AI-driven operations.
    """
    
    def __init__(self, aiops_config: Dict[str, Any]):
        """
        Initialize AIOps stream.
        
        Args:
            aiops_config: AIOps configuration
        """
        self.aiops_config = aiops_config
        self.event_count = 0
        self.anomaly_count = 0
        logger.info(f"AIOpsStream initialized")
    
    def stream_event(self, event: Event) -> bool:
        """
        Stream event to AIOps platform.
        
        Args:
            event: Event to stream
            
        Returns:
            True if successful
        """
        try:
            # Convert to AIOps-specific format
            aiops_event = self._format_for_aiops(event)
            
            # In production, this would send to actual AIOps platform
            logger.debug(f"AIOps event: {json.dumps(aiops_event)}")
            
            self.event_count += 1
            if 'anomaly' in event.tags:
                self.anomaly_count += 1
            
            return True
        except Exception as e:
            logger.error(f"Error streaming to AIOps: {e}", exc_info=True)
            return False
    
    def _format_for_aiops(self, event: Event) -> Dict[str, Any]:
        """Format event for AIOps ingestion."""
        return {
            'event_id': event.event_id,
            'type': event.event_type,
            'severity': event.severity.value,
            'source': event.source,
            'timestamp': event.timestamp,
            'metrics': event.data,
            'tags': event.tags,
            'metadata': {
                'platform': 'adaptive_engineer',
                'version': '2.0'
            }
        }
    
    def close(self) -> None:
        """Close AIOps stream."""
        logger.info(f"AIOpsStream closed ({self.event_count} events, {self.anomaly_count} anomalies)")


class EventIntegrationManager:
    """
    Manages event-driven integrations with external tools.
    
    Provides:
    - Event filtering and routing
    - Multiple stream management
    - Real-time event streaming
    """
    
    def __init__(self):
        """Initialize event integration manager."""
        self.streams: Dict[str, EventStream] = {}
        self.filters: Dict[str, EventFilter] = {}
        self.event_buffer: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self.total_events = 0
        logger.info("EventIntegrationManager initialized")
    
    def add_stream(self, name: str, stream: EventStream, 
                   event_filter: Optional[EventFilter] = None) -> None:
        """
        Add an event stream.
        
        Args:
            name: Stream name/identifier
            stream: EventStream instance
            event_filter: Optional filter for this stream
        """
        with self._lock:
            self.streams[name] = stream
            if event_filter:
                self.filters[name] = event_filter
        logger.info(f"Added event stream: {name}")
    
    def remove_stream(self, name: str) -> None:
        """
        Remove an event stream.
        
        Args:
            name: Stream name to remove
        """
        with self._lock:
            if name in self.streams:
                self.streams[name].close()
                del self.streams[name]
                if name in self.filters:
                    del self.filters[name]
        logger.info(f"Removed event stream: {name}")
    
    def publish_event(self, event: Event) -> int:
        """
        Publish event to all matching streams.
        
        Args:
            event: Event to publish
            
        Returns:
            Number of streams event was sent to
        """
        with self._lock:
            streams = dict(self.streams)
            filters = dict(self.filters)
        
        # Buffer event for replay
        self.event_buffer.append(event.to_dict())
        self.total_events += 1
        
        # Route to matching streams
        sent_count = 0
        for name, stream in streams.items():
            # Check filter if present
            if name in filters and not filters[name].matches(event):
                continue
            
            # Stream event
            try:
                if stream.stream_event(event):
                    sent_count += 1
            except Exception as e:
                logger.error(f"Error streaming to {name}: {e}", exc_info=True)
        
        logger.debug(f"Event {event.event_id} sent to {sent_count} streams")
        return sent_count
    
    def publish_from_message(self, message: Message) -> int:
        """
        Convert message to event and publish.
        
        Args:
            message: Message to convert and publish
            
        Returns:
            Number of streams event was sent to
        """
        # Map message to event
        severity_map = {
            MessageType.ALERT: EventSeverity.WARNING,
            MessageType.ANOMALY: EventSeverity.WARNING,
            MessageType.REMEDIATION: EventSeverity.INFO,
            MessageType.EVENT: EventSeverity.INFO,
        }
        
        severity = severity_map.get(message.message_type, EventSeverity.INFO)
        
        # Determine tags
        tags = []
        if message.message_type == MessageType.ANOMALY:
            tags.append('anomaly')
        if message.message_type == MessageType.ALERT:
            tags.append('alert')
        if message.priority.value == 'critical':
            tags.append('critical')
        
        event = Event(
            event_id=message.message_id,
            event_type=message.message_type.value,
            severity=severity,
            source=f"node_{message.source_node_id}" if message.source_node_id else "unknown",
            timestamp=message.timestamp,
            data=message.payload,
            tags=tags
        )
        
        return self.publish_event(event)
    
    def get_recent_events(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent events from buffer.
        
        Args:
            count: Number of events to retrieve
            
        Returns:
            List of recent events
        """
        with self._lock:
            events = list(self.event_buffer)
        
        return events[-count:]
    
    def close_all(self) -> None:
        """Close all event streams."""
        with self._lock:
            for name, stream in self.streams.items():
                try:
                    stream.close()
                except Exception as e:
                    logger.error(f"Error closing stream {name}: {e}", exc_info=True)
            self.streams.clear()
            self.filters.clear()
        
        logger.info(f"All streams closed ({self.total_events} total events processed)")


# Convenience functions
def create_webhook_integration(url: str, 
                               event_filter: Optional[EventFilter] = None) -> tuple:
    """
    Create a webhook integration.
    
    Args:
        url: Webhook URL
        event_filter: Optional event filter
        
    Returns:
        Tuple of (stream, filter)
    """
    stream = WebhookStream(url)
    return stream, event_filter


def create_siem_integration(siem_config: Dict[str, Any],
                            severity_threshold: EventSeverity = EventSeverity.WARNING) -> tuple:
    """
    Create a SIEM integration with severity filtering.
    
    Args:
        siem_config: SIEM configuration
        severity_threshold: Minimum severity to stream
        
    Returns:
        Tuple of (stream, filter)
    """
    stream = SIEMStream(siem_config)
    
    # Filter for warning and above
    if severity_threshold == EventSeverity.WARNING:
        severities = [EventSeverity.WARNING, EventSeverity.ERROR, EventSeverity.CRITICAL]
    elif severity_threshold == EventSeverity.ERROR:
        severities = [EventSeverity.ERROR, EventSeverity.CRITICAL]
    elif severity_threshold == EventSeverity.CRITICAL:
        severities = [EventSeverity.CRITICAL]
    else:
        severities = None
    
    event_filter = EventFilter(severities=severities) if severities else None
    return stream, event_filter


def create_aiops_integration(aiops_config: Dict[str, Any],
                             tags: Optional[List[str]] = None) -> tuple:
    """
    Create an AIOps integration with tag filtering.
    
    Args:
        aiops_config: AIOps configuration
        tags: Optional tags to filter on
        
    Returns:
        Tuple of (stream, filter)
    """
    stream = AIOpsStream(aiops_config)
    event_filter = EventFilter(tags=tags) if tags else None
    return stream, event_filter
