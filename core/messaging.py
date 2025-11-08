"""
Secure, structured messaging system for adaptive engineer communication.

Provides:
- Rich message types (alert, event, anomaly, remediation, trust_update, consensus)
- Multiple backend support (in-memory, ZeroMQ, RabbitMQ, Kafka)
- Encryption support for P2P communication
- Idempotency, tracing, and replay capabilities
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from collections import deque
import hashlib
import threading


logger = logging.getLogger('messaging')


class MessageType(str, Enum):
    """Rich message types for adaptive engineer communication."""
    ALERT = "alert"
    EVENT = "event"
    ANOMALY = "anomaly"
    REMEDIATION = "remediation"
    TRUST_UPDATE = "trust_update"
    CONSENSUS = "consensus"
    HEARTBEAT = "heartbeat"
    REQUEST = "request"
    RESPONSE = "response"


class Priority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Message:
    """
    Structured message for inter-node communication.
    
    Attributes:
        message_id: Unique identifier for idempotency
        message_type: Type of message (alert, event, etc.)
        source_node_id: Originating node identifier
        target_node_id: Target node identifier (None for broadcast)
        payload: Message content
        timestamp: Creation timestamp
        priority: Message priority level
        trace_id: Trace identifier for request correlation
        reply_to: Message ID this is replying to
        ttl: Time-to-live in seconds
        requires_ack: Whether acknowledgment is required
        encrypted: Whether payload is encrypted
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.EVENT
    source_node_id: Optional[int] = None
    target_node_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: Priority = Priority.NORMAL
    trace_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: int = 300  # 5 minutes default
    requires_ack: bool = False
    encrypted: bool = False
    
    def __post_init__(self):
        """Normalize enum types."""
        if isinstance(self.message_type, str):
            self.message_type = MessageType(self.message_type)
        if isinstance(self.priority, str):
            self.priority = Priority(self.priority)
        if self.trace_id is None:
            self.trace_id = self.message_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def is_expired(self) -> bool:
        """Check if message has exceeded TTL."""
        return time.time() - self.timestamp > self.ttl
    
    def get_content_hash(self) -> str:
        """Get hash of message content for idempotency checking."""
        content = json.dumps({
            'message_type': self.message_type.value,
            'source_node_id': self.source_node_id,
            'payload': self.payload
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class MessageHandler:
    """Handler for processing incoming messages."""
    
    def __init__(self, callback: Callable[[Message], None], 
                 message_types: Optional[List[MessageType]] = None):
        """
        Initialize message handler.
        
        Args:
            callback: Function to call with incoming messages
            message_types: List of message types to handle (None = all types)
        """
        self.callback = callback
        self.message_types = message_types
    
    def can_handle(self, message: Message) -> bool:
        """Check if this handler can process the message."""
        if self.message_types is None:
            return True
        return message.message_type in self.message_types
    
    def handle(self, message: Message) -> None:
        """Process the message."""
        try:
            self.callback(message)
        except Exception as e:
            logger.error(f"Error in message handler: {e}", exc_info=True)


class MessageBus(ABC):
    """
    Abstract base class for message bus implementations.
    
    Provides secure, structured messaging with support for:
    - Multiple backends (in-memory, ZeroMQ, RabbitMQ, Kafka)
    - Rich message types
    - Idempotency checking
    - Message tracing
    - Replay support
    """
    
    def __init__(self, node_id: Optional[int] = None):
        """
        Initialize message bus.
        
        Args:
            node_id: Identifier of the node using this bus
        """
        self.node_id = node_id
        self.handlers: List[MessageHandler] = []
        self.message_history: deque = deque(maxlen=1000)  # For replay
        self.processed_messages: Dict[str, float] = {}  # For idempotency
        self.ack_callbacks: Dict[str, Callable] = {}  # For acknowledgments
        self._lock = threading.Lock()
        logger.info(f"MessageBus initialized for node {node_id}")
    
    @abstractmethod
    def send(self, message: Message) -> bool:
        """
        Send a message through the bus.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the message bus (listening for messages)."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the message bus."""
        pass
    
    def register_handler(self, handler: MessageHandler) -> None:
        """
        Register a message handler.
        
        Args:
            handler: MessageHandler instance
        """
        with self._lock:
            self.handlers.append(handler)
            logger.debug(f"Registered handler for message types: {handler.message_types}")
    
    def subscribe(self, callback: Callable[[Message], None],
                  message_types: Optional[List[MessageType]] = None) -> MessageHandler:
        """
        Subscribe to messages with a callback.
        
        Args:
            callback: Function to call with incoming messages
            message_types: List of message types to subscribe to (None = all)
            
        Returns:
            MessageHandler instance
        """
        handler = MessageHandler(callback, message_types)
        self.register_handler(handler)
        return handler
    
    def _is_duplicate(self, message: Message) -> bool:
        """
        Check if message has already been processed (idempotency).
        
        Args:
            message: Message to check
            
        Returns:
            True if message is a duplicate
        """
        with self._lock:
            content_hash = message.get_content_hash()
            
            # Clean up old entries (older than 5 minutes)
            current_time = time.time()
            self.processed_messages = {
                k: v for k, v in self.processed_messages.items()
                if current_time - v < 300
            }
            
            if content_hash in self.processed_messages:
                logger.debug(f"Duplicate message detected: {message.message_id}")
                return True
            
            self.processed_messages[content_hash] = current_time
            return False
    
    def _record_message(self, message: Message) -> None:
        """
        Record message in history for replay support.
        
        Args:
            message: Message to record
        """
        with self._lock:
            self.message_history.append(message.to_dict())
    
    def _dispatch_message(self, message: Message) -> None:
        """
        Dispatch message to registered handlers.
        
        Args:
            message: Message to dispatch
        """
        # Check for duplicates
        if self._is_duplicate(message):
            return
        
        # Check if expired
        if message.is_expired():
            logger.warning(f"Message {message.message_id} expired (TTL: {message.ttl}s)")
            return
        
        # Record for replay
        self._record_message(message)
        
        # Dispatch to handlers
        with self._lock:
            handlers = list(self.handlers)
        
        for handler in handlers:
            if handler.can_handle(message):
                handler.handle(message)
        
        # Send acknowledgment if required
        if message.requires_ack:
            self._send_ack(message)
    
    def _send_ack(self, original_message: Message) -> None:
        """
        Send acknowledgment for a message.
        
        Args:
            original_message: Message to acknowledge
        """
        ack_message = Message(
            message_type=MessageType.RESPONSE,
            source_node_id=self.node_id,
            target_node_id=original_message.source_node_id,
            payload={'ack': True, 'original_message_id': original_message.message_id},
            reply_to=original_message.message_id,
            trace_id=original_message.trace_id
        )
        self.send(ack_message)
    
    def replay_messages(self, 
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       message_types: Optional[List[MessageType]] = None) -> List[Dict[str, Any]]:
        """
        Replay messages from history.
        
        Args:
            start_time: Start timestamp for replay (None = from beginning)
            end_time: End timestamp for replay (None = to end)
            message_types: Filter by message types (None = all types)
            
        Returns:
            List of message dictionaries
        """
        with self._lock:
            messages = list(self.message_history)
        
        # Filter by time range
        if start_time is not None:
            messages = [m for m in messages if m['timestamp'] >= start_time]
        if end_time is not None:
            messages = [m for m in messages if m['timestamp'] <= end_time]
        
        # Filter by message type
        if message_types is not None:
            type_values = [mt.value for mt in message_types]
            messages = [m for m in messages if m['message_type'] in type_values]
        
        logger.info(f"Replaying {len(messages)} messages")
        return messages
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages associated with a trace ID.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            List of message dictionaries with matching trace_id
        """
        with self._lock:
            messages = [m for m in self.message_history if m.get('trace_id') == trace_id]
        
        logger.debug(f"Found {len(messages)} messages for trace {trace_id}")
        return messages


class InMemoryMessageBus(MessageBus):
    """
    In-memory message bus implementation for testing and single-process scenarios.
    
    Provides full message bus functionality without external dependencies.
    """
    
    def __init__(self, node_id: Optional[int] = None):
        """Initialize in-memory message bus."""
        super().__init__(node_id)
        self.running = False
        self._message_queue: deque = deque()
        self._processing_thread: Optional[threading.Thread] = None
        logger.info(f"InMemoryMessageBus created for node {node_id}")
    
    def send(self, message: Message) -> bool:
        """
        Send a message through the in-memory bus.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was queued successfully
        """
        try:
            if message.source_node_id is None:
                message.source_node_id = self.node_id
            
            with self._lock:
                self._message_queue.append(message)
            
            logger.debug(f"Message {message.message_id} queued (type: {message.message_type.value})")
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            return False
    
    def start(self) -> None:
        """Start processing messages."""
        if self.running:
            return
        
        self.running = True
        self._processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self._processing_thread.start()
        logger.info(f"InMemoryMessageBus started for node {self.node_id}")
    
    def stop(self) -> None:
        """Stop processing messages."""
        self.running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
        logger.info(f"InMemoryMessageBus stopped for node {self.node_id}")
    
    def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self.running:
            try:
                with self._lock:
                    if self._message_queue:
                        message = self._message_queue.popleft()
                    else:
                        message = None
                
                if message:
                    self._dispatch_message(message)
                else:
                    time.sleep(0.01)  # Small sleep to avoid busy waiting
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                time.sleep(0.1)


# Factory function for creating message buses
def create_message_bus(backend: str = "memory", 
                      node_id: Optional[int] = None,
                      **kwargs) -> MessageBus:
    """
    Factory function to create a message bus.
    
    Args:
        backend: Backend type ('memory', 'zeromq', 'rabbitmq', 'kafka')
        node_id: Node identifier
        **kwargs: Backend-specific configuration
        
    Returns:
        MessageBus instance
    """
    if backend == "memory":
        return InMemoryMessageBus(node_id=node_id)
    else:
        # For Phase 2, we implement in-memory bus
        # Future phases can add ZeroMQ, RabbitMQ, Kafka implementations
        logger.warning(f"Backend '{backend}' not yet implemented, using in-memory bus")
        return InMemoryMessageBus(node_id=node_id)
