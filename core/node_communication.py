"""
Node Communication Module

This module handles all communication and signaling functionality for AliveLoopNode,
including signal processing, queue management, and message handling.

Extracted from AliveLoopNode to improve modularity and maintainability.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional

from core.social_signals import SocialSignal
from core.time_manager import get_timestamp

logger = logging.getLogger('node_communication')
logger.setLevel(logging.WARNING)


class NodeCommunication:
    """
    Handles communication and signaling for nodes.
    
    This class manages:
    - Signal sending and receiving
    - Communication queues
    - Signal history tracking
    - Duplicate detection
    - Signal validation
    """
    
    def __init__(self, node_id: int, max_queue_size: int = 20, 
                 max_history: int = 100):
        """
        Initialize node communication system.
        
        Args:
            node_id: ID of the node this communication system belongs to
            max_queue_size: Maximum size of communication queue
            max_history: Maximum size of signal history
        """
        self.node_id = node_id
        
        # Communication queues
        self.communication_queue = deque(maxlen=max_queue_size)
        self.signal_history = deque(maxlen=max_history)
        
        # Signal processing tracking
        self.processed_signals = set()  # Track processed signal IDs
        self.signal_attempts = {}  # Track signal processing attempts
        
        # Partition queue for distributed processing
        self.partition_queues = {}
        
    def add_signal_to_queue(self, signal: SocialSignal) -> bool:
        """
        Add a signal to the communication queue.
        
        Args:
            signal: Signal to add to queue
            
        Returns:
            True if signal was added, False if rejected (e.g., duplicate)
        """
        # Check for duplicates
        if self._is_duplicate_signal(signal):
            logger.debug(f"Node {self.node_id}: Rejecting duplicate signal")
            return False
        
        # Validate signal schema
        if not self._validate_signal_schema(signal):
            logger.warning(f"Node {self.node_id}: Invalid signal schema")
            return False
        
        # Add to queue
        self.communication_queue.append(signal)
        
        # Record in history
        self.signal_history.append({
            'signal': signal,
            'timestamp': get_timestamp(),
            'action': 'received'
        })
        
        return True
    
    def record_sent_signal(self, signal: SocialSignal) -> None:
        """
        Record a signal that was sent.
        
        Args:
            signal: Signal that was sent
        """
        self.signal_history.append({
            'signal': signal,
            'timestamp': get_timestamp(),
            'action': 'sent'
        })
    
    def _is_duplicate_signal(self, signal: SocialSignal) -> bool:
        """
        Check if a signal is a duplicate of a recently processed one.
        
        Args:
            signal: Signal to check
            
        Returns:
            True if signal is a duplicate
        """
        # Create signal fingerprint
        signal_id = f"{signal.source_id}_{signal.signal_type}_{signal.timestamp}"
        
        if signal_id in self.processed_signals:
            return True
        
        # Check recent history for similar signals
        for entry in list(self.signal_history)[-10:]:  # Check last 10
            hist_signal = entry.get('signal')
            if hist_signal:
                if (hist_signal.source_id == signal.source_id and
                    hist_signal.signal_type == signal.signal_type and
                    abs(hist_signal.timestamp - signal.timestamp) < 1.0):
                    return True
        
        return False
    
    def _validate_signal_schema(self, signal: SocialSignal) -> bool:
        """
        Validate that a signal has the required schema.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        # Basic validation - signal should have required attributes
        if not hasattr(signal, 'source_id'):
            return False
        if not hasattr(signal, 'signal_type'):
            return False
        if not hasattr(signal, 'timestamp'):
            return False
        
        return True
    
    def record_signal_processed(self, signal: SocialSignal) -> None:
        """
        Record that a signal has been processed.
        
        Args:
            signal: Signal that was processed
        """
        signal_id = f"{signal.source_id}_{signal.signal_type}_{signal.timestamp}"
        self.processed_signals.add(signal_id)
        
        # Limit size of processed signals set
        if len(self.processed_signals) > 1000:
            # Remove oldest entries (approximate - just clear half)
            processed_list = list(self.processed_signals)
            self.processed_signals = set(processed_list[-500:])
    
    def record_signal_attempt(self, signal: SocialSignal, 
                             success: bool) -> None:
        """
        Record an attempt to process a signal.
        
        Args:
            signal: Signal that was attempted
            success: Whether processing was successful
        """
        signal_id = f"{signal.source_id}_{signal.signal_type}_{signal.timestamp}"
        
        if signal_id not in self.signal_attempts:
            self.signal_attempts[signal_id] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0
            }
        
        self.signal_attempts[signal_id]['attempts'] += 1
        if success:
            self.signal_attempts[signal_id]['successes'] += 1
        else:
            self.signal_attempts[signal_id]['failures'] += 1
    
    def add_to_partition_queue(self, partition_id: int, 
                               signal: SocialSignal) -> None:
        """
        Add a signal to a specific partition queue.
        
        Args:
            partition_id: ID of the partition
            signal: Signal to add
        """
        if partition_id not in self.partition_queues:
            self.partition_queues[partition_id] = deque(maxlen=50)
        
        self.partition_queues[partition_id].append(signal)
    
    def get_queue_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about communication queues.
        
        Returns:
            Dictionary of queue metrics
        """
        return {
            'queue_size': len(self.communication_queue),
            'queue_capacity': self.communication_queue.maxlen,
            'queue_utilization': len(self.communication_queue) / self.communication_queue.maxlen if self.communication_queue.maxlen > 0 else 0,
            'history_size': len(self.signal_history),
            'processed_count': len(self.processed_signals),
            'partition_count': len(self.partition_queues),
            'total_attempts': sum(a['attempts'] for a in self.signal_attempts.values()),
            'total_successes': sum(a['successes'] for a in self.signal_attempts.values()),
            'total_failures': sum(a['failures'] for a in self.signal_attempts.values())
        }
    
    def get_signal_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent signal history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of recent signal history entries
        """
        return list(self.signal_history)[-limit:]
    
    def clear_old_data(self, max_age: float = 3600.0) -> None:
        """
        Clear old data from tracking structures.
        
        Args:
            max_age: Maximum age in seconds for data to keep
        """
        current_time = get_timestamp()
        
        # Clear old history entries
        while self.signal_history:
            oldest = self.signal_history[0]
            if current_time - oldest['timestamp'] > max_age:
                self.signal_history.popleft()
            else:
                break
        
        # Clear old partition queues if empty
        empty_partitions = [
            pid for pid, queue in self.partition_queues.items()
            if len(queue) == 0
        ]
        for pid in empty_partitions:
            del self.partition_queues[pid]
