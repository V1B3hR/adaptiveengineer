"""
Time management module for synchronized timing across nodes.
"""

import time
from typing import Optional


class TimeManager:
    """Centralized time manager for simulation."""
    
    def __init__(self):
        self.current_time = 0
        self.use_real_time = False
        
    def get_time(self) -> int:
        """Get current simulation time."""
        if self.use_real_time:
            return int(time.time())
        return self.current_time
    
    def advance(self, delta: int = 1) -> None:
        """Advance simulation time."""
        self.current_time += delta
    
    def set_time(self, new_time: int) -> None:
        """Set simulation time."""
        self.current_time = new_time


# Global time manager instance
_time_manager: Optional[TimeManager] = None


def get_time_manager() -> TimeManager:
    """Get or create the global time manager."""
    global _time_manager
    if _time_manager is None:
        _time_manager = TimeManager()
        _time_manager.use_real_time = True  # Default to real time
    return _time_manager


def get_timestamp() -> int:
    """Get current timestamp."""
    return get_time_manager().get_time()
