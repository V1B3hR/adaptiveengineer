"""
Time management module for synchronized timing across nodes.
"""

import time
from typing import Optional


class TimeManager:
    """Centralized time manager for simulation."""

    def __init__(self):
        self.current_time = 0
        self.simulation_step = 0
        self.circadian_time = 0
        self.use_real_time = False

    def get_time(self) -> int:
        """Get current simulation time."""
        if self.use_real_time:
            return int(time.time())
        return self.current_time

    def advance(self, delta: int = 1) -> None:
        """Advance simulation time."""
        self.current_time += delta
        self.simulation_step += delta
        self.circadian_time = self.simulation_step % 24  # 24-hour cycle

    def advance_simulation(self, steps: int) -> None:
        """Advance simulation by specified number of steps."""
        self.advance(steps)

    def set_time(self, new_time: int) -> None:
        """Set simulation time."""
        self.current_time = new_time
        self.simulation_step = new_time
        self.circadian_time = new_time % 24

    def reset(self) -> None:
        """Reset time manager to initial state."""
        self.current_time = 0
        self.simulation_step = 0
        self.circadian_time = 0


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
