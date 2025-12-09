"""
NeuralLive - Artificial Life System with Self-Awareness & Swarm Intelligence.

This package represents the modular refactoring of the adaptiveengineer
codebase as part of Phase 1: Foundation Refactoring.

Version: 2.0.0
"""

from typing import Any, List, Optional

__version__ = "2.0.0"
__author__ = "V1B3hR"

# For backward compatibility, maintain imports from the original structure
# This allows gradual migration without breaking existing code
AliveLoopNode: Optional[Any] = None
Memory: Optional[Any] = None
SocialSignal: Optional[Any] = None

try:
    from adaptiveengineer import AliveLoopNode as _AliveLoopNode
    from adaptiveengineer import Memory as _Memory
    from adaptiveengineer import SocialSignal as _SocialSignal

    AliveLoopNode = _AliveLoopNode
    Memory = _Memory
    SocialSignal = _SocialSignal
except ImportError:
    # Handle case where adaptiveengineer hasn't been fully set up yet
    pass

__all__: List[str] = [
    "__version__",
    "__author__",
    "AliveLoopNode",
    "Memory",
    "SocialSignal",
]
