"""
Memory utility functions for NeuralLive.

This module provides utility functions for memory estimation and management.
Extracted from adaptiveengineer.py as part of Phase 1 refactoring.
"""

import sys
from typing import Any


def estimate_size_mb(content: Any) -> float:
    """
    Conservative size estimation in MB for common types.

    This function estimates the memory size of various Python objects,
    useful for memory management and capacity planning.

    Args:
        content: Any Python object to estimate size for

    Returns:
        Estimated size in megabytes (MB)

    Examples:
        >>> estimate_size_mb(None)
        0.0
        >>> estimate_size_mb("hello") > 0
        True
        >>> estimate_size_mb(b"test") > 0
        True
    """
    if content is None:
        return 0.0

    # Handle bytes-like objects
    if isinstance(content, (bytes, bytearray)):
        return len(content) / (1024 * 1024)

    # Handle strings
    if isinstance(content, str):
        return len(content.encode("utf-8")) / (1024 * 1024)

    # Generic size estimation
    try:
        size = sys.getsizeof(content)
    except Exception:
        size = 0

    # Convert to MB (shallow estimate)
    return max(0.0, size / (1024 * 1024))
