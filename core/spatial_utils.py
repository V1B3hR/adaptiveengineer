"""
Spatial utilities for multi-dimensional operations.
"""

import numpy as np
from typing import List, Any


def validate_spatial_dimensions(vectors: List[Any], expected_dims: int) -> None:
    """
    Validate that all vectors have the expected number of dimensions.
    
    Args:
        vectors: List of vectors to validate
        expected_dims: Expected number of dimensions
        
    Raises:
        ValueError: If any vector has incorrect dimensions
    """
    for i, vec in enumerate(vectors):
        if hasattr(vec, '__len__'):
            if len(vec) != expected_dims:
                raise ValueError(f"Vector {i} has {len(vec)} dimensions, expected {expected_dims}")


def zero_vector(dims: int) -> np.ndarray:
    """
    Create a zero vector of specified dimensions.
    
    Args:
        dims: Number of dimensions
        
    Returns:
        Zero vector as numpy array
    """
    return np.zeros(dims, dtype=float)
