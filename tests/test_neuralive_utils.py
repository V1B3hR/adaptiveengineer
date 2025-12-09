"""
Tests for NeuralLive utility modules.

This test file validates the utility functions extracted as part of
Phase 1 modular refactoring.
"""

import pytest

from neuralive.utils import estimate_size_mb


def test_estimate_size_mb_none():
    """Test that None returns 0.0 MB."""
    assert estimate_size_mb(None) == 0.0


def test_estimate_size_mb_empty_string():
    """Test that empty string returns minimal size."""
    size = estimate_size_mb("")
    assert size >= 0.0
    assert size < 0.001  # Should be very small


def test_estimate_size_mb_string():
    """Test size estimation for strings."""
    # A string with 1000 characters should be roughly 1KB
    test_string = "x" * 1000
    size = estimate_size_mb(test_string)
    assert size > 0.0
    assert size < 0.01  # Should be less than 10KB


def test_estimate_size_mb_bytes():
    """Test size estimation for bytes."""
    # 1MB of bytes should return approximately 1.0
    test_bytes = b"x" * (1024 * 1024)
    size = estimate_size_mb(test_bytes)
    assert 0.9 < size < 1.1  # Allow some margin


def test_estimate_size_mb_bytearray():
    """Test size estimation for bytearray."""
    test_bytearray = bytearray(b"x" * (1024 * 512))  # 512KB
    size = estimate_size_mb(test_bytearray)
    assert 0.4 < size < 0.6  # Should be roughly 0.5MB


def test_estimate_size_mb_list():
    """Test size estimation for generic objects (list)."""
    test_list = [1, 2, 3, 4, 5]
    size = estimate_size_mb(test_list)
    assert size >= 0.0  # Should return some value


def test_estimate_size_mb_dict():
    """Test size estimation for dictionary."""
    test_dict = {"key1": "value1", "key2": "value2"}
    size = estimate_size_mb(test_dict)
    assert size >= 0.0


def test_estimate_size_mb_exception_handling():
    """Test that function handles objects that can't be sized."""

    class UnsizableObject:
        """Object that raises TypeError on getsizeof."""

        def __sizeof__(self):
            raise TypeError("Cannot determine size")

    obj = UnsizableObject()
    # Should not raise exception, should return 0.0
    size = estimate_size_mb(obj)
    assert size >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
