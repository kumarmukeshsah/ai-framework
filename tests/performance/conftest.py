"""Pytest configuration and fixtures for performance tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def large_text() -> str:
    """Generate a large text for benchmarking."""
    return " ".join([f"This is sentence number {i}." for i in range(1000)])
