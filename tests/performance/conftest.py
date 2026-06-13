"""Pytest configuration and fixtures for performance tests.

The ``benchmark`` fixture wraps :func:`pytest_benchmark.fixture.benchmark`
so that async callables are properly handled by running them in a
separate thread with its own event loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable

import pytest


@pytest.fixture
def benchmark(benchmark: Any) -> Any:
    """Wrap pytest-benchmark fixture to support async callables."""

    def _wrapper(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if asyncio.iscoroutinefunction(func):

            def _sync() -> Any:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(lambda: asyncio.run(func(*args, **kwargs)))
                    return fut.result()

            return benchmark(_sync)
        return benchmark(func, *args, **kwargs)

    return _wrapper


@pytest.fixture
def large_text() -> str:
    """Generate a large text for benchmarking."""
    return " ".join([f"This is sentence number {i}." for i in range(1000)])
