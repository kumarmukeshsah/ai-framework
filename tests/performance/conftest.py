"""Pytest configuration and fixtures for performance tests.

This file provides a minimal ``benchmark`` fixture so the performance tests
can run without requiring the optional ``pytest-benchmark`` plugin. The
fixture simply times a callable, runs it a fixed number of times, and
returns the last result — sufficient to exercise the code under test and
assert functional correctness, while still producing throughput numbers
in test output.

The fixture supports both sync and async callables: if the supplied
``func`` returns a coroutine, it is awaited inside the timed loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import time
from typing import Any, Callable

import pytest


class _SimpleBenchmark:
    """Lightweight drop-in replacement for pytest-benchmark's fixture."""

    def __init__(self, iterations: int = 5) -> None:
        self.iterations = iterations

    def __call__(self, func: Callable[[], Any]) -> Any:
        if inspect.iscoroutinefunction(func) or _returns_coroutine(func):
            return self._run_async(func)
        return self._run_sync(func)

    # ── internal helpers ─────────────────────────────────────────────────

    def _run_sync(self, func: Callable[[], Any]) -> Any:
        last_result: Any = None
        start = time.perf_counter()
        for _ in range(self.iterations):
            last_result = func()
        elapsed = time.perf_counter() - start
        per_iter_ms = (elapsed / self.iterations) * 1000
        print(
            f"\n[benchmark] iterations={self.iterations} "
            f"total={elapsed:.4f}s avg={per_iter_ms:.3f}ms/iter"
        )
        return last_result

    def _run_async(self, func: Callable[[], Any]) -> Any:
        # Run the coroutine in a *separate* thread with its own event loop
        # so we don't conflict with any loop pytest-asyncio has already
        # started in the main test thread.
        def _runner() -> Any:
            last: Any = None
            loop = asyncio.new_event_loop()
            try:
                for _ in range(self.iterations):
                    last = loop.run_until_complete(func())
            finally:
                loop.close()
            return last

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            last_result = ex.submit(_runner).result()
        elapsed = time.perf_counter() - start
        per_iter_ms = (elapsed / self.iterations) * 1000
        print(
            f"\n[benchmark] iterations={self.iterations} "
            f"total={elapsed:.4f}s avg={per_iter_ms:.3f}ms/iter"
        )
        return last_result


def _returns_coroutine(func: Callable[[], Any]) -> bool:
    """Return True if calling ``func`` returns a coroutine object."""
    try:
        result = func()
    except Exception:  # noqa: BLE001
        return False
    is_coro = inspect.iscoroutine(result)
    if is_coro:
        try:
            result.close()  # avoid 'never awaited' warnings
        except Exception:  # noqa: BLE001
            pass
    return is_coro


@pytest.fixture
def benchmark() -> _SimpleBenchmark:
    """Provide a simple benchmark fixture.

    Acts as a small drop-in for the optional ``pytest-benchmark`` plugin.
    Runs ``func`` ``self.iterations`` times, returns the last result, and
    prints the average iteration time.
    """
    return _SimpleBenchmark()


@pytest.fixture
def large_text() -> str:
    """Generate a large text for benchmarking."""
    return " ".join([f"This is sentence number {i}." for i in range(1000)])
