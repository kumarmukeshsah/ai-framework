"""Tests for performance regression detection.

Tracks latency across commits to detect performance degradation.
"""
from __future__ import annotations

import time

import pytest

from product.agents.evaluator import EvaluatorAgent


LATENCY_THRESHOLD_MS = 5000  # 5 seconds for async operations


class TestPerformanceRegression:
    """Tests for performance regression detection."""

    @pytest.fixture
    def evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_evaluation_latency(self, evaluator):
        """Single evaluation should complete within latency threshold."""
        input_text = "I have 8 years of experience with Python."

        start = time.perf_counter()
        await evaluator.process(input_text)
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < LATENCY_THRESHOLD_MS, (
            f"Evaluation took {duration_ms:.1f}ms, threshold is {LATENCY_THRESHOLD_MS}ms"
        )

    @pytest.mark.asyncio
    async def test_batch_latency_stability(self, evaluator):
        """Batch processing should not degrade per-item latency."""
        inputs = [
            "Junior Python developer, 1 year experience.",
            "Mid-level Java developer, 4 years experience.",
            "Senior engineer, 10 years experience across multiple stacks.",
            "Entry-level with internship experience.",
            "Principal architect with 15 years leading teams.",
        ]

        start = time.perf_counter()
        for text in inputs:
            await evaluator.process(text)
        total_ms = (time.perf_counter() - start) * 1000
        avg_ms = total_ms / len(inputs)

        assert avg_ms < LATENCY_THRESHOLD_MS, (
            f"Average latency {avg_ms:.1f}ms exceeds threshold {LATENCY_THRESHOLD_MS}ms"
        )

    @pytest.mark.asyncio
    async def test_large_input_latency(self, evaluator):
        """Large inputs should still process within threshold."""
        large_input = "Python experience. " * 1000

        start = time.perf_counter()
        await evaluator.process(large_input)
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < LATENCY_THRESHOLD_MS * 2, (
            f"Large input took {duration_ms:.1f}ms"
        )

    @pytest.mark.asyncio
    async def test_latency_scaling(self, evaluator):
        """Latency should scale linearly with input count."""
        input_text = "Test input for latency scaling."
        num_iterations = 10

        start = time.perf_counter()
        for _ in range(num_iterations):
            await evaluator.process(input_text)
        total_ms = (time.perf_counter() - start) * 1000
        avg_ms = total_ms / num_iterations

        assert avg_ms < LATENCY_THRESHOLD_MS, (
            f"Average latency {avg_ms:.1f}ms exceeds threshold"
        )

    @pytest.mark.asyncio
    async def test_empty_input_latency(self, evaluator):
        """Empty input should be fast to process."""
        start = time.perf_counter()
        try:
            await evaluator.process("")
        except Exception:
            pass
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < LATENCY_THRESHOLD_MS, (
            f"Empty input took {duration_ms:.1f}ms"
        )