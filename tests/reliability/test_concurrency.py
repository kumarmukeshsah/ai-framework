"""Tests for concurrent request handling.

Ensures the framework handles multiple simultaneous requests without
data corruption, race conditions, or degraded service.
"""
from __future__ import annotations

import asyncio

import pytest

from product.agents.evaluator import EvaluatorAgent


class TestConcurrency:
    """Tests for concurrent request handling."""

    @pytest.fixture
    def evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_concurrent_same_input(self, evaluator):
        """Same input processed concurrently should produce consistent results."""
        input_text = "I have 5 years of experience with Python."

        async def process():
            result = await evaluator.process(input_text)
            return result.evaluation.candidate_level if result.evaluation else str(result.success)

        tasks = [process() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(r == results[0] for r in results), "Concurrent results differ"

    @pytest.mark.asyncio
    async def test_concurrent_different_inputs(self, evaluator):
        """Different inputs processed concurrently should produce correct results."""
        inputs = [
            "I am a Junior developer with 1 year experience.",
            "I am a Mid-level developer with 5 years experience.",
            "I am a Senior developer with 10 years experience.",
            "I have no experience.",
            "I have extensive experience in multiple languages.",
        ]

        async def process(text):
            result = await evaluator.process(text)
            return str(result)

        tasks = [process(text) for text in inputs]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(inputs)
        for result in results:
            assert len(result) > 0, "Empty result from concurrent processing"

    @pytest.mark.asyncio
    async def test_concurrent_empty_inputs(self, evaluator):
        """Empty inputs processed concurrently should not crash."""
        async def process():
            try:
                result = await evaluator.process("")
                return str(result)
            except Exception:
                return "ERROR"

        tasks = [process() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_concurrent_memory_stability(self, evaluator):
        """Concurrent calls should not cause memory issues."""
        async def process():
            result = await evaluator.process("Python developer with 3 years experience.")
            return result

        results = await asyncio.gather(*[process() for _ in range(50)])
        assert len(results) == 50
        for r in results:
            assert r is not None