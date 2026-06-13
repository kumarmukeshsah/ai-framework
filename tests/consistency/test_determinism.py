"""Tests for output determinism and reproducibility.

For production AI systems, deterministic behavior (at temperature=0) is critical
for testing, auditing, and debugging.
"""

from __future__ import annotations

import pytest

from product.agents.evaluator import EvaluatorAgent


class TestDeterminism:
    """Tests for output determinism."""

    @pytest.fixture
    def evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_same_input_same_output(self, evaluator):
        """Same input should produce identical evaluation across multiple calls."""
        input_text = (
            "I have 8 years of Python experience, leading a team of 5 engineers. "
            "Expertise in FastAPI, PostgreSQL, and cloud infrastructure."
        )

        results = []
        for _ in range(5):
            result = await evaluator.process(input_text)
            if result.evaluation:
                results.append(result.evaluation.candidate_level)
            else:
                results.append(str(result.success))

        assert all(
            r == results[0] for r in results
        ), "Rule-based evaluator produced different evaluation for same input"

    @pytest.mark.asyncio
    async def test_batch_consistency(self, evaluator):
        """Processing the same input in batch should produce consistent results."""
        input_text = "I have 3 years of experience with Java and Spring Boot."

        result1 = await evaluator.process(input_text)
        result2 = await evaluator.process(input_text)
        result3 = await evaluator.process(input_text)

        # Compare evaluation results (ignoring timing differences)
        e1 = result1.evaluation
        e2 = result2.evaluation
        e3 = result3.evaluation
        if e1 and e2 and e3:
            assert e1.candidate_level == e2.candidate_level == e3.candidate_level

    @pytest.mark.asyncio
    async def test_empty_input_consistency(self, evaluator):
        """Empty input handling should be consistent."""
        results = []
        for _ in range(3):
            try:
                result = await evaluator.process("")
                results.append(result)
            except Exception:
                results.append(None)

        # All results should be None (same outcome) or all non-None
        none_results = [r is None for r in results]
        assert all(none_results) or not any(none_results)

    @pytest.mark.asyncio
    async def test_determinism_across_different_inputs(self, evaluator):
        """Multiple distinct inputs should each produce deterministic results."""
        inputs = [
            "I am a junior developer with 1 year of experience.",
            "I am a senior engineer with 10 years of experience.",
            "I am a mid-level developer with 5 years of experience.",
        ]

        for input_text in inputs:
            r1 = await evaluator.process(input_text)
            r2 = await evaluator.process(input_text)
            e1 = r1.evaluation
            e2 = r2.evaluation
            if e1 and e2:
                assert (
                    e1.candidate_level == e2.candidate_level
                ), f"Non-deterministic for: {input_text[:50]}"
