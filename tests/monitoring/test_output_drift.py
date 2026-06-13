"""Tests for output drift detection.

Monitors whether LLM output distribution changes over time,
which could indicate model degradation, data drift, or prompt issues.
"""

from __future__ import annotations

import pytest

from product.agents.evaluator import EvaluatorAgent


class TestOutputDrift:
    """Tests for output drift detection."""

    @pytest.fixture
    def evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_output_distribution_stability(self, evaluator):
        """Output distribution should remain stable over time."""
        input_text = "I have 5 years of Python and Java experience."
        results = []
        for _ in range(10):
            result = await evaluator.process(input_text)
            results.append(str(result))

        assert all(r == results[0] for r in results), "Output drift detected"

    @pytest.mark.asyncio
    async def test_no_unexpected_output_categories(self, evaluator):
        """Output should only contain valid categories."""
        inputs = [
            "Junior developer with 1 year experience.",
            "Mid-level with 4 years experience.",
            "Senior with 10 years experience.",
        ]
        valid_levels = {"Junior", "Mid", "Senior"}
        for text in inputs:
            result = await evaluator.process(text)
            if result.evaluation:
                assert result.evaluation.candidate_level in valid_levels

    @pytest.mark.asyncio
    async def test_output_format_consistency(self, evaluator):
        """Output format should remain consistent across calls."""
        inputs = [
            "1 year experience Python.",
            "10 years experience multiple languages.",
            "5 years experience team lead.",
        ]
        for text in inputs:
            result = await evaluator.process(text)
            assert result is not None
            assert result.success is True

    @pytest.mark.asyncio
    async def test_regression_detection(self, evaluator):
        """Changes in input should produce expected changes in output."""
        junior = await evaluator.process("Entry-level developer, 0 years experience.")
        senior = await evaluator.process("Principal engineer, 15 years experience leading teams.")

        j_level = junior.evaluation.candidate_level if junior.evaluation else ""
        s_level = senior.evaluation.candidate_level if senior.evaluation else ""

        if j_level and s_level:
            scores = {"Junior": 1, "Mid": 2, "Senior": 3}
            assert scores.get(s_level, 0) >= scores.get(
                j_level, 0
            ), "Regression: senior scored lower"
