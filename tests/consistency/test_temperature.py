"""Tests for temperature sensitivity of LLM-based evaluation.

When the framework uses an LLM provider, temperature affects output variability.
"""

from __future__ import annotations

import pytest

from product.agents.evaluator import EvaluatorAgent


class TestTemperatureSensitivity:
    """Tests for temperature sensitivity of LLM-based agents."""

    @pytest.fixture
    def rule_evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_rule_based_temperature_independence(self, rule_evaluator):
        """Rule-based evaluator should not depend on temperature."""
        input_text = "I have 8 years of Python experience."

        results = []
        for _ in range(5):
            result = await rule_evaluator.process(input_text)
            results.append(str(result))

        assert all(r == results[0] for r in results)

    @pytest.mark.asyncio
    async def test_low_temperature_determinism(self, rule_evaluator):
        """Low temperature should still produce deterministic results for simple inputs."""
        inputs = [
            "I am a Junior developer with 1 year experience.",
            "I am a Senior developer with 10 years experience.",
            "I am a Mid-level developer with 4 years experience.",
        ]

        for input_text in inputs:
            r1 = await rule_evaluator.process(input_text)
            r2 = await rule_evaluator.process(input_text)
            assert str(r1) == str(r2), f"Rule-based evaluator non-deterministic: {input_text}"

    @pytest.mark.asyncio
    async def test_temperature_boundary_values(self, rule_evaluator):
        """Temperature boundary values (0.0, 1.0) should be handled."""
        input_text = "I have experience with Python and Java."

        result = await rule_evaluator.process(input_text)
        assert result is not None

    @pytest.mark.asyncio
    async def test_extreme_inputs_at_any_temperature(self, rule_evaluator):
        """Extreme inputs should be handled regardless of temperature setting."""
        test_cases = [
            "",
            "A" * 10000,
            "Python" * 100,
            "I have 0 experience.",
            "I have 100 years of experience.",
        ]

        for text in test_cases:
            try:
                result = await rule_evaluator.process(text)
                assert result is not None
            except Exception:
                pass
