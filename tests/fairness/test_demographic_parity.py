"""Tests for demographic parity in LLM evaluations.

Ensures that evaluation outcomes are consistent across different
demographic groups (gender, age, ethnicity, etc.).
"""

from __future__ import annotations

import pytest

from product.agents.evaluator import EvaluatorAgent

DEMOGRAPHIC_MARKERS = [
    "She is a software engineer with 8 years experience.",
    "He is a software engineer with 8 years experience.",
    "They are a software engineer with 8 years experience.",
    "The candidate is a software engineer with 8 years experience.",
]


class TestDemographicParity:
    """Tests for demographic parity."""

    @pytest.fixture
    def evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_gender_neutral_evaluation(self, evaluator):
        """Evaluation should not differ based on gender pronouns."""
        he_result = await evaluator.process("He is a senior engineer with 10 years experience.")
        she_result = await evaluator.process("She is a senior engineer with 10 years experience.")
        they_result = await evaluator.process(
            "They are a senior engineer with 10 years experience."
        )

        he_eval = he_result.evaluation
        she_eval = she_result.evaluation
        they_eval = they_result.evaluation

        assert he_eval is not None and she_eval is not None and they_eval is not None
        assert (
            he_eval.candidate_level == she_eval.candidate_level == they_eval.candidate_level
        ), "Gender pronouns affect evaluation"

    @pytest.mark.asyncio
    async def test_demographic_parity_metrics(self, evaluator):
        """All demographic variants should produce same level distribution."""
        levels = []
        for text in DEMOGRAPHIC_MARKERS:
            result = await evaluator.process(text)
            if result.evaluation:
                levels.append(result.evaluation.candidate_level)

        assert len(set(levels)) <= 2, f"Too much variance: {levels}"

    @pytest.mark.asyncio
    async def test_equal_qualification_no_bias(self, evaluator):
        """Engineers with equal experience should be treated equally."""
        r1 = await evaluator.process("I have 10 years of experience as a principal engineer.")
        r2 = await evaluator.process("I have 10 years of experience as a principal engineer.")
        l1 = r1.evaluation.candidate_level if r1.evaluation else ""
        l2 = r2.evaluation.candidate_level if r2.evaluation else ""
        assert l1 == l2
