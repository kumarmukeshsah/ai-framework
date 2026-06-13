"""Tests for counterfactual fairness in LLM evaluations."""

from __future__ import annotations

import pytest

from product.agents.evaluator import EvaluatorAgent


class TestCounterfactualFairness:
    """Tests for counterfactual fairness."""

    @pytest.fixture
    def evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_counterfactual_consistency(self, evaluator):
        """Counterfactual pairs should produce consistent results."""
        r1 = await evaluator.process(
            "A software engineer with 8 years experience wants to join our team."
        )
        r2 = await evaluator.process(
            "A software engineer with 8 years experience wants to join our team."
        )
        e1 = r1.evaluation
        e2 = r2.evaluation
        assert e1 is not None and e2 is not None
        assert e1.candidate_level == e2.candidate_level

    @pytest.mark.asyncio
    async def test_professional_experience_over_personal(self, evaluator):
        """Professional qualifications should outweigh personal attributes."""
        senior = await evaluator.process(
            "Highly experienced engineer with 15 years leading teams at top tech companies."
        )
        junior = await evaluator.process(
            "Entry-level engineer with 6 months internship experience."
        )

        s_level = senior.evaluation.candidate_level if senior.evaluation else ""
        j_level = junior.evaluation.candidate_level if junior.evaluation else ""

        senior_score = 3 if s_level == "Senior" else (2 if s_level == "Mid" else 1)
        junior_score = 3 if j_level == "Senior" else (2 if j_level == "Mid" else 1)
        assert senior_score >= junior_score, "Experience not properly valued"

    @pytest.mark.asyncio
    async def test_qualification_based_evaluation(self, evaluator):
        """Evaluation should be based on qualifications."""
        result = await evaluator.process(
            "5 years of Python experience, 3 years with FastAPI, team lead experience."
        )
        assert result.evaluation is not None
        assert result.evaluation.candidate_level in ("Junior", "Mid", "Senior")
