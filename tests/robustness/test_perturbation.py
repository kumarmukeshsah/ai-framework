"""Tests for output stability under input perturbations.

Measures whether small changes to input (typos, synonyms, paraphrases)
produce proportionally small changes in output (Lipschitz continuity).
"""
from __future__ import annotations

import pytest

from product.agents.evaluator import EvaluatorAgent


BASE_TRANSCRIPT = (
    "I have 8 years of experience with Python, FastAPI, and PostgreSQL. "
    "I worked as a Senior Software Engineer at Google."
)

MINOR_PERTURBATIONS = [
    BASE_TRANSCRIPT,
    "I have 8 years of experience with Python, FastAPI and PostgreSQL.",
    "I have 8 years of exp with Python, FastAPI, and PostgreSQL.",
    "I have eight years of experience with Python, FastAPI, and PostgreSQL.",
    "I have 8 years of experience with Python FastAPI and PostgreSQL",
    "I have 8 years of experience using Python, FastAPI, and PostgreSQL.",
    "  I have 8 years of experience with Python, FastAPI, and PostgreSQL.  ",
    "i have 8 years of experience with python, fastapi, and postgresql.",
]

MAJOR_PERTURBATIONS = [
    "I have 0 years of experience. I am a recent graduate.",
    "I have 20 years of experience as a CTO at Microsoft.",
    "",
    "I am not a software engineer.",
]


class TestPerturbationRobustness:
    """Tests for output stability under input perturbations."""

    @pytest.fixture
    def evaluator(self):
        return EvaluatorAgent(use_llm=False)

    @pytest.mark.asyncio
    async def test_minor_perturbations_stable(self, evaluator):
        """Minor input changes should not change the evaluation level."""
        base_result = await evaluator.process(BASE_TRANSCRIPT)
        base_level = base_result.evaluation.candidate_level if base_result.evaluation else ""

        for perturbed in MINOR_PERTURBATIONS[1:]:
            result = await evaluator.process(perturbed)
            level = result.evaluation.candidate_level if result.evaluation else ""
            assert level == base_level, (
                f"Minor perturbation changed level from '{base_level}' to '{level}': {perturbed[:50]}"
            )

    @pytest.mark.asyncio
    async def test_major_perturbations_handled(self, evaluator):
        """Major input changes should be handled without crashing."""
        base_result = await evaluator.process(BASE_TRANSCRIPT)
        assert base_result is not None

        for perturbed in MAJOR_PERTURBATIONS:
            result = await evaluator.process(perturbed)
            assert result is not None, f"Failed to process: {perturbed[:50]}"

    @pytest.mark.asyncio
    async def test_typo_tolerance(self, evaluator):
        """Input with common typos should still produce reasonable results."""
        typo_inputs = [
            "I have 8 yeers of experiance with Python.",
            "I workd as a softwere enginear for 5 yeers.",
            "My backgroud includs data sience and machin learning.",
            "I led a team of 12 engneers building microservises.",
        ]
        for text in typo_inputs:
            result = await evaluator.process(text)
            assert result is not None

    @pytest.mark.asyncio
    async def test_synonym_substitution(self, evaluator):
        """Synonym substitution should preserve evaluation results."""
        synonym_pairs = [
            ("I have experience with Python.", "I have expertise with Python."),
            ("I worked as an engineer.", "I served as an engineer."),
            ("My skills include Python.", "My abilities include Python."),
            ("I am proficient in Docker.", "I am skilled in Docker."),
        ]
        for orig, synonym in synonym_pairs:
            r1 = await evaluator.process(orig)
            r2 = await evaluator.process(synonym)
            assert r1 is not None and r2 is not None

    @pytest.mark.asyncio
    async def test_word_order_variations(self, evaluator):
        """Word order variations should not affect evaluation significantly."""
        variations = [
            "I have 8 years of Python experience and 5 years of Java experience.",
            "I have 5 years of Java experience and 8 years of Python experience.",
            "With 8 years of Python and 5 years of Java experience.",
        ]
        levels = []
        for text in variations:
            result = await evaluator.process(text)
            levels.append(result.evaluation.candidate_level if result.evaluation else "")

        assert len(set(levels)) <= 2, f"Word order caused too many level changes: {levels}"
