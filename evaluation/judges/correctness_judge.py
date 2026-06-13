"""Correctness judge - evaluates factual accuracy of outputs."""

from __future__ import annotations

from typing import Any

from .base import BaseJudge, JudgeResult


class CorrectnessJudge(BaseJudge):
    """Evaluates factual accuracy of generated outputs.

    This judge compares the actual output against the expected output
    to determine factual correctness using simple matching heuristics.
    In production, this would use an LLM provider for semantic comparison.
    """

    def __init__(self, name: str = "correctness"):
        super().__init__(name)

    async def evaluate(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        """Evaluate factual correctness.

        Uses keyword overlap and exact match heuristics.
        """
        if not expected_output:
            return JudgeResult(
                score=0.5,
                reasoning="No expected output provided for comparison",
                feedback="Cannot verify correctness without reference output",
            )

        actual_lower = actual_output.lower()
        expected_lower = expected_output.lower()

        # Check for exact match
        if actual_lower == expected_lower:
            return JudgeResult(
                score=1.0,
                reasoning="Actual output exactly matches expected output",
                feedback="Perfect factual accuracy",
            )

        # Calculate word overlap (Jaccard similarity)
        actual_words = set(actual_lower.split())
        expected_words = set(expected_lower.split())

        if not expected_words:
            return JudgeResult(score=0.5, reasoning="Expected output is empty")

        intersection = actual_words & expected_words
        union = actual_words | expected_words
        overlap_score = len(intersection) / len(union) if union else 0.0

        # Check for key fact presence
        key_facts = [w for w in expected_words if len(w) > 4]
        facts_present = sum(1 for f in key_facts if f in actual_lower)
        fact_score = facts_present / len(key_facts) if key_facts else 0.5

        # Combined score (weighted)
        final_score = (overlap_score * 0.4) + (fact_score * 0.6)
        final_score = max(0.0, min(1.0, final_score))

        reasoning = (
            f"Word overlap: {overlap_score:.2f}, "
            f"Key facts present: {facts_present}/{len(key_facts)}"
        )

        feedback = (
            "High factual accuracy"
            if final_score >= 0.8
            else ("Moderate factual accuracy" if final_score >= 0.5 else "Low factual accuracy")
        )

        return JudgeResult(
            score=final_score,
            reasoning=reasoning,
            feedback=feedback,
            metadata={
                "word_overlap": overlap_score,
                "facts_present": facts_present,
                "total_facts": len(key_facts),
            },
        )
