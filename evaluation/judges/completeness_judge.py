"""Completeness judge - evaluates output completeness."""

from __future__ import annotations

from typing import Any

from .base import BaseJudge, JudgeResult


class CompletenessJudge(BaseJudge):
    """Evaluates whether the output covers all required aspects.

    Measures output length, structure, and coverage against expected output.
    """

    def __init__(self, name: str = "completeness"):
        super().__init__(name)

    async def evaluate(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        """Evaluate output completeness.

        Criteria:
        - Minimum viable length
        - Content diversity (not just repetition)
        - Coverage of expected elements (if expected_output provided)
        """
        if not actual_output:
            return JudgeResult(
                score=0.0,
                reasoning="Empty output",
                feedback="Output is empty, zero completeness",
            )

        actual_lower = actual_output.lower()
        actual_words = actual_lower.split()

        # Length sufficiency (min 10 words for a meaningful response)
        word_count = len(actual_words)
        length_score = min(1.0, word_count / 50.0)  # 50 words = full score

        # Content diversity (unique word ratio)
        unique_words = len(set(actual_words))
        diversity_ratio = unique_words / word_count if word_count > 0 else 0.0
        diversity_score = min(1.0, diversity_ratio * 2.0)  # 50% unique = full score

        # Coverage against expected output
        coverage_score = 0.5  # Neutral if no expected output
        if expected_output:
            expected_lower = expected_output.lower()
            expected_sentences = [s.strip() for s in expected_lower.split(".") if s.strip()]
            if expected_sentences:
                sentences_covered = sum(
                    1
                    for sentence in expected_sentences
                    if any(word in actual_lower for word in sentence.split() if len(word) > 3)
                )
                coverage_score = sentences_covered / len(expected_sentences)

        # Combined score (weighted)
        final_score = (length_score * 0.3) + (diversity_score * 0.3) + (coverage_score * 0.4)
        final_score = max(0.0, min(1.0, final_score))

        reasoning = (
            f"Length: {word_count} words (score: {length_score:.2f}), "
            f"Diversity: {diversity_ratio:.2%} (score: {diversity_score:.2f}), "
            f"Coverage: {coverage_score:.2f}"
        )

        feedback = (
            "Complete and comprehensive"
            if final_score >= 0.8
            else ("Partially complete" if final_score >= 0.5 else "Incomplete")
        )

        return JudgeResult(
            score=final_score,
            reasoning=reasoning,
            feedback=feedback,
            metadata={
                "word_count": word_count,
                "unique_words": unique_words,
                "diversity_ratio": diversity_ratio,
                "coverage_score": coverage_score,
            },
        )
