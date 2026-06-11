"""Relevance judge - evaluates response relevance to input."""
from __future__ import annotations

from typing import Any

from .base import BaseJudge, JudgeResult


class RelevanceJudge(BaseJudge):
    """Evaluates how relevant a generated output is to the input prompt.

    Measures keyword and concept overlap between input and output.
    """

    def __init__(self, name: str = "relevance"):
        super().__init__(name)

    async def evaluate(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        """Evaluate output relevance to input.

        Relevance is measured by:
        - Keyword overlap between input and output
        - Concept coverage (output addresses input topics)
        """
        if not actual_output:
            return JudgeResult(
                score=0.0,
                reasoning="No output to evaluate",
                feedback="Empty output has no relevance",
            )

        input_lower = input_text.lower()
        output_lower = actual_output.lower()

        # Extract significant words from input (length > 3)
        input_keywords = {w for w in input_lower.split() if len(w) > 3}
        if not input_keywords:
            return JudgeResult(
                score=0.5,
                reasoning="Input has no significant keywords",
                feedback="Cannot determine relevance from minimal input",
            )

        # Count how many input keywords appear in output
        keywords_covered = sum(1 for kw in input_keywords if kw in output_lower)
        keyword_coverage = keywords_covered / len(input_keywords) if input_keywords else 0.0

        # Output should not be empty or purely repetition
        output_words = set(output_lower.split())
        if len(output_words) < 3:
            relevance_penalty = 0.3
        else:
            relevance_penalty = 0.0

        # Higher score for covering more input concepts
        final_score = max(0.0, min(1.0, keyword_coverage - relevance_penalty))

        reasoning = (
            f"Input keywords covered: {keywords_covered}/{len(input_keywords)}, "
            f"Output length: {len(output_words)} unique words"
        )

        feedback = "Highly relevant" if final_score >= 0.7 else (
            "Moderately relevant" if final_score >= 0.4 else
            "Poorly relevant"
        )

        return JudgeResult(
            score=final_score,
            reasoning=reasoning,
            feedback=feedback,
            metadata={
                "keyword_coverage": keyword_coverage,
                "keywords_covered": keywords_covered,
                "total_keywords": len(input_keywords),
                "output_unique_words": len(output_words),
            },
        )
