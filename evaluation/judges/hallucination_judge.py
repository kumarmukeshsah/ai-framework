"""Hallucination judge - detects when the LLM fabricates facts not in the source.

Hallucination detection is a critical LLM evaluation concern. This judge implements
a "groundedness" check: every claim in the output should be supported by either:
1. The original input (input_text)
2. The retrieved context (passed via context kwarg)
3. The expected/reference output (expected_output)

Score interpretation:
- 1.0 = All claims grounded in source material (no hallucination)
- 0.5 = Some claims partially grounded
- 0.0 = Output is entirely fabricated or contradicts source

This implementation uses heuristic-based claim extraction and overlap scoring
suitable for CI/CD pipelines. In production, this can be augmented with an
LLM-as-judge pass or a specialized model like HHEM or FActScore.
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseJudge, JudgeResult

# Claim boundary patterns - sentences containing verifiable factual claims
_CLAIM_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|"  # sentence boundary
    r"\n+|"  # newlines
    r"(?:^|\s)(?:-|\*|\d+\.)\s+"  # list markers
)

# Hedging / uncertainty markers - claims with these are treated as "soft" claims
_HEDGING_MARKERS = [
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "probably",
    "likely",
    "appears to",
    "seems to",
    "suggests",
    "indicates",
    "i think",
    "i believe",
    "in my opinion",
]

# Strong assertion markers - these signal definite claims that need grounding
_ASSERTION_MARKERS = [
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "will",
    "must",
    "definitely",
    "certainly",
    "always",
    "never",
]


def _extract_claims(text: str) -> list[str]:
    """Split output into individual claim-like sentences.

    Filters out:
    - Empty fragments
    - Pure questions (don't need grounding)
    - Very short fragments (likely noise)
    """
    if not text:
        return []

    raw_fragments = _CLAIM_SPLIT_PATTERN.split(text.strip())
    claims: list[str] = []
    for frag in raw_fragments:
        frag = frag.strip().strip("-*•").strip()
        # Must be at least 4 words and not end with a question mark
        if len(frag.split()) < 4:
            continue
        if frag.endswith("?"):
            continue
        claims.append(frag)
    return claims


def _is_soft_claim(claim: str) -> bool:
    """Check if a claim uses hedging language (less strict grounding required)."""
    claim_lower = claim.lower()
    return any(marker in claim_lower for marker in _HEDGING_MARKERS)


def _claim_grounded(claim: str, sources: list[str]) -> tuple[bool, float]:
    """Check if a claim is grounded in any of the source texts.

    Returns:
        Tuple of (is_grounded, coverage_score)
        - is_grounded: True if at least 40% of key terms appear in any source
        - coverage_score: 0-1 fraction of key terms found
    """
    claim_lower = claim.lower()
    # Extract key terms: words longer than 3 chars, excluding common stopwords
    stopwords = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "had",
        "has",
        "this",
        "that",
        "with",
        "from",
        "they",
        "have",
        "been",
        "their",
        "than",
        "such",
        "also",
        "into",
        "more",
        "some",
        "these",
        "would",
        "could",
        "should",
        "about",
    }
    key_terms = [
        w.strip(".,;:!?()[]\"'") for w in claim_lower.split() if len(w) > 3 and w not in stopwords
    ]
    if not key_terms:
        return True, 1.0  # Nothing to verify

    # Check coverage against any source
    sources_text = " ".join(s.lower() for s in sources if s)
    found_terms = sum(1 for term in key_terms if term in sources_text)
    coverage = found_terms / len(key_terms)

    # Grounded threshold: 40% of key terms must appear in source
    return coverage >= 0.4, coverage


class HallucinationJudge(BaseJudge):
    """Detects hallucinations by verifying output claims against source material.

    The judge requires either:
    - ``expected_output`` (reference answer)
    - ``context`` kwarg (retrieved RAG documents)

    If neither is provided, the judge cannot verify grounding and returns a
    neutral score with a warning.
    """

    # Default threshold below which a claim is considered hallucinated
    DEFAULT_HALLUCINATION_THRESHOLD = 0.5

    def __init__(
        self,
        name: str = "hallucination",
        hallucination_threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    ):
        super().__init__(name)
        self.hallucination_threshold = hallucination_threshold

    async def evaluate(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        """Evaluate whether ``actual_output`` contains hallucinated claims.

        Args:
            input_text: The original prompt/question.
            actual_output: The LLM-generated response to evaluate.
            expected_output: Optional reference answer (acts as a source).
            **kwargs: Must include ``context`` (list[str]) for RAG sources.

        Returns:
            JudgeResult with:
                - score: 1.0 = no hallucination, 0.0 = fully hallucinated
                - metadata.hallucinated_claims: list of ungrounded claims
                - metadata.soft_claims: list of hedged claims (lenient)
        """
        context: list[str] = list(kwargs.get("context", []) or [])

        # Build the list of source texts we can ground against
        sources: list[str] = []
        if expected_output:
            sources.append(expected_output)
        sources.extend(context)
        if input_text:
            sources.append(input_text)

        if not sources:
            return JudgeResult(
                score=0.5,
                reasoning=(
                    "No source material (context or expected_output) provided. "
                    "Cannot verify grounding."
                ),
                feedback=(
                    "Pass `context=[...]` or `expected_output` to enable "
                    "hallucination detection."
                ),
                metadata={"hallucinated_claims": [], "soft_claims": []},
            )

        if not actual_output:
            return JudgeResult(
                score=1.0,
                reasoning="Empty output - no claims to verify",
                feedback="No hallucination (no output)",
                metadata={"hallucinated_claims": [], "soft_claims": []},
            )

        claims = _extract_claims(actual_output)
        if not claims:
            return JudgeResult(
                score=1.0,
                reasoning="No extractable claims in output",
                feedback="No verifiable claims - vacuously grounded",
                metadata={"hallucinated_claims": [], "soft_claims": []},
            )

        hallucinated: list[dict[str, Any]] = []
        soft_claims: list[str] = []
        grounded_count = 0
        coverage_sum = 0.0

        for claim in claims:
            if _is_soft_claim(claim):
                # Hedged claims get a pass; record for transparency
                soft_claims.append(claim)
                grounded_count += 1
                coverage_sum += 1.0
                continue

            is_grounded, coverage = _claim_grounded(claim, sources)
            coverage_sum += coverage
            if is_grounded:
                grounded_count += 1
            else:
                hallucinated.append(
                    {
                        "claim": claim,
                        "coverage": round(coverage, 3),
                    }
                )

        total_verifiable = len(claims) - len(soft_claims) + len(soft_claims)
        # Groundedness ratio (1.0 = all claims grounded)
        groundedness = grounded_count / total_verifiable if total_verifiable else 1.0
        avg_coverage = coverage_sum / total_verifiable if total_verifiable else 1.0

        # Final score is groundedness, penalized by average coverage
        final_score = round(min(1.0, groundedness * (0.5 + 0.5 * avg_coverage)), 4)

        if final_score >= 0.8:
            feedback = "Output appears well-grounded in source material"
        elif final_score >= 0.5:
            feedback = "Some claims may not be fully grounded; review recommended"
        else:
            feedback = "Significant hallucination detected - output contains ungrounded claims"

        return JudgeResult(
            score=final_score,
            reasoning=(
                f"Analyzed {len(claims)} claims: "
                f"{grounded_count} grounded, {len(hallucinated)} ungrounded, "
                f"{len(soft_claims)} hedged. Avg coverage: {avg_coverage:.2f}"
            ),
            feedback=feedback,
            metadata={
                "total_claims": len(claims),
                "grounded_claims": grounded_count,
                "hallucinated_claims": hallucinated,
                "soft_claims": soft_claims,
                "avg_coverage": round(avg_coverage, 3),
                "threshold": self.hallucination_threshold,
            },
        )
