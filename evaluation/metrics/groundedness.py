"""Groundedness metric for RAG - measures if generated answer is faithful to retrieved context.

Industry-standard metric (used by RAGAS, TruLens, ARES) for evaluating
RAG (Retrieval-Augmented Generation) systems.

The metric answers the question: "Is every claim in the generated answer
supported by the retrieved context documents?"

Scoring:
- 1.0 = All claims grounded in context
- 0.0 = No claims grounded (full hallucination)

Computation:
1. Extract claims from the generated answer
2. For each claim, check if it's supported by the retrieved context
3. Compute fraction of grounded claims

The heuristic used here is N-gram overlap (term coverage) for
reproducibility in CI/CD. For higher accuracy, this can be combined
with an LLM-as-judge pass or NLI model.
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseMetric, MetricResult

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Common English stopwords - filtered out before n-gram extraction
_STOPWORDS = frozenset(
    {
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
        "very",
        "when",
        "what",
        "where",
        "which",
        "while",
        "those",
        "there",
        "here",
        "will",
        "your",
        "yours",
        "them",
        "then",
    }
)


def _split_into_claims(text: str) -> list[str]:
    """Split answer into individual factual claim sentences.

    Skips:
    - Empty / whitespace-only segments
    - Pure questions (no claim to verify)
    - Very short fragments (< 4 words)
    """
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 4 and not s.endswith("?")]


def _extract_terms(text: str) -> set[str]:
    """Extract content-bearing terms from a text (lowercase, no stopwords)."""
    return {
        w.lower().strip(".,;:!?()[]\"'")
        for w in text.split()
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }


def _claim_supported(claim: str, context_terms: set[str], context_text: str) -> tuple[bool, float]:
    """Check if a claim is supported by the context.

    Args:
        claim: The claim sentence to verify.
        context_terms: Set of terms extracted from context.
        context_text: Full context text (lowercased) for substring checks.

    Returns:
        Tuple of (is_supported, coverage_score).
        - is_supported: True if >= 50% of claim terms appear in context.
        - coverage_score: Fraction of claim terms found in context (0-1).
    """
    claim_terms = _extract_terms(claim)
    if not claim_terms:
        return True, 1.0

    claim.lower()
    found = 0
    for term in claim_terms:
        if term in context_text or term in context_terms:
            found += 1
    coverage = found / len(claim_terms) if claim_terms else 0.0
    return coverage >= 0.5, coverage


class GroundednessMetric(BaseMetric):
    """Measures faithfulness of generated answers to retrieved context.

    This is a deterministic, reproducible metric suitable for unit tests
    and CI/CD quality gates. For higher-fidelity evaluation, consider
    using an LLM-as-judge in combination with this metric.

    Args:
        name: Optional metric name override.
        coverage_threshold: Minimum fraction of claim terms that must appear
            in context for a claim to be considered "supported". Default 0.5.
    """

    def __init__(
        self,
        name: str = "groundedness",
        coverage_threshold: float = 0.5,
    ):
        super().__init__(name)
        self.coverage_threshold = coverage_threshold

    def compute(
        self,
        answer: str,
        context: str | list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute groundedness of ``answer`` against ``context``.

        Args:
            answer: The LLM-generated answer to evaluate.
            context: Either a single context string or a list of context
                documents (e.g., retrieved chunks from a RAG system).
            **kwargs: Reserved for future use (e.g., custom tokenizers).

        Returns:
            MetricResult with:
                - value: Float 0-1 (higher = more grounded)
                - details: Per-claim support, unsupported claims, coverage
        """
        # Normalize context to a single string + term set
        if isinstance(context, list):
            context_docs = [c for c in context if c]
        else:
            context_docs = [context] if context else []

        if not context_docs:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={
                    "error": "no_context_provided",
                    "message": "Cannot compute groundedness without context",
                },
            )

        context_text = " ".join(context_docs).lower()
        context_terms = _extract_terms(context_text)

        if not answer:
            # No answer = vacuously grounded (or vacuously unsupported).
            # We return 0.0 so this is treated as a failure in quality gates.
            return MetricResult(
                name=self.name,
                value=0.0,
                details={
                    "answer_empty": True,
                    "supported_claims": 0,
                    "unsupported_claims": [],
                    "total_claims": 0,
                },
            )

        claims = _split_into_claims(answer)
        if not claims:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={
                    "no_claims_extracted": True,
                    "supported_claims": 0,
                    "unsupported_claims": [],
                    "total_claims": 0,
                    "message": "No verifiable claims - vacuously grounded",
                },
            )

        supported: list[str] = []
        unsupported: list[dict[str, Any]] = []
        coverage_values: list[float] = []

        for claim in claims:
            is_supported, coverage = _claim_supported(claim, context_terms, context_text)
            coverage_values.append(coverage)
            if is_supported:
                supported.append(claim)
            else:
                unsupported.append(
                    {
                        "claim": claim,
                        "coverage": round(coverage, 3),
                    }
                )

        # Score: fraction of supported claims
        groundedness = len(supported) / len(claims)
        # Penalize lightly if average coverage is low even among "supported" claims
        avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        adjusted_score = round(groundedness * (0.7 + 0.3 * avg_coverage), 4)

        return MetricResult(
            name=self.name,
            value=adjusted_score,
            details={
                "total_claims": len(claims),
                "supported_claims": len(supported),
                "unsupported_claims": unsupported,
                "groundedness_ratio": round(groundedness, 4),
                "avg_coverage": round(avg_coverage, 3),
                "coverage_threshold": self.coverage_threshold,
                "context_doc_count": len(context_docs),
            },
        )
