"""nDCG (Normalized Discounted Cumulative Gain) metric for retrieval evaluation."""
from __future__ import annotations

import math
from typing import Any

from .base import BaseMetric, MetricResult


class NormalizedDiscountedCumulativeGain(BaseMetric):
    """nDCG: Normalized Discounted Cumulative Gain for ranking quality."""

    def __init__(self, k: int = 5):
        super().__init__(f"ndcg@{k}")
        self.k = k

    def compute(
        self,
        relevance_scores: list[float],
        ideal_relevance_scores: list[float] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute nDCG@k.

        Args:
            relevance_scores: Relevance scores for retrieved documents (in order)
            ideal_relevance_scores: Ideal relevance scores sorted descending

        Returns:
            MetricResult with nDCG@k value
        """
        if not relevance_scores:
            return MetricResult(name=self.name, value=0.0, details={"k": self.k})

        k = min(self.k, len(relevance_scores))
        top_scores = relevance_scores[:k]

        dcg = self._dcg(top_scores)

        if ideal_relevance_scores is not None:
            ideal_top = sorted(ideal_relevance_scores, reverse=True)[:k]
        else:
            ideal_top = sorted(relevance_scores, reverse=True)[:k]

        idcg = self._dcg(ideal_top)

        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        return MetricResult(
            name=self.name,
            value=ndcg,
            details={
                "k": self.k,
                "dcg": round(dcg, 4),
                "idcg": round(idcg, 4),
                "actual_top_scores": top_scores,
                "ideal_top_scores": ideal_top,
            },
        )

    @staticmethod
    def _dcg(scores: list[float]) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = scores[0] if scores else 0.0
        for i, score in enumerate(scores[1:], start=2):
            dcg += score / math.log2(i)
        return dcg