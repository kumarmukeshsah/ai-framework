"""Recall@k metric for retrieval evaluation."""
from __future__ import annotations

from typing import Any

from .base import BaseMetric, MetricResult


class RecallAtK(BaseMetric):
    """Recall@k metric: fraction of relevant documents in top-k results."""

    def __init__(self, k: int = 5):
        super().__init__(f"recall@{k}")
        self.k = k

    def compute(
        self,
        relevant_ids: set[Any],
        retrieved_ids: list[Any],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Recall@k.

        Args:
            relevant_ids: Set of relevant document IDs
            retrieved_ids: List of retrieved document IDs (ordered by rank)

        Returns:
            MetricResult with recall@k value
        """
        if not relevant_ids:
            return MetricResult(
                name=self.name,
                value=1.0,  # No relevant docs means perfect recall
                details={"k": self.k, "relevant_count": 0, "retrieved_count": len(retrieved_ids)},
            )

        top_k = retrieved_ids[: self.k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        recall = relevant_retrieved / len(relevant_ids)

        return MetricResult(
            name=self.name,
            value=recall,
            details={
                "k": self.k,
                "relevant_in_top_k": relevant_retrieved,
                "total_relevant": len(relevant_ids),
                "top_k_retrieved": len(top_k),
            },
        )
