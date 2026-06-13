"""Mean Reciprocal Rank (MRR) metric for retrieval evaluation."""

from __future__ import annotations

from typing import Any

from .base import BaseMetric, MetricResult


class MeanReciprocalRank(BaseMetric):
    """MRR: Average of reciprocal ranks of first relevant document."""

    def __init__(self):
        super().__init__("mrr")

    def compute(
        self,
        relevant_ids: set[Any],
        retrieved_ids: list[Any],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute MRR.

        Args:
            relevant_ids: Set of relevant document IDs
            retrieved_ids: List of retrieved document IDs (ordered by rank)

        Returns:
            MetricResult with MRR value
        """
        if not retrieved_ids or not relevant_ids:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={"note": "No retrieved or relevant documents"},
            )

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                mrr = 1.0 / rank
                return MetricResult(
                    name=self.name,
                    value=mrr,
                    details={"first_relevant_rank": rank, "total_retrieved": len(retrieved_ids)},
                )

        return MetricResult(
            name=self.name,
            value=0.0,
            details={
                "note": "No relevant document found in results",
                "total_retrieved": len(retrieved_ids),
            },
        )
