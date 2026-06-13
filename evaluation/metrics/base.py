"""Base metric for evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Result from a metric evaluation."""

    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute the metric value."""
        ...
