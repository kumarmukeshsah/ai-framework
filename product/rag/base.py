"""Base retriever abstraction for RAG."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from product.core.errors import RetrievalError


class RetrievedDocument(BaseModel):
    """A document retrieved from the vector store."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""

    documents: list[RetrievedDocument]
    query: str
    total_found: int = 0
    duration_ms: float = 0.0


class BaseRetriever(ABC):
    """Abstract base class for vector store retrievers."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        ...

    @abstractmethod
    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
