"""Base retriever abstraction for RAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


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
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult: ...

    @abstractmethod
    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool: ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
