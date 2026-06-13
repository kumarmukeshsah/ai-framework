"""Qdrant vector store retriever implementation."""

from __future__ import annotations

import time
from typing import Any

from product.core.errors import IndexingError, RetrievalError
from product.core.logging import get_logger

logger = get_logger(__name__)


class QdrantRetriever:
    """Retriever backed by Qdrant vector database.

    Requires a callable embedding function to be provided at initialization.
    Without one, it uses a mock client for testing only.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = "ai_framework",
        embedding_dim: int = 1536,
        preferred: bool = True,
        embed_fn: callable | None = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.embedding_dim = embedding_dim
        self.embed_fn = embed_fn
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, api_key=self.api_key)
            logger.info(f"Connected to Qdrant at {self.url}")
        except ImportError:
            logger.warning(
                "qdrant_client not installed. Install it with: pip install qdrant-client\n"
                "Using mock client for testing — searches will return placeholder results."
            )
            self._client = _MockQdrantClient()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.url}: {e}")
            raise
        return self._client

    async def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding vector for a text string.

        Uses the configured embed_fn if available, otherwise returns
        a placeholder zero vector. For production, provide an embed_fn.
        """
        if self.embed_fn is not None:
            result = await self.embed_fn(text)
            if hasattr(result, "embeddings") and result.embeddings:
                return result.embeddings[0]
            if isinstance(result, list) and result:
                return result[0] if isinstance(result[0], (list, tuple)) else result
            raise RetrievalError(
                "Embedding function returned unexpected format. "
                "Expected object with .embeddings[0] or a list of floats."
            )
        logger.warning(
            "No embed_fn provided to QdrantRetriever. "
            "Using zero-vector placeholder — search results will be meaningless. "
            "Provide embed_fn via constructor for real embeddings."
        )
        return [0.0] * self.embedding_dim

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> dict:
        """Retrieve documents similar to the query.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            score_threshold: Minimum similarity score threshold.

        Returns:
            A dict with keys: documents, query, total_found, duration_ms.
        """
        start = time.monotonic()
        try:
            client = self._get_client()
            query_vector = await self._compute_embedding(query)

            search_result = client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
            documents = [
                {
                    "id": str(p.id),
                    "content": p.payload.get("text", ""),
                    "score": p.score,
                    "metadata": p.payload,
                }
                for p in search_result
            ]
            duration = (time.monotonic() - start) * 1000
            logger.info(f"Retrieved {len(documents)} documents in {duration:.0f}ms")
            return {
                "documents": documents,
                "query": query,
                "total_found": len(documents),
                "duration_ms": round(duration, 1),
            }
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Qdrant retrieval failed: {e}")

    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Index a document into Qdrant.

        Args:
            document_id: Unique document identifier.
            content: Document text content.
            metadata: Optional metadata dict.

        Returns:
            True on success, False on failure.
        """
        try:
            client = self._get_client()
            embedding = await self._compute_embedding(content)

            from qdrant_client.http import models as qdrant_models

            point = qdrant_models.PointStruct(
                id=document_id,
                vector=embedding,
                payload={"text": content, **(metadata or {})},
            )
            client.upsert(collection_name=self.collection, points=[point])
            logger.info(f"Indexed document {document_id}")
            return True
        except Exception as e:
            raise IndexingError(f"Qdrant indexing failed: {e}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from Qdrant.

        Args:
            document_id: Document identifier to delete.

        Returns:
            True on success, False on failure.
        """
        try:
            client = self._get_client()
            client.delete(collection_name=self.collection, points_selector=[document_id])
            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable.

        Returns:
            True if Qdrant is healthy, False otherwise.
        """
        try:
            client = self._get_client()
            return client.get_collections() is not None
        except Exception:
            return False


class _MockQdrantClient:
    """Mock client used when qdrant_client is not installed."""

    class MockPoint:
        def __init__(self, id, score=0.5, payload=None):
            self.id = id
            self.score = score
            self.payload = payload or {"text": "Mock document content"}

    def search(self, collection_name, query_vector, limit=5, score_threshold=None):
        return [
            self.MockPoint(
                id="mock-1", score=0.9, payload={"text": "Mock result 1", "source": "mock"}
            ),
        ]

    def upsert(self, collection_name, points):
        pass

    def delete(self, collection_name, points_selector):
        pass

    def get_collections(self):
        return True
