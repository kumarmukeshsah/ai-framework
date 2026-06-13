"""RAG pipeline components.

Provides:
- ``DocumentChunker`` — split documents into chunks.
- ``BaseRetriever`` — abstract retrieval interface.
- ``QdrantRetriever`` — Qdrant vector store integration.
"""

from product.rag.base import BaseRetriever, RetrievalResult, RetrievedDocument
from product.rag.chunkers import ChunkResult, DocumentChunker

__all__ = [
    "DocumentChunker",
    "ChunkResult",
    "BaseRetriever",
    "RetrievedDocument",
    "RetrievalResult",
]
