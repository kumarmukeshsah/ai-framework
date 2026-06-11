"""RAG pipeline components.

Provides:
- ``DocumentChunker`` — split documents into chunks.
- ``BaseRetriever`` — abstract retrieval interface.
- ``QdrantRetriever`` — Qdrant vector store integration.
"""
from product.rag.chunkers import DocumentChunker, ChunkResult
from product.rag.base import BaseRetriever, RetrievedDocument, RetrievalResult

__all__ = [
    "DocumentChunker",
    "ChunkResult",
    "BaseRetriever",
    "RetrievedDocument",
    "RetrievalResult",
]
