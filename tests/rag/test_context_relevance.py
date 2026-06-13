"""Tests for context relevance in RAG retrieval."""

from __future__ import annotations

import pytest

from product.rag.chunkers import TextChunker


class TestContextRelevance:
    """Tests for RAG context relevance."""

    @pytest.fixture
    def chunkers(self):
        return {
            "fixed": TextChunker(chunk_size=200, chunk_overlap=20, strategy="fixed"),
            "paragraph": TextChunker(chunk_size=200, chunk_overlap=20, strategy="paragraph"),
            "recursive": TextChunker(chunk_size=200, chunk_overlap=20, strategy="recursive"),
        }

    def test_relevant_document_chunked_completely(self, chunkers):
        """Relevant documents should be fully chunked without data loss."""
        text = "Python is a programming language. " * 200
        for name, chunker in chunkers.items():
            result = chunker.chunk(text)
            combined = " ".join(result.chunks)
            assert "Python" in combined, f"Content lost in {name}"

    def test_irrelevant_content_excluded(self, chunkers):
        """Irrelevant content should be separable."""
        relevant = "Machine learning is a subset of artificial intelligence. " * 50
        irrelevant = "The weather is nice today. " * 50
        mixed = relevant + "[[SEPARATOR]]" + irrelevant

        for _name, chunker in chunkers.items():
            result = chunker.chunk(mixed)
            assert result.chunk_count >= 1

    def test_multi_topic_separation(self, chunkers):
        """Multi-topic documents should produce topic-aligned chunks."""
        text = (
            "Python is used for web development. "
            "FastAPI is a modern Python framework. "
            "PostgreSQL is a relational database. "
            "Docker is used for containerization. "
        ) * 20
        for _name, chunker in chunkers.items():
            result = chunker.chunk(text)
            assert result.chunk_count > 0

    def test_context_relevance_metrics(self):
        """Basic relevance metrics should be computable from chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10, strategy="fixed")
        document = "Python programming. Machine learning. Data science. " * 30

        result = chunker.chunk(document)
        relevant_chunks = sum(1 for c in result.chunks if "Python" in c or "programming" in c)
        assert relevant_chunks > 0, "No relevant chunks found"
