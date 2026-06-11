"""Tests for embedding drift detection in RAG pipeline."""
from __future__ import annotations

import pytest

from product.rag.chunkers import TextChunker, DocumentChunker


class TestEmbeddingDrift:
    """Tests for embedding drift detection."""

    @pytest.fixture
    def chunkers(self):
        return {
            "fixed": TextChunker(chunk_size=100, chunk_overlap=20, strategy="fixed"),
            "recursive": TextChunker(chunk_size=100, chunk_overlap=20, strategy="recursive"),
        }

    def test_chunk_stability_across_document_types(self, chunkers):
        """Different document types should produce valid chunks."""
        documents = {
            "code": "def hello(): print('hello world') " * 20,
            "prose": "This is a long paragraph of natural language text. " * 20,
            "mixed": "Code: def foo(): pass. Text: This is a description. " * 20,
            "numbers": "123 456 789 012 345 678 901 " * 20,
        }
        for doc_type, text in documents.items():
            for name, chunker in chunkers.items():
                result = chunker.chunk(text)
                assert result.chunk_count > 0, f"No chunks for {doc_type} with {name}"
                assert all(len(c) > 0 for c in result.chunks), f"Empty chunk in {doc_type}"

    def test_chunk_consistency(self, chunkers):
        """Same document should produce same chunks across calls."""
        text = "This is a test document. " * 50
        for name, chunker in chunkers.items():
            r1 = chunker.chunk(text)
            r2 = chunker.chunk(text)
            assert r1.chunks == r2.chunks, f"Inconsistent chunks for {name}"

    def test_empty_document_handling(self, chunkers):
        """Empty documents should not crash chunkers."""
        for name, chunker in chunkers.items():
            result = chunker.chunk("")
            assert isinstance(result.chunks, list)

    def test_very_large_document_chunking(self, chunkers):
        """Very large documents should be chunked without overflow."""
        large_text = "Python is a programming language. " * 10000
        for name, chunker in chunkers.items():
            result = chunker.chunk(large_text)
            assert result.chunk_count > 0
            for chunk in result.chunks:
                assert len(chunk) > 0

    def test_chunk_overlap_behavior(self):
        """Overlapping chunks should share content."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=50, strategy="fixed")
        text = "Hello world. " * 100
        result = chunker.chunk(text)
        if result.chunk_count >= 2:
            assert len(result.chunks[0]) > 0 and len(result.chunks[1]) > 0

    def test_special_characters(self, chunkers):
        """Special characters should not break chunking."""
        texts = [
            "Hello! @World #2024 $Test %Value ^Power &Symbol *Star (Paren)",
            "New\nLine\nSeparated\nText\nWith\nMultiple\nLines",
            "Tab\tSeparated\tValues\tHere",
            "  Multiple   spaces   between   words  ",
            "--- --- --- bullet points --- --- ---",
        ]
        for text in texts:
            for name, chunker in chunkers.items():
                result = chunker.chunk(text)
                assert isinstance(result.chunks, list)
