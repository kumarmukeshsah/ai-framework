"""Tests for comparing different chunking strategies.

Evaluates trade-offs between chunking approaches for RAG:
- Fixed-size vs recursive vs paragraph vs sentence chunking
- Impact of chunk size and overlap on retrieval quality
"""
from __future__ import annotations

import pytest

from product.rag.chunkers import TextChunker


class TestChunkingStrategies:
    """Tests for comparing chunking strategies."""

    @pytest.fixture
    def strategies(self):
        return {
            "fixed": TextChunker(chunk_size=200, chunk_overlap=20, strategy="fixed"),
            "recursive": TextChunker(chunk_size=200, chunk_overlap=20, strategy="recursive"),
            "paragraph": TextChunker(chunk_size=200, chunk_overlap=20, strategy="paragraph"),
            "sentence": TextChunker(chunk_size=200, chunk_overlap=20, strategy="sentence"),
        }

    def test_all_strategies_produce_chunks(self, strategies):
        """All strategies should produce at least one chunk."""
        text = "Python is great. " * 50
        for name, chunker in strategies.items():
            result = chunker.chunk(text)
            assert result.chunk_count > 0, f"{name} produced no chunks"

    def test_chunk_count_variance(self, strategies):
        """Different strategies should produce different chunk counts."""
        text = (
            "Paragraph one contains multiple sentences. "
            "It has several ideas. And more content.\n\n"
            "Paragraph two is about different topics. "
            "It also has multiple sentences. "
            "And even more details than before.\n\n"
            "Paragraph three concludes the document. "
            "It wraps everything up."
        )
        counts = {}
        for name, chunker in strategies.items():
            result = chunker.chunk(text)
            counts[name] = result.chunk_count

        unique_counts = set(counts.values())
        assert len(unique_counts) >= 2, (
            f"All strategies produced same chunk count: {counts}"
        )

    def test_chunk_size_configuration(self):
        """Different chunk sizes should affect output."""
        text = "Hello world. " * 200
        small = TextChunker(chunk_size=50, chunk_overlap=5, strategy="fixed")
        large = TextChunker(chunk_size=500, chunk_overlap=50, strategy="fixed")

        small_chunks = small.chunk(text)
        large_chunks = large.chunk(text)

        assert small_chunks.chunk_count > large_chunks.chunk_count, (
            f"Small chunks ({small_chunks.chunk_count}) should produce more chunks than large ({large_chunks.chunk_count})"
        )

    def test_overlap_configuration(self):
        """Overlap configuration should affect adjacent chunk content."""
        text = "Python programming. Machine learning. Data science. " * 50
        no_overlap = TextChunker(chunk_size=200, chunk_overlap=0, strategy="fixed")
        with_overlap = TextChunker(chunk_size=200, chunk_overlap=50, strategy="fixed")

        chunks_no = no_overlap.chunk(text)
        chunks_yes = with_overlap.chunk(text)

        assert chunks_no.chunks != chunks_yes.chunks

    def test_empty_edge_cases(self, strategies):
        """Edge cases should be handled by all strategies."""
        edge_cases = [
            "",
            " ",
            ".",
            "a",
            "\n\n\n",
        ]
        for text in edge_cases:
            for name, chunker in strategies.items():
                result = chunker.chunk(text)
                assert isinstance(result.chunks, list), f"{name} didn't return list for: {text!r}"
