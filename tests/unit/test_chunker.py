"""Unit tests for text chunker."""

from product.rag.chunkers import TextChunker


class TestTextChunker:
    """Test the text chunker."""

    def test_fixed_chunk(self):
        """Test fixed-size chunking."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, strategy="fixed")
        text = "A" * 200
        result = chunker.chunk(text)
        assert len(result.chunks) > 1
        assert result.chunk_count == len(result.chunks)

    def test_paragraph_chunk(self):
        """Test paragraph chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, strategy="paragraph")
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunker.chunk(text)
        assert result.chunk_count > 0

    def test_sentence_chunk(self):
        """Test sentence chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, strategy="sentence")
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = chunker.chunk(text)
        assert result.chunk_count > 0

    def test_recursive_chunk(self):
        """Test recursive chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10, strategy="recursive")
        text = "\n\n".join(["A" * 50 for _ in range(5)])
        result = chunker.chunk(text)
        assert result.chunk_count > 0

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        result = chunker.chunk("")
        assert result.chunk_count == 0
        assert len(result.chunks) == 0

    def test_small_text(self):
        """Test chunking text smaller than chunk size."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=0, strategy="fixed")
        result = chunker.chunk("Small text")
        assert result.chunk_count == 1
        assert result.chunks[0] == "Small text"

    def test_metadata(self):
        """Test chunk metadata."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0, strategy="fixed")
        result = chunker.chunk("A" * 100, metadata={"source": "test"})
        assert len(result.metadata) == result.chunk_count
        assert all(m["source"] == "test" for m in result.metadata)
        assert all("chunk_index" in m for m in result.metadata)

    def test_merge_chunks(self):
        """Test merging chunks."""
        chunker = TextChunker()
        merged = chunker.merge_chunks(["First", "Second", "Third"])
        assert merged == "First\n\nSecond\n\nThird"
