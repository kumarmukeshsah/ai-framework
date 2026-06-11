"""Performance benchmarks for text chunker."""
import pytest

from product.rag.chunkers import TextChunker


class TestChunkerPerformance:
    """Performance benchmarks for text chunking."""

    @pytest.fixture
    def large_text(self):
        """Generate a large text for benchmarking."""
        return " ".join(["This is sentence number " + str(i) + "." for i in range(1000)])

    def test_fixed_chunk_throughput(self, large_text, benchmark):
        """Benchmark fixed-size chunking throughput."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="fixed")

        def run():
            return chunker.chunk(large_text)

        result = benchmark(run)
        assert result.chunk_count > 0

    def test_paragraph_chunk_throughput(self, large_text, benchmark):
        """Benchmark paragraph chunking throughput."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=0, strategy="paragraph")
        paragraph_text = "\n\n".join([f"This is paragraph {i} with some content." for i in range(100)])

        def run():
            return chunker.chunk(paragraph_text)

        result = benchmark(run)
        assert result.chunk_count > 0

    def test_recursive_chunk_throughput(self, large_text, benchmark):
        """Benchmark recursive chunking throughput."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="recursive")

        def run():
            return chunker.chunk(large_text)

        result = benchmark(run)
        assert result.chunk_count > 0

    def test_sentence_chunk_throughput(self, large_text, benchmark):
        """Benchmark sentence chunking throughput."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=0, strategy="sentence")

        def run():
            return chunker.chunk(large_text)

        result = benchmark(run)
        assert result.chunk_count > 0

    def test_merge_chunks_throughput(self, benchmark):
        """Benchmark merge chunks throughput."""
        chunker = TextChunker()
        chunks = ["Chunk " + str(i) for i in range(500)]

        def run():
            return chunker.merge_chunks(chunks)

        result = benchmark(run)
        assert len(result) > 0

    def test_empty_text_throughput(self, benchmark):
        """Benchmark empty text handling."""
        chunker = TextChunker()

        def run():
            return chunker.chunk("")

        result = benchmark(run)
        assert result.chunk_count == 0
