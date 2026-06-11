"""Document chunking for RAG pipelines."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from product.core.errors import ChunkingError


class ChunkResult(BaseModel):
    """Result from a chunking operation."""

    chunks: List[str]
    chunk_count: int
    original_length: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunker:
    """Splits documents into chunks for embedding and indexing.

    Supports:
    - Fixed-size chunking with overlap
    - Recursive splitting on sentence/paragraph boundaries
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """Split text into chunks using recursive character splitting."""
        if not text:
            return ChunkResult(chunks=[], chunk_count=0, original_length=0, metadata=metadata or {})

        chunks = self._recursive_split(text)
        return ChunkResult(
            chunks=chunks,
            chunk_count=len(chunks),
            original_length=len(text),
            metadata=metadata or {},
        )

    def chunk_by_tokens(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """Split text by approximate token count (~4 chars per token)."""
        approx_tokens = len(text) // 4
        if approx_tokens <= self.chunk_size:
            return ChunkResult(
                chunks=[text],
                chunk_count=1,
                original_length=len(text),
                metadata=metadata or {},
            )
        return self.chunk_text(text, metadata)

    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text using separators list."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        current = text

        for sep in self.separators:
            if sep in current:
                parts = current.split(sep)
                for part in parts:
                    if part:  # non-empty
                        if len(part) <= self.chunk_size:
                            chunks.append(part)
                        else:
                            # Still too large, move to next separator
                            chunks.extend(self._split_fixed(part))
                result = []
                # Merge small chunks
                buffer = ""
                for chunk in chunks:
                    if len(buffer) + len(chunk) + len(sep) <= self.chunk_size:
                        buffer = (buffer + sep + chunk) if buffer else chunk
                    else:
                        if buffer:
                            result.append(buffer)
                        buffer = chunk
                if buffer:
                    result.append(buffer)
                return result

        # Fall back to fixed-size splitting
        return self._split_fixed(text)

    def _split_fixed(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
            if start >= len(text):
                break
        return chunks or [text]


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""
    chunk_index: int
    total_chunks: int
    chunk_size: int
    class Config:
        extra = "allow"


class TextChunkResult(BaseModel):
    """Result from TextChunker chunking."""
    chunks: List[str]
    metadata: List[Dict[str, Any]]
    chunk_count: int


class TextChunker:
    """Text chunker with configurable size, overlap, and strategy.

    Supports strategies: fixed, paragraph, sentence, recursive.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "recursive",
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> TextChunkResult:
        """Split text into chunks using the configured strategy.

        Args:
            text: The text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            TextChunkResult with chunks, metadata, and chunk_count.
        """
        if not text:
            return TextChunkResult(chunks=[], metadata=[], chunk_count=0)

        if self.strategy == "fixed":
            chunks = self._fixed_chunk(text)
        elif self.strategy == "paragraph":
            chunks = self._paragraph_chunk(text)
        elif self.strategy == "sentence":
            chunks = self._sentence_chunk(text)
        else:
            chunks = self._recursive_chunk(text)

        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            meta = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                **(metadata or {}),
            }
            chunk_metadata.append(meta)

        return TextChunkResult(chunks=chunks, metadata=chunk_metadata, chunk_count=len(chunks))

    def merge_chunks(self, chunks: List[str], separator: str = "\n\n") -> str:
        """Merge chunks back into text."""
        return separator.join(chunks)

    def _fixed_chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _paragraph_chunk(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            current_len = sum(len(p) for p in current_chunk)
            if current_len + len(para) > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
            else:
                current_chunk.append(para)
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks if chunks else [text]

    def _sentence_chunk(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            current_len = sum(len(s) for s in current_chunk)
            if current_len + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks if chunks else [text]

    def _recursive_chunk(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks: List[str] = []
        self._recursive_split(text, self.separators, 0, chunks)
        return chunks

    def _recursive_split(self, text: str, separators: List[str], depth: int, chunks: List[str]) -> None:
        if not text:
            return
        if len(text) <= self.chunk_size:
            chunks.append(text)
            return
        if depth >= len(separators):
            chunks.append(text[:self.chunk_size])
            remaining = text[self.chunk_size - self.chunk_overlap:]
            self._recursive_split(remaining, separators, depth, chunks)
            return
        separator = separators[depth]
        if not separator:
            self._recursive_split(text, separators, depth + 1, chunks)
            return
        parts = text.split(separator)
        if len(parts) == 1:
            self._recursive_split(text, separators, depth + 1, chunks)
            return
        current_chunk: List[str] = []
        current_size = 0
        for part in parts:
            part_text = part + separator if separator else part
            part_len = len(part_text)
            if current_size + part_len > self.chunk_size and current_chunk:
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)
                overlap_idx = max(0, len(current_chunk) - 2)
                overlap_text = separator.join(current_chunk[overlap_idx:]) if overlap_idx < len(current_chunk) else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text) if overlap_text else 0
            current_chunk.append(part)
            current_size += part_len
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) <= self.chunk_size:
                chunks.append(chunk_text)
            else:
                self._recursive_split(chunk_text, separators, depth + 1, chunks)
