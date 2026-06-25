# Retrieval-Augmented Generation (RAG)

## Overview

The RAG system provides document chunking, embedding, and retrieval capabilities for building knowledge-augmented LLM applications.

## Architecture

```
Document → Chunker → Chunks → Embedding Model → Vector Store
                                                      │
User Query ───→ Embedding Model ───→ Vector Search ───┘
                                           │
                                     Retrieved Documents
                                           │
                                     LLM Generation
```

## Document Chunking

The `DocumentChunker` class provides multiple chunking strategies:

```python
class DocumentChunker(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
```

### Chunking Strategies

| Strategy | Method | Best For |
|----------|--------|----------|
| **Fixed** | Split by character count | Simple text, logs |
| **Paragraph** | Split on double newlines | Articles, documentation |
| **Sentence** | Split on sentence boundaries | Prose, reports |
| **Recursive** | Try multiple separators recursively | Code, mixed content |

### Usage

```python
from product.rag.chunkers import DocumentChunker

chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)

# Fixed-size chunks
chunks = chunker.chunk_text(long_text, strategy="fixed")

# Paragraph-based chunks
chunks = chunker.chunk_text(article_text, strategy="paragraph")

# Sentence-aware chunks
chunks = chunker.chunk_text(prose_text, strategy="sentence")

# Recursive (tries paragraph → sentence → fixed)
chunks = chunker.chunk_text(mixed_text, strategy="recursive")
```

## Vector Store Interface

### BaseRetriever

```python
class BaseRetriever(ABC):
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult: ...

    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool: ...

    async def delete_document(self, document_id: str) -> bool: ...

    async def health_check(self) -> bool: ...
```

### Data Models

```python
class RetrievedDocument(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    documents: list[RetrievedDocument]
    query: str
    total_found: int = 0
    duration_ms: float = 0.0
```

## Qdrant Implementation

The `QdrantRetriever` implements `BaseRetriever` using Qdrant vector database:

```python
from product.rag.retrieval.qdrant_retriever import QdrantRetriever

retriever = QdrantRetriever(
    url="http://localhost:6333",
    collection="my_docs",
    embedding_dim=1536,
)
```

### Configuration

```bash
VECTOR_DB__PROVIDER=qdrant
VECTOR_DB__URL=http://localhost:6333
VECTOR_DB__COLLECTION=ai_framework
VECTOR_DB__EMBEDDING_DIM=1536
VECTOR_DB__CHUNK_SIZE=512
VECTOR_DB__CHUNK_OVERLAP=64
VECTOR_DB__TOP_K=5
VECTOR_DB__SCORE_THRESHOLD=0.7
```

## RAG Pipeline

A complete RAG pipeline combines chunking, embedding, retrieval, and generation:

```python
async def rag_pipeline(query: str) -> str:
    # 1. Generate embedding for query
    embedding = await provider.embeddings([query])

    # 2. Retrieve relevant documents
    results = await retriever.retrieve(
        query=query,
        top_k=5,
        score_threshold=0.7,
    )

    # 3. Build context from retrieved documents
    context = "\n\n".join([
        f"[{doc.score:.2f}] {doc.content}"
        for doc in results.documents
    ])

    # 4. Generate response with context
    messages = [
        Message(role="system", content="Answer based on the provided context."),
        Message(role="user", content=f"Context:\n{context}\n\nQuery: {query}"),
    ]
    response = await provider.generate(messages)
    return response.content
```

## Testing RAG

```python
import pytest
from product.rag.chunkers import DocumentChunker

class TestChunker:
    def test_fixed_chunking(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "A " * 100
        chunks = chunker.chunk_text(text, strategy="fixed")
        assert len(chunks) > 1
        assert all(len(c) <= 50 for c in chunks)

    def test_empty_text(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("", strategy="fixed")
        assert chunks == []

    def test_paragraph_chunking(self):
        chunker = DocumentChunker(chunk_size=100)
        text = "Para1.\n\nPara2.\n\nPara3."
        chunks = chunker.chunk_text(text, strategy="paragraph")
        assert len(chunks) == 3
```

## Error Handling

| Error | When Raised |
|-------|-------------|
| `ChunkingError` | Document chunking fails |
| `EmbeddingError` | Embedding generation fails |
| `RetrievalError` | Vector search fails |
| `IndexingError` | Document indexing fails |