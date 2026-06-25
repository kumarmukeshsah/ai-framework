# Vector Databases & Embeddings

## Overview

Vector databases are the backbone of semantic search and RAG. This document covers embeddings, similarity search, and vector database concepts in depth.

## Embeddings Fundamentals

### What are Embeddings?

Embeddings are numerical vector representations of data that capture semantic meaning:

```
"Python programming"  →  [0.12, -0.34, 0.56, ..., 0.89]  (1536 dimensions)
"Java development"    →  [0.10, -0.30, 0.52, ..., 0.85]  (similar → nearby)
"Baking a cake"       →  [0.89, 0.12, -0.45, ..., 0.23]  (different → far away)
```

### Properties of Good Embeddings

1. **Semantic proximity**: Similar meanings → nearby vectors
2. **Fixed dimensionality**: All vectors have same length
3. **Dense representations**: Most dimensions carry information
4. **Model-specific**: Each model has its own embedding space

### How Embeddings Are Generated

```python
# Framework's embedding flow
from product.providers.base import EmbeddingResponse

# 1. Text is tokenized
# 2. Passed through transformer model
# 3. Output pooled to single vector
response = await provider.embeddings(
    ["Text to embed"],
    model="text-embedding-3-small"  # optional, defaults to config
)

vector = response.embeddings[0]  # list[float], e.g. 1536 dimensions
```

## Similarity Search

### Distance Metrics

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| **Cosine Similarity** | cos(θ) = A·B / (‖A‖ × ‖B‖) | [-1, 1] | Text similarity (default) |
| **Dot Product** | A · B = Σ(ai × bi) | [-∞, ∞] | Normalized vectors |
| **Euclidean Distance** | ‖A - B‖₂ | [0, ∞] | Clustering |
| **Manhattan Distance** | ‖A - B‖₁ | [0, ∞] | Sparse vectors |

### Cosine Similarity in Practice

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = sum(ai * ai for ai in a) ** 0.5
    norm_b = sum(bi * bi for bi in b) ** 0.5
    return dot_product / (norm_a * norm_b)

# Usage in the framework
from product.rag.base import BaseRetriever

class MyRetriever(BaseRetriever):
    async def retrieve(self, query, top_k=5, **kwargs):
        query_embedding = await provider.embeddings([query])
        # Framework handles similarity calculation internally
        results = await self.vector_store.search(
            query_embedding.embeddings[0],
            limit=top_k,
        )
        return results
```

### Score Threshold

```python
results = await retriever.retrieve(
    query="Python microservices",
    top_k=5,
    score_threshold=0.7,  # Only return results with similarity > 0.7
)
```

## Vector Database Concepts

### Index Structures

Different index types for different use cases:

| Index Type | Search Speed | Memory | Accuracy | Best For |
|-----------|-------------|--------|----------|----------|
| **Flat (Brute Force)** | Slow | Low | 100% | Small datasets (< 10K) |
| **IVF (Inverted File)** | Fast | Medium | 95-99% | Large datasets |
| **HNSW (Hierarchical)** | Very Fast | High | 98-100% | Production (default) |
| **PQ (Product Quantization)** | Fast | Very Low | 85-95% | Memory-constrained |

### HNSW (Hierarchical Navigable Small World)

The most common production index:

```
Layer 3:    ●────●────●      (Coarse, long-range connections)
               ↓
Layer 2:    ●──●──●──●──●    (Medium granularity)
               ↓
Layer 1:    ●●─●●─●●─●●─●●  (Fine-grained, base layer)
```

**Parameters:**
- **M**: Number of connections per node (higher = more accurate but more memory)
- **ef_construction**: Build time vs quality tradeoff
- **ef_search**: Search time vs quality tradeoff

### IVF (Inverted File Index)

```
Cluster 1: [●●●●●]  →  centroid₁
Cluster 2: [●●●●●]  →  centroid₂    Search only nearest
Cluster 3: [●●●●●]  →  centroid₃     2-3 clusters
Cluster 4: [●●●●●]  →  centroid₄
```

## Qdrant Deep Dive

### Collection Configuration

```python
from product.rag.retrieval.qdrant_retriever import QdrantRetriever

retriever = QdrantRetriever(
    url="http://localhost:6333",
    collection="my_docs",
    embedding_dim=1536,

    # Optional: HNSW configuration
    hnsw_config={
        "m": 16,           # Connections per node
        "ef_construct": 100, # Build quality
        "full_scan_threshold": 10000,  # Threshold for full scan
    },

    # Optional: Quantization for memory reduction
    quantization_config={
        "binary": True,     # Compress to 1-bit per dimension
        "always_ram": True, # Keep in memory
    },
)
```

### Filtering

Vector search combined with metadata filtering:

```python
results = await retriever.retrieve(
    query="Python",
    top_k=10,
    filters={
        "category": "backend",
        "difficulty": {"$gte": 3},
        "date": {"$gte": "2024-01-01"},
    },
)
```

### Payload Storage

```python
await retriever.index_document(
    document_id="doc_123",
    content="Python is a versatile programming language...",
    metadata={
        "title": "Python Guide",
        "author": "John Doe",
        "category": "programming",
        "tags": ["python", "tutorial"],
        "word_count": 1500,
        "created_at": "2024-06-14",
    },
)
```

## Chunking Strategies Deep Dive

### Fixed-Size Chunking

```python
Text: "AAA...AABBB...BBCCC...CC"
Chunk 1: "AAA...AA" (512 chars)
Chunk 2: "BBB...BB" (512 chars)
Chunk 3: "CCC...CC" (512 chars)
```

**Pros:** Simple, predictable
**Cons:** May split in middle of sentences/paragraphs

### Overlapping Chunks

```python
Text: "AAA...AABBB...BBCCC...CC"
Chunk 1: "AAA...AA" (512 chars)
             overlap=64
Chunk 2:      "AABBB...BB" (512 chars)
                  overlap=64
Chunk 3:           "BBCCC...CC" (512 chars)
```

**Benefits:**
- Preserves context at boundaries
- Ensures important information isn't lost at split points
- Slightly more storage but significantly better retrieval

### Semantic Chunking

```python
class SemanticChunker:
    def chunk_by_semantic_boundary(self, text: str) -> list[str]:
        # Split by topic changes using embedding similarity
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = [sentences[0]]

        for sentence in sentences[1:]:
            # Check if sentence is semantically related to current chunk
            similarity = self._embedding_similarity(
                " ".join(current_chunk), sentence
            )
            if similarity > 0.8:
                current_chunk.append(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]

        return chunks
```

## Embedding Quality & Best Practices

### When to Re-embed

```python
# Good: Same model for indexing and querying
index_embedder = "text-embedding-3-small"
query_embedder = "text-embedding-3-small"  # Same model ✓

# Bad: Different models for indexing and querying
index_embedder = "text-embedding-3-small"
query_embedder = "text-embedding-3-large"  # Different model ✗
```

### Handling Long Documents

```python
# Strategy 1: Truncate
embedding = await provider.embeddings([text[:8000]])  # First 8K chars

# Strategy 2: Chunk and average
chunks = chunker.chunk_text(long_text, strategy="paragraph")
embeddings = await provider.embeddings(chunks)
avg_embedding = [sum(dim) / len(embeddings) for dim in zip(*embeddings)]

# Strategy 3: Use model's sliding window
# (Some models handle long text natively)
```

### Common Pitfalls

| Pitfall | Impact | Solution |
|---------|--------|----------|
| **Mixing embedding models** | Zero retrieval accuracy | Always use same model |
| **Not normalizing vectors** | Inconsistent similarity scores | Normalize to unit length |
| **Too large chunks** | Diluted semantics | Keep 256-512 tokens |
| **Too small chunks** | Missing context | Minimum 50 tokens |
| **Wrong distance metric** | Poor results | Use cosine for text |

## Performance Optimization

### Batch Embedding

```python
# Slow: one at a time
for doc in documents:
    emb = await provider.embeddings([doc])

# Fast: batch all at once
embeddings = await provider.embeddings(documents)  # Parallel processing
```

### Caching Embeddings

```python
from functools import lru_cache

class CachedRetriever(BaseRetriever):
    def __init__(self):
        self.cache = {}

    async def retrieve(self, query, **kwargs):
        # Cache by query hash
        query_hash = hash(query)
        if query_hash in self.cache:
            return self.cache[query_hash]

        result = await super().retrieve(query, **kwargs)
        self.cache[query_hash] = result
        return result
```

## Storage Requirements

| Scale | Documents | Embeddings Size | Metadata |
|-------|-----------|----------------|----------|
| Small | 1K | ~6 MB | ~1 MB |
| Medium | 100K | ~600 MB | ~100 MB |
| Large | 1M | ~6 GB | ~1 GB |
| Enterprise | 10M+ | ~60 GB+ | ~10 GB+ |

*(Assuming 1536-dim float32 embeddings)*