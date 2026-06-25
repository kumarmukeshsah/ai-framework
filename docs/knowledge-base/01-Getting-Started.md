# Getting Started with AI Platform Framework

## Overview

The AI Platform Framework is a production-grade, enterprise-ready AI application framework built with Python. It provides a clean, modular architecture for building LLM-powered applications with strong emphasis on:

- **Extensibility** — Plugin-style provider architecture
- **Observability** — Built-in tracing, metrics, and logging
- **Security** — Prompt injection detection, rate limiting, output filtering
- **Testability** — Rule-based fallbacks, dependency injection, comprehensive test suite

## Prerequisites

- Python 3.10+
- pip / poetry
- (Optional) Docker & Docker Compose for infrastructure services

## Quick Start

### 1. Clone and Install

```bash
git clone <repo-url>
cd ai-framework
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your LLM API keys:
# LLM__API_KEY=sk-...
# LLM__PROVIDER=openai
```

### 3. Run the API Server

```bash
uvicorn product.api.app:app --reload --port 8000
```

### 4. Test the Health Endpoint

```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"1.0.0"}
```

## Basic Usage Examples

### Using the API (cURL)

```bash
# Evaluate a candidate
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "I have 5 years Python experience...",
    "use_llm": false
  }'

# List available prompts
curl http://localhost:8000/prompts

# Get a specific prompt version
curl "http://localhost:8000/prompts/candidate_evaluation?version=v2"
```

### Using the Agent Programmatically

```python
import asyncio
from product.agents.evaluator import EvaluatorAgent

async def main():
    agent = EvaluatorAgent()
    result = await agent.process(
        "I have 8 years of experience building Python microservices."
    )
    print(f"Score: {result.score}/10")
    print(f"Level: {result.candidate_level}")
    print(f"Recommendation: {result.recommendation}")

asyncio.run(main())
```

## Project Structure

```
├── product/                  # Main application package
│   ├── api/                  # FastAPI endpoints & middleware
│   ├── agents/               # Agent implementations
│   ├── providers/            # LLM provider interface & implementations
│   ├── models/               # Pydantic data models
│   ├── services/             # Business logic (prompt management)
│   ├── rag/                  # RAG components
│   └── core/                 # Config, DI, errors, telemetry
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── contract/             # Contract tests
│   ├── security/             # Security tests
│   ├── performance/          # Performance benchmarks
│   └── e2e/                  # End-to-end tests
├── evaluation/               # LLM evaluation framework
├── infra/                    # Docker & infrastructure
├── docs/                     # Documentation
└── pyproject.toml            # Project configuration
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **LLMProvider** | Abstract interface for LLM backends (OpenAI, Anthropic, etc.) |
| **BaseAgent** | Abstract agent with memory, tools, and optional LLM integration |
| **PromptManager** | Versioned prompt template system with YAML storage |
| **Container** | Lightweight dependency injection container |
| **Chunker** | Document chunking strategies for RAG pipelines |
| **Retriever** | Vector store interface (Qdrant, etc.) |
| **DatasetRunner** | Batch evaluation runner for golden datasets |

## Next Steps

- Read [Architecture Deep Dive](02-Architecture.md) for system design
- Read [LLM Provider System](03-LLM-Providers.md) to understand provider abstraction
- Read [Agent Framework](04-Agents.md) for agent development
- Read [API Layer](05-API.md) for endpoint documentation