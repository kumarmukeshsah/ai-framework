# AI Framework

[![CI](https://github.com/kumarmukeshsah/ai-framework/actions/workflows/pr.yml/badge.svg)](https://github.com/kumarmukeshsah/ai-framework/actions/workflows/pr.yml)
[![Nightly Tests](https://github.com/kumarmukeshsah/ai-framework/actions/workflows/nightly.yml/badge.svg)](https://github.com/kumarmukeshsah/ai-framework/actions/workflows/nightly.yml)
[![Release](https://github.com/kumarmukeshsah/ai-framework/actions/workflows/release.yml/badge.svg)](https://github.com/kumarmukeshsah/ai-framework/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, extensible AI framework for building robust LLM-powered applications with built-in evaluation, observability, multi-provider support, and comprehensive safety testing.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Running the API Server](#running-the-api-server)
  - [Docker Deployment](#docker-deployment)
  - [Using Multi-Provider LLM Support](#using-multi-provider-llm-support)
- [Providers](#providers)
- [RAG Pipeline](#rag-pipeline)
- [Evaluation Framework](#evaluation-framework)
- [Observability](#observability)
- [Testing](#testing)
  - [Test Categories](#test-categories)
  - [Running Tests](#running-tests)
- [CI/CD Pipelines](#cicd-pipelines)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

AI Framework is a comprehensive, modular framework designed for building and deploying LLM-powered applications in production. It provides a unified interface for multiple LLM providers, a built-in RAG pipeline, an extensible evaluation framework, robust observability, and a full suite of safety and reliability tests — covering unit, integration, end-to-end, contract, performance, security, fairness, robustness, and monitoring test categories.

## Architecture

The framework follows a layered, dependency-injected architecture:

```
┌──────────────────────────────────────────────────────┐
│                    API Layer                          │
│           FastAPI + Middleware (Auth, Rate Limiting)   │
├──────────────────────────────────────────────────────┤
│                 Service Layer                         │
│      Prompt Service  │  Agent Registry  │  RAG       │
├──────────────────────────────────────────────────────┤
│                  Agent Layer                          │
│     Base Agent  │  Evaluator  │  Interview Agent     │
├──────────────────────────────────────────────────────┤
│                Provider Layer                         │
│   OpenAI │ Anthropic │ Azure │ Gemini │ Ollama │ vLLM│
├──────────────────────────────────────────────────────┤
│                  Core Layer                           │
│   Config │ DI │ Errors │ Logging │ Telemetry          │
├──────────────────────────────────────────────────────┤
│               Observability Layer                     │
│          Prometheus │ Structured Logging               │
└──────────────────────────────────────────────────────┘
```

## Key Features

- **Multi-Provider LLM Support** — Unified interface for OpenAI, Anthropic, Azure OpenAI, Google Gemini, Ollama, and vLLM with automatic failover and circuit breaker patterns.
- **Dependency Injection** — Lightweight DI container for testable, decoupled components.
- **RAG Pipeline** — Document chunking (recursive, semantic, fixed-size), Qdrant vector store integration, and context retrieval.
- **Evaluation Framework** — LLM-as-judge evaluation with correctness, relevance, completeness, hallucination, safety, and fairness judges, plus traditional metrics (Recall, MRR, NDCG, groundedness).
- **Prompt Management** — Versioned prompt templates (v1, v2, v3) with hot-reload via Prompt Service.
- **Observability** — Prometheus metrics, structured logging, telemetry, and monitoring for token budgets, output drift, and performance regressions.
- **API Layer** — FastAPI-based REST API with authentication middleware, rate limiting, request validation, and OpenAPI docs.
- **Comprehensive Testing** — Unit, integration, contract, end-to-end, performance (benchmarks), security (prompt injection, jailbreak, adversarial, data leakage), fairness (toxicity, demographic parity, counterfactual, stereotypes), robustness (perturbation, OOD detection, calibration), reliability (fallback, circuit breaker, degradation, concurrency), monitoring (drift, perf regression, token budget), and consistency (determinism, cross-provider, temperature).
- **Pre-commit Hooks** — Linting, formatting, and type checking via pre-commit.
- **Docker & Docker Compose** — Production-ready containerization with Prometheus monitoring.
- **CI/CD Pipelines** — GitHub Actions for PR checks, nightly regression tests, and automated releases.

## Project Structure

```
ai-framework/
├── product/                    # Core application code
│   ├── agents/                 # Agent abstractions and implementations
│   │   ├── base.py             # Base agent class
│   │   ├── evaluator.py        # Evaluation agent
│   │   ├── memory.py           # Agent memory management
│   │   └── tools.py            # Agent tools
│   ├── api/                    # FastAPI application and routes
│   │   ├── app.py              # Application factory
│   │   ├── api_v1_routes.py    # v1 API endpoints
│   │   ├── endpoints.py        # Endpoint definitions
│   │   ├── middleware.py        # Auth, rate limiting, logging
│   │   └── schemas.py          # Pydantic request/response models
│   ├── core/                   # Core infrastructure
│   │   ├── config.py           # Pydantic settings configuration
│   │   ├── di.py               # Dependency injection container
│   │   ├── errors.py           # Custom error hierarchy
│   │   ├── logging.py          # Structured logging setup
│   │   └── telemetry.py        # Telemetry and metrics
│   ├── models/                 # Domain models
│   │   └── candidate.py        # Candidate evaluation models
│   ├── observability/          # Monitoring and observability
│   ├── providers/              # LLM provider integrations
│   │   ├── base.py             # Abstract provider interface
│   │   ├── openai.py           # OpenAI provider
│   │   ├── anthropic.py        # Anthropic provider
│   │   ├── azure_openai.py     # Azure OpenAI provider
│   │   ├── gemini.py           # Google Gemini provider
│   │   ├── ollama.py           # Ollama provider
│   │   ├── vllm.py             # vLLM provider
│   │   └── registry.py         # Provider registry with failover
│   ├── prompts/                # Versioned prompt templates
│   │   ├── v1/
│   │   ├── v2/
│   │   └── v3/
│   ├── rag/                    # Retrieval-Augmented Generation
│   │   ├── base.py             # RAG base abstractions
│   │   ├── chunkers.py         # Document chunking strategies
│   │   └── retrieval/          # Vector store retrieval
│   │       └── qdrant_retriever.py
│   └── services/               # Business logic services
│       └── prompt_service.py   # Prompt template management
├── evaluation/                 # Evaluation framework
│   ├── datasets/               # Golden evaluation datasets
│   │   └── golden_dataset.json
│   ├── judges/                 # LLM-as-judge evaluators
│   │   ├── base.py
│   │   ├── correctness_judge.py
│   │   ├── relevance_judge.py
│   │   ├── completeness_judge.py
│   │   ├── hallucination_judge.py
│   │   ├── safety_judge.py
│   │   └── fairness_judge.py
│   ├── metrics/                # Traditional evaluation metrics
│   │   ├── base.py
│   │   ├── groundedness.py
│   │   ├── mrr.py
│   │   ├── ndcg.py
│   │   └── recall.py
│   ├── runners/                # Evaluation runners
│   │   └── dataset_runner.py
│   └── reports/                # Report generation
├── tests/                      # Comprehensive test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── e2e/                    # End-to-end tests
│   ├── contract/               # Provider contract tests
│   ├── performance/            # Benchmark tests
│   ├── security/               # Security & adversarial tests
│   ├── fairness/               # Bias and fairness tests
│   ├── robustness/             # Robustness & perturbation tests
│   ├── reliability/            # Fallback, circuit breaker, concurrency
│   ├── consistency/            # Cross-provider consistency tests
│   ├── monitoring/             # Drift and regression monitoring
│   ├── agents/                 # Agent-specific tests
│   └── conftest.py             # Shared pytest fixtures
├── infra/                      # Infrastructure
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── prometheus.yml
│   └── scripts/
├── docs/                       # Documentation
│   ├── Architecture.md
│   ├── DeveloperGuide.md
│   ├── EvaluationFramework.md
│   └── TestingStrategy.md
├── .github/workflows/          # CI/CD pipelines
│   ├── pr.yml                  # PR checks (lint, type check, test)
│   ├── nightly.yml             # Nightly regression suite
│   └── release.yml             # Automated release pipeline
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Getting Started

### Prerequisites

- **Python** 3.10+
- **Poetry** or **pip** for dependency management
- **Docker** & **Docker Compose** (for containerized deployment)
- **Qdrant** (optional, for RAG vector store)
- **Prometheus** (optional, for observability)

### Installation

```bash
# Clone the repository
git clone https://github.com/kumarmukeshsah/ai-framework.git
cd ai-framework

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys and configuration
# At minimum, set at least one provider API key:
#   OPENAI_API_KEY=your-key
#   ANTHROPIC_API_KEY=your-key
#   AZURE_OPENAI_API_KEY=your-key
#   GEMINI_API_KEY=your-key
```

See [`.env.example`](.env.example) for all available configuration options.

## Usage

### Running the API Server

```bash
# Start the FastAPI server
uvicorn product.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Docker Deployment

```bash
# Build and start all services
cd infra/docker
docker compose up --build -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

The Docker Compose setup includes:
- **API server** on port `8000`
- **Prometheus** on port `9090` for metrics collection

### Using Multi-Provider LLM Support

```python
from product.providers.registry import ProviderRegistry

# The registry automatically manages provider failover
registry = ProviderRegistry()

# Use a specific provider
response = await registry.generate(
    provider="openai",
    prompt="Evaluate this candidate...",
    model="gpt-4o"
)

# With automatic failover
response = await registry.generate_with_fallback(
    prompt="Evaluate this candidate...",
    preferred="openai",
    fallbacks=["anthropic", "azure_openai"]
)
```

## Providers

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-4o, GPT-4, GPT-3.5-turbo | ✅ Supported |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | ✅ Supported |
| **Azure OpenAI** | GPT-4o, GPT-4 | ✅ Supported |
| **Google Gemini** | Gemini 1.5 Pro, Gemini 1.5 Flash | ✅ Supported |
| **Ollama** | Llama 3, Mistral, Phi-3, etc. | ✅ Supported |
| **vLLM** | Any HuggingFace model | ✅ Supported |

All providers implement a common `BaseProvider` interface, enabling seamless switching and fallback. The `ProviderRegistry` handles provider selection, circuit breaking, and automatic degradation.

## RAG Pipeline

The framework includes a full RAG (Retrieval-Augmented Generation) pipeline:

```python
from product.rag.chunkers import Chunker
from product.rag.retrieval.qdrant_retriever import QdrantRetriever

# Chunk documents using different strategies
chunker = Chunker(strategy="recursive", chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(documents)

# Retrieve relevant context from Qdrant
retriever = QdrantRetriever(collection_name="documents")
results = retriever.retrieve(query="candidate experience", top_k=5)
```

**Chunking Strategies:**
- **Recursive** — Hierarchical text splitting with overlap
- **Semantic** — Sentence/paragraph-aware chunking
- **Fixed-size** — Uniform chunk sizes with configurable overlap

## Evaluation Framework

The evaluation framework provides both LLM-as-judge and traditional metrics:

### LLM-as-Judge Evaluators

| Judge | Purpose |
|-------|---------|
| **Correctness Judge** | Factual accuracy of outputs |
| **Relevance Judge** | Relevance to the input query |
| **Completeness Judge** | Coverage of all required aspects |
| **Hallucination Judge** | Detection of fabricated content |
| **Safety Judge** | Harmful or unsafe content detection |
| **Fairness Judge** | Bias and stereotype detection |

### Traditional Metrics

| Metric | Description |
|--------|-------------|
| **Recall** | Proportion of relevant items retrieved |
| **MRR** | Mean Reciprocal Rank |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **Groundedness** | Fact-grounding score |

### Running Evaluations

```bash
# Run the dataset evaluation runner
python -m evaluation.runners.dataset_runner --dataset evaluation/datasets/golden_dataset.json
```

See [docs/EvaluationFramework.md](docs/EvaluationFramework.md) for detailed evaluation documentation.

## Observability

The framework provides built-in observability:

- **Prometheus Metrics** — Request counts, latency histograms, error rates, token usage
- **Structured Logging** — JSON-formatted logs with request tracing
- **Telemetry** — Distributed tracing and span tracking
- **Monitoring Tests** — Output drift detection, performance regression, token budget enforcement

Prometheus configuration is included in `infra/docker/prometheus.yml`.

## Testing

### Test Categories

| Category | Description | Directory |
|----------|-------------|-----------|
| **Unit** | Component-level tests | `tests/unit/` |
| **Integration** | API and service integration | `tests/integration/` |
| **End-to-End** | Full workflow tests | `tests/e2e/` |
| **Contract** | Provider API contract tests | `tests/contract/` |
| **Performance** | Benchmarks and load tests | `tests/performance/` |
| **Security** | Prompt injection, jailbreak, adversarial, data leakage | `tests/security/` |
| **Fairness** | Toxicity, demographic parity, counterfactual, stereotypes | `tests/fairness/` |
| **Robustness** | Perturbation, OOD detection, calibration | `tests/robustness/` |
| **Reliability** | Fallback, circuit breaker, degradation, concurrency | `tests/reliability/` |
| **Consistency** | Determinism, cross-provider, temperature | `tests/consistency/` |
| **Monitoring** | Drift detection, perf regression, token budget | `tests/monitoring/` |
| **RAG** | Embedding drift, context relevance, chunking | `tests/rag/` |

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests
pytest tests/security/                # Security tests
pytest tests/fairness/                # Fairness tests
pytest tests/performance/ -s          # Performance benchmarks
pytest tests/reliability/             # Reliability tests

# Run with coverage
pytest --cov=product --cov-report=html

# Run with verbose output
pytest -v --tb=short

# Run nightly test suite (all categories)
pytest tests/ --ignore=tests/performance
```

## CI/CD Pipelines

### Pull Request Pipeline (`pr.yml`)
Triggered on every pull request:
- Code linting (ruff)
- Type checking (mypy)
- Unit and integration tests
- Security tests
- Contract tests
- Coverage reporting

### Nightly Pipeline (`nightly.yml`)
Runs nightly:
- Full test suite execution
- Performance benchmarks
- Fairness and robustness evaluations
- Security adversarial testing
- Provider consistency checks

### Release Pipeline (`release.yml`)
Triggered on tags:
- Full test suite
- Package build and publish
- Docker image build and push

## Configuration

All configuration is managed via environment variables and `pyproject.toml`. Key configuration areas:

- **Provider API Keys** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `GEMINI_API_KEY`
- **Application Settings** — Host, port, debug mode, log level
- **RAG Settings** — Qdrant URL, collection names, chunking parameters
- **Evaluation Settings** — Judge models, metric thresholds
- **Security Settings** — Rate limits, authentication tokens

See [`.env.example`](.env.example) and [docs/DeveloperGuide.md](docs/DeveloperGuide.md) for full configuration reference.

## Documentation

- [Architecture Overview](docs/Architecture.md) — System design and component relationships
- [Developer Guide](docs/DeveloperGuide.md) — Setup, development workflow, and conventions
- [Evaluation Framework](docs/EvaluationFramework.md) — Evaluation pipeline and judge documentation
- [Testing Strategy](docs/TestingStrategy.md) — Test categories, patterns, and guidelines

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure pre-commit hooks pass (`pre-commit run --all-files`)
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please read [docs/DeveloperGuide.md](docs/DeveloperGuide.md) for detailed contribution guidelines.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.