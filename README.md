# AI Platform Framework

Production-grade enterprise AI application framework supporting AI Agents, RAG, multi-provider LLMs, evaluation, observability, CI/CD, and production deployment.

## Features

- **Multi-LLM Provider Support**: OpenAI, Anthropic, Azure OpenAI, Gemini, Ollama, vLLM
- **Agent Framework**: Extensible BaseAgent with memory, tools, prompt rendering, structured output
- **Prompt Versioning**: YAML-based prompts with version tracking
- **RAG**: Document chunking, embedding, Qdrant vector store integration
- **Structured Output**: Pydantic-validated responses with schema enforcement
- **Security**: Prompt injection detection, rate limiting, output filtering
- **Observability**: OpenTelemetry tracing, Prometheus metrics, Grafana dashboards
- **Evaluation**: Golden datasets, automated scoring, report generation
- **Testing**: Full test pyramid (unit, contract, integration, E2E, security, performance)
- **CI/CD**: GitHub Actions pipelines for PR, nightly, and release workflows
- **Docker**: Multi-stage Docker build with Docker Compose for all services

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn product.api.app:app --reload --port 8000

# Check health
curl http://localhost:8000/health

# Evaluate a candidate
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"transcript": "I have 8 years of Python experience...", "use_llm": false}'
```

## Docker Deployment

```bash
docker compose -f infra/docker/docker-compose.yml up -d
```

## Project Structure

```
├── product/           # Core framework
│   ├── api/           # FastAPI application
│   ├── agents/        # Agent implementations
│   ├── providers/     # LLM provider implementations
│   ├── models/        # Pydantic schemas
│   ├── services/      # Business logic
│   ├── rag/           # RAG components
│   ├── core/          # Configuration, errors, DI, logging, telemetry
│   ├── observability/ # OpenTelemetry & Prometheus
│   └── prompts/       # YAML prompt templates
├── tests/             # Test suite
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── e2e/
│   ├── security/
│   └── performance/
├── evaluation/        # Evaluation framework
│   ├── datasets/
│   ├── runners/
│   └── reports/
├── infra/             # Infrastructure
│   └── docker/
├── .github/           # CI/CD
│   └── workflows/
├── docs/              # Documentation
├── requirements.txt
└── pyproject.toml
```

## Architecture

See [Architecture Guide](docs/Architecture.md) for detailed architecture documentation.

## Documentation

- [Architecture Guide](docs/Architecture.md)
- [Developer Guide](docs/DeveloperGuide.md)
- [Testing Strategy](docs/TestingStrategy.md)
- [Evaluation Framework](docs/EvaluationFramework.md)

## Configuration

All configuration is managed via environment variables with YAML overrides. See `product/core/config.py` for all options.

```bash
# Provider selection
LLM__PROVIDER=openai
LLM__API_KEY=sk-...
LLM__MODEL_NAME=gpt-4o

# Vector DB
VECTOR_DB__URL=http://localhost:6333
```

## License

MIT