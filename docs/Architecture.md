# Architecture Guide

## Overview

The AI Platform Framework is a production-grade, enterprise-ready AI application framework designed for extensibility, maintainability, and observability.

## Core Principles

1. **Clean Architecture**: Separation of concerns with distinct layers
2. **SOLID Principles**: Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion
3. **Provider Abstraction**: No application code depends directly on any provider SDK
4. **Configuration-Driven**: All behavior controlled via YAML + env vars
5. **Observability by Default**: Tracing, metrics, and logging built into every component

## Architecture Layers

```
┌─────────────────────────────────────────────┐
│                  API Layer                   │
│     FastAPI + Middleware + Exception Handlers│
├─────────────────────────────────────────────┤
│                Service Layer                 │
│     Agents  │  Prompt Manager  │  RAG       │
├─────────────────────────────────────────────┤
│              Provider Abstraction            │
│  LLMProvider Interface + Factory + 6 Impls  │
├─────────────────────────────────────────────┤
│           Infrastructure Layer               │
│   Vector DBs  │  Observability  │  Security  │
└─────────────────────────────────────────────┘
```

## Directory Structure

```
product/
├── api/          # FastAPI endpoints, DI, middleware
├── agents/       # BaseAgent + Agent implementations
├── providers/    # LLM provider interface + implementations
├── models/       # Pydantic models for structured output
├── services/     # Business logic services (prompt, etc.)
├── rag/          # RAG components (chunkers, retrieval)
├── middleware/    # Security, observability middleware
├── tools/        # Agent tool framework
└── config.py     # Configuration management
```

## Key Design Decisions

### Provider Abstraction
- `LLMProvider` interface defines `generate()`, `structured_generate()`, `embeddings()`, `stream()`, `count_tokens()`, `health_check()`
- `ProviderFactory.create_from_config()` selects provider via configuration only
- Six implementations: OpenAI, Anthropic, Azure OpenAI, Gemini, vLLM, Ollama
- Adding a new provider requires only implementing the interface

### Prompt Versioning
- Prompts stored as YAML in `product/prompts/v{1..N}/`
- `PromptManager` handles loading, caching, rendering
- Every response records `prompt_version` for traceability
- Versions are immutable - updates create new versions

### Agent Framework
- `BaseAgent` provides memory, tools, prompt rendering, structured output
- Agents can operate without LLM (rule-based) or with LLM
- `AgentResult` captures execution trace ID, duration, prompt version

### Security
- Prompt injection detection with 30+ known attack patterns
- Rate limiting per IP
- Input size validation
- Sensitive output filtering (API keys, tokens)
- All configurable via `SecurityConfig`

### Observability
- OpenTelemetry tracing for API, agent, retriever, LLM calls
- Prometheus metrics: latency, tokens, request count, prompt versions
- Grafana dashboards for visualization
- Structured logging with loguru

## Data Flow

```
Request → SecurityMiddleware → RequestIDMiddleware → Router
  → ExceptionHandler → Dependency Injection
  → Agent.run() → PromptManager.render()
  → LLMProvider.generate() / structured_generate()
  → Pydantic Validation
  → Response ← Metrics/Tracing