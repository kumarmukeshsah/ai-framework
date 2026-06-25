# Architecture Deep Dive

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Middleware  │  │   Routes     │  │ Error Handlers   │  │
│  │▪ Security  │  │  /health     │  │▪ FrameworkError  │  │
│  │▪ RequestID │  │  /evaluate   │  │▪ ValidationError │  │
│  │▪ RateLimit │  │  /prompts    │  │▪ HTTPException   │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                  Service Layer                            │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   Agents     │  │ PromptManager  │  │    RAG       │  │
│  │▪ Evaluator   │  │▪ YAML storage │  │▪ Chunkers    │  │
│  │▪ BaseAgent   │  │▪ Versioning   │  │▪ Retrievers  │  │
│  │▪ Memory/Tools│  │▪ LRU Cache    │  │              │  │
│  └──────────────┘  └────────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│                Provider Abstraction Layer                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │              LLMProvider Interface                  │  │
│  │  generate() │ structured_generate() │ embeddings() │  │
│  │  stream()   │ count_tokens()        │ health_check()│  │
│  ├────────┬────────┬────────┬────────┬────────┬───────┤  │
│  │ OpenAI │Anthropic│Azure   │ Gemini │ vLLM   │Ollama │  │
│  └────────┴────────┴────────┴────────┴────────┴───────┘  │
├─────────────────────────────────────────────────────────┤
│                Infrastructure Layer                       │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ Observability│  │   Security     │  │  Storage     │  │
│  │▪ OpenTelemetry│  │▪ Injection    │  │▪ Qdrant      │  │
│  │▪ Prometheus  │  │  Detection    │  │▪ PostgreSQL  │  │
│  │▪ Structured  │  │▪ Rate Limiting │  │              │  │
│  │  Logging     │  │▪ Output Filter│  │              │  │
│  └──────────────┘  └────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Clean Architecture (Layered Separation)

Each layer communicates only with the layer directly below it:

```
API Layer → Service Layer → Provider Layer → Infrastructure
```

- **API Layer**: HTTP concerns only — routing, request validation, serialization
- **Service Layer**: Business logic — agents, prompt management, RAG pipelines
- **Provider Layer**: External service abstraction — LLM APIs, vector databases
- **Infrastructure Layer**: Cross-cutting concerns — monitoring, security, storage

### 2. SOLID Principles

| Principle | Implementation |
|-----------|----------------|
| **S**ingle Responsibility | Each module has one clear purpose (e.g., `PromptManager` only manages prompts) |
| **O**pen/Closed | New providers added by implementing `LLMProvider` — no existing code changes |
| **L**iskov Substitution | All provider implementations are interchangeable via the interface |
| **I**nterface Segregation | `LLMProvider` has focused methods, not a monolithic interface |
| **D**ependency Inversion | High-level code depends on `LLMProvider` abstract, not concrete implementations |

### 3. Dependency Injection (DI)

The `Container` class provides lightweight DI:

```python
container = Container()
container.register(LLMProvider, factory=lambda: OpenAIProvider(api_key="..."))
container.singleton(Settings, Settings())

@inject(container)
async def handle(request: Request, llm: LLMProvider) -> Response:
    # llm is automatically resolved from the container
    ...
```

Key features:
- **Type-based resolution**: Parameters resolved by their type annotation
- **Singleton support**: Shared instances for config, connections
- **Parent containers**: Scoped resolution hierarchies
- **Test overrides**: `container.override(LLMProvider, mock_provider)` for tests

### 4. Configuration-Driven Architecture

All behavior is controlled via layered configuration:

```
1. Default values (in code)
2. YAML config files (config/{env}.yaml)
3. Environment variables (highest priority)
```

Environment variables use `__` as nested delimiter:
- `LLM__API_KEY` → `settings.llm.api_key`
- `VECTOR_DB__URL` → `settings.vector_db.url`
- `SECURITY__RATE_LIMIT_PER_MINUTE` → `settings.security.rate_limit_per_minute`

### 5. Error Handling Hierarchy

```
FrameworkException (Base)
├── ConfigurationError
├── ProviderException
│   ├── InvalidProviderError
│   │   └── ProviderNotFoundError
│   ├── ProviderConnectionError
│   ├── ProviderAPIError
│   ├── ProviderRateLimitError
│   ├── ProviderAuthError
│   └── ProviderInitializationError
├── AgentException
│   ├── AgentToolError
│   └── AgentMemoryError
├── RAGException
│   ├── ChunkingError
│   ├── EmbeddingError
│   ├── RetrievalError
│   └── IndexingError
├── ServiceException
│   ├── PromptNotFoundError
│   └── PromptRenderError
├── APIException
│   ├── ValidationError
│   └── NotFoundError
└── SecurityException
    ├── PromptInjectionError
    ├── RateLimitExceededError
    └── UnauthorizedError
```

Each exception carries:
- `error_code`: Machine-readable string for API responses
- `http_status`: Appropriate HTTP status code
- `to_dict()`: Structured error serialization

## Request Lifecycle

```
Client Request
    │
    ▼
┌─────────────────────┐
│ SecurityMiddleware  │  → Prompt injection detection
│                     │  → Input size validation
│                     │  → Rate limit check
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ RequestIDMiddleware │  → X-Request-ID generation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FastAPI Router     │  → Route matching
│  Exception Handlers │  → Error → JSON response
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Service Layer      │  → Agent.process()
│                     │  → PromptManager.render()
│                     │  → LLMProvider.generate()
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Response           │  → Pydantic validation
│  + Metrics/Tracing  │  → Duration measurement
└─────────────────────┘