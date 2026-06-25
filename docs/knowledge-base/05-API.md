# API Layer

## Overview

The API layer is built with FastAPI and provides RESTful endpoints for interacting with the platform. It includes middleware for security, observability, and request tracking.

## Application Setup

```python
# product/api/app.py
app = FastAPI(
    title="AI Framework API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware stack
app.add_middleware(SecurityMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)

# Exception handlers
app.add_exception_handler(FrameworkException, framework_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
```

## API Endpoints

### Health Check
```http
GET /health

Response 200:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "providers": {
    "llm": "openai",
    "vector_db": "qdrant"
  }
}
```

### Evaluate Candidate
```http
POST /evaluate

Request:
{
  "transcript": "string (required)",
  "context": "string (optional, job context)",
  "use_llm": "boolean (optional, default: false)",
  "prompt_version": "string (optional)"
}

Response 200:
{
  "success": true,
  "evaluation": {
    "candidate_level": "Senior",
    "score": 8.5,
    "recommendation": "Strong Hire",
    "skills": ["Python", "FastAPI", "Docker"],
    "experience_years": 8.0,
    "rubric": {
      "technical_depth": 2.5,
      "problem_solving": 2.5,
      "communication": 1.5,
      "experience_relevance": 2.0
    },
    "feedback": "Candidate is at Senior level...",
    "strengths": ["Microservices", "Leadership"],
    "weaknesses": [],
    "chain_of_thought": null
  },
  "stages": [...],
  "duration_ms": 150.0
}
```

### List Prompts
```http
GET /prompts

Response 200:
[
  {
    "name": "candidate_evaluation",
    "version": "v3",
    "description": "Evaluates candidate transcripts"
  },
  ...
]
```

### Get Prompt
```http
GET /prompts/{name}?version=v2

Response 200:
{
  "name": "candidate_evaluation",
  "version": "v3",
  "description": "...",
  "system_prompt": "...",
  "user_template": "...",
  "output_schema": {...}
}
```

### Metrics
```http
GET /metrics

Response 200:
# HELP ai_framework_requests_total Total requests
# TYPE ai_framework_requests_total counter
ai_framework_requests_total{method="POST",endpoint="/evaluate"} 42
...
```

## Middleware

### SecurityMiddleware

The security middleware handles:

1. **Prompt Injection Detection**
   ```python
   INJECTION_PATTERNS = [
       r"ignore\s+(all\s+)?(previous|above)\s+instructions",
       r"you\s+are\s+(now|free|an?\s+AI)",
       r"system\s+prompt",
       # ... 30+ patterns
   ]
   ```

2. **Input Size Validation**
   - Max input length: 32,000 characters (configurable)

3. **Rate Limiting**
   - Configurable requests per minute
   - Per-IP tracking

4. **Output Filtering**
   - Strips sensitive patterns (API keys, tokens, etc.)
   - Regex-based sanitization

### RequestIDMiddleware

- Generates `X-Request-ID` header for every request
- Passes through client-provided IDs
- Enables distributed tracing correlation

## Error Handling

All errors follow a structured format:

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid input: transcript is required",
  "detail": {
    "field": "transcript",
    "reason": "field required"
  }
}
```

### Exception Handlers

| Exception | HTTP Status | Error Code |
|-----------|-------------|------------|
| `ValidationError` | 422 | `VALIDATION_ERROR` |
| `NotFoundError` | 404 | `NOT_FOUND` |
| `ProviderConnectionError` | 502 | `PROVIDER_CONNECTION_ERROR` |
| `ProviderRateLimitError` | 429 | `PROVIDER_RATE_LIMITED` |
| `PromptInjectionError` | 400 | `PROMPT_INJECTION_DETECTED` |
| `UnauthorizedError` | 401 | `UNAUTHORIZED` |

## Dependency Injection

The API uses the DI container for service resolution:

```python
from product.core.di import Container, inject

container = Container()
container.register(EvaluatorAgent, factory=lambda: EvaluatorAgent())
container.singleton(Settings, get_settings())

@inject(container)
async def evaluate_endpoint(
    request: EvaluateRequest,
    agent: EvaluatorAgent,
    settings: Settings,
):
    ...
```

## Request Validation

Pydantic models validate all requests:

```python
class EvaluateRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=32000)
    context: str | None = Field(None, max_length=1000)
    use_llm: bool = False
    prompt_version: str | None = None