# Observability

## Overview

The observability stack provides comprehensive monitoring, tracing, logging, and metrics for all system components. Built on OpenTelemetry, Prometheus, and structured logging.

## Three Pillars

### 1. Tracing (OpenTelemetry)

Distributed tracing across all layers:

```python
from opentelemetry import trace
from product.core.telemetry import span, track_agent_execution

# Automatic tracing by decorator
@track_agent_execution(agent_name="EvaluatorAgent")
async def process(self, transcript: str):
    ...

# Manual span creation
with span("evaluator.parse"):
    parse_result = await self._run_parse_stage(transcript)
```

**Traced operations:**
- API requests (method, endpoint, status code)
- Agent execution (name, duration, success/failure)
- LLM calls (provider, model, tokens, latency)
- RAG retrievals (query, results count, duration)
- External service calls (vector DB, HTTP clients)

### 2. Prometheus Metrics

Available at `/metrics` endpoint:

```prometheus
# Request metrics
ai_framework_requests_total{method="POST",endpoint="/evaluate"} 42
ai_framework_request_duration_seconds{endpoint="/evaluate"} 0.15
ai_framework_requests_in_progress 3

# LLM metrics
ai_framework_llm_calls_total{provider="openai",model="gpt-4o"} 128
ai_framework_llm_tokens_total{type="prompt"} 524288
ai_framework_llm_tokens_total{type="completion"} 65536
ai_framework_llm_duration_seconds{provider="openai"} 0.85

# Agent metrics
ai_framework_agent_runs_total{agent="EvaluatorAgent",status="success"} 95
ai_framework_agent_runs_total{agent="EvaluatorAgent",status="failure"} 5

# System metrics
ai_framework_prompt_versions{name="candidate_evaluation",version="v3"} 42
```

### 3. Structured Logging (Loguru)

```python
from product.core.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing evaluation", agent="EvaluatorAgent", transcript_len=512)
logger.error("Provider failed", provider="openai", error="Connection timeout")
```

**Log format:**
```
2026-06-14 10:30:00.123 | INFO     | product.agents.evaluator:process:96 - Processing evaluation (agent=EvaluatorAgent, transcript_len=512)
2026-06-14 10:30:00.456 | ERROR   | product.providers.openai:generate:60 - Provider failed (provider=openai, error="Connection timeout")
```

## Configuration

```bash
# Observability settings
OBSERVABILITY__ENABLED=true
OBSERVABILITY__METRICS_PORT=8001
OBSERVABILITY__ENABLE_TRACE_EXPORT=false
OBSERVABILITY__TRACE_ENDPOINT=http://localhost:4318
OBSERVABILITY__SERVICE_NAME=ai-framework
OBSERVABILITY__ENVIRONMENT=production
```

## Grafana Dashboards

Pre-configured dashboards are available in `infra/docker/`:

```yaml
# infra/docker/prometheus.yml
scrape_configs:
  - job_name: 'ai-framework'
    scrape_interval: 10s
    static_configs:
      - targets: ['app:8001']
```

## Docker Compose

```yaml
# infra/docker/docker-compose.yml
services:
  app:
    build:
      context: ../..
      dockerfile: infra/docker/Dockerfile
    ports:
      - "8000:8000"
      - "8001:8001"  # Metrics

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## Key Practices

1. **Every component is instrumented**: API, agents, providers, RAG
2. **Consistent attributes**: service name, environment, component
3. **Error tracking**: All exceptions logged with stack traces
4. **Performance monitoring**: Latency histograms for all operations
5. **Token accounting**: Track prompt/completion token usage
6. **Version tracking**: Monitor prompt version adoption