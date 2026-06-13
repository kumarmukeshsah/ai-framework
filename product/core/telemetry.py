"""Observability module with decorator-driven auto-instrumentation.

Provides:
- ``@track_llm_call`` — automatically measures latency, counts calls, tracks tokens.
- ``@track_agent_execution`` — wraps agent runs with timing and status.
- ``@track_api_endpoint`` — wraps API handlers with request counting.
- ``init_telemetry`` — one-time setup of OpenTelemetry + Prometheus.

All decorators are no-op when telemetry is disabled, so they can be safely
applied in any environment without overhead.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from product.core.logging import get_logger

logger = get_logger(__name__)

# ── Prometheus metrics (lazy-loaded; only registered when init_telemetry runs) ──

_METRICS_INITIALIZED = False

# Will hold references after init_telemetry()
_llm_calls_counter: Any = None
_llm_latency_histogram: Any = None
_llm_tokens_counter: Any = None
_agent_executions_counter: Any = None
_agent_latency_histogram: Any = None
_retrieval_calls_counter: Any = None
_retrieval_latency_histogram: Any = None
_api_requests_counter: Any = None
_api_latency_histogram: Any = None
_active_requests_gauge: Any = None
_prompt_versions_counter: Any = None

# OpenTelemetry
_tracer: Any = None
_meter: Any = None


def init_telemetry(
    service_name: str = "ai-framework",
    environment: str = "development",
    metrics_port: int = 8001,
    enable_trace_export: bool = False,
    trace_endpoint: str | None = None,
) -> None:
    """Initialize OpenTelemetry and Prometheus metrics.

    Safe to call multiple times — only the first call has an effect.
    """
    global _METRICS_INITIALIZED, _tracer, _meter
    global _llm_calls_counter, _llm_latency_histogram, _llm_tokens_counter
    global _agent_executions_counter, _agent_latency_histogram
    global _retrieval_calls_counter, _retrieval_latency_histogram
    global _api_requests_counter, _api_latency_histogram, _active_requests_gauge
    global _prompt_versions_counter

    if _METRICS_INITIALIZED:
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram, start_http_server

        # ── Prometheus metrics ─────────────────────────────────────────────
        _llm_calls_counter = Counter(
            "llm_calls_total",
            "Total LLM calls",
            ["provider", "model", "operation"],
        )
        _llm_latency_histogram = Histogram(
            "llm_latency_seconds",
            "LLM call latency",
            ["provider", "model"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        _llm_tokens_counter = Counter(
            "llm_tokens_total",
            "Total LLM tokens",
            ["provider", "model", "type"],
        )
        _agent_executions_counter = Counter(
            "agent_executions_total",
            "Agent executions",
            ["agent_name", "status"],
        )
        _agent_latency_histogram = Histogram(
            "agent_latency_seconds",
            "Agent execution latency",
            ["agent_name"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )
        _retrieval_calls_counter = Counter(
            "retrieval_calls_total",
            "Retrieval calls",
            ["retriever"],
        )
        _retrieval_latency_histogram = Histogram(
            "retrieval_latency_seconds",
            "Retrieval latency",
            ["retriever"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
        )
        _api_requests_counter = Counter(
            "api_requests_total",
            "API requests",
            ["endpoint", "method", "status"],
        )
        _api_latency_histogram = Histogram(
            "api_latency_seconds",
            "API request latency",
            ["endpoint", "method"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )
        _active_requests_gauge = Gauge("active_requests", "Active requests", ["endpoint"])
        _prompt_versions_counter = Counter(
            "prompt_versions_total",
            "Prompt version usage",
            ["prompt_name", "version"],
        )

        # Start Prometheus HTTP server
        try:
            start_http_server(metrics_port)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")

        # ── OpenTelemetry ──────────────────────────────────────────────────
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.environment": environment,
                }
            )
            tracer_provider = TracerProvider(resource=resource)
            if enable_trace_export and trace_endpoint:
                exporter = OTLPSpanExporter(endpoint=trace_endpoint)
                tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            _tracer = tracer_provider.get_tracer(service_name)
            trace.set_tracer_provider(tracer_provider)
            logger.info("OpenTelemetry tracing initialized")
        except ImportError:
            _tracer = None
            logger.info("OpenTelemetry not installed — tracing disabled")
        except Exception as e:
            _tracer = None
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")

        _METRICS_INITIALIZED = True
        logger.info(f"Telemetry initialized: service={service_name}, env={environment}")

    except ImportError:
        logger.info("prometheus_client not installed — metrics disabled")


# ── Decorators ─────────────────────────────────────────────────────────────


F = TypeVar("F", bound=Callable[..., Any])


def track_llm_call(provider: str = "unknown", model: str = "unknown") -> Callable[[F], F]:
    """Decorator that tracks LLM call metrics.

    Expects the wrapped async function to return an object with a
    ``tokens_used`` attribute (or an integer if it returns just tokens).

    Usage::

        @track_llm_call(provider="openai", model="gpt-4")
        async def generate(prompt: str) -> LLMResponse:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                latency = time.monotonic() - start
                if _llm_calls_counter is not None:
                    _llm_calls_counter.labels(
                        provider=provider,
                        model=model,
                        operation=func.__name__,
                    ).inc()
                    _llm_latency_histogram.labels(
                        provider=provider,
                        model=model,
                    ).observe(latency)
                tokens = getattr(result, "tokens_used", None) or getattr(
                    result, "token_count", None
                )
                if tokens and _llm_tokens_counter is not None:
                    _llm_tokens_counter.labels(
                        provider=provider,
                        model=model,
                        type="total",
                    ).inc(tokens)
                return result
            except Exception:
                latency = time.monotonic() - start
                if _llm_calls_counter is not None:
                    _llm_calls_counter.labels(
                        provider=provider,
                        model=model,
                        operation=func.__name__,
                    ).inc()
                    _llm_latency_histogram.labels(
                        provider=provider,
                        model=model,
                    ).observe(latency)
                raise

        if hasattr(wrapper, "__wrapped__"):
            wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def track_agent_execution(agent_name: str = "unknown") -> Callable[[F], F]:
    """Decorator that tracks agent execution metrics."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                latency = time.monotonic() - start
                if _agent_executions_counter is not None:
                    _agent_executions_counter.labels(
                        agent_name=agent_name,
                        status="success",
                    ).inc()
                    _agent_latency_histogram.labels(agent_name=agent_name).observe(latency)
                return result
            except Exception:
                latency = time.monotonic() - start
                if _agent_executions_counter is not None:
                    _agent_executions_counter.labels(
                        agent_name=agent_name,
                        status="error",
                    ).inc()
                    _agent_latency_histogram.labels(agent_name=agent_name).observe(latency)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def track_api_endpoint(endpoint: str, method: str = "POST") -> Callable[[F], F]:
    """Decorator that tracks API endpoint metrics."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            if _active_requests_gauge is not None:
                _active_requests_gauge.labels(endpoint=endpoint).inc()
            try:
                result = await func(*args, **kwargs)
                latency = time.monotonic() - start
                status = (
                    getattr(result, "status_code", 200) if hasattr(result, "status_code") else 200
                )
                if _api_requests_counter is not None:
                    _api_requests_counter.labels(
                        endpoint=endpoint,
                        method=method,
                        status=str(status),
                    ).inc()
                    _api_latency_histogram.labels(endpoint=endpoint, method=method).observe(latency)
                return result
            except Exception:
                latency = time.monotonic() - start
                if _api_requests_counter is not None:
                    _api_requests_counter.labels(
                        endpoint=endpoint,
                        method=method,
                        status="500",
                    ).inc()
                    _api_latency_histogram.labels(endpoint=endpoint, method=method).observe(latency)
                raise
            finally:
                if _active_requests_gauge is not None:
                    _active_requests_gauge.labels(endpoint=endpoint).dec()

        return wrapper  # type: ignore[return-value]

    return decorator


def track_prompt_version(prompt_name: str, version: str) -> None:
    """Record prompt version usage (non-decorator helper)."""
    if _prompt_versions_counter is not None:
        _prompt_versions_counter.labels(prompt_name=prompt_name, version=version).inc()


# ── Context managers ──────────────────────────────────────────────────────


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Any:
    """Context manager that creates an OpenTelemetry span.

    If telemetry is not initialized or OpenTelemetry is unavailable,
    this is a no-op.
    """
    if _tracer is not None:
        with _tracer.start_as_current_span(name, attributes=attributes):
            yield
    else:
        yield
