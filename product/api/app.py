"""FastAPI application factory for the AI Platform."""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from product.api.api_v1_routes import router as api_v1_router
from product.api.endpoints import router
from product.api.middleware import SecurityMiddleware
from product.core.config import get_settings
from product.core.errors import FrameworkException
from product.core.logging import get_logger, setup_logging
from product.core.telemetry import init_telemetry

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add X-Request-ID and X-Response-Time-Ms headers to every response.

    The integration / e2e test suites assert that these headers are present,
    so we install a small middleware that runs around every request.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start = time.monotonic()
        # Reuse an incoming X-Request-ID if provided, otherwise mint a new one.
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance ready to serve.
    """
    settings = get_settings()
    setup_logging(level=settings.api.log_level, env=settings.env)
    init_telemetry(
        service_name=settings.observability.service_name,
        environment=settings.observability.environment,
        metrics_port=settings.observability.metrics_port,
        enable_trace_export=settings.observability.enable_trace_export,
        trace_endpoint=settings.observability.trace_endpoint,
    )

    app = FastAPI(
        title="AI Platform Framework",
        description="Production-grade AI application framework with multi-provider LLM, RAG, agents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    origins = settings.security.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Security ──────────────────────────────────────────────────────────
    app.add_middleware(
        SecurityMiddleware,
        rate_limit=settings.security.rate_limit_per_minute,
        max_input=settings.security.max_input_length,
        enable_injection=settings.security.enable_injection_detection,
        enable_filtering=settings.security.enable_output_filtering,
    )

    # ── Request context (request-id, response-time) ──────────────────────
    app.add_middleware(RequestContextMiddleware)

    # ── Exception handler ─────────────────────────────────────────────────
    @app.exception_handler(FrameworkException)
    async def framework_exception_handler(request: Request, exc: FrameworkException) -> JSONResponse:
        logger.error(f"Framework error: {exc.message}", exc_info=exc.cause)
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_SERVER_ERROR", "message": str(exc)},
        )

    # ── Include routes ────────────────────────────────────────────────────
    # Canonical routes (no prefix): /, /health, /evaluate, /chat, /index, /metrics, /prompts
    app.include_router(router)
    # Compatibility /api/* routes used by tests/integration/test_api.py
    app.include_router(api_v1_router, prefix="/api")

    logger.info(f"App created: env={settings.env}")
    return app


# Module-level app for `uvicorn api.app:app`
app = create_app()
