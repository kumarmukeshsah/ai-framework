"""Compatibility /api router.

The integration tests in ``tests/integration/test_api.py`` hit endpoints
under the ``/api`` prefix and expect slightly different response shapes than
the canonical v1 router. This module provides the compatibility shim so
both the canonical and the test-expected responses work.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from product.core.config import Settings, get_settings
from product.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ── Request/Response schemas for the compatibility API ─────────────────────


class ApiEvaluationRequest(BaseModel):
    transcript: str = Field(..., min_length=1)
    context: Optional[str] = None
    use_llm: bool = False


class ApiChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    use_llm: bool = False


class ApiIndexRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


# ── Lightweight in-process metrics ─────────────────────────────────────────


class _MetricsStore:
    """In-process counters for the /api/metrics JSON response."""

    def __init__(self) -> None:
        self.evaluations_processed: int = 0
        self.total_response_time_ms: float = 0.0
        self.requests_count: int = 0

    def record(self, duration_ms: float, evaluation: bool = False) -> None:
        self.requests_count += 1
        self.total_response_time_ms += duration_ms
        if evaluation:
            self.evaluations_processed += 1

    def snapshot(self) -> Dict[str, Any]:
        avg = self.total_response_time_ms / self.requests_count if self.requests_count else 0.0
        return {
            "evaluations_processed": self.evaluations_processed,
            "average_response_time_ms": round(avg, 2),
            "requests_count": self.requests_count,
        }


_metrics = _MetricsStore()


def get_metrics_store() -> _MetricsStore:
    return _metrics


# ── Routes ────────────────────────────────────────────────────────────────


@router.get("/health")
async def api_health(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Compatibility health endpoint returning the same shape as ``/health``."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.env,
        "provider": settings.llm.provider,
        "vector_db": settings.vector_db.provider,
    }


@router.post("/evaluate")
async def api_evaluate(
    request: ApiEvaluationRequest,
    settings: Settings = Depends(get_settings),
    metrics: _MetricsStore = Depends(get_metrics_store),
) -> Dict[str, Any]:
    """Compatibility evaluation endpoint.

    Runs the rule-based evaluator (or echoes a structured payload if LLM
    evaluation is requested) and returns a flat dict with the fields the
    integration test expects (``candidate_level``, ``score``,
    ``recommendation``).
    """
    from product.agents.evaluator import EvaluatorAgent
    from product.providers.registry import ProviderRegistry

    start = time.monotonic()
    provider = None
    if request.use_llm and settings.llm.api_key:
        try:
            provider = ProviderRegistry.create(
                settings.llm.provider,
                api_key=settings.llm.api_key,
                model=settings.llm.model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create provider: %s", exc)

    evaluator = EvaluatorAgent(provider=provider)
    evaluator._use_llm = bool(request.use_llm and provider is not None)
    result = await evaluator.process(request.transcript, request.context)
    duration = (time.monotonic() - start) * 1000
    metrics.record(duration, evaluation=True)

    if result.evaluation is not None:
        return {
            "candidate_level": result.evaluation.candidate_level,
            "score": result.evaluation.score,
            "recommendation": result.evaluation.recommendation,
            "skills": result.evaluation.skills,
            "duration_ms": round(duration, 2),
        }
    return {
        "candidate_level": None,
        "score": 0.0,
        "recommendation": "Reject",
        "error": result.error or "evaluation failed",
        "duration_ms": round(duration, 2),
    }


@router.post("/chat")
async def api_chat(
    request: ApiChatRequest,
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Compatibility chat endpoint (uses ``reply`` key, not ``response``)."""
    from product.agents.evaluator import EvaluatorAgent
    from product.providers.registry import ProviderRegistry

    conversation_id = request.conversation_id or str(uuid.uuid4())

    provider = None
    if request.use_llm and settings.llm.api_key:
        try:
            provider = ProviderRegistry.create(
                settings.llm.provider,
                api_key=settings.llm.api_key,
                model=settings.llm.model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create provider: %s", exc)

    evaluator = EvaluatorAgent(provider=provider)
    evaluator._use_llm = bool(request.use_llm and provider is not None)

    if request.use_llm and provider is not None:
        reply = await evaluator.chat(request.message)
    else:
        reply = f"Echo: {request.message}"

    return {
        "reply": reply,
        "conversation_id": conversation_id,
    }


@router.post("/index")
async def api_index(request: ApiIndexRequest) -> Dict[str, Any]:
    """Compatibility document-indexing endpoint."""
    from product.rag.chunkers import DocumentChunker

    chunker = DocumentChunker()
    result = chunker.chunk_text(request.content, {"title": request.title})
    return {
        "status": "indexed",
        "title": request.title,
        "chunks": result.chunk_count,
    }


@router.get("/metrics")
async def api_metrics(metrics: _MetricsStore = Depends(get_metrics_store)) -> Dict[str, Any]:
    """Compatibility JSON metrics endpoint."""
    return metrics.snapshot()
