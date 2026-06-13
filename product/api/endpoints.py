"""API endpoint definitions for the AI Platform."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from product.agents.evaluator import EvaluatorAgent
from product.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
)
from product.core.config import Settings, get_settings
from product.core.logging import get_logger
from product.core.telemetry import track_api_endpoint
from product.providers.registry import ProviderRegistry
from product.rag.chunkers import DocumentChunker
from product.services.prompt_service import PromptManager

logger = get_logger(__name__)

router = APIRouter()


# ── Dependencies ──────────────────────────────────────────────────────────


def _get_settings() -> Settings:
    return get_settings()


def _get_evaluator(settings: Settings = Depends(_get_settings)) -> EvaluatorAgent:
    provider = None
    if settings.llm.api_key:
        try:
            provider = ProviderRegistry.create(
                settings.llm.provider,
                api_key=settings.llm.api_key,
                model=settings.llm.model,
            )
        except Exception as e:
            logger.warning(f"Failed to create provider: {e}")
    return EvaluatorAgent(provider=provider)


def _get_prompt_manager(settings: Settings = Depends(_get_settings)) -> PromptManager:
    return PromptManager(prompts_dir=settings.prompts_dir)


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/", response_model=dict)
async def root() -> dict:
    return {"name": "AI Platform Framework", "version": "1.0.0", "status": "operational"}


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(_get_settings)) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.env,
        provider=settings.llm.provider,
        vector_db=settings.vector_db.provider,
    )


@router.post("/evaluate", response_model=EvaluationResponse)
@track_api_endpoint(endpoint="/evaluate", method="POST")
async def evaluate(
    request: EvaluationRequest,
    evaluator: EvaluatorAgent = Depends(_get_evaluator),
    prompt_manager: PromptManager = Depends(_get_prompt_manager),
) -> EvaluationResponse:
    """Evaluate an interview transcript using the multi-stage pipeline."""
    start = time.monotonic()
    try:
        evaluator._use_llm = request.use_llm and evaluator.provider is not None
        result = await evaluator.process(request.transcript, request.context)

        duration = (time.monotonic() - start) * 1000
        return EvaluationResponse(
            success=result.success,
            evaluation=result.evaluation.model_dump() if result.evaluation else None,
            stages=[s.model_dump() for s in result.stages],
            error=result.error,
            duration_ms=round(duration, 1),
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return EvaluationResponse(
            success=False,
            error=str(e),
            duration_ms=round((time.monotonic() - start) * 1000, 1),
        )


@router.post("/chat", response_model=ChatResponse)
@track_api_endpoint(endpoint="/chat", method="POST")
async def chat(
    request: ChatRequest,
    evaluator: EvaluatorAgent = Depends(_get_evaluator),
) -> ChatResponse:
    """Chat with the AI agent."""
    start = time.monotonic()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    try:
        if request.use_llm and evaluator.provider:
            response_text = await evaluator.chat(request.message)
        else:
            response_text = f"Echo: {request.message}"

        duration = (time.monotonic() - start) * 1000
        return ChatResponse(
            success=True,
            response=response_text,
            conversation_id=conversation_id,
            duration_ms=round(duration, 1),
        )
    except Exception as e:
        return ChatResponse(
            success=False,
            error=str(e),
            conversation_id=conversation_id,
            duration_ms=round((time.monotonic() - start) * 1000, 1),
        )


@router.post("/index", response_model=IndexResponse)
@track_api_endpoint(endpoint="/index", method="POST")
async def index_document(request: IndexRequest) -> IndexResponse:
    """Index a document for RAG retrieval."""
    start = time.monotonic()
    try:
        chunker = DocumentChunker()
        chunk_result = chunker.chunk_text(request.content, request.metadata)
        duration = (time.monotonic() - start) * 1000
        return IndexResponse(
            success=True,
            document_id=request.document_id,
            chunks_indexed=chunk_result.chunk_count,
            duration_ms=round(duration, 1),
        )
    except Exception as e:
        return IndexResponse(
            success=False,
            document_id=request.document_id,
            error=str(e),
            duration_ms=round((time.monotonic() - start) * 1000, 1),
        )


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@router.get("/prompts")
async def list_prompts(
    prompt_manager: PromptManager = Depends(_get_prompt_manager),
) -> dict:
    """List all available prompt templates."""
    return {"prompts": prompt_manager.list_prompts()}


@router.get("/prompts/{prompt_name}")
async def get_prompt(
    prompt_name: str,
    version: str | None = None,
    prompt_manager: PromptManager = Depends(_get_prompt_manager),
) -> dict:
    """Get a specific prompt template."""
    template = prompt_manager.get_prompt(prompt_name, version)
    return {
        "name": prompt_name,
        "version": template.version,
        "description": template.description,
        "system_prompt": template.system_prompt,
        "user_template": template.user_template,
    }
