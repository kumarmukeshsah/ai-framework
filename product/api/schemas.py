"""API request/response models for the AI Platform."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Request to evaluate an interview transcript."""

    transcript: str = Field(..., min_length=1, description="Interview transcript")
    context: str | None = Field(default=None, description="Job context / requirements")
    use_llm: bool = Field(default=False, description="Use LLM for evaluation")
    prompt_version: str | None = Field(default=None, description="Specific prompt version")


class EvaluationResponse(BaseModel):
    """Response from an evaluation."""

    success: bool
    evaluation: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    prompt_version: str | None = None
    duration_ms: float = 0.0


class ChatRequest(BaseModel):
    """Request to chat with an agent."""

    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    use_llm: bool = Field(default=False)


class ChatResponse(BaseModel):
    """Response from a chat."""

    success: bool
    response: str | None = None
    error: str | None = None
    conversation_id: str | None = None
    duration_ms: float = 0.0


class IndexRequest(BaseModel):
    """Request to index a document."""

    document_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class IndexResponse(BaseModel):
    """Response from indexing a document."""

    success: bool
    document_id: str
    chunks_indexed: int = 0
    error: str | None = None
    duration_ms: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    provider: str | None = None
    vector_db: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
