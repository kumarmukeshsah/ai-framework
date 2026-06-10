"""API request/response models for the AI Platform."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Request to evaluate an interview transcript."""

    transcript: str = Field(..., min_length=1, description="Interview transcript")
    context: Optional[str] = Field(default=None, description="Job context / requirements")
    use_llm: bool = Field(default=False, description="Use LLM for evaluation")
    prompt_version: Optional[str] = Field(default=None, description="Specific prompt version")


class EvaluationResponse(BaseModel):
    """Response from an evaluation."""

    success: bool
    evaluation: Optional[Dict[str, Any]] = None
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    prompt_version: Optional[str] = None
    duration_ms: float = 0.0


class ChatRequest(BaseModel):
    """Request to chat with an agent."""

    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    use_llm: bool = Field(default=False)


class ChatResponse(BaseModel):
    """Response from a chat."""

    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    conversation_id: Optional[str] = None
    duration_ms: float = 0.0


class IndexRequest(BaseModel):
    """Request to index a document."""

    document_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class IndexResponse(BaseModel):
    """Response from indexing a document."""

    success: bool
    document_id: str
    chunks_indexed: int = 0
    error: Optional[str] = None
    duration_ms: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    provider: Optional[str] = None
    vector_db: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str