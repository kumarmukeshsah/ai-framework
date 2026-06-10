"""Base LLM provider interface.

Defines the abstract contract that all LLM providers must implement.
No application code should depend directly on any provider SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    name: Optional[str] = None

    model_config = {"frozen": True}


class LLMResponse(BaseModel):
    """Response from an LLM call."""

    content: str
    model: str
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    provider: str = "unknown"
    prompt_version: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response from an embedding call."""

    embeddings: List[List[float]]
    model: str
    tokens_used: Optional[int] = None
    provider: str = "unknown"


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement:
    - ``generate()`` — synchronous completion
    - ``structured_generate()`` — structured output validated against a Pydantic model
    - ``embeddings()`` — text-to-vector embeddings
    - ``stream()`` — streaming completion
    - ``count_tokens()`` — token counting
    - ``health_check()`` — provider health status
    """

    provider_name: str = "unknown"

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def structured_generate(
        self,
        messages: List[Message],
        response_model: type[BaseModel],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        ...

    @abstractmethod
    async def embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> EmbeddingResponse:
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...