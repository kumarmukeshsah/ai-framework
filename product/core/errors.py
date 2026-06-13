"""Exception hierarchy for the AI Framework.

Base exception -> typed sub-exceptions -> specific errors.
All exceptions carry a structured error_code for API responses.
"""

from __future__ import annotations

from typing import Any


class FrameworkException(Exception):
    """Base exception for all framework errors."""

    error_code: str = "FRAMEWORK_ERROR"
    http_status: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        detail: Any | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message or str(self),
            "detail": self.detail,
        }


# ── Configuration ──────────────────────────────────────────────────────────


class ConfigurationError(FrameworkException):
    """Raised when configuration is invalid or missing."""

    error_code: str = "CONFIGURATION_ERROR"
    http_status: int = 500


# ── Providers ──────────────────────────────────────────────────────────────


class ProviderException(FrameworkException):
    """Base for provider errors."""

    error_code: str = "PROVIDER_ERROR"
    http_status: int = 502


class InvalidProviderError(ProviderException):
    """Raised when an invalid provider is specified.

    Used for both:
    - Unknown provider name (factory rejected the name), and
    - Provider factory was given insufficient/invalid arguments (e.g. a
      cloud provider created without an ``api_key``).
    """

    error_code: str = "INVALID_PROVIDER"
    http_status: int = 400


class ProviderNotFoundError(InvalidProviderError):
    """Raised when a requested provider is not registered.

    Inherits from ``InvalidProviderError`` so that callers using a broad
    ``except InvalidProviderError`` clause will also catch the not-found
    case (they are conceptually both "the provider name you supplied is
    not valid for this factory").
    """

    error_code: str = "PROVIDER_NOT_FOUND"
    http_status: int = 404


class ProviderConnectionError(ProviderException):
    """Raised when connection to the provider API fails."""

    error_code: str = "PROVIDER_CONNECTION_ERROR"
    http_status: int = 502


class ProviderAPIError(ProviderException):
    """Raised when the provider API returns an error."""

    error_code: str = "PROVIDER_API_ERROR"
    http_status: int = 502


class ProviderRateLimitError(ProviderException):
    """Raised when the provider rate-limits the request."""

    error_code: str = "PROVIDER_RATE_LIMITED"
    http_status: int = 429


class ProviderAuthError(ProviderException):
    """Raised when provider authentication fails."""

    error_code: str = "PROVIDER_AUTH_ERROR"
    http_status: int = 401


class ProviderInitializationError(ProviderException):
    """Raised when provider initialization fails."""

    error_code: str = "PROVIDER_INITIALIZATION_ERROR"
    http_status: int = 500


# ── Agents ─────────────────────────────────────────────────────────────────


class AgentException(FrameworkException):
    """Base for agent errors."""

    error_code: str = "AGENT_ERROR"
    http_status: int = 500


class AgentToolError(AgentException):
    """Raised when an agent tool execution fails."""

    error_code: str = "AGENT_TOOL_ERROR"
    http_status: int = 500


class AgentMemoryError(AgentException):
    """Raised when agent memory operations fail."""

    error_code: str = "AGENT_MEMORY_ERROR"
    http_status: int = 500


# ── RAG ────────────────────────────────────────────────────────────────────


class RAGException(FrameworkException):
    """Base for RAG errors."""

    error_code: str = "RAG_ERROR"
    http_status: int = 500


class ChunkingError(RAGException):
    """Raised when document chunking fails."""

    error_code: str = "CHUNKING_ERROR"
    http_status: int = 500


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""

    error_code: str = "EMBEDDING_ERROR"
    http_status: int = 502


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""

    error_code: str = "RETRIEVAL_ERROR"
    http_status: int = 502


class IndexingError(RAGException):
    """Raised when document indexing fails."""

    error_code: str = "INDEXING_ERROR"
    http_status: int = 500


# ── Services ───────────────────────────────────────────────────────────────


class ServiceException(FrameworkException):
    """Base for service-layer errors."""

    error_code: str = "SERVICE_ERROR"
    http_status: int = 500


class PromptNotFoundError(ServiceException):
    """Raised when a prompt template is not found."""

    error_code: str = "PROMPT_NOT_FOUND"
    http_status: int = 404


class PromptRenderError(ServiceException):
    """Raised when prompt template rendering fails."""

    error_code: str = "PROMPT_RENDER_ERROR"
    http_status: int = 500


# ── API ────────────────────────────────────────────────────────────────────


class APIException(FrameworkException):
    """Base for API-layer errors."""

    error_code: str = "API_ERROR"
    http_status: int = 400


class ValidationError(APIException):
    """Raised when request validation fails."""

    error_code: str = "VALIDATION_ERROR"
    http_status: int = 422


class NotFoundError(APIException):
    """Raised when a requested resource is not found."""

    error_code: str = "NOT_FOUND"
    http_status: int = 404


# ── Security ───────────────────────────────────────────────────────────────


class SecurityException(FrameworkException):
    """Base for security errors."""

    error_code: str = "SECURITY_ERROR"
    http_status: int = 403


class PromptInjectionError(SecurityException):
    """Raised when prompt injection is detected."""

    error_code: str = "PROMPT_INJECTION_DETECTED"
    http_status: int = 400


class RateLimitExceededError(SecurityException):
    """Raised when rate limit is exceeded."""

    error_code: str = "RATE_LIMIT_EXCEEDED"
    http_status: int = 429


class UnauthorizedError(SecurityException):
    """Raised when authentication/authorization fails."""

    error_code: str = "UNAUTHORIZED"
    http_status: int = 401
