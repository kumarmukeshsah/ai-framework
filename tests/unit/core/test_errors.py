"""Tests for core.errors module."""
from __future__ import annotations

from product.core.errors import (
    FrameworkException,
    ProviderException,
    ProviderNotFoundError,
    ProviderConnectionError,
    ProviderAPIError,
    AgentException,
    AgentToolError,
    RAGException,
    RetrievalError,
    ServiceException,
    PromptNotFoundError,
    APIException,
    ValidationError,
    SecurityException,
    PromptInjectionError,
    RateLimitExceededError,
)


class TestFrameworkException:
    def test_base_exception_defaults(self) -> None:
        exc = FrameworkException("something broke")
        assert exc.message == "something broke"
        assert exc.detail is None
        assert exc.cause is None
        assert exc.error_code == "FRAMEWORK_ERROR"
        assert exc.http_status == 500

    def test_base_exception_with_detail_and_cause(self) -> None:
        cause = ValueError("root cause")
        exc = FrameworkException("wrapped", detail={"key": "val"}, cause=cause)
        assert exc.message == "wrapped"
        assert exc.detail == {"key": "val"}
        assert exc.cause is cause

    def test_to_dict(self) -> None:
        exc = FrameworkException("test error", detail="more info")
        d = exc.to_dict()
        assert d["error"] == "FRAMEWORK_ERROR"
        assert d["message"] == "test error"
        assert d["detail"] == "more info"


class TestExceptionHierarchy:
    def test_provider_exception(self) -> None:
        assert issubclass(ProviderException, FrameworkException)
        assert ProviderException.error_code == "PROVIDER_ERROR"
        assert ProviderException.http_status == 502

    def test_provider_not_found(self) -> None:
        assert issubclass(ProviderNotFoundError, ProviderException)
        assert ProviderNotFoundError.http_status == 404

    def test_provider_connection_error(self) -> None:
        assert issubclass(ProviderConnectionError, ProviderException)
        exc = ProviderConnectionError("connection refused")
        assert exc.message == "connection refused"

    def test_provider_api_error(self) -> None:
        assert issubclass(ProviderAPIError, ProviderException)

    def test_agent_exception(self) -> None:
        assert issubclass(AgentException, FrameworkException)
        assert AgentException.error_code == "AGENT_ERROR"
        assert AgentException.http_status == 500

    def test_agent_tool_error(self) -> None:
        assert issubclass(AgentToolError, AgentException)
        assert AgentToolError.error_code == "AGENT_TOOL_ERROR"

    def test_rag_exception(self) -> None:
        assert issubclass(RAGException, FrameworkException)
        assert RAGException.error_code == "RAG_ERROR"

    def test_retrieval_error(self) -> None:
        assert issubclass(RetrievalError, RAGException)
        assert RetrievalError.error_code == "RETRIEVAL_ERROR"
        assert RetrievalError.http_status == 502

    def test_service_exception(self) -> None:
        assert issubclass(ServiceException, FrameworkException)
        exc = PromptNotFoundError("prompt v2 not found")
        assert exc.error_code == "PROMPT_NOT_FOUND"
        assert exc.http_status == 404

    def test_api_exception(self) -> None:
        assert issubclass(APIException, FrameworkException)
        exc = ValidationError("invalid input")
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.http_status == 422

    def test_security_exception(self) -> None:
        assert issubclass(SecurityException, FrameworkException)
        exc = PromptInjectionError("injection detected")
        assert exc.error_code == "PROMPT_INJECTION_DETECTED"
        assert exc.http_status == 400

    def test_rate_limit_exceeded(self) -> None:
        assert issubclass(RateLimitExceededError, SecurityException)
        assert RateLimitExceededError.http_status == 429