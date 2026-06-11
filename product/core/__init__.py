"""AI Framework - Core module.

Provides the foundational building blocks:
- Configuration management (Pydantic-based with env + YAML support)
- Exception hierarchy with typed error codes
- Structured logging with context propagation
- Dependency injection container
- Telemetry (tracing + metrics) with decorator-driven auto-instrumentation
"""

from product.core.config import Settings, get_settings
from product.core.errors import (
    FrameworkException,
    ProviderException,
    AgentException,
    RAGException,
    ServiceException,
    APIException,
    SecurityException,
)
from product.core.logging import get_logger, setup_logging
from product.core.di import Container, inject, provider

__all__ = [
    "Settings",
    "get_settings",
    "FrameworkException",
    "ProviderException",
    "AgentException",
    "RAGException",
    "ServiceException",
    "APIException",
    "SecurityException",
    "get_logger",
    "setup_logging",
    "Container",
    "inject",
    "provider",
]
