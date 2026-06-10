"""Plugin-based provider registry.

Providers self-register via the ``@register_provider`` decorator,
making discovery automatic and the factory extensible without modification.
"""
from __future__ import annotations

from typing import Dict, Optional, Type

from product.core.errors import (
    InvalidProviderError,
    ProviderInitializationError,
    ProviderNotFoundError,
)
from product.providers.base import LLMProvider

_registry: Dict[str, Type[LLMProvider]] = {}

# Providers that require an API key. Local / on-prem providers
# (ollama, vllm) do not.
_API_KEY_REQUIRED: Dict[str, bool] = {
    "openai": True,
    "anthropic": True,
    "azure_openai": True,
    "gemini": True,
    "ollama": False,
    "vllm": False,
}


def register_provider(name: str) -> callable:
    """Decorator that registers an LLM provider class under *name*.

    Usage::

        @register_provider("openai")
        class OpenAIProvider(LLMProvider):
            provider_name = "openai"
            ...
    """
    def decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        if not issubclass(cls, LLMProvider):
            raise TypeError(f"{cls.__name__} must subclass LLMProvider")
        cls.provider_name = name
        _registry[name] = cls
        return cls

    return decorator


class ProviderRegistry:
    """Registry and factory for LLM providers."""

    @staticmethod
    def get_names() -> list[str]:
        """Return all registered provider names."""
        return list(_registry.keys())

    @staticmethod
    def get_class(name: str) -> Type[LLMProvider]:
        """Get the provider class by name.

        Raises:
            ProviderNotFoundError: If the name is not registered.
        """
        cls = _registry.get(name)
        if cls is None:
            raise ProviderNotFoundError(
                f"Unknown provider '{name}'. Available: {', '.join(_registry.keys())}",
            )
        return cls

    @staticmethod
    def create(name: str, **kwargs) -> LLMProvider:
        """Create a provider instance by name.

        Args:
            name: Provider name (e.g. "openai", "anthropic").
            **kwargs: Arguments forwarded to the provider constructor.

        Returns:
            An initialized LLMProvider instance.

        Raises:
            ProviderNotFoundError: If the name is not registered.
            InvalidProviderError: If a required argument (e.g. ``api_key``)
                is missing for a cloud provider.
        """
        if name not in _registry:
            raise ProviderNotFoundError(
                f"Unknown provider '{name}'. Available: {', '.join(_registry.keys())}",
            )
        # Enforce api_key for cloud providers
        if _API_KEY_REQUIRED.get(name, False) and not kwargs.get("api_key"):
            raise InvalidProviderError(
                f"Provider '{name}' requires a non-empty 'api_key' argument",
            )
        cls = _registry[name]
        try:
            return cls(**kwargs)
        except TypeError as e:
            raise InvalidProviderError(
                f"Invalid arguments for provider '{name}': {e}",
            ) from e
        except Exception as e:  # noqa: BLE001
            raise ProviderInitializationError(
                f"Failed to initialize provider '{name}': {e}",
            ) from e

    @staticmethod
    def is_registered(name: str) -> bool:
        """Check if a provider name is registered."""
        return name in _registry

    @staticmethod
    def register(name: str, cls: Type[LLMProvider]) -> None:
        """Manually register a provider class (alternative to the decorator)."""
        if not issubclass(cls, LLMProvider):
            raise TypeError(f"{cls.__name__} must subclass LLMProvider")
        cls.provider_name = name
        _registry[name] = cls


# Alias for backward compatibility
ProviderFactory = ProviderRegistry
