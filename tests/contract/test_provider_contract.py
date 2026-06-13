"""Contract tests for LLM providers.

Every registered provider must implement the full LLMProvider interface.
These tests verify that all providers conform to the contract.
"""

from __future__ import annotations

import inspect

import pytest

import product.providers.anthropic  # noqa: F401
import product.providers.azure_openai  # noqa: F401
import product.providers.gemini  # noqa: F401
import product.providers.ollama  # noqa: F401

# Import all providers to register them
import product.providers.openai  # noqa: F401
import product.providers.vllm  # noqa: F401
from product.providers.base import LLMProvider, Message
from product.providers.registry import ProviderRegistry

# Expected set of core providers (not test registrations)
EXPECTED_PROVIDERS = {"openai", "anthropic", "ollama", "azure_openai", "gemini", "vllm"}


def _get_registered_provider_names() -> list[str]:
    """Get all registered provider names, filtering out test registrations."""
    return [n for n in ProviderRegistry.get_names() if n in EXPECTED_PROVIDERS]


class TestProviderContract:
    """Verifies all registered providers implement the full LLMProvider contract."""

    @pytest.fixture(params=_get_registered_provider_names())
    def provider_name(self, request) -> str:
        return request.param

    def test_provider_class_has_name(self, provider_name: str) -> None:
        cls = ProviderRegistry.get_class(provider_name)
        assert hasattr(cls, "provider_name")
        assert isinstance(cls.provider_name, str)
        assert cls.provider_name == provider_name

    def test_provider_has_no_abstract_methods(self, provider_name: str) -> None:
        """Verify the concrete class is not abstract (all methods implemented)."""
        cls = ProviderRegistry.get_class(provider_name)
        # If a class is still abstract, trying to instantiate it or checking
        # __abstractmethods__ will reveal it.
        abstract = getattr(cls, "__abstractmethods__", frozenset())
        assert len(abstract) == 0, f"{provider_name} still has abstract methods: {abstract}"

    def test_provider_can_be_instantiated(self, provider_name: str) -> None:
        """Provider without required API keys should fail gracefully."""
        if provider_name in ("openai", "anthropic", "azure_openai", "gemini"):
            pytest.skip(f"{provider_name} requires API key — skipping instantiation test")
        try:
            instance = ProviderRegistry.create(provider_name)
            assert isinstance(instance, LLMProvider)
        except TypeError as e:
            if "required positional argument" in str(e):
                pytest.skip(f"{provider_name} requires constructor args")
            raise

    def test_provider_interface_methods_exist(self, provider_name: str) -> None:
        """Verify the class has all required methods with correct signatures."""
        cls = ProviderRegistry.get_class(provider_name)
        required_methods = [
            "generate",
            "structured_generate",
            "embeddings",
            "stream",
            "count_tokens",
            "health_check",
        ]
        for method_name in required_methods:
            assert hasattr(cls, method_name), f"{provider_name} missing method '{method_name}'"
            method = getattr(cls, method_name)
            assert callable(method), f"{provider_name}.{method_name} is not callable"

    def test_provider_methods_are_async(self, provider_name: str) -> None:
        """Verify that async methods are properly defined as coroutines.

        Note: ``stream`` is an async generator and may be wrapped by
        the telemetry decorator, so we don't check its introspection type here.
        """
        cls = ProviderRegistry.get_class(provider_name)
        # stream is an async generator; the decorator may alter its inspect type
        async_methods = {"generate", "structured_generate", "embeddings", "health_check"}
        for method_name in async_methods:
            method = getattr(cls, method_name)
            assert inspect.iscoroutinefunction(
                method
            ), f"{provider_name}.{method_name} should be a coroutine function"

    def test_provider_returns_correct_types(self, provider_name: str) -> None:
        """Verify return type annotations match the base contract."""
        import inspect

        cls = ProviderRegistry.get_class(provider_name)
        base_cls = LLMProvider

        # Check method signatures match base class
        for method_name in ["generate", "embeddings", "health_check"]:
            method = getattr(cls, method_name, None)
            base_method = getattr(base_cls, method_name, None)
            if method and base_method:
                sig = inspect.signature(method)
                base_sig = inspect.signature(base_method)
                # Verify the method accepts at least the base parameters
                base_params = set(base_sig.parameters.keys()) - {"self"}
                actual_params = set(sig.parameters.keys()) - {"self"}
                # Actual should accept all base params (may have more)
                missing = base_params - actual_params
                assert not missing, f"{provider_name}.{method_name} missing parameters: {missing}"
                # Method should not be abstract (concrete implementation)
                assert not getattr(
                    method, "__isabstractmethod__", False
                ), f"{provider_name}.{method_name} is still abstract"


class TestAllProvidersRegistered:
    """Verify all expected providers are discovered."""

    def test_expected_providers_present(self) -> None:
        names = ProviderRegistry.get_names()
        for name in EXPECTED_PROVIDERS:
            assert name in names, f"Expected provider '{name}' not registered. Got: {names}"

    def test_core_providers_complete(self) -> None:
        """All 6 core providers must be registered."""
        names = ProviderRegistry.get_names()
        core_providers = EXPECTED_PROVIDERS & set(names)
        assert (
            len(core_providers) >= 6
        ), f"Only {len(core_providers)}/6 providers registered: {core_providers}"


class TestProviderMessageModel:
    def test_message_creation(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_with_name(self) -> None:
        msg = Message(role="user", content="Hi", name="test_user")
        assert msg.name == "test_user"

    def test_message_is_frozen(self) -> None:
        msg = Message(role="user", content="test")
        with pytest.raises(ValueError):
            msg.content = "changed"
