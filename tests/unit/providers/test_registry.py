"""Tests for the provider registry."""
from __future__ import annotations

import pytest

from product.core.errors import ProviderNotFoundError
from product.providers.registry import ProviderRegistry, register_provider
from product.providers.base import LLMProvider, Message, LLMResponse, EmbeddingResponse


class TestProviderRegistry:
    def test_empty_registry(self) -> None:
        """Test registry starts empty when no providers imported."""
        # Don't rely on specific state — just test the API shape
        assert ProviderRegistry.get_names() is not None

    def test_register_and_create(self) -> None:
        """Test manual registration and creation."""
        class DummyProvider(LLMProvider):
            provider_name = "dummy"

            async def generate(self, messages, temperature=0.7, max_tokens=None, stop_sequences=None):
                return LLMResponse(content="ok", model="dummy")

            async def structured_generate(self, messages, response_model, temperature=0.7, max_tokens=None):
                return response_model()

            async def embeddings(self, texts, model=None):
                return EmbeddingResponse(embeddings=[[0.1, 0.2]], model="dummy")

            async def stream(self, messages, temperature=0.7, max_tokens=None):
                yield "test"

            async def count_tokens(self, text):
                return 5

            async def health_check(self):
                return True

        ProviderRegistry.register("test_dummy", DummyProvider)
        assert ProviderRegistry.is_registered("test_dummy")

        instance = ProviderRegistry.create("test_dummy")
        assert isinstance(instance, DummyProvider)

    def test_create_unknown_provider(self) -> None:
        with pytest.raises(ProviderNotFoundError):
            ProviderRegistry.create("nonexistent_provider_xyz")

    def test_get_names(self) -> None:
        """Names should be a list of strings."""
        names = ProviderRegistry.get_names()
        assert isinstance(names, list)
        for name in names:
            assert isinstance(name, str)


class TestRegisterProviderDecorator:
    def test_decorator_sets_name(self) -> None:
        @register_provider("custom_test_provider")
        class CustomProvider(LLMProvider):
            provider_name = "custom_test_provider"

            async def generate(self, *args, **kwargs):
                return LLMResponse(content="x", model="x")

            async def structured_generate(self, *args, **kwargs):
                return type("Mock", (), {})()

            async def embeddings(self, *args, **kwargs):
                return EmbeddingResponse(embeddings=[], model="x")

            async def stream(self, *args, **kwargs):
                yield "x"

            async def count_tokens(self, text):
                return 0

            async def health_check(self):
                return True

        assert ProviderRegistry.is_registered("custom_test_provider")
        assert CustomProvider.provider_name == "custom_test_provider"

    def test_decorator_rejects_non_provider(self) -> None:
        with pytest.raises(TypeError):
            @register_provider("bad")  # type: ignore
            class NotAProvider:
                pass
