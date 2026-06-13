"""Unit tests for providers."""

import pytest

from product.core.errors import InvalidProviderError
from product.providers.anthropic import AnthropicProvider
from product.providers.base import Message
from product.providers.openai import OpenAIProvider
from product.providers.registry import ProviderFactory


class TestProviderFactory:
    """Test provider factory."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = ProviderFactory.create("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = ProviderFactory.create("anthropic", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = ProviderFactory.create("ollama")
        assert provider is not None

    def test_invalid_provider(self):
        """Test invalid provider type."""
        with pytest.raises(InvalidProviderError):
            ProviderFactory.create("invalid_provider")

    def test_openai_requires_api_key(self):
        """Test OpenAI provider requires API key."""
        with pytest.raises(InvalidProviderError):
            ProviderFactory.create("openai")

    def test_anthropic_requires_api_key(self):
        """Test Anthropic provider requires API key."""
        with pytest.raises(InvalidProviderError):
            ProviderFactory.create("anthropic")


class TestMessage:
    """Test Message model."""

    def test_message_creation(self):
        """Test creating message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_validation(self):
        """Test message validation."""
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"
