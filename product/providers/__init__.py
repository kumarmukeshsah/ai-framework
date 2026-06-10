"""Provider abstraction layer.

Provides the ``LLMProvider`` interface and a plugin-based registry
that auto-discovers available implementations.
"""
from product.providers.base import LLMProvider, Message, LLMResponse, EmbeddingResponse
from product.providers.registry import ProviderRegistry, ProviderFactory

# Import providers so they self-register via the @register_provider decorator.
# These imports have side effects: they populate the registry.
from product.providers.openai import OpenAIProvider  # noqa: F401
from product.providers.anthropic import AnthropicProvider  # noqa: F401
from product.providers.ollama import OllamaProvider  # noqa: F401
from product.providers.azure_openai import AzureOpenAIProvider  # noqa: F401
from product.providers.gemini import GeminiProvider  # noqa: F401
from product.providers.vllm import VLLMProvider  # noqa: F401

__all__ = [
    "LLMProvider",
    "Message",
    "LLMResponse",
    "EmbeddingResponse",
    "ProviderRegistry",
    "ProviderFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "AzureOpenAIProvider",
    "GeminiProvider",
    "VLLMProvider",
]
