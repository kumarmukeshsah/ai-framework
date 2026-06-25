# LLM Provider System

## Overview

The provider abstraction layer decouples all application code from specific LLM SDKs. The `LLMProvider` interface defines a contract that all providers must implement, enabling seamless switching between backends via configuration.

## Interface Definition

```python
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_name: str = "unknown"

    # Core generation
    async def generate(self, messages: list[Message],
                       temperature: float = 0.7,
                       max_tokens: int | None = None,
                       stop_sequences: list[str] | None = None) -> LLMResponse

    # Structured output (Pydantic model validation)
    async def structured_generate(self, messages: list[Message],
                                   response_model: type[BaseModel],
                                   temperature: float = 0.7,
                                   max_tokens: int | None = None) -> BaseModel

    # Embeddings
    async def embeddings(self, texts: list[str],
                         model: str | None = None) -> EmbeddingResponse

    # Streaming
    async def stream(self, messages: list[Message],
                     temperature: float = 0.7,
                     max_tokens: int | None = None) -> AsyncGenerator[str, None]

    # Utilities
    async def count_tokens(self, text: str) -> int
    async def health_check(self) -> bool
```

## Data Models

### Message
```python
class Message(BaseModel):
    role: str        # "user", "assistant", "system"
    content: str
    name: str | None = None
```

### LLMResponse
```python
class LLMResponse(BaseModel):
    content: str                  # Generated text
    model: str                    # Model identifier
    tokens_used: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    stop_reason: str | None
    provider: str = "unknown"
    prompt_version: str | None
```

### EmbeddingResponse
```python
class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    tokens_used: int | None
    provider: str = "unknown"
```

## Available Providers

| Provider | Module | Config Key | API Key Env Var |
|----------|--------|------------|-----------------|
| OpenAI | `product.providers.openai` | `openai` | `OPENAI_API_KEY` |
| Anthropic | `product.providers.anthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `product.providers.azure_openai` | `azure_openai` | `AZURE_OPENAI_API_KEY` |
| Google Gemini | `product.providers.gemini` | `gemini` | `GEMINI_API_KEY` |
| vLLM | `product.providers.vllm` | `vllm` | — |
| Ollama | `product.providers.ollama` | `ollama` | — |

## Provider Factory

The `ProviderFactory` creates providers from configuration:

```python
from product.providers.registry import ProviderFactory
from product.core.config import get_settings

settings = get_settings()
provider = ProviderFactory.create_from_config(settings.llm)
```

### Registration Pattern

```python
from product.providers.registry import register_provider

@register_provider("my_provider")
class MyProvider(LLMProvider):
    provider_name = "my_provider"
    ...
```

## Configuration

### Environment Variables
```bash
# Required
LLM__PROVIDER=openai
LLM__API_KEY=sk-...

# Optional
LLM__API_BASE=https://api.openai.com/v1
LLM__API_VERSION=2024-02-01
LLM__MODEL=gpt-4o
LLM__EMBEDDING_MODEL=text-embedding-3-small
LLM__MAX_TOKENS=4096
LLM__TEMPERATURE=0.7
LLM__TIMEOUT_SECONDS=60
LLM__MAX_RETRIES=3
```

### YAML Configuration
```yaml
# config/default.yaml
llm:
  provider: openai
  model: gpt-4o
  max_tokens: 4096
  temperature: 0.7
```

## Implementing a New Provider

To add a new provider, you need to:

1. Create a new file in `product/providers/`
2. Implement the `LLMProvider` interface
3. Register with the `@register_provider` decorator
4. Add configuration support in `LLMConfig`

### Template

```python
"""New provider implementation."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from product.providers.base import LLMProvider, LLMResponse, EmbeddingResponse, Message
from product.providers.registry import register_provider


@register_provider("my_provider")
class MyProvider(LLMProvider):
    """My custom LLM provider."""

    provider_name = "my_provider"

    def __init__(self, **kwargs: Any) -> None:
        self.api_key = kwargs.get("api_key", "")
        self.model = kwargs.get("model", "default-model")
        # Initialize client SDK here

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        # Call provider API
        # Return LLMResponse
        raise NotImplementedError

    async def structured_generate(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> BaseModel:
        # Generate structured output
        raise NotImplementedError

    async def embeddings(self, texts: list[str],
                         model: str | None = None) -> EmbeddingResponse:
        # Generate embeddings
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        # Stream tokens
        raise NotImplementedError

    async def count_tokens(self, text: str) -> int:
        # Count tokens (use tiktoken or provider API)
        raise NotImplementedError

    async def health_check(self) -> bool:
        # Check provider availability
        raise NotImplementedError
```

## Error Handling

Providers throw typed exceptions from `product.core.errors`:

- `ProviderConnectionError`: Network/connection failures
- `ProviderAPIError`: API returned error response
- `ProviderRateLimitError`: Rate limited (HTTP 429)
- `ProviderAuthError`: Authentication failure (HTTP 401)
- `ProviderInitializationError`: Provider setup failed

## Testing Providers

```python
import pytest
from product.providers.openai import OpenAIProvider

@pytest.mark.asyncio
async def test_openai_generate():
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
    response = await provider.generate([
        Message(role="user", content="Hello")
    ])
    assert isinstance(response.content, str)
```

### Mock Provider for Tests

```python
from product.providers.base import LLMProvider, LLMResponse

class MockProvider(LLMProvider):
    provider_name = "mock"

    async def generate(self, messages, **kwargs):
        return LLMResponse(
            content="Mock response",
            model="mock",
            provider="mock"
        )
    # ... implement other methods