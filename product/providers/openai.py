"""OpenAI provider implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import BaseModel

from product.core.errors import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from product.core.telemetry import track_llm_call
from product.providers.base import EmbeddingResponse, LLMProvider, LLMResponse, Message
from product.providers.registry import register_provider


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        api_base: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.api_base = api_base.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def _request(self, endpoint: str, payload: dict) -> dict:
        client = self._get_client()
        try:
            response = await client.post(endpoint, json=payload)
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Failed to connect to OpenAI: {e}")
        except httpx.TimeoutException as e:
            raise ProviderConnectionError(f"OpenAI request timed out: {e}")

        if response.status_code == 401:
            raise ProviderAuthError("OpenAI authentication failed. Check your API key.")
        if response.status_code == 429:
            raise ProviderRateLimitError("OpenAI rate limit exceeded.")
        if response.status_code >= 500:
            raise ProviderAPIError(f"OpenAI server error: {response.status_code} {response.text}")
        if response.status_code >= 400:
            raise ProviderAPIError(f"OpenAI API error: {response.status_code} {response.text}")

        return response.json()

    @track_llm_call(provider="openai", model="gpt-4o-mini")
    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop_sequences:
            payload["stop"] = stop_sequences

        data = await self._request("/chat/completions", payload)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            tokens_used=usage.get("total_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            stop_reason=choice.get("finish_reason"),
            provider="openai",
        )

    @track_llm_call(provider="openai", model="gpt-4o-mini")
    async def structured_generate(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                },
            },
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = await self._request("/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        import json

        parsed = json.loads(content)
        return response_model.model_validate(parsed)

    async def embeddings(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        payload = {
            "model": model or self.embedding_model,
            "input": texts,
        }
        data = await self._request("/embeddings", payload)
        embeddings = [item["embedding"] for item in data["data"]]
        usage = data.get("usage", {})
        return EmbeddingResponse(
            embeddings=embeddings,
            model=data["model"],
            tokens_used=usage.get("total_tokens"),
            provider="openai",
        )

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        client = self._get_client()
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise ProviderAPIError(f"OpenAI streaming error: {response.status_code} {text}")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        import json

                        data = json.loads(chunk)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Streaming connection failed: {e}")

    async def count_tokens(self, text: str) -> int:
        # Approximate count using tiktoken if available
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: ~4 chars per token
            return len(text) // 4

    async def health_check(self) -> bool:
        try:
            result = await self.generate([Message(role="user", content="ping")], max_tokens=5)
            return bool(result.content)
        except Exception:
            return False
