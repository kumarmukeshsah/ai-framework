"""Anthropic provider implementation."""

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


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """LLM provider for Anthropic (Claude)."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-3-haiku-20240307",
        api_base: str = "https://api.anthropic.com/v1",
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
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
            raise ProviderConnectionError(f"Failed to connect to Anthropic: {e}")
        except httpx.TimeoutException as e:
            raise ProviderConnectionError(f"Anthropic request timed out: {e}")

        if response.status_code == 401:
            raise ProviderAuthError("Anthropic authentication failed. Check your API key.")
        if response.status_code == 429:
            raise ProviderRateLimitError("Anthropic rate limit exceeded.")
        if response.status_code >= 500:
            raise ProviderAPIError(f"Anthropic server error: {response.status_code}")
        if response.status_code >= 400:
            raise ProviderAPIError(f"Anthropic API error: {response.status_code} {response.text}")

        return response.json()

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        system = ""
        converted = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                converted.append({"role": m.role, "content": m.content})
        return system, converted

    @track_llm_call(provider="anthropic", model="claude-3-haiku-20240307")
    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        system, converted = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if system:
            payload["system"] = system
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences

        data = await self._request("/messages", payload)
        content_block = data["content"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=content_block.get("text", ""),
            model=data["model"],
            tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            stop_reason=data.get("stop_reason"),
            provider="anthropic",
        )

    @track_llm_call(provider="anthropic", model="claude-3-haiku-20240307")
    async def structured_generate(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        system, converted = self._convert_messages(messages)
        schema_prompt = f"\n\nYou must respond with valid JSON matching this schema:\n{schema}"
        if system:
            system += schema_prompt
        else:
            system = schema_prompt

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if system:
            payload["system"] = system

        data = await self._request("/messages", payload)
        content = data["content"][0].get("text", "")
        import json

        parsed = json.loads(content)
        return response_model.model_validate(parsed)

    async def embeddings(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        raise NotImplementedError("Anthropic does not provide an embeddings API")

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        system, converted = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }
        if system:
            payload["system"] = system

        client = self._get_client()
        try:
            async with client.stream("POST", "/messages", json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise ProviderAPIError(f"Anthropic streaming error: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        import json

                        data = json.loads(chunk)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if "text" in delta:
                                yield delta["text"]
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Anthropic streaming failed: {e}")

    async def count_tokens(self, text: str) -> int:
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)
            return client.count_tokens(text)
        except ImportError:
            return len(text) // 4

    async def health_check(self) -> bool:
        try:
            result = await self.generate([Message(role="user", content="ping")], max_tokens=5)
            return bool(result.content)
        except Exception:
            return False
