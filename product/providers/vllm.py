"""vLLM provider implementation (OpenAI-compatible API)."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, List, Optional, Type

import httpx
from pydantic import BaseModel

from product.core.errors import ProviderAPIError, ProviderConnectionError
from product.core.telemetry import track_llm_call
from product.providers.base import EmbeddingResponse, LLMProvider, LLMResponse, Message
from product.providers.registry import register_provider


@register_provider("vllm")
class VLLMProvider(LLMProvider):
    """LLM provider for vLLM (OpenAI-compatible local server)."""

    provider_name = "vllm"

    def __init__(
        self,
        api_base: str = "http://localhost:8000",
        model: str = "mistral-7b-instruct",
        timeout: float = 120.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        return self._client

    async def _request(self, endpoint: str, payload: dict) -> dict:
        client = self._get_client()
        try:
            response = await client.post(endpoint, json=payload)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderConnectionError(f"vLLM connection failed: {e}")

        if response.status_code >= 400:
            raise ProviderAPIError(f"vLLM error: {response.status_code} {response.text}")
        return response.json()

    @track_llm_call(provider="vllm", model="mistral-7b-instruct")
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
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

        data = await self._request("/v1/chat/completions", payload)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            tokens_used=usage.get("total_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            stop_reason=choice.get("finish_reason"),
            provider="vllm",
        )

    @track_llm_call(provider="vllm", model="mistral-7b-instruct")
    async def structured_generate(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        prompt = f"Respond with valid JSON matching:\n{json.dumps(schema, indent=2)}"
        system_msg = Message(role="system", content=prompt)
        result = await self.generate([system_msg] + messages, temperature=temperature, max_tokens=max_tokens)
        parsed = json.loads(result.content)
        return response_model.model_validate(parsed)

    async def embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> EmbeddingResponse:
        payload = {
            "model": model or self.model,
            "input": texts,
        }
        data = await self._request("/v1/embeddings", payload)
        embeddings = [item["embedding"] for item in data["data"]]
        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", self.model),
            provider="vllm",
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
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
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise ProviderAPIError(f"vLLM streaming error: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        data = json.loads(chunk)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"vLLM streaming failed: {e}")

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.get("/v1/models")
            return response.status_code == 200
        except Exception:
            return False