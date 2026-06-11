"""Ollama provider implementation."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, List, Optional, Type

import httpx
from pydantic import BaseModel

from product.core.errors import ProviderAPIError, ProviderConnectionError
from product.core.telemetry import track_llm_call
from product.providers.base import EmbeddingResponse, LLMProvider, LLMResponse, Message
from product.providers.registry import register_provider


@register_provider("ollama")
class OllamaProvider(LLMProvider):
    """LLM provider for Ollama (local models)."""

    provider_name = "ollama"

    def __init__(
        self,
        api_base: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 120.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.api_base, timeout=self.timeout)
        return self._client

    async def _request(self, endpoint: str, payload: dict) -> dict:
        client = self._get_client()
        try:
            response = await client.post(endpoint, json=payload)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderConnectionError(f"Ollama connection failed: {e}")

        if response.status_code >= 400:
            raise ProviderAPIError(f"Ollama error: {response.status_code} {response.text}")
        return response.json()

    @track_llm_call(provider="ollama", model="llama3.1")
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        data = await self._request("/api/chat", payload)
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            tokens_used=data.get("eval_count"),
            provider="ollama",
        )

    @track_llm_call(provider="ollama", model="llama3.1")
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
        all_messages = [system_msg] + messages
        result = await self.generate(all_messages, temperature=temperature, max_tokens=max_tokens)
        parsed = json.loads(result.content)
        return response_model.model_validate(parsed)

    async def embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            model: Optional model override.

        Returns:
            EmbeddingResponse with embeddings for each text.
        """
        if not texts:
            return EmbeddingResponse(embeddings=[], model=model or self.model, provider="ollama")

        all_embeddings = []
        used_model = model or self.model
        for text in texts:
            payload = {"model": used_model, "prompt": text}
            data = await self._request("/api/embeddings", payload)
            all_embeddings.append(data["embedding"])

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=data.get("model", used_model),
            provider="ollama",
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": temperature},
            "stream": True,
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        client = self._get_client()
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise ProviderAPIError(f"Ollama streaming error: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        if data.get("done", False):
                            break
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Ollama streaming failed: {e}")

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
