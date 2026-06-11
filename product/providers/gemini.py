"""Google Gemini provider implementation."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, List, Optional, Type

import httpx
from pydantic import BaseModel

from product.core.errors import ProviderAPIError, ProviderAuthError, ProviderConnectionError, ProviderRateLimitError
from product.core.telemetry import track_llm_call
from product.providers.base import EmbeddingResponse, LLMProvider, LLMResponse, Message
from product.providers.registry import register_provider


@register_provider("gemini")
class GeminiProvider(LLMProvider):
    """LLM provider for Google Gemini."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-1.5-pro",
        embedding_model: str = "text-embedding-004",
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _chat_url(self) -> str:
        return f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"

    def _stream_url(self) -> str:
        return f"{self.api_base}/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                contents.append({
                    "role": "user" if m.role == "user" else "model",
                    "parts": [{"text": m.content}],
                })
        result = {"contents": contents}
        if system_instruction:
            result["system_instruction"] = {"parts": [{"text": system_instruction}]}
        return result

    async def _request(self, url: str, payload: dict) -> dict:
        client = self._get_client()
        try:
            response = await client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Failed to connect to Gemini: {e}")
        except httpx.TimeoutException as e:
            raise ProviderConnectionError(f"Gemini request timed out: {e}")

        if response.status_code == 403 or response.status_code == 401:
            raise ProviderAuthError("Gemini authentication failed. Check your API key.")
        if response.status_code == 429:
            raise ProviderRateLimitError("Gemini rate limit exceeded.")
        if response.status_code >= 400:
            raise ProviderAPIError(f"Gemini error: {response.status_code} {response.text}")
        return response.json()

    @track_llm_call(provider="gemini", model="gemini-1.5-pro")
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        payload = self._convert_messages(messages)
        payload["generationConfig"] = {"temperature": temperature}
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if stop_sequences:
            payload["generationConfig"]["stopSequences"] = stop_sequences

        data = await self._request(self._chat_url(), payload)
        candidate = data["candidates"][0]
        content = candidate["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=usage.get("totalTokenCount"),
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            stop_reason=candidate.get("finishReason"),
            provider="gemini",
        )

    @track_llm_call(provider="gemini", model="gemini-1.5-pro")
    async def structured_generate(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        payload = self._convert_messages(messages)
        payload["generationConfig"] = {"temperature": temperature}
        payload["generationConfig"]["response_mime_type"] = "application/json"
        payload["generationConfig"]["response_schema"] = schema
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        data = await self._request(self._chat_url(), payload)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        return response_model.model_validate(parsed)

    async def embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> EmbeddingResponse:
        model_name = model or self.embedding_model
        url = f"{self.api_base}/models/{model_name}:embedContent?key={self.api_key}"
        payload = {"model": f"models/{model_name}", "content": {"parts": [{"text": texts[0]}]}}
        data = await self._request(url, payload)
        embedding = data["embedding"]["values"]
        return EmbeddingResponse(
            embeddings=[embedding],
            model=model_name,
            provider="gemini",
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        payload = self._convert_messages(messages)
        payload["generationConfig"] = {"temperature": temperature}
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        client = self._get_client()
        try:
            async with client.stream("POST", self._stream_url(), json=payload) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise ProviderAPIError(f"Gemini streaming error: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        data = json.loads(chunk)
                        try:
                            text = data["candidates"][0]["content"]["parts"][0]["text"]
                            yield text
                        except (KeyError, IndexError):
                            pass
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Gemini streaming failed: {e}")

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def health_check(self) -> bool:
        try:
            result = await self.generate([Message(role="user", content="ping")], max_tokens=5)
            return bool(result.content)
        except Exception:
            return False
