"""Azure OpenAI provider implementation."""
from __future__ import annotations

from typing import Any, AsyncGenerator, List, Optional, Type

import httpx
from pydantic import BaseModel

from product.core.errors import ProviderAPIError, ProviderAuthError, ProviderConnectionError, ProviderRateLimitError
from product.core.telemetry import track_llm_call
from product.providers.base import EmbeddingResponse, LLMProvider, LLMResponse, Message
from product.providers.registry import register_provider


@register_provider("azure_openai")
class AzureOpenAIProvider(LLMProvider):
    """LLM provider for Azure OpenAI."""

    provider_name = "azure_openai"

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        api_version: str = "2024-02-15-preview",
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.api_version = api_version
        self.model = model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def _request(self, url: str, payload: dict) -> dict:
        client = self._get_client()
        try:
            response = await client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise ProviderConnectionError(f"Failed to connect to Azure OpenAI: {e}")
        except httpx.TimeoutException as e:
            raise ProviderConnectionError(f"Azure OpenAI request timed out: {e}")

        if response.status_code == 401:
            raise ProviderAuthError("Azure OpenAI authentication failed.")
        if response.status_code == 429:
            raise ProviderRateLimitError("Azure OpenAI rate limit exceeded.")
        if response.status_code >= 400:
            raise ProviderAPIError(f"Azure OpenAI error: {response.status_code} {response.text}")
        return response.json()

    def _chat_url(self) -> str:
        return (
            f"{self.api_base}/openai/deployments/{self.model}"
            f"/chat/completions?api-version={self.api_version}"
        )

    @track_llm_call(provider="azure_openai", model="gpt-4o")
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop_sequences:
            payload["stop"] = stop_sequences

        data = await self._request(self._chat_url(), payload)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            tokens_used=usage.get("total_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            stop_reason=choice.get("finish_reason"),
            provider="azure_openai",
        )

    @track_llm_call(provider="azure_openai", model="gpt-4o")
    async def structured_generate(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        payload: dict[str, Any] = {
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

        data = await self._request(self._chat_url(), payload)
        content = data["choices"][0]["message"]["content"]
        import json
        parsed = json.loads(content)
        return response_model.model_validate(parsed)

    async def embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> EmbeddingResponse:
        model_name = model or self.embedding_model
        url = (
            f"{self.api_base}/openai/deployments/{model_name}"
            f"/embeddings?api-version={self.api_version}"
        )
        payload = {"input": texts}
        data = await self._request(url, payload)
        embeddings = [item["embedding"] for item in data["data"]]
        return EmbeddingResponse(
            embeddings=embeddings,
            model=data["model"],
            tokens_used=data.get("usage", {}).get("total_tokens"),
            provider="azure_openai",
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        client = self._get_client()
        try:
            async with client.stream("POST", self._chat_url(), json=payload) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise ProviderAPIError(f"Azure OpenAI streaming error: {response.status_code}")
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
            raise ProviderConnectionError(f"Azure OpenAI streaming failed: {e}")

    async def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            return len(text) // 4

    async def health_check(self) -> bool:
        try:
            result = await self.generate([Message(role="user", content="ping")], max_tokens=5)
            return bool(result.content)
        except Exception:
            return False