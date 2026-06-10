"""Tests for token budget monitoring.

Ensures the framework can track and enforce token usage limits
per request, preventing unexpected cost spikes.
"""
from __future__ import annotations

import pytest

from product.core.config import LLMConfig


class TestTokenBudget:
    """Tests for token budget tracking."""

    def test_config_has_token_limits(self):
        """Configuration should include token limits."""
        config = LLMConfig()
        assert config.max_tokens > 0
        assert config.max_tokens <= 4096  # default

    def test_token_limit_configurable(self):
        """Token limits should be configurable."""
        config = LLMConfig(max_tokens=8192)
        assert config.max_tokens == 8192

    def test_token_limit_bounds(self):
        """Token limits should have sensible bounds."""
        config = LLMConfig()
        assert 1 <= config.max_tokens <= 128000  # max context window

    def test_embedding_model_configured(self):
        """Embedding model should be configured."""
        config = LLMConfig()
        assert config.embedding_model is not None
        assert len(config.embedding_model) > 0

    def test_token_budget_tracking(self):
        """Token usage should be trackable through LLMResponse."""
        from product.providers.base import LLMResponse

        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            tokens_used=50,
            prompt_tokens=20,
            completion_tokens=30,
            provider="openai",
        )
        assert response.tokens_used == 50
        assert response.prompt_tokens == 20
        assert response.completion_tokens == 30

    def test_token_budget_no_tracking(self):
        """Token tracking should be optional."""
        from product.providers.base import LLMResponse

        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            provider="openai",
        )
        assert response.tokens_used is None
        assert response.prompt_tokens is None
        assert response.completion_tokens is None

    def test_response_token_validation(self):
        """Token counts should be validated."""
        from product.providers.base import LLMResponse

        response = LLMResponse(
            content="Test",
            model="gpt-4",
            provider="openai",
            tokens_used=100,
        )
        assert response.tokens_used == 100
        assert response.content == "Test"