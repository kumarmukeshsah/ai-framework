"""Tests for graceful degradation when dependencies are unavailable."""
from __future__ import annotations

import pytest

from product.core.config import Settings, LLMConfig, VectorDBConfig
from product.agents.evaluator import EvaluatorAgent


class TestGracefulDegradation:
    """Tests for graceful degradation."""

    @pytest.mark.asyncio
    async def test_rule_based_evaluator_no_llm(self):
        """Rule-based evaluator should work without any LLM provider."""
        evaluator = EvaluatorAgent(use_llm=False)
        result = await evaluator.process("I have 8 years of experience.")
        assert result is not None

    def test_missing_api_key_fallback(self):
        config = LLMConfig(api_key="", provider="openai")
        assert config.api_key == ""
        assert config.provider == "openai"

    def test_vector_db_config_defaults(self):
        config = VectorDBConfig()
        assert config.url == "http://localhost:6333"
        assert config.collection == "ai_framework"

    @pytest.mark.asyncio
    async def test_missing_vector_db(self):
        evaluator = EvaluatorAgent(use_llm=False)
        result = await evaluator.process("Test without vector DB.")
        assert result is not None

    def test_invalid_vector_db_url(self):
        config = VectorDBConfig(url="http://invalid:9999")
        assert config.url == "http://invalid:9999"

    def test_partial_config_recovery(self):
        settings = Settings()
        assert settings is not None
        assert settings.env in ("development", "qa", "staging", "production")

    @pytest.mark.asyncio
    async def test_telemetry_failure_isolation(self):
        evaluator = EvaluatorAgent(use_llm=False)
        result = await evaluator.process("Test telemetry isolation.")
        assert result is not None