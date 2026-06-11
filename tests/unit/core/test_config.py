"""Tests for core.config module."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from product.core.config import (
    Settings,
    LLMConfig,
    APIConfig,
    SecurityConfig,
    get_settings,
)


class TestLLMConfig:
    def test_default_values(self) -> None:
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_env_prefix(self) -> None:
        with patch.dict(os.environ, {"LLM__PROVIDER": "anthropic", "LLM__MODEL": "claude-3-opus-20240229"}):
            config = LLMConfig()
            assert config.provider == "anthropic"
            assert config.model == "claude-3-opus-20240229"


class TestAPIConfig:
    def test_default_values(self) -> None:
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "INFO"


class TestSecurityConfig:
    def test_default_values(self) -> None:
        config = SecurityConfig()
        assert config.rate_limit_per_minute == 60
        assert config.max_input_length == 32_000
        assert config.enable_injection_detection is True


class TestSettings:
    def test_default_settings(self) -> None:
        settings = Settings()
        assert settings.env == "development"
        assert settings.debug is False
        assert isinstance(settings.llm, LLMConfig)
        assert isinstance(settings.api, APIConfig)

    def test_debug_disabled_in_production(self) -> None:
        with pytest.raises(ValidationError):
            Settings(env="production", debug=True)

    def test_nested_env_override(self) -> None:
        with patch.dict(os.environ, {"LLM__TEMPERATURE": "0.5", "VECTOR_DB__TOP_K": "10"}):
            settings = Settings()
            assert settings.llm.temperature == 0.5
            assert settings.vector_db.top_k == 10

    def test_load_yaml_not_found(self, tmp_path: Path) -> None:
        """Should not crash when no YAML config exists."""
        settings = Settings(config_dir=str(tmp_path))
        # Just verify no exception
        assert settings.env == "development"

    def test_get_settings_singleton(self) -> None:
        # Clear any cached instance
        if hasattr(get_settings, "_settings"):
            del get_settings._settings  # type: ignore[attr-defined]
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
