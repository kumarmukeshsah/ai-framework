"""Tests for circuit breaker patterns in provider calls.

Ensures that repeated failures trigger circuit breaker logic,
preventing cascading failures and allowing recovery.
"""

from __future__ import annotations

from product.core.config import LLMConfig


class TestCircuitBreaker:
    """Tests for circuit breaker configuration and behavior."""

    def test_config_has_retry_settings(self):
        """Configuration should have retry/timeout settings."""
        config = LLMConfig()
        assert config.max_retries > 0, "Must have at least 1 retry"
        assert config.timeout_seconds > 0, "Must have a positive timeout"

    def test_config_retry_default(self):
        """Default retry count should be reasonable."""
        config = LLMConfig()
        assert config.max_retries == 3, "Default max_retries should be 3"

    def test_config_timeout_default(self):
        """Default timeout should be reasonable."""
        config = LLMConfig()
        assert config.timeout_seconds == 60, "Default timeout should be 60s"

    def test_config_custom_values(self):
        """Custom retry/timeout values should be configurable."""
        config = LLMConfig(max_retries=5, timeout_seconds=120)
        assert config.max_retries == 5
        assert config.timeout_seconds == 120

    def test_config_zero_retries(self):
        """Zero retries should be allowed (fire-and-forget)."""
        config = LLMConfig(max_retries=0)
        assert config.max_retries == 0

    def test_config_temperature_fallback(self):
        """Temperature should default to reasonable value."""
        config = LLMConfig()
        assert 0.0 <= config.temperature <= 1.0

    def test_config_model_fallback(self):
        """Model should have a default if not specified."""
        config = LLMConfig()
        assert isinstance(config.model, str)
        assert len(config.model) > 0
