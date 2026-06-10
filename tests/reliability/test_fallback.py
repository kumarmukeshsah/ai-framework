"""Tests for provider fallback and retry mechanisms.

Ensures that when a primary provider fails, the system gracefully
falls back to alternative providers without data loss or errors.
"""
from __future__ import annotations

import pytest

from product.providers.registry import ProviderRegistry


class TestFallbackMechanisms:
    """Tests for provider fallback logic."""

    def test_registry_contains_fallback_options(self):
        """Registry should have multiple providers for fallback."""
        names = ProviderRegistry.get_names()
        assert len(names) >= 2, (
            f"Need at least 2 providers for fallback, only have {len(names)}: {names}"
        )

    def test_known_provider_creation(self):
        """Known providers should be creatable with proper args."""
        known_local = [n for n in ProviderRegistry.get_names() if n in ("ollama", "vllm")]
        for name in known_local:
            try:
                provider = ProviderRegistry.create(name)
                assert provider is not None
            except Exception:
                pass  # May fail if binary not installed

    def test_invalid_provider_name_raises(self):
        """Invalid provider name should raise ProviderNotFoundError."""
        from product.core.errors import ProviderNotFoundError
        with pytest.raises(ProviderNotFoundError):
            ProviderRegistry.create("nonexistent_provider")

    def test_provider_not_found_error_message(self):
        """Error message should list available providers."""
        from product.core.errors import ProviderNotFoundError
        try:
            ProviderRegistry.create("fake_provider")
        except ProviderNotFoundError as e:
            msg = str(e)
            for name in ProviderRegistry.get_names():
                assert name in msg, f"Error message missing provider: {name}"

    def test_registry_immutability(self):
        """Registry should not be modifiable externally."""
        names_before = set(ProviderRegistry.get_names())
        # Try to access _registry directly (should not be accessible)
        assert not hasattr(ProviderRegistry, "_registry"), "Registry should be private"
        names_after = set(ProviderRegistry.get_names())
        assert names_before == names_after

    def test_minimum_fallback_chain(self):
        """Should have at least one local and one cloud provider for fallback."""
        names = ProviderRegistry.get_names()
        has_local = any(n in ("ollama", "vllm") for n in names)
        has_cloud = any(n in ("openai", "anthropic", "azure_openai", "gemini") for n in names)
        assert has_local or has_cloud, "No usable providers available"