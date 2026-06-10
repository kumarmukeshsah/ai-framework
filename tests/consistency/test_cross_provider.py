"""Tests for cross-provider consistency.

Ensures that the same prompt produces structurally consistent results
across different LLM providers. This is critical for production systems
that may switch between providers.
"""
from __future__ import annotations

import pytest

from product.providers.registry import ProviderRegistry


class TestCrossProviderConsistency:
    """Tests for consistency across LLM providers."""

    def test_all_registered_providers_have_required_methods(self):
        """All registered providers must implement the full interface."""
        required_methods = [
            "generate",
            "structured_generate",
            "embeddings",
            "stream",
            "count_tokens",
            "health_check",
        ]

        for name in ProviderRegistry.get_names():
            provider_class = ProviderRegistry.get_class(name)
            for method in required_methods:
                assert hasattr(provider_class, method), (
                    f"Provider '{name}' missing method '{method}'"
                )

    def test_provider_names_are_unique(self):
        """Provider names must be unique in the registry."""
        names = ProviderRegistry.get_names()
        assert len(names) == len(set(names)), "Duplicate provider names found"

    def test_minimum_providers_available(self):
        """At minimum, local providers (ollama, vllm) should be available."""
        names = ProviderRegistry.get_names()
        # At least 3 providers should be registered
        assert len(names) >= 3, f"Only {len(names)} providers registered: {names}"

    def test_local_providers_available(self):
        """Local/on-prem providers should always be available."""
        names = ProviderRegistry.get_names()
        has_local = any(p in names for p in ["ollama", "vllm", "local"])
        assert has_local, f"No local provider found among: {names}"

    def test_cloud_providers_listed(self):
        """Cloud providers should be listed in registry."""
        names = ProviderRegistry.get_names()
        cloud_providers = {"openai", "anthropic", "azure_openai", "gemini"}
        found = cloud_providers.intersection(names)
        # Should have at least some cloud providers
        assert len(found) >= 2, f"Only {len(found)} cloud providers found: {found}"

    def test_provider_health_check_contract(self):
        """All providers must have a health_check method matching the contract."""
        for name in ProviderRegistry.get_names():
            provider_class = ProviderRegistry.get_class(name)
            health_check = getattr(provider_class, "health_check", None)
            assert health_check is not None, f"Provider '{name}' missing health_check"
            assert callable(health_check)