"""Unit tests for the prompt service."""
import pytest
from product.services.prompt_service import PromptManager
from product.core.errors import PromptNotFoundError, PromptRenderError


class TestPromptManager:
    """Test the prompt manager."""

    def test_load_prompts(self):
        """Test prompt loading."""
        manager = PromptManager(prompts_dir="product/prompts")
        prompts = manager.list_prompts()
        assert len(prompts) > 0
        assert any(p["name"] == "candidate_evaluation" for p in prompts)

    def test_get_prompt_latest(self):
        """Test getting latest prompt version."""
        manager = PromptManager(prompts_dir="product/prompts")
        template = manager.get_prompt("candidate_evaluation")
        assert template.version is not None
        assert template.system_prompt
        assert template.user_template

    def test_get_prompt_specific_version(self):
        """Test getting specific prompt version."""
        manager = PromptManager(prompts_dir="product/prompts")
        template = manager.get_prompt("candidate_evaluation", version="v1")
        assert template.version == "1.0"

    def test_get_prompt_invalid(self):
        """Test getting invalid prompt."""
        manager = PromptManager(prompts_dir="product/prompts")
        with pytest.raises(PromptNotFoundError):
            manager.get_prompt("nonexistent_prompt")

    def test_render_prompt(self):
        """Test prompt rendering."""
        manager = PromptManager(prompts_dir="product/prompts")
        system_prompt, user_prompt, version = manager.render_prompt(
            "candidate_evaluation",
            {"transcript": "Test transcript", "context": "Test context"},
        )
        assert "Test transcript" in user_prompt
        assert "Test context" in user_prompt
        assert version is not None

    def test_render_prompt_missing_variable(self):
        """Test rendering with missing variables."""
        manager = PromptManager(prompts_dir="product/prompts")
        with pytest.raises(PromptRenderError):
            manager.render_prompt("candidate_evaluation", {})

    def test_get_latest_version(self):
        """Test getting latest version."""
        manager = PromptManager(prompts_dir="product/prompts")
        version = manager.get_latest_version("candidate_evaluation")
        assert version is not None
        assert version.startswith("v")