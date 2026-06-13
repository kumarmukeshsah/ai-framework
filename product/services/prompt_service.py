"""Prompt management service with versioning and LRU caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from product.core.errors import PromptNotFoundError, PromptRenderError
from product.core.logging import get_logger
from product.core.telemetry import track_prompt_version

logger = get_logger(__name__)


class PromptTemplate(BaseModel):
    """A versioned prompt template loaded from YAML."""

    version: str
    description: str
    system_prompt: str
    user_template: str
    output_schema: dict[str, Any] | None = None

    model_config = {"frozen": True}


class PromptManager:
    """Manages prompt templates with versioning support and LRU cache.

    Prompts are stored as YAML in ``prompts/v{N}/`` directories.
    Versions are inferred from the directory name (``v1``, ``v2``, etc.).
    """

    def __init__(self, prompts_dir: str = "product/prompts", max_cache: int = 128):
        self._prompts_dir = Path(prompts_dir)
        self._max_cache = max_cache
        self._cache: dict[str, PromptTemplate] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Scan the prompts directory and load all YAML files."""
        if not self._prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self._prompts_dir}")
            return

        for version_dir in sorted(self._prompts_dir.iterdir()):
            if not version_dir.is_dir() or not version_dir.name.startswith("v"):
                continue
            for yaml_file in version_dir.glob("*.yaml"):
                prompt_name = yaml_file.stem
                key = f"{prompt_name}@{version_dir.name}"
                try:
                    with yaml_file.open() as f:
                        data = yaml.safe_load(f)
                    template = PromptTemplate(
                        version=data.get("version", version_dir.name),
                        description=data.get("description", ""),
                        system_prompt=data.get("system_prompt", ""),
                        user_template=data.get("user_template", ""),
                        output_schema=data.get("output_schema"),
                    )
                    self._cache[key] = template
                    logger.debug(f"Loaded prompt: {key}")
                except Exception as e:
                    logger.error(f"Failed to load prompt {yaml_file}: {e}")

    def get_prompt(self, prompt_name: str, version: str | None = None) -> PromptTemplate:
        """Get a prompt template by name and optional version.

        If version is None, returns the latest version.
        """
        if version:
            key = f"{prompt_name}@{version}"
            if key in self._cache:
                track_prompt_version(prompt_name, version)
                return self._cache[key]
            raise PromptNotFoundError(f"Prompt '{key}' not found")

        # Find the latest version
        matching = sorted(
            (k for k in self._cache if k.startswith(f"{prompt_name}@")),
            key=self._version_sort_key,
            reverse=True,
        )
        if not matching:
            raise PromptNotFoundError(f"No prompts found for '{prompt_name}'")
        latest_key = matching[0]
        _, version_str = latest_key.split("@")
        track_prompt_version(prompt_name, version_str)
        return self._cache[latest_key]

    def render_prompt(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        version: str | None = None,
    ) -> tuple:
        """Render a prompt template with variables.

        Returns:
            Tuple of (system_prompt, user_prompt, version_string).
        """
        try:
            template = self.get_prompt(prompt_name, version)
            user_prompt = template.user_template.format(**variables)
            return template.system_prompt, user_prompt, template.version
        except KeyError as e:
            raise PromptRenderError(f"Missing template variable: {e}")
        except Exception as e:
            raise PromptRenderError(f"Failed to render prompt '{prompt_name}': {e}")

    def get_latest_version(self, prompt_name: str) -> str | None:
        matching = sorted(
            (k for k in self._cache if k.startswith(f"{prompt_name}@")),
            key=self._version_sort_key,
            reverse=True,
        )
        if not matching:
            return None
        return matching[0].split("@")[1]

    def list_prompts(self) -> list[dict[str, str]]:
        result = []
        for key, template in self._cache.items():
            name, version = key.split("@")
            result.append(
                {
                    "name": name,
                    "version": version,
                    "description": template.description,
                }
            )
        return result

    @staticmethod
    def _version_sort_key(key: str) -> tuple:
        try:
            v_part = key.split("@")[1]
            return tuple(int(x) for x in v_part[1:].split("."))
        except (IndexError, ValueError):
            return (0,)
