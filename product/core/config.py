"""Configuration management for the AI Framework.

Uses Pydantic Settings with layered loading:
1. Default values
2. YAML config file (environment-specific)
3. Environment variables (highest priority)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "ollama", "azure_openai", "gemini", "vllm"] = "openai"
    api_key: str = ""
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60
    max_retries: int = 3

    model_config = {"env_prefix": "LLM__"}


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""

    provider: Literal["qdrant", "pinecone", "weaviate"] = "qdrant"
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection: str = "ai_framework"
    embedding_dim: int = 1536
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    score_threshold: float = 0.7

    model_config = {"env_prefix": "VECTOR_DB__"}


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = "sqlite:///./ai_framework.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    model_config = {"env_prefix": "DATABASE__"}


class SecurityConfig(BaseSettings):
    """Security configuration."""

    rate_limit_per_minute: int = 60
    max_input_length: int = 32_000
    enable_injection_detection: bool = True
    enable_output_filtering: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = {"env_prefix": "SECURITY__"}


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""

    enabled: bool = True
    metrics_port: int = 8001
    enable_trace_export: bool = False
    trace_endpoint: Optional[str] = None
    service_name: str = "ai-framework"
    environment: str = "development"

    model_config = {"env_prefix": "OBSERVABILITY__"}


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"

    model_config = {"env_prefix": "API__"}


class Settings(BaseSettings):
    """Root settings object.

    Loads from environment variables (prefixed) and optional YAML config.
    Environment variables take precedence over YAML values.
    """

    # Environment
    env: Literal["development", "qa", "staging", "production"] = "development"
    debug: bool = False

    # Sub-configurations
    llm: LLMConfig = LLMConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    api: APIConfig = APIConfig()

    # Paths
    prompts_dir: str = "product/prompts"
    datasets_dir: str = "evaluation/datasets"
    reports_dir: str = "evaluation/reports"
    config_dir: str = "config"

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @field_validator("debug")
    @classmethod
    def _validate_debug(cls, v: bool, info: Any) -> bool:
        if info.data.get("env") == "production" and v:
            raise ValueError("Debug mode is not allowed in production")
        return v

    def load_yaml(self, config_path: Optional[str] = None) -> Settings:
        """Merge YAML config values into this settings instance.

        Args:
            config_path: Explicit path to a YAML file. If None, attempts to
                         load ``config/{env}.yaml`` and ``config/default.yaml``.
        """
        candidates: list[str] = []
        if config_path:
            candidates.append(config_path)
        else:
            env_config = Path(self.config_dir) / f"{self.env}.yaml"
        default_config = Path(self.config_dir) / "default.yaml"

        for candidate in (env_config, default_config):
            if candidate.exists():
                candidates.append(str(candidate))

        for path in candidates:
            p = Path(path)
            if p.exists():
                with open(p) as f:
                    data: dict = yaml.safe_load(f) or {}
                self._merge_dict(data)

        return self

    def _merge_dict(self, data: dict, prefix: str = "") -> None:
        """Recursively merge a dictionary into the settings."""
        for key, value in data.items():
            attr_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and hasattr(self, key):
                child = getattr(self, key)
                if isinstance(child, BaseSettings):
                    for sub_key, sub_value in value.items():
                        if hasattr(child, sub_key):
                            setattr(child, sub_key, sub_value)
                else:
                    setattr(self, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)


def get_settings() -> Settings:
    """Get application settings singleton (cached)."""
    if not hasattr(get_settings, "_settings"):
        settings = Settings()
        settings.load_yaml()
        get_settings._settings = settings  # type: ignore[attr-defined]
    return get_settings._settings  # type: ignore[attr-defined]
