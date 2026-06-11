"""Structured logging for the AI Framework.

Provides context-aware logging with JSON formatting option,
and a standard logging interface compatible with loguru-style usage.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}
_initialized = False


class ContextFilter(logging.Filter):
    """Adds contextual fields (service, env, request_id) to every log record."""

    def __init__(self, service: str = "ai-framework", env: str = "development") -> None:
        super().__init__()
        self.service = service
        self.env = env

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = getattr(record, "service", self.service)
        record.env = getattr(record, "env", self.env)
        record.request_id = getattr(record, "request_id", "-")
        return True


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    service: str = "ai-framework",
    env: str = "development",
) -> None:
    """Configure the root logger once.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, emit JSON-structured logs (requires ``python-json-logger``).
        service: Service name for context.
        env: Deployment environment for context.
    """
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplication
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if json_format:
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore[import-untyped]

            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(service)s %(env)s %(request_id)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        except ImportError:
            # Fall back to standard format if json-logger not installed
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.addFilter(ContextFilter(service=service, env=env))

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Usually ``__name__`` from the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger
