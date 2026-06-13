"""Security middleware for the AI Platform API.

Implements:
- Prompt injection detection
- Input size validation
- Rate limiting
- Output filtering
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from product.core.logging import get_logger

logger = get_logger(__name__)

# ── Compile injection patterns once ───────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior|the)\s+(instructions|prompts|commands|directions|content)",
    r"disregard\s+(all\s+)?(previous|above|prior|the)\s+(instructions|prompts|commands|directions|safety|security|content|policy|policies|guidelines|rules)",
    r"forget\s+(all\s+)?(previous|above|prior|the|your)\s+(instructions|prompts|commands|directions)",
    r"reveal\s+(your\s+)?(the\s+)?(system\s+)?(prompt|hidden|secrets?|restricted|confidential)",
    r"show\s+(your\s+)?(the\s+)?(system\s+)?(prompt|instructions|secrets?|hidden|tokens?)",
    r"print\s+(your\s+)?(the\s+)?(system\s+)?(prompt|instructions)",
    r"output\s+(your\s+)?(the\s+)?(system\s+)?(prompt|instructions|secrets?|api\s+keys?|tokens?)",
    r"what\s+(is|are)\s+(your\s+)?(the\s+)?(system\s+)?(prompt|instructions?)",
    r"tell\s+me\s+(your\s+)?(the\s+)?(system\s+)?(prompt|secrets?)",
    r"leak\s+(the\s+)?(hidden\s+)?(secrets?|api\s+keys?)",
    r"expose\s+(the\s+)?(internal\s+)?(secrets?|configuration|system\s+configuration|api\s+keys?)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"from\s+now\s+on\s+",
    r"new\s+instruction",
    r"override\s+",
    r"you\s+must\s+ignore",
    r"do\s+not\s+follow",
    r"DAN\b",
    r"do\s+anything\s+now",
    r"no\s+(restrictions|limits|boundaries|filtering)",
    r"you\s+(have\s+)?no\s+(rules|restrictions|limitations)",
    r"jailbreak",
    r"bypass\s+(the\s+)?(filter|safety|policy|restriction|content)",
    r"system\s+prompt\s+is:",
    r"initialization\s+",
    r"hidden\s+(instructions?|initialization)",
    r"confidential\s+instructions?",
    r"share\s+(secrets?|api\s+keys?|tokens?)",
    r"forget\s+your\s+previous",
    r"show\s+me\s+the\s+secret",
    r"show\s+me\s+the\s+system",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

SENSITIVE_OUTPUT_PATTERNS = [
    r"sk-[A-Za-z0-9]{32,}",
    r"api_key['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    r"secret['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    r"password['\"]?\s*[:=]\s*['\"][A-Za-z0-9_!@#$%^&*()]{8,}['\"]",
    r"Bearer\s+[A-Za-z0-9\-._~+/]+={0,2}",
    r"ghp_[A-Za-z0-9]{36}",
    r"AKIA[A-Z0-9]{16}",
]


class SecurityConfig:
    """Configuration for security middleware."""

    def __init__(
        self,
        rate_limit: int = 60,
        rate_limit_window: int = 60,
        max_input_length: int = 32000,
        enable_injection_detection: bool = True,
        enable_output_filtering: bool = True,
        injection_threshold: float = 0.3,
    ):
        self.rate_limit = rate_limit
        self.rate_limit_window = rate_limit_window
        self.max_input_length = max_input_length
        self.enable_injection_detection = enable_injection_detection
        self.enable_output_filtering = enable_output_filtering
        self.injection_threshold = injection_threshold


def detect_prompt_injection(text: str) -> dict[str, Any]:
    """Check text for prompt injection attempts.

    Returns:
        Dict with ``detected``, ``severity``, and ``matched_patterns``.
    """
    if not text:
        return {"detected": False, "severity": 0.0, "matched_patterns": []}
    lower = text.lower()
    matched = []
    for i, pattern in enumerate(COMPILED_PATTERNS):
        if pattern.search(lower):
            matched.append(INJECTION_PATTERNS[i][:50])
    if matched:
        return {
            "detected": True,
            "severity": min(1.0, len(matched) / 5.0),
            "matched_patterns": matched,
        }
    return {"detected": False, "severity": 0.0, "matched_patterns": []}


def filter_sensitive_output(text: str) -> str:
    """Redact sensitive data patterns from text."""
    result = text
    for pattern in SENSITIVE_OUTPUT_PATTERNS:
        result = re.sub(pattern, "[REDACTED]", result)
    return result


class SecurityMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for security checks."""

    def __init__(
        self,
        app: Any,
        config: SecurityConfig | None = None,
        rate_limit: int = 60,
        rate_window: int = 60,
        max_input: int = 32_000,
        enable_injection: bool = True,
        enable_filtering: bool = True,
    ) -> None:
        super().__init__(app)
        # Support both old-style kwargs and new config object
        if config:
            self.config = config
        else:
            self.config = SecurityConfig(
                rate_limit=rate_limit,
                rate_limit_window=rate_window,
                max_input_length=max_input,
                enable_injection_detection=enable_injection,
                enable_output_filtering=enable_filtering,
            )
        self._store: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"

        # Rate-limit check
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "detail": f"Max {self.config.rate_limit} req/min",
                },
            )

        # Input validation for write methods
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                body_text = body.decode("utf-8", errors="ignore")
                if len(body_text) > self.config.max_input_length:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "INPUT_TOO_LARGE",
                            "detail": f"Max {self.config.max_input_length} bytes",
                        },
                    )
                if self.config.enable_injection_detection:
                    detection = self._detect_injection(body_text)
                    if detection["detected"]:
                        logger.warning(f"Injection from {client_ip}")
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": "PROMPT_INJECTION_DETECTED",
                                "detail": "Input contains disallowed patterns",
                                "severity": detection["severity"],
                            },
                        )
            except Exception as e:
                logger.error(f"Security check failed: {e}")

        response = await call_next(request)

        # Output filtering
        if self.config.enable_output_filtering and isinstance(response, JSONResponse):
            body = response.body.decode("utf-8", errors="ignore")
            filtered = self._filter_output(body)
            if filtered != body:
                import json as _json

                response = JSONResponse(
                    status_code=response.status_code,
                    content=_json.loads(filtered),
                    headers=dict(response.headers),
                )
        return response

    def _check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.config.rate_limit_window
        self._store[client_ip] = [t for t in self._store[client_ip] if t > window_start]
        if len(self._store[client_ip]) >= self.config.rate_limit:
            return False
        self._store[client_ip].append(now)
        return True

    def _detect_injection(self, text: str) -> dict[str, Any]:
        """Detect prompt injection in text."""
        return detect_prompt_injection(text)

    def _filter_output(self, text: str) -> str:
        """Filter sensitive output."""
        return filter_sensitive_output(text)
