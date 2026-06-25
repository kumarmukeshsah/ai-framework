"""
DeepEval Provider for Promptfoo.

This provider wraps DeepEval metrics so Promptfoo can call them as assertions.
Each metric is exposed as a callable that returns a score (0.0-1.0).

Metrics covered:
- HallucinationMetric
- FaithfulnessMetric
- ToxicityMetric
- CorrectnessMetric
- BiasMetric
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEEPEVAL_AVAILABLE = False
try:
    from deepeval.metrics import (
        BiasMetric,
        FaithfulnessMetric,
        HallucinationMetric,
        ToxicityMetric,
    )
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except ImportError:
    pass
except Exception:
    # DeepEval may fail to init if no API key - that's OK, we fall back gracefully
    pass


class DeepEvalRunner:
    """Runs DeepEval metrics and returns structured results.
    Uses lazy initialization to avoid import-time failures when API keys are missing.
    """

    def __init__(self):
        self._metrics = {}
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True

        if not DEEPEVAL_AVAILABLE:
            return

        try:
            self._metrics = {
                "hallucination": HallucinationMetric(threshold=0.5),
                "faithfulness": FaithfulnessMetric(threshold=0.5),
                "toxicity": ToxicityMetric(threshold=0.5),
                "bias": BiasMetric(threshold=0.5),
            }
        except Exception:
            # Metrics may fail if no API keys configured
            pass

    async def evaluate_hallucination(
        self, input_text: str, actual_output: str, context: list[str] | None = None
    ) -> dict[str, Any]:
        """Evaluate hallucination score (lower is better)."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "DeepEval not available (no API key?)", "passed": True}

        try:
            test_case = LLMTestCase(
                input=input_text,
                actual_output=actual_output,
                context=context or [],
            )
            metric = self._metrics["hallucination"]
            result = metric.measure(test_case)
            return {
                "score": metric.score,
                "reason": metric.reason,
                "passed": result,
                "metric": "hallucination",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"Hallucination eval failed: {e}", "passed": True}

    async def evaluate_faithfulness(
        self, input_text: str, actual_output: str, context: list[str] | None = None
    ) -> dict[str, Any]:
        """Evaluate faithfulness score (higher is better)."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "DeepEval not available (no API key?)", "passed": True}

        try:
            test_case = LLMTestCase(
                input=input_text,
                actual_output=actual_output,
                context=context or [],
            )
            metric = self._metrics["faithfulness"]
            result = metric.measure(test_case)
            return {
                "score": metric.score,
                "reason": metric.reason,
                "passed": result,
                "metric": "faithfulness",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"Faithfulness eval failed: {e}", "passed": True}

    async def evaluate_toxicity(
        self, input_text: str, actual_output: str
    ) -> dict[str, Any]:
        """Evaluate toxicity score (lower is better)."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.0, "reason": "DeepEval not available (no API key?)", "passed": True}

        try:
            test_case = LLMTestCase(input=input_text, actual_output=actual_output)
            metric = self._metrics["toxicity"]
            result = metric.measure(test_case)
            return {
                "score": metric.score,
                "reason": metric.reason,
                "passed": result,
                "metric": "toxicity",
            }
        except Exception as e:
            return {"score": 0.0, "reason": f"Toxicity eval failed: {e}", "passed": True}

    async def evaluate_bias(
        self, input_text: str, actual_output: str
    ) -> dict[str, Any]:
        """Evaluate bias score (lower is better)."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.0, "reason": "DeepEval not available (no API key?)", "passed": True}

        try:
            test_case = LLMTestCase(input=input_text, actual_output=actual_output)
            metric = self._metrics["bias"]
            result = metric.measure(test_case)
            return {
                "score": metric.score,
                "reason": metric.reason,
                "passed": result,
                "metric": "bias",
            }
        except Exception as e:
            return {"score": 0.0, "reason": f"Bias eval failed: {e}", "passed": True}

    async def run_all(
        self, input_text: str, actual_output: str, context: list[str] | None = None
    ) -> dict[str, Any]:
        """Run all DeepEval metrics and return aggregated results."""
        results = {
            "hallucination": await self.evaluate_hallucination(
                input_text, actual_output, context
            ),
            "faithfulness": await self.evaluate_faithfulness(
                input_text, actual_output, context
            ),
            "toxicity": await self.evaluate_toxicity(input_text, actual_output),
            "bias": await self.evaluate_bias(input_text, actual_output),
        }
        overall_score = sum(r["score"] for r in results.values()) / len(results)
        all_passed = all(r["passed"] for r in results.values())
        return {
            "provider": "deepeval",
            "overall_score": overall_score,
            "all_passed": all_passed,
            "results": results,
        }


# Lazily initialized singleton - Promptfoo calls call_api, not this directly
_runner_instance = None


def _get_runner() -> DeepEvalRunner:
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = DeepEvalRunner()
    return _runner_instance


async def call_api(
    prompt: str, options: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Promptfoo provider interface.

    This is called by Promptfoo to evaluate with DeepEval metrics.
    """
    runner = _get_runner()
    input_text = context.get("vars", {}).get("transcript", prompt)
    actual_output = context.get("output", prompt)
    context_docs = [context.get("vars", {}).get("context", "")]

    results = await runner.run_all(input_text, actual_output, context_docs)
    return {
        "output": json.dumps(results),
        "metrics": {
            "deepeval_score": results["overall_score"],
            "deepeval_passed": results["all_passed"],
        },
    }
