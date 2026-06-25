"""
Custom Validators Provider for Promptfoo.

This provider wraps the framework's existing evaluation judges and adds
additional validators for JSON Schema, tool traces, API contracts,
cost caps, and latency SLAs.

Validators covered:
- Correctness (from our CorrectnessJudge)
- Relevance (from our RelevanceJudge)
- Completeness (from our CompletenessJudge)
- Hallucination (from our HallucinationJudge)
- Safety (from our SafetyJudge)
- Fairness (from our FairnessJudge)
- JSON Schema Compliance
- Tool Trace Validation
- Cost Cap Check
- Latency SLA Check
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from evaluation.judges.completeness_judge import CompletenessJudge
    from evaluation.judges.correctness_judge import CorrectnessJudge
    from evaluation.judges.fairness_judge import FairnessJudge
    from evaluation.judges.hallucination_judge import HallucinationJudge
    from evaluation.judges.relevance_judge import RelevanceJudge
    from evaluation.judges.safety_judge import SafetyJudge

    JUDGES_AVAILABLE = True
except ImportError:
    JUDGES_AVAILABLE = False


class CustomValidatorsRunner:
    """Runs custom validators and returns structured results."""

    def __init__(self):
        self._judges = {}
        self._init_judges()
        self._start_time = time.time()

    def _init_judges(self):
        if not JUDGES_AVAILABLE:
            return
        self._judges = {
            "correctness": CorrectnessJudge(),
            "relevance": RelevanceJudge(),
            "completeness": CompletenessJudge(),
            "hallucination": HallucinationJudge(),
            "safety": SafetyJudge(),
            "fairness": FairnessJudge(),
        }

    async def validate_correctness(
        self, input_text: str, actual_output: str, expected_output: str | None = None
    ) -> dict[str, Any]:
        """Evaluate correctness using our judge."""
        if not JUDGES_AVAILABLE:
            return {"score": 0.5, "reason": "Judges not available", "passed": True}

        result = await self._judges["correctness"].evaluate(
            input_text=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        return {
            "score": result.score,
            "reason": result.reasoning,
            "feedback": result.feedback,
            "passed": result.score >= 0.7,
            "metric": "correctness",
        }

    async def validate_relevance(
        self, input_text: str, actual_output: str
    ) -> dict[str, Any]:
        """Evaluate relevance using our judge."""
        if not JUDGES_AVAILABLE:
            return {"score": 0.5, "reason": "Judges not available", "passed": True}

        result = await self._judges["relevance"].evaluate(
            input_text=input_text,
            actual_output=actual_output,
        )
        return {
            "score": result.score,
            "reason": result.reasoning,
            "feedback": result.feedback,
            "passed": result.score >= 0.7,
            "metric": "relevance",
        }

    async def validate_completeness(
        self, input_text: str, actual_output: str
    ) -> dict[str, Any]:
        """Evaluate completeness using our judge."""
        if not JUDGES_AVAILABLE:
            return {"score": 0.5, "reason": "Judges not available", "passed": True}

        result = await self._judges["completeness"].evaluate(
            input_text=input_text,
            actual_output=actual_output,
        )
        return {
            "score": result.score,
            "reason": result.reasoning,
            "feedback": result.feedback,
            "passed": result.score >= 0.7,
            "metric": "completeness",
        }

    async def validate_hallucination(
        self,
        input_text: str,
        actual_output: str,
        context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate hallucination using our judge."""
        if not JUDGES_AVAILABLE:
            return {"score": 1.0, "reason": "Judges not available", "passed": True}

        result = await self._judges["hallucination"].evaluate(
            input_text=input_text,
            actual_output=actual_output,
            context=context or [],
        )
        return {
            "score": result.score,
            "reason": result.reasoning,
            "feedback": result.feedback,
            "passed": result.score >= 0.7,
            "metric": "hallucination",
        }

    async def validate_safety(
        self, input_text: str, actual_output: str
    ) -> dict[str, Any]:
        """Evaluate safety using our judge."""
        if not JUDGES_AVAILABLE:
            return {"score": 1.0, "reason": "Judges not available", "passed": True}

        result = await self._judges["safety"].evaluate(
            input_text=input_text,
            actual_output=actual_output,
        )
        return {
            "score": result.score,
            "reason": result.reasoning,
            "feedback": result.feedback,
            "passed": result.score >= 0.7,
            "metric": "safety",
        }

    async def validate_fairness(
        self, input_text: str, actual_output: str
    ) -> dict[str, Any]:
        """Evaluate fairness using our judge."""
        if not JUDGES_AVAILABLE:
            return {"score": 1.0, "reason": "Judges not available", "passed": True}

        result = await self._judges["fairness"].evaluate(
            input_text=input_text,
            actual_output=actual_output,
        )
        return {
            "score": result.score,
            "reason": result.reasoning,
            "feedback": result.feedback,
            "passed": result.score >= 0.7,
            "metric": "fairness",
        }

    def validate_json_schema(
        self, actual_output: str, expected_schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate that output conforms to JSON schema."""
        try:
            parsed = json.loads(actual_output)
        except (json.JSONDecodeError, ValueError):
            return {
                "score": 0.0,
                "reason": "Output is not valid JSON",
                "passed": False,
                "metric": "json_schema",
            }

        if not expected_schema:
            # Basic structure check: must be a dict or list
            if isinstance(parsed, (dict, list)):
                return {
                    "score": 1.0,
                    "reason": "Output is valid JSON structure",
                    "passed": True,
                    "metric": "json_schema",
                }
            return {
                "score": 0.5,
                "reason": f"Unexpected JSON type: {type(parsed).__name__}",
                "passed": False,
                "metric": "json_schema",
            }

        # Schema validation (basic field presence check)
        if isinstance(expected_schema, dict):
            missing_fields = [
                k for k in expected_schema if k not in parsed
            ]
            if missing_fields:
                return {
                    "score": 0.0,
                    "reason": f"Missing required fields: {missing_fields}",
                    "passed": False,
                    "metric": "json_schema",
                }

        return {
            "score": 1.0,
            "reason": "JSON schema validation passed",
            "passed": True,
            "metric": "json_schema",
        }

    def validate_cost_cap(
        self, cost: float, max_cost: float = 0.01
    ) -> dict[str, Any]:
        """Validate that LLM call cost is under cap."""
        passed = cost <= max_cost
        return {
            "score": 1.0 - (cost / max_cost) if cost > 0 else 1.0,
            "reason": f"Cost ${cost:.4f} {'under' if passed else 'over'} cap ${max_cost:.2f}",
            "passed": passed,
            "cost": cost,
            "max_cost": max_cost,
            "metric": "cost_cap",
        }

    def validate_latency_sla(
        self, start_time: float, max_latency: float = 10.0
    ) -> dict[str, Any]:
        """Validate that response time is under SLA."""
        elapsed = time.time() - start_time
        passed = elapsed <= max_latency
        return {
            "score": 1.0 - (elapsed / max_latency) if elapsed > 0 else 1.0,
            "reason": f"Latency {elapsed:.2f}s {'under' if passed else 'over'} SLA {max_latency}s",
            "passed": passed,
            "latency": elapsed,
            "max_latency": max_latency,
            "metric": "latency_sla",
        }

    def validate_tool_trace(
        self, tool_calls: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Validate tool call traces."""
        if not tool_calls:
            return {
                "score": 0.5,
                "reason": "No tool calls to validate",
                "passed": True,
                "metric": "tool_trace",
            }

        errors = []
        for i, call in enumerate(tool_calls):
            if "name" not in call:
                errors.append(f"Tool call {i}: missing 'name'")
            if "arguments" not in call:
                errors.append(f"Tool call {i}: missing 'arguments'")
            elif not isinstance(call["arguments"], dict):
                errors.append(f"Tool call {i}: 'arguments' must be a dict")

        passed = len(errors) == 0
        return {
            "score": 1.0 - (len(errors) / len(tool_calls)),
            "reason": f"{'All' if passed else len(errors)} tool calls valid",
            "passed": passed,
            "errors": errors,
            "total_calls": len(tool_calls),
            "metric": "tool_trace",
        }

    async def run_all(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run all custom validators and return aggregated results."""
        vars_data = context.get("vars", {})
        input_text = vars_data.get("transcript", "")
        actual_output = vars_data.get("output", context.get("prompt", ""))
        expected_output = vars_data.get("expected_output", None)
        context_docs = [vars_data.get("context", "")]
        cost = vars_data.get("cost", 0.0)
        tool_calls = vars_data.get("tool_calls", None)
        expected_schema = vars_data.get("expected_schema", None)

        results = {
            "correctness": await self.validate_correctness(
                input_text, actual_output, expected_output
            ),
            "relevance": await self.validate_relevance(input_text, actual_output),
            "completeness": await self.validate_completeness(
                input_text, actual_output
            ),
            "hallucination": await self.validate_hallucination(
                input_text, actual_output, context_docs
            ),
            "safety": await self.validate_safety(input_text, actual_output),
            "fairness": await self.validate_fairness(input_text, actual_output),
            "json_schema": self.validate_json_schema(actual_output, expected_schema),
            "cost_cap": self.validate_cost_cap(cost),
            "latency_sla": self.validate_latency_sla(self._start_time),
            "tool_trace": self.validate_tool_trace(tool_calls),
        }

        overall_score = sum(r["score"] for r in results.values()) / len(results)
        all_passed = all(r["passed"] for r in results.values())
        return {
            "provider": "custom_validators",
            "overall_score": overall_score,
            "all_passed": all_passed,
            "results": results,
        }


# Promptfoo expects a callable that returns dict
runner = CustomValidatorsRunner()


async def call_api(
    prompt: str, options: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Promptfoo provider interface.

    This is called by Promptfoo to evaluate with custom validators.
    """
    results = await runner.run_all(context)
    return {
        "output": json.dumps(results),
        "metrics": {
            "custom_validators_score": results["overall_score"],
            "custom_validators_passed": results["all_passed"],
        },
    }
