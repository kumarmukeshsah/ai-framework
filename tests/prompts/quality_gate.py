"""
Quality Gate for Promptfoo Evaluation Pipeline.

Aggregates results from all 3 evaluation tracks (DeepEval, RAGAS, Custom Validators)
and enforces threshold-based pass/fail decisions.

Usage:
    python tests/prompts/quality_gate.py --results tests/prompts/results/eval_results.json
    python tests/prompts/quality_gate.py --results tests/prompts/results/eval_results.json --threshold 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Thresholds for different evaluation tracks
DEFAULT_THRESHOLDS = {
    "overall": 0.8,  # Overall pass threshold
    "deepeval": {
        "hallucination": 0.7,  # Lower is better (inverted score)
        "faithfulness": 0.8,
        "toxicity": 0.7,  # Lower is better
        "bias": 0.7,  # Lower is better
    },
    "ragas": {
        "mrr": 0.5,
        "ndcg": 0.5,
        "recall": 0.5,
        "context_precision": 0.6,
        "context_recall": 0.6,
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
    },
    "custom_validators": {
        "correctness": 0.7,
        "relevance": 0.7,
        "completeness": 0.7,
        "hallucination": 0.7,
        "safety": 0.8,
        "fairness": 0.8,
        "json_schema": 0.8,
        "cost_cap": 0.8,
        "latency_sla": 0.8,
        "tool_trace": 0.8,
    },
}


class QualityGate:
    """Evaluates Promptfoo results and enforces quality thresholds."""

    def __init__(self, thresholds: dict[str, Any] | None = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.results: dict[str, Any] = {}
        self.violations: list[str] = []

    def load_results(self, results_path: str) -> dict[str, Any]:
        """Load Promptfoo evaluation results from JSON file."""
        path = Path(results_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(path) as f:
            self.results = json.load(f)
        return self.results

    def _check_deepeval(self, deepeval_results: dict[str, Any]) -> float:
        """Check DeepEval metrics against thresholds."""
        score_sum = 0.0
        count = 0
        thresholds = self.thresholds.get("deepeval", {})

        for metric, value in deepeval_results.items():
            if isinstance(value, dict):
                score = value.get("score", 0.0)
                metric_name = value.get("metric", metric)
                threshold = thresholds.get(metric_name, 0.7)

                # For hallucination, toxicity, bias - lower is better, invert score
                if metric_name in ("hallucination", "toxicity", "bias"):
                    score = 1.0 - score

                if score < threshold:
                    self.violations.append(
                        f"DeepEval {metric_name}: {score:.2f} < {threshold:.2f} (threshold)"
                    )
                score_sum += score
                count += 1

        return score_sum / count if count > 0 else 0.0

    def _check_ragas(self, ragas_results: dict[str, Any]) -> float:
        """Check RAGAS metrics against thresholds."""
        score_sum = 0.0
        count = 0
        thresholds = self.thresholds.get("ragas", {})

        for metric, value in ragas_results.items():
            if isinstance(value, dict):
                score = value.get("score", 0.0)
                metric_name = value.get("metric", metric)
                threshold = thresholds.get(metric_name, 0.5)

                if score < threshold:
                    self.violations.append(
                        f"RAGAS {metric_name}: {score:.2f} < {threshold:.2f} (threshold)"
                    )
                score_sum += score
                count += 1

        return score_sum / count if count > 0 else 0.0

    def _check_custom_validators(self, validator_results: dict[str, Any]) -> float:
        """Check custom validator metrics against thresholds."""
        score_sum = 0.0
        count = 0
        thresholds = self.thresholds.get("custom_validators", {})

        for metric, value in validator_results.items():
            if isinstance(value, dict):
                score = value.get("score", 0.0)
                metric_name = value.get("metric", metric)
                threshold = thresholds.get(metric_name, 0.7)

                if score < threshold:
                    self.violations.append(
                        f"Custom Validator {metric_name}: {score:.2f} < {threshold:.2f} (threshold)"
                    )
                score_sum += score
                count += 1

        return score_sum / count if count > 0 else 0.0

    def evaluate(self) -> dict[str, Any]:
        """Run quality gate evaluation against all tracks."""
        if not self.results:
            raise ValueError("No results loaded. Call load_results() first.")

        self.violations = []
        track_scores = {}

        # Check if results contain per-provider data
        results_data = self.results

        # DeepEval track
        deepeval_data = results_data.get("deepeval", results_data.get("DeepEval", {}))
        if deepeval_data:
            deepeval_inner = deepeval_data.get("results", deepeval_data)
            track_scores["deepeval"] = self._check_deepeval(deepeval_inner)
        else:
            track_scores["deepeval"] = 0.0

        # RAGAS track
        ragas_data = results_data.get("ragas", results_data.get("RAGAS", {}))
        if ragas_data:
            ragas_inner = ragas_data.get("results", ragas_data)
            track_scores["ragas"] = self._check_ragas(ragas_inner)
        else:
            track_scores["ragas"] = 0.0

        # Custom Validators track
        custom_data = results_data.get(
            "custom_validators", results_data.get("Custom Validators", {})
        )
        if custom_data:
            custom_inner = custom_data.get("results", custom_data)
            track_scores["custom_validators"] = self._check_custom_validators(
                custom_inner
            )
        else:
            track_scores["custom_validators"] = 0.0

        # Overall score
        overall = sum(track_scores.values()) / len(track_scores) if track_scores else 0.0

        threshold = self.thresholds.get("overall", 0.8)
        passed = overall >= threshold and len(self.violations) == 0

        return {
            "passed": passed,
            "overall_score": round(overall, 3),
            "threshold": threshold,
            "track_scores": {k: round(v, 3) for k, v in track_scores.items()},
            "violations": self.violations,
            "violation_count": len(self.violations),
            "summary": "✓ All quality gates passed!"
            if passed
            else f"✗ {len(self.violations)} quality gate violation(s) detected",
        }

    def generate_pr_comment(self, result: dict[str, Any]) -> str:
        """Generate a Markdown comment suitable for GitHub PRs."""
        lines = [
            "## 🤖 Promptfoo Quality Gate Results",
            "",
            f"**Overall Status:** {'✅ PASSED' if result['passed'] else '❌ FAILED'}",
            f"**Overall Score:** {result['overall_score']:.2f} / {result['threshold']:.2f}",
            "",
            "### Track Scores",
            "| Track | Score | Status |",
            "|-------|-------|--------|",
        ]

        for track, score in result["track_scores"].items():
            emoji = "✅" if score >= result["threshold"] else "⚠️"
            lines.append(f"| {track.title()} | {score:.2f} | {emoji} |")

        lines.extend([
            "",
            "### Violations",
        ])

        if result["violations"]:
            for v in result["violations"]:
                lines.append(f"- ❌ {v}")
        else:
            lines.append("- No violations found ✨")

        lines.extend([
            "",
            "### Details",
            "- Total tests evaluated across 3 tracks (DeepEval, RAGAS, Custom Validators)",
            f"- Threshold: {result['threshold']:.0%}",
            "- [View full results](tests/prompts/results/eval_results.json)",
        ])

        return "\n".join(lines)

    def generate_cli_report(self, result: dict[str, Any]) -> str:
        """Generate a CLI-friendly report."""
        lines = [
            "=" * 60,
            "  Promptfoo Quality Gate Report",
            "=" * 60,
            "",
            f"  Overall: {'✅ PASSED' if result['passed'] else '❌ FAILED'}",
            f"  Score:   {result['overall_score']:.2f} / {result['threshold']:.2f}",
            "",
            "  Track Scores:",
        ]

        for track, score in result["track_scores"].items():
            emoji = "✅" if score >= result["threshold"] else "⚠️"
            lines.append(f"    {emoji} {track.title()}: {score:.2f}")

        if result["violations"]:
            lines.extend([
                "",
                "  Violations:",
            ])
            for v in result["violations"]:
                lines.append(f"    ❌ {v}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Promptfoo Quality Gate - Aggregates and enforces evaluation thresholds"
    )
    parser.add_argument(
        "--results",
        default="tests/prompts/results/eval_results.json",
        help="Path to Promptfoo results JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Overall pass threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--format",
        choices=["cli", "pr-comment", "json"],
        default="cli",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        help="Path to write output file (optional)",
    )

    args = parser.parse_args()

    gate = QualityGate(thresholds={**DEFAULT_THRESHOLDS, "overall": args.threshold})

    try:
        gate.load_results(args.results)
        result = gate.evaluate()

        if args.format == "json":
            output = json.dumps(result, indent=2)
        elif args.format == "pr-comment":
            output = gate.generate_pr_comment(result)
        else:
            output = gate.generate_cli_report(result)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Report written to {args.output}")
        else:
            print(output)

        sys.exit(0 if result["passed"] else 1)

    except Exception as e:
        print(f"Error running quality gate: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
