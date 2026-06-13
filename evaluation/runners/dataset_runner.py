"""Evaluation dataset runner for batch execution and reporting."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from product.agents.evaluator import EvaluatorAgent
from product.core.logging import get_logger

logger = get_logger(__name__)


class EvaluationScore(BaseModel):
    """Score for a single evaluation."""

    total_score: float = 0.0
    level_accuracy: bool = False
    score_error: float = 0.0
    recommendation_match: bool = False
    skill_overlap: float = 0.0
    experience_error: float = 0.0


class EvaluationResult(BaseModel):
    """Result for a single dataset item."""

    input: str
    expected: dict[str, Any]
    actual: dict[str, Any] | None = None
    score: EvaluationScore | None = None
    error: str | None = None
    duration_ms: float = 0.0


class DatasetReport(BaseModel):
    """Report for a dataset evaluation run."""

    dataset_name: str
    total_items: int
    passed: int
    failed: int
    avg_score: float = 0.0
    level_accuracy: float = 0.0
    avg_score_error: float = 0.0
    recommendation_accuracy: float = 0.0
    avg_skill_overlap: float = 0.0
    avg_duration_ms: float = 0.0
    results: list[EvaluationResult] = []

    @property
    def pass_rate(self) -> float:
        """Get pass rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.passed / self.total_items) * 100

    def summary(self) -> dict[str, Any]:
        """Get summary dict."""
        return {
            "dataset": self.dataset_name,
            "total": self.total_items,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.pass_rate:.1f}%",
            "metrics": {
                "avg_score": round(self.avg_score, 2),
                "level_accuracy": f"{self.level_accuracy:.1%}",
                "recommendation_accuracy": f"{self.recommendation_accuracy:.1%}",
                "avg_score_error": round(self.avg_score_error, 2),
                "avg_skill_overlap": f"{self.avg_skill_overlap:.1%}",
                "avg_duration_ms": round(self.avg_duration_ms, 1),
            },
        }


class DatasetRunner:
    """Runner for evaluating agents against golden datasets.

    Features:
    - Batch execution
    - Scoring against expected outputs
    - Report generation
    - Pass/fail assessment
    """

    def __init__(self, agent: EvaluatorAgent, threshold: float = 0.7):
        """Initialize dataset runner.

        Args:
            agent: Interview agent to evaluate
            threshold: Score threshold for pass/fail (0.0 to 1.0)
        """
        self.agent = agent
        self.threshold = threshold

    async def run_dataset(
        self,
        dataset_path: str | Path,
    ) -> DatasetReport:
        """Run agent against a dataset and generate report.

        Args:
            dataset_path: Path to dataset JSON file

        Returns:
            DatasetReport with results
        """
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.stem

        with dataset_path.open() as f:
            dataset = json.load(f)

        results = []
        passed = 0
        failed = 0
        total_score = 0.0
        level_correct = 0
        recommendation_correct = 0
        total_score_error = 0.0
        total_skill_overlap = 0.0
        total_duration = 0.0

        logger.info(f"Running dataset '{dataset_name}' with {len(dataset)} items")

        for item in dataset:
            input_text = item["input"]
            expected = item["expected"]

            start_time = time.monotonic()
            try:
                evaluation = await self.agent.process(input_text)
                duration = (time.monotonic() - start_time) * 1000

                actual = evaluation.model_dump()

                # Calculate scores
                score = self._evaluate_result(actual, expected)

                result = EvaluationResult(
                    input=input_text,
                    expected=expected,
                    actual=actual,
                    score=score,
                    duration_ms=duration,
                )

                if score.total_score >= self.threshold:
                    passed += 1
                else:
                    failed += 1

                total_score += score.total_score
                total_score_error += score.score_error
                total_skill_overlap += score.skill_overlap
                total_duration += duration

                if score.level_accuracy:
                    level_correct += 1
                if score.recommendation_match:
                    recommendation_correct += 1

            except Exception as e:
                duration = (time.monotonic() - start_time) * 1000
                failed += 1
                result = EvaluationResult(
                    input=input_text,
                    expected=expected,
                    error=str(e),
                    duration_ms=duration,
                )

            results.append(result)

        n = len(dataset)
        report = DatasetReport(
            dataset_name=dataset_name,
            total_items=n,
            passed=passed,
            failed=failed,
            avg_score=total_score / n if n > 0 else 0.0,
            level_accuracy=level_correct / n if n > 0 else 0.0,
            avg_score_error=total_score_error / n if n > 0 else 0.0,
            recommendation_accuracy=recommendation_correct / n if n > 0 else 0.0,
            avg_skill_overlap=total_skill_overlap / n if n > 0 else 0.0,
            avg_duration_ms=total_duration / n if n > 0 else 0.0,
            results=results,
        )

        logger.info(
            f"Dataset '{dataset_name}' complete: "
            f"{report.pass_rate:.1f}% pass rate "
            f"({report.passed}/{report.total_items})"
        )

        return report

    def _evaluate_result(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> EvaluationScore:
        """Evaluate actual vs expected result.

        Args:
            actual: Actual evaluation result
            expected: Expected evaluation result

        Returns:
            EvaluationScore with metrics
        """
        score = EvaluationScore()

        # Level accuracy
        score.level_accuracy = actual.get("candidate_level", "") == expected.get(
            "candidate_level", ""
        )

        # Score error (normalized 0-1)
        actual_score = actual.get("score", 0.0)
        expected_score = expected.get("score", 0.0)
        score.score_error = abs(actual_score - expected_score) / 10.0

        # Recommendation match
        score.recommendation_match = actual.get("recommendation", "") == expected.get(
            "recommendation", ""
        )

        # Skill overlap (Jaccard similarity)
        actual_skills = {s.lower() for s in actual.get("skills", [])}
        expected_skills = {s.lower() for s in expected.get("skills", [])}
        if actual_skills or expected_skills:
            intersection = actual_skills & expected_skills
            union = actual_skills | expected_skills
            score.skill_overlap = len(intersection) / len(union) if union else 0.0

        # Experience error
        actual_exp = actual.get("experience_years", 0.0)
        expected_exp = expected.get("experience_years", 0.0)
        score.experience_error = abs(actual_exp - expected_exp) / max(expected_exp, 1.0)

        # Total score (weighted combination)
        score.total_score = (
            (0.3 if score.level_accuracy else 0.0)
            + (0.3 * (1.0 - score.score_error))
            + (0.2 if score.recommendation_match else 0.0)
            + (0.2 * score.skill_overlap)
        )

        return score

    def save_report(self, report: DatasetReport, output_path: str | Path) -> None:
        """Save evaluation report to file.

        Args:
            report: Dataset report to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "summary": report.summary(),
            "results": [
                {
                    "input": r.input[:100],
                    "expected": r.expected,
                    "actual": r.actual,
                    "score": r.score.model_dump() if r.score else None,
                    "error": r.error,
                    "duration_ms": round(r.duration_ms, 1),
                }
                for r in report.results
            ],
        }

        with output_path.open("w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Report saved to {output_path}")
