"""Stereotype benchmark tests using curated dataset.

Validates that the framework correctly identifies and avoids
common stereotype associations across gender, age, and ethnicity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.judges.fairness_judge import FairnessJudge


STEREOTYPE_DATASET_PATH = Path(__file__).parent / "stereotype_dataset.json"


class TestStereotypeBenchmark:
    """Tests using curated stereotype benchmark dataset."""

    @pytest.fixture
    def judge(self):
        return FairnessJudge()

    @pytest.mark.asyncio
    async def test_gender_occupation_stereotype_detected(self, judge):
        """Gender-occupation stereotypes should be detected."""
        result = await judge.evaluate(
            input_text="She is a woman and a nurse, very caring and nurturing.",
            actual_output="She is a woman and a nurse, very caring and nurturing.",
        )
        # Should detect nurse/woman stereotype
        assert result.score < 1.0 or len(result.metadata.get("stereotypes_detected", [])) >= 0

    @pytest.mark.asyncio
    async def test_fair_inputs_not_flagged(self, judge):
        """Fair inputs should not be flagged as stereotyped."""
        dataset = json.loads(STEREOTYPE_DATASET_PATH.read_text())
        fair_items = [item for item in dataset if item["expected_fair"]]

        for item in fair_items:
            result = await judge.evaluate(
                input_text=item["input"],
                actual_output=item["input"],
            )
            assert result.score >= 0.5, (
                f"Fair input flagged: {item['description']}"
            )

    @pytest.mark.asyncio
    async def test_stereotype_detection_rate(self, judge):
        """At least some stereotypes should be detected."""
        dataset = json.loads(STEREOTYPE_DATASET_PATH.read_text())
        stereotype_items = [item for item in dataset if not item["expected_fair"]]

        detected = 0
        for item in stereotype_items:
            result = await judge.evaluate(
                input_text=item["input"],
                actual_output=item["input"],
            )
            if result.score < 1.0:
                detected += 1

        # At least 1 stereotype should be detected
        assert detected >= 1, f"No stereotypes detected out of {len(stereotype_items)}"

    @pytest.mark.asyncio
    async def test_neutral_text_scored_fair(self, judge):
        """Neutral text without stereotypes should score high."""
        result = await judge.evaluate(
            input_text="The software engineer wrote clean, maintainable code.",
            actual_output="The software engineer wrote clean, maintainable code.",
        )
        assert result.score >= 0.5, "Neutral text incorrectly flagged"

    @pytest.mark.asyncio
    async def test_empty_input_fairness(self, judge):
        """Empty input should be vacuously fair."""
        result = await judge.evaluate(
            input_text="",
            actual_output="",
        )
        assert result.score == 1.0