"""Contract tests for Pydantic schema validation.

Build must fail if contract changes.
These tests ensure response schemas remain stable.
"""
import pytest
from pydantic import ValidationError

from product.models.candidate import (
    CandidateEvaluationV1,
    CandidateEvaluationV2,
    CandidateEvaluationV3,
    RubricBreakdown,
    RubricV3,
)


class TestCandidateEvaluationV1Contract:
    """Contract tests for CandidateEvaluationV1 schema."""

    def test_valid_evaluation(self):
        """Test valid evaluation data."""
        data = {
            "candidate_level": "Senior",
            "score": 8.5,
            "recommendation": "Hire",
            "skills": ["Python", "FastAPI"],
            "experience_years": 6.0,
            "feedback": "Excellent candidate",
        }
        evaluation = CandidateEvaluationV1(**data)
        assert evaluation.candidate_level == "Senior"
        assert evaluation.score == 8.5

    def test_minimal_evaluation(self):
        """Test minimal valid input."""
        data = {
            "candidate_level": "Junior",
            "score": 5.0,
            "recommendation": "Consider",
        }
        evaluation = CandidateEvaluationV1(**data)
        assert evaluation.skills == []
        assert evaluation.feedback == ""

    def test_invalid_level(self):
        """Test invalid candidate level."""
        with pytest.raises(ValidationError):
            CandidateEvaluationV1(
                candidate_level="Invalid",
                score=5.0,
                recommendation="Consider",
            )

    def test_invalid_score_range(self):
        """Test score out of range."""
        with pytest.raises(ValidationError):
            CandidateEvaluationV1(
                candidate_level="Senior",
                score=15.0,
                recommendation="Hire",
            )

    def test_invalid_recommendation(self):
        """Test invalid recommendation."""
        with pytest.raises(ValidationError):
            CandidateEvaluationV1(
                candidate_level="Senior",
                score=8.0,
                recommendation="Maybe",
            )


class TestCandidateEvaluationV2Contract:
    """Contract tests for CandidateEvaluationV2 schema."""

    def test_valid_with_rubric(self):
        """Test evaluation with rubric breakdown."""
        data = {
            "candidate_level": "Senior",
            "score": 8.5,
            "recommendation": "Hire",
            "skills": ["Python"],
            "rubric_breakdown": {
                "technical_skills": 2.5,
                "problem_solving": 2.5,
                "communication": 1.5,
                "experience": 2.0,
            },
            "strengths": ["Strong technical skills"],
            "weaknesses": [],
        }
        evaluation = CandidateEvaluationV2(**data)
        assert evaluation.rubric_breakdown is not None
        assert evaluation.rubric_breakdown.technical_skills == 2.5

    def test_v2_valid_recommendations(self):
        """Test V2 recommendation values."""
        for rec in ["Strong Hire", "Hire", "Consider", "Reject"]:
            data = {
                "candidate_level": "Senior",
                "score": 8.0,
                "recommendation": rec,
            }
            evaluation = CandidateEvaluationV2(**data)
            assert evaluation.recommendation == rec


class TestCandidateEvaluationV3Contract:
    """Contract tests for CandidateEvaluationV3 schema."""

    def test_valid_v3_full(self):
        """Test V3 with all fields."""
        data = {
            "candidate_level": "Senior",
            "score": 8.5,
            "recommendation": "Strong Hire",
            "skills": ["Python", "FastAPI"],
            "experience_years": 8.0,
            "rubric": {
                "technical_depth": 2.5,
                "problem_solving": 2.5,
                "communication": 1.5,
                "experience_relevance": 2.0,
            },
            "feedback": "Excellent",
            "strengths": ["Leadership"],
            "weaknesses": [],
        }
        evaluation = CandidateEvaluationV3(**data)
        assert evaluation.rubric.technical_depth == 2.5

    def test_v3_required_fields(self):
        """Test V3 required fields."""
        data = {
            "candidate_level": "Lead",
            "score": 9.0,
            "recommendation": "Strong Hire",
            "skills": ["Python"],
            "experience_years": 10.0,
            "rubric": {
                "technical_depth": 3.0,
                "problem_solving": 3.0,
                "communication": 2.0,
                "experience_relevance": 2.0,
            },
            "feedback": "Great",
        }
        evaluation = CandidateEvaluationV3(**data)
        assert evaluation.strengths == []
        assert evaluation.chain_of_thought is None


class TestRubricContracts:
    """Contract tests for rubric schemas."""

    def test_rubric_breakdown_bounds(self):
        """Test rubric breakdown bounds."""
        with pytest.raises(ValidationError):
            RubricBreakdown(
                technical_depth=5.0,
                problem_solving=2.0,
                communication=1.0,
                experience_relevance=1.0,
            )

    def test_rubric_v3_bounds(self):
        """Test rubric V3 bounds."""
        with pytest.raises(ValidationError):
            RubricV3(
                technical_depth=5.0,
                problem_solving=1.0,
                communication=1.0,
                experience_relevance=1.0,
            )