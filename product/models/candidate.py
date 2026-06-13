"""Candidate evaluation domain models."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator


class RubricBreakdown(BaseModel):
    """Detailed rubric scores for candidate evaluation."""

    technical_depth: float = Field(default=0, ge=0, le=3)
    problem_solving: float = Field(default=0, ge=0, le=3)
    communication: float = Field(default=0, ge=0, le=2)
    experience_relevance: float = Field(default=0, ge=0, le=2)


# ── Versioned schemas (V1, V2, V3) ────────────────────────────────────────


class RubricV1(BaseModel):
    """V1 rubric: equal-weight technical / problem / communication / experience (0-3 each)."""

    technical_skills: float = Field(default=0, ge=0, le=3)
    problem_solving: float = Field(default=0, ge=0, le=3)
    communication: float = Field(default=0, ge=0, le=3)
    experience: float = Field(default=0, ge=0, le=3)


class RubricV2(BaseModel):
    """V2 rubric: same as V1 but reused under V2 contract."""

    technical_skills: float = Field(default=0, ge=0, le=3)
    problem_solving: float = Field(default=0, ge=0, le=3)
    communication: float = Field(default=0, ge=0, le=3)
    experience: float = Field(default=0, ge=0, le=3)


class RubricV3(BaseModel):
    """V3 rubric: matches RubricBreakdown (technical_depth, problem_solving, communication, experience_relevance)."""

    technical_depth: float = Field(default=0, ge=0, le=3)
    problem_solving: float = Field(default=0, ge=0, le=3)
    communication: float = Field(default=0, ge=0, le=2)
    experience_relevance: float = Field(default=0, ge=0, le=2)


class CandidateEvaluationV1(BaseModel):
    """V1 candidate evaluation schema.

    Simple, flat schema with equal-weight rubric.
    """

    VALID_LEVELS: ClassVar[tuple[str, ...]] = ("Junior", "Mid", "Senior")
    VALID_RECOMMENDATIONS: ClassVar[tuple[str, ...]] = ("Hire", "Consider", "Reject")

    candidate_level: str = Field(..., description="Seniority level (Junior, Mid, Senior)")
    score: float = Field(..., ge=0, le=10, description="Overall evaluation score /10")
    recommendation: str = Field(..., description="Hiring recommendation")
    skills: list[str] = Field(default_factory=list, description="Identified technical skills")
    experience_years: float | None = Field(default=None, ge=0)
    feedback: str = Field(default="", description="Detailed evaluation feedback")

    @field_validator("candidate_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        if v not in cls.VALID_LEVELS:
            raise ValueError(f"candidate_level must be one of {cls.VALID_LEVELS}, got '{v}'")
        return v

    @field_validator("recommendation")
    @classmethod
    def _validate_recommendation(cls, v: str) -> str:
        if v not in cls.VALID_RECOMMENDATIONS:
            raise ValueError(
                f"recommendation must be one of {cls.VALID_RECOMMENDATIONS}, got '{v}'"
            )
        return v


class CandidateEvaluationV2(BaseModel):
    """V2 candidate evaluation schema.

    Adds rubric_breakdown, strengths/weaknesses, and extended recommendation set.
    """

    VALID_LEVELS: ClassVar[tuple[str, ...]] = ("Junior", "Mid", "Senior", "Lead")
    VALID_RECOMMENDATIONS: ClassVar[tuple[str, ...]] = (
        "Strong Hire",
        "Hire",
        "Consider",
        "Reject",
    )

    candidate_level: str = Field(..., description="Seniority level")
    score: float = Field(..., ge=0, le=10, description="Overall evaluation score /10")
    recommendation: str = Field(..., description="Hiring recommendation")
    skills: list[str] = Field(default_factory=list)
    rubric_breakdown: RubricV2 | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    @field_validator("candidate_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        if v not in cls.VALID_LEVELS:
            raise ValueError(f"candidate_level must be one of {cls.VALID_LEVELS}, got '{v}'")
        return v


class CandidateEvaluationV3(BaseModel):
    """V3 candidate evaluation schema.

    Final production schema with V3 rubric, chain-of-thought, and full provenance.
    """

    VALID_LEVELS: ClassVar[tuple[str, ...]] = ("Junior", "Mid", "Senior", "Lead")
    VALID_RECOMMENDATIONS: ClassVar[tuple[str, ...]] = (
        "Strong Hire",
        "Hire",
        "Consider",
        "Reject",
    )

    candidate_level: str = Field(..., description="Seniority level")
    score: float = Field(..., ge=0, le=10, description="Overall evaluation score /10")
    recommendation: str = Field(..., description="Hiring recommendation")
    skills: list[str] = Field(default_factory=list)
    experience_years: float | None = Field(default=None, ge=0)
    rubric: RubricV3 = Field(default_factory=RubricV3)
    feedback: str = Field(default="", description="Detailed evaluation feedback")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    chain_of_thought: str | None = Field(default=None, description="LLM reasoning trace")

    @field_validator("candidate_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        if v not in cls.VALID_LEVELS:
            raise ValueError(f"candidate_level must be one of {cls.VALID_LEVELS}, got '{v}'")
        return v

    @field_validator("recommendation")
    @classmethod
    def _validate_recommendation(cls, v: str) -> str:
        if v not in cls.VALID_RECOMMENDATIONS:
            raise ValueError(
                f"recommendation must be one of {cls.VALID_RECOMMENDATIONS}, got '{v}'"
            )
        return v


class CandidateEvaluation(BaseModel):
    """Full candidate evaluation result."""

    candidate_level: str = Field(..., description="Seniority level (Junior, Mid, Senior, Lead)")
    score: float = Field(..., ge=0, le=10, description="Overall evaluation score /10")
    recommendation: str = Field(..., description="Hiring recommendation (Hire, Consider, Reject)")
    skills: list[str] = Field(default_factory=list, description="Identified technical skills")
    experience_years: float | None = Field(default=None, ge=0)
    rubric: RubricBreakdown = Field(default_factory=RubricBreakdown)
    feedback: str = Field(default="", description="Detailed evaluation feedback")
    strengths: list[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Areas for improvement")
    chain_of_thought: str | None = Field(default=None, description="LLM reasoning trace")


class EvaluationStageResult(BaseModel):
    """Result from a single evaluation stage."""

    stage_name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluationPipelineResult(BaseModel):
    """Result from the complete multi-stage evaluation pipeline.

    Exposes the most common evaluation fields (candidate_level, score,
    recommendation, skills, strengths, weaknesses) as direct properties so
    callers and tests can do ``result.candidate_level`` instead of
    ``result.evaluation.candidate_level`` while still keeping the full
    evaluation available via ``result.evaluation``.
    """

    success: bool
    evaluation: CandidateEvaluation | None = None
    stages: list[EvaluationStageResult] = Field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None
    prompt_version: str | None = None

    # ── Convenience proxies for ergonomic access in tests/agents ──────────
    @property
    def candidate_level(self) -> str | None:  # type: ignore[override]
        return self.evaluation.candidate_level if self.evaluation else None

    @property
    def score(self) -> float | None:  # type: ignore[override]
        return self.evaluation.score if self.evaluation else None

    @property
    def recommendation(self) -> str | None:  # type: ignore[override]
        return self.evaluation.recommendation if self.evaluation else None

    @property
    def skills(self) -> list[str]:  # type: ignore[override]
        return list(self.evaluation.skills) if self.evaluation else []

    @property
    def strengths(self) -> list[str]:  # type: ignore[override]
        return list(self.evaluation.strengths) if self.evaluation else []

    @property
    def weaknesses(self) -> list[str]:  # type: ignore[override]
        return list(self.evaluation.weaknesses) if self.evaluation else []
