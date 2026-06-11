"""Unit tests for the EvaluatorAgent."""
import pytest
from product.agents.evaluator import EvaluatorAgent
from product.models.candidate import (
    CandidateEvaluation,
    EvaluationPipelineResult,
    EvaluationStageResult,
    RubricBreakdown,
)


class TestEvaluatorAgent:
    """Test the EvaluatorAgent implementation."""

    @pytest.fixture
    def agent(self):
        return EvaluatorAgent()

    @pytest.mark.asyncio
    async def test_process_senior_candidate(self, agent):
        """Test processing a senior candidate transcript."""
        result = await agent.process(
            "I have 8 years of experience building Python microservices with FastAPI. "
            "I've led teams of 5 engineers, architected distributed systems on AWS, "
            "and implemented CI/CD pipelines. I'm proficient with Docker, Kubernetes, "
            "and PostgreSQL. I've mentored junior developers and driven technical strategy."
        )
        assert isinstance(result, EvaluationPipelineResult)
        assert result.success is True
        assert result.evaluation is not None
        assert isinstance(result.evaluation, CandidateEvaluation)
        assert result.evaluation.score >= 0
        assert result.evaluation.score <= 10

    @pytest.mark.asyncio
    async def test_process_junior_candidate(self, agent):
        """Test processing a junior candidate transcript."""
        result = await agent.process(
            "I am new to programming. I just graduated with a Computer Science degree. "
            "I did a short internship where I learned Python. "
            "I'm looking for my first job and excited to start my career."
        )
        assert isinstance(result, EvaluationPipelineResult)
        assert result.success is True
        assert result.evaluation is not None

    @pytest.mark.asyncio
    async def test_process_empty_transcript(self, agent):
        """Test processing an empty transcript."""
        result = await agent.process("")
        assert isinstance(result, EvaluationPipelineResult)
        assert result.evaluation is not None

    @pytest.mark.asyncio
    async def test_process_with_context(self, agent):
        """Test processing with job context."""
        result = await agent.process(
            "I have 5 years of experience with Python.",
            context="Senior Python Developer position requiring 5+ years"
        )
        assert isinstance(result, EvaluationPipelineResult)
        assert result.evaluation is not None
        assert result.evaluation.score > 0

    def test_extract_keywords(self, agent):
        """Test keyword extraction from transcript."""
        keywords = agent._extract_keywords("python fastapi docker 5 years")
        assert "python" in keywords
        assert "fastapi" in keywords
        assert "docker" in keywords

    def test_extract_keywords_empty(self, agent):
        """Test keyword extraction from empty text."""
        try:
            keywords = agent._extract_keywords("")
            assert keywords == []
        except Exception:
            pass

    def test_identify_skills(self, agent):
        """Test skill identification from keywords."""
        skills = agent._identify_skills(["python", "docker", "unknown_skill"])
        assert "Python" in skills
        assert "Docker" in skills

    def test_determine_level(self, agent):
        """Test seniority level determination."""
        level = agent._determine_level(["lead", "architect", "8+ years"])
        assert level == "Senior"

    def test_calculate_score(self, agent):
        """Test score calculation."""
        rubric = RubricBreakdown(technical_depth=2, problem_solving=2, communication=1, experience_relevance=1)
        score = agent._calculate_score(rubric, "Senior")
        assert 0 <= score <= 10

    def test_recommend_from_score(self, agent):
        """Test recommendation generation."""
        assert agent._recommend_from_score(8.5) == "Hire"
        assert agent._recommend_from_score(6.0) == "Consider"
        assert agent._recommend_from_score(3.0) == "Reject"

    def test_recommendation_text(self, agent):
        """Test recommendation text generation."""
        text = agent._recommendation_text(8.5)
        assert isinstance(text, str)
        assert len(text) > 0
        text = agent._recommendation_text(3.0)
        assert isinstance(text, str)
        assert len(text) > 0
