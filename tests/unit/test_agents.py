"""Unit tests for agents."""

import pytest

from product.agents.evaluator import EvaluatorAgent


class TestInterviewAgent:
    """Test EvaluatorAgent."""

    def setup_method(self):
        """Setup test fixtures."""
        self.agent = EvaluatorAgent()

    @pytest.mark.asyncio
    async def test_process_junior_developer(self):
        """Test evaluating junior developer."""
        transcript = "I have 2 years of experience with Python and entry-level knowledge."
        result = await self.agent.process(transcript)

        assert result.candidate_level.lower() == "junior"
        assert result.recommendation in ["Hire", "Consider", "Reject"]
        assert result.score is not None and 0 <= result.score <= 10

    @pytest.mark.asyncio
    async def test_process_senior_developer(self):
        """Test evaluating senior developer."""
        transcript = (
            "I have 10+ years of experience, architected multiple systems, and mentored teams."
        )
        result = await self.agent.process(transcript)

        assert result.candidate_level.lower() == "senior"
        assert result.score is not None and 0 <= result.score <= 10

    @pytest.mark.asyncio
    async def test_skill_extraction(self):
        """Test skill extraction."""
        transcript = "I'm proficient in Python, FastAPI, Docker, and Kubernetes."
        result = await self.agent.process(transcript)

        assert len(result.skills) > 0

    @pytest.mark.asyncio
    async def test_recommendation_logic(self):
        """Test recommendation logic."""
        low_score_transcript = "I just started learning programming."
        result = await self.agent.process(low_score_transcript)

        if result.score is not None and result.score < 5.0:
            assert result.recommendation != "Hire"

    def test_keyword_extraction(self):
        """Test keyword extraction."""
        transcript = "I have 8 years of Python experience."
        keywords = self.agent._extract_keywords(transcript)

        assert any("python" in kw.lower() for kw in keywords) or "8_years_experience" in keywords
        assert any("8" in kw for kw in keywords)
