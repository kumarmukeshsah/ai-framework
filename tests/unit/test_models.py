"""Unit tests for models."""
import pytest

from product.models.candidate import CandidateEvaluation
from product.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
)


class TestCandidateEvaluation:
    """Test CandidateEvaluation model."""

    def test_valid_evaluation(self):
        """Test creating valid evaluation."""
        eval = CandidateEvaluation(
            candidate_level="Senior",
            score=8.5,
            recommendation="Hire",
            skills=["Python", "FastAPI"],
        )
        assert eval.candidate_level == "Senior"
        assert eval.score == 8.5

    def test_evaluation_defaults(self):
        """Test evaluation with defaults."""
        eval = CandidateEvaluation(
            candidate_level="Mid",
            score=6.0,
            recommendation="Consider",
        )
        assert eval.feedback == ""


class TestSchemas:
    """Test API schemas."""

    def test_evaluation_request(self):
        """Test evaluation request schema."""
        req = EvaluationRequest(transcript="I have 5 years of experience")
        assert req.transcript == "I have 5 years of experience"
        assert req.context is None

    def test_evaluation_response(self):
        """Test evaluation response schema."""
        resp = EvaluationResponse(
            success=True,
            evaluation={"candidate_level": "Mid", "score": 6.5, "recommendation": "Consider"},
        )
        assert resp.success is True
        assert resp.evaluation is not None

    def test_chat_request(self):
        """Test chat request schema."""
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.conversation_id is None

    def test_chat_response(self):
        """Test chat response schema."""
        resp = ChatResponse(success=True, response="Hi there", conversation_id="conv-123")
        assert resp.response == "Hi there"
        assert resp.conversation_id == "conv-123"

    def test_index_request(self):
        """Test document index request."""
        req = IndexRequest(
            document_id="doc-1",
            content="Content here",
        )
        assert req.document_id == "doc-1"
        assert req.metadata is None

    def test_health_response(self):
        """Test health response."""
        resp = HealthResponse(
            status="healthy",
            version="1.0.0",
            environment="development",
        )
        assert resp.status == "healthy"
