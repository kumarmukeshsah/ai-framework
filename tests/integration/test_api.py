"""Integration tests for API endpoints."""

from fastapi.testclient import TestClient

from product.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test health endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestEvaluationEndpoint:
    """Test evaluation endpoint."""

    def test_evaluate_success(self):
        """Test successful evaluation."""
        response = client.post(
            "/api/evaluate",
            json={"transcript": "I have 5 years of Python experience."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "candidate_level" in data
        assert "score" in data
        assert "recommendation" in data

    def test_evaluate_missing_transcript(self):
        """Test evaluation with missing transcript."""
        response = client.post("/api/evaluate", json={})
        assert response.status_code == 422

    def test_evaluate_junior_vs_senior(self):
        """Test junior vs senior evaluation."""
        junior_response = client.post(
            "/api/evaluate",
            json={"transcript": "I'm new to programming."},
        )

        senior_response = client.post(
            "/api/evaluate",
            json={"transcript": "I have 10+ years experience and have led teams."},
        )

        junior_data = junior_response.json()
        senior_data = senior_response.json()

        assert senior_data["score"] >= junior_data["score"]


class TestChatEndpoint:
    """Test chat endpoint."""

    def test_chat_success(self):
        """Test successful chat."""
        response = client.post(
            "/api/chat",
            json={"message": "Hello, how are you?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert "conversation_id" in data

    def test_chat_with_conversation_id(self):
        """Test chat with conversation ID."""
        response = client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "conversation_id": "conv-123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv-123"


class TestIndexEndpoint:
    """Test document indexing endpoint."""

    def test_index_document(self):
        """Test document indexing."""
        response = client.post(
            "/api/index",
            json={
                "title": "Test Document",
                "content": "This is test content.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["title"] == "Test Document"


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics(self):
        """Test metrics endpoint."""
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "evaluations_processed" in data
        assert "average_response_time_ms" in data
