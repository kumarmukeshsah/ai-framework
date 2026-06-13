"""Performance benchmarks for API endpoints (sync client for benchmark compatibility)."""

import pytest
from starlette.testclient import TestClient

from product.api.app import app


class TestAPIPerformance:
    """Performance benchmarks for API endpoints."""

    @pytest.fixture
    def client(self):
        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client, benchmark):
        """Benchmark health check endpoint."""
        result = benchmark(client.get, "/health")
        assert result.status_code == 200

    def test_evaluate_endpoint(self, client, benchmark):
        """Benchmark evaluate endpoint."""

        def run():
            return client.post(
                "/evaluate",
                json={
                    "transcript": "I have 5 years of experience with Python and FastAPI.",
                    "use_llm": False,
                },
            )

        result = benchmark(run)
        assert result.status_code == 200

    def test_evaluate_senior_endpoint(self, client, benchmark):
        """Benchmark senior candidate evaluation."""

        def run():
            return client.post(
                "/evaluate",
                json={
                    "transcript": (
                        "I have 8 years of experience building Python microservices with FastAPI. "
                        "I've led teams of 5 engineers, architected distributed systems on AWS, "
                        "and implemented CI/CD pipelines. I'm proficient with Docker, Kubernetes, "
                        "and PostgreSQL. I've mentored junior developers and driven technical strategy."
                    ),
                    "use_llm": False,
                },
            )

        result = benchmark(run)
        assert result.status_code == 200

    def test_prompts_list_endpoint(self, client, benchmark):
        """Benchmark prompts listing endpoint."""
        result = benchmark(client.get, "/prompts")
        assert result.status_code == 200

    def test_root_endpoint(self, client, benchmark):
        """Benchmark root endpoint."""
        result = benchmark(client.get, "/")
        assert result.status_code == 200
