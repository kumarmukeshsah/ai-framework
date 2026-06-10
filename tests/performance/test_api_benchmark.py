"""Performance benchmarks for API endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport

from product.api.app import app


@pytest.mark.asyncio
class TestAPIPerformance:
    """Performance benchmarks for API endpoints."""

    @pytest.fixture
    async def client(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    async def test_health_endpoint(self, client, benchmark):
        """Benchmark health check endpoint."""
        async def run():
            response = await client.get("/health")
            return response

        result = benchmark(run)
        assert result.status_code == 200

    async def test_evaluate_endpoint(self, client, benchmark):
        """Benchmark evaluate endpoint."""
        async def run():
            response = await client.post(
                "/evaluate",
                json={
                    "transcript": "I have 5 years of experience with Python and FastAPI.",
                    "use_llm": False,
                },
            )
            return response

        result = benchmark(run)
        assert result.status_code == 200

    async def test_evaluate_senior_endpoint(self, client, benchmark):
        """Benchmark senior candidate evaluation."""
        async def run():
            response = await client.post(
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
            return response

        result = benchmark(run)
        assert result.status_code == 200

    async def test_prompts_list_endpoint(self, client, benchmark):
        """Benchmark prompts listing endpoint."""
        async def run():
            response = await client.get("/prompts")
            return response

        result = benchmark(run)
        assert result.status_code == 200

    async def test_root_endpoint(self, client, benchmark):
        """Benchmark root endpoint."""
        async def run():
            response = await client.get("/")
            return response

        result = benchmark(run)
        assert result.status_code == 200