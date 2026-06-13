"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_transcript():
    """Sample interview transcript."""
    return """I've been working as a software developer for 8 years.
    I have extensive experience with Python, FastAPI, Docker, and Kubernetes.
    I've led teams of engineers and architected several microservices systems."""


@pytest.fixture
def junior_transcript():
    """Junior developer transcript."""
    return """I'm a junior developer with about 1 year of experience.
    I'm learning Python and have some basic web development knowledge."""
