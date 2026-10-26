"""Isolated API fixtures; real user data and saved benchmarks remain untouched."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app


@pytest.fixture
def api_settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        published_dir=Path(__file__).resolve().parents[1] / "docs/benchmark_results",
    )


@pytest.fixture
def client(api_settings):
    with TestClient(create_app(api_settings)) as client:
        yield client


@pytest.fixture
def index(client):
    response = client.post("/api/v1/indexes", json={"name": "test", "dimension": 2})
    assert response.status_code == 201
    return response.json()["id"]
