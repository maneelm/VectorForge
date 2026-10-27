"""Phase 5 configuration/security tests; no Docker daemon or cloud required."""

import json
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app
from backend.schemas import JobInfo, QuickRun
from backend.services.benchmark_manager import BenchmarkManager
from backend.services.reports import ReportService


@pytest.mark.parametrize(
    "name,value",
    [
        ("VECTORFORGE_ENV", "staging"),
        ("VECTORFORGE_API_PORT", "0"),
        ("VECTORFORGE_API_PORT", "65536"),
        ("VECTORFORGE_API_HOST", "untrusted.example"),
        ("VECTORFORGE_ALLOWED_HOSTS", "*"),
        ("VECTORFORGE_ALLOWED_HOSTS", "*.example.com"),
        ("VECTORFORGE_ALLOWED_HOSTS", "localhost:8000"),
        ("VECTORFORGE_ALLOWED_HOSTS", ""),
        ("VECTORFORGE_CORS_ORIGINS", "*"),
        ("VECTORFORGE_CORS_ORIGINS", "https://example.com/path"),
        ("VECTORFORGE_CORS_ORIGINS", "https://secret:password@example.com"),
        ("VECTORFORGE_CORS_ORIGINS", "http://localhost:invalid"),
        ("VECTORFORGE_ENABLE_DOCS", "yes"),
        ("VECTORFORGE_ENABLE_BENCHMARKS", "1"),
        ("VECTORFORGE_JOB_TIMEOUT", "181"),
        ("VECTORFORGE_MAX_INDEXES", "17"),
        ("VECTORFORGE_MAX_PENDING_JOBS", "3"),
    ],
)
def test_invalid_environment_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings()


def test_production_defaults_are_narrow(monkeypatch):
    for name in ("VECTORFORGE_CORS_ORIGINS", "VECTORFORGE_ALLOWED_HOSTS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(environment="production")
    assert settings.origins == () and settings.allowed_hosts == ("localhost", "127.0.0.1")
    with pytest.raises(ValueError):
        Settings(environment="production", api_port=80)


def test_custom_environment_paths_hosts_and_limits(monkeypatch, tmp_path):
    values = {
        "VECTORFORGE_ENV": "production",
        "VECTORFORGE_API_PORT": "8123",
        "VECTORFORGE_API_HOST": "0.0.0.0",
        "VECTORFORGE_ALLOWED_HOSTS": "example.com,backend,127.0.0.1",
        "VECTORFORGE_CORS_ORIGINS": "https://example.com",
        "VECTORFORGE_DATA_DIR": str(tmp_path / "archives"),
        "VECTORFORGE_RESULTS_DIR": str(tmp_path / "reports"),
        "VECTORFORGE_JOB_TIMEOUT": "30",
        "VECTORFORGE_MAX_PENDING_JOBS": "1",
        "VECTORFORGE_MAX_INDEXES": "4",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = Settings()
    assert (
        settings.api_port,
        settings.job_timeout,
        settings.max_pending_jobs,
        settings.max_indexes,
    ) == (8123, 30, 1, 4)
    assert settings.origins == ("https://example.com",)
    with TestClient(create_app(settings), base_url="http://example.com") as client:
        health = client.get("/api/v1/health").json()
        assert health["benchmark_limits"] == {"max_pending_jobs": 1, "job_timeout_seconds": 30}
        assert (
            client.get("/api/v1/health", headers={"host": "untrusted.example"}).status_code == 400
        )
    assert settings.data_dir.is_dir() and settings.results_dir.is_dir()


def test_production_save_load_survives_app_recreation(api_settings):
    settings = replace(api_settings, environment="production", allowed_hosts=("testserver",))
    with TestClient(create_app(settings)) as client:
        info = client.post("/api/v1/indexes", json={"name": "persist", "dimension": 2}).json()
        path = "/api/v1/indexes/" + info["id"]
        client.post(path + "/vectors", json={"vectors": [[1, 0]], "ids": ["9223372036854775807"]})
        expected = client.post(path + "/search", json={"query": [1, 0]}).json()["matches"]
        saved = client.post(path + "/save").json()
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/indexes").json() == []
        loaded = client.post(
            "/api/v1/indexes/load", json={"archive_id": saved["archive_id"]}
        ).json()
        assert (
            client.post(
                "/api/v1/indexes/" + loaded["id"] + "/search", json={"query": [1, 0]}
            ).json()["matches"]
            == expected
        )
        assert client.get("/api/v1/benchmarks/runs").json()["runs"][0]["source"] == "published"


def test_readonly_or_invalid_storage_fails_at_startup(api_settings, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    settings = replace(api_settings, environment="production", data_dir=blocked)
    with pytest.raises(OSError):
        with TestClient(create_app(settings)):
            pass


def test_disabled_docs_and_benchmarks(api_settings, monkeypatch):
    from backend.services import benchmark_manager as module

    monkeypatch.setattr(module, "missing_dependencies", lambda: [])
    settings = replace(api_settings, enable_docs=False, enable_benchmarks=False)
    with TestClient(create_app(settings)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        health = client.get("/api/v1/health").json()
        assert not health["can_run_benchmarks"] and not health["benchmark_enabled"]
        assert (
            client.post("/api/v1/benchmarks/runs", json={}).json()["error"]["code"]
            == "benchmark_disabled"
        )


def test_production_report_errors_and_metadata_are_redacted(api_settings):
    settings = replace(api_settings, environment="production", allowed_hosts=("testserver",))
    with TestClient(create_app(settings)) as client:
        run = client.get("/api/v1/benchmarks/runs").json()["runs"][0]
        report = client.get("/api/v1/benchmarks/runs/" + run["id"] + "/results").json()
        assert "output_dir" not in report["metadata"]["config"]
        invalid = settings.results_dir / "private-report"
        invalid.mkdir()
        (invalid / "summary.json").write_text(
            json.dumps({"benchmark_schema_version": 1, "results": []})
        )
        listing = client.get("/api/v1/benchmarks/runs").json()
        assert listing["issues"] and str(settings.results_dir) not in json.dumps(listing)
        issue = listing["issues"][0]
        error = client.get("/api/v1/benchmarks/runs/" + issue["id"] + "/results").json()
        assert "FileNotFound" not in json.dumps(error) and str(
            settings.results_dir
        ) not in json.dumps(error)


def test_production_job_errors_and_logs_are_redacted(api_settings):
    settings = replace(api_settings, environment="production")
    jobs = BenchmarkManager(settings, ReportService(settings))
    identifier = uuid4()
    jobs.jobs[identifier] = JobInfo(
        id=identifier,
        status="failed",
        created_at="now",
        error="/private/secret.txt password",
        log="Traceback at /private/data",
    )
    assert "/private" not in jobs.get(identifier).model_dump_json()
    assert jobs.list()[0].log == ""


def test_single_pending_job_limit_is_enforced(api_settings, monkeypatch):
    from backend.services import benchmark_manager as module

    monkeypatch.setattr(module, "missing_dependencies", lambda: [])
    settings = replace(api_settings, max_pending_jobs=1)
    jobs = BenchmarkManager(settings, ReportService(settings))
    identifier = uuid4()
    jobs.jobs[identifier] = JobInfo(id=identifier, status="running", created_at="now")
    with pytest.raises(Exception) as error:
        jobs.submit(QuickRun())
    assert error.value.status == 409


def test_server_entrypoint_uses_config_and_single_worker(monkeypatch):
    from backend import serve

    captured = {}
    monkeypatch.setenv("VECTORFORGE_API_PORT", "8123")
    monkeypatch.setattr(serve.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))
    serve.main()
    assert captured["workers"] == 1 and captured["port"] == 8123 and not captured["proxy_headers"]
