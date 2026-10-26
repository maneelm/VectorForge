import json
import shutil
import subprocess
import time
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from backend.core.config import confined
from backend.core.errors import AppError
from backend.services import benchmark_manager as module

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "docs/benchmark_results/benchmark_20261025T025027Z_4e5164f0"
)


def wait_job(client, identifier):
    for _ in range(300):
        job = client.get(f"/api/v1/benchmarks/jobs/{identifier}").json()
        if job["status"] not in ("running", "queued"):
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_discovery_real_published_data_without_faiss(client, monkeypatch):
    monkeypatch.setattr(module, "missing_dependencies", lambda: ["faiss"])
    listing = client.get("/api/v1/benchmarks/runs").json()
    assert len(listing["runs"]) == 1 and not listing["issues"]
    run = listing["runs"][0]
    assert run["source"] == "published" and run["size"] == 10000
    assert client.get(f"/api/v1/benchmarks/runs/{run['id']}").json() == run
    results = client.get(f"/api/v1/benchmarks/runs/{run['id']}/results").json()
    assert len(results["results"]) == 12
    assert results["results"][0]["latency_mean_ms"] > 0
    assert client.post("/api/v1/benchmarks/runs", json={}).status_code == 503
    assert client.get(f"/api/v1/benchmarks/runs/{uuid4()}").status_code == 404


@pytest.mark.parametrize(
    "change", ["oversize", "bad_json", "mixed_workload", "missing_field", "version", "nonfinite"]
)
def test_invalid_reports_are_explicit(client, api_settings, change):
    target = api_settings.results_dir / "invalid"
    shutil.copytree(SOURCE, target)
    path = target / "summary.json"
    if change == "oversize":
        path.write_bytes(b" " * (api_settings.report_bytes + 1))
    elif change == "bad_json":
        path.write_text("bad")
    else:
        data = json.loads(path.read_text())
        if change == "mixed_workload":
            data["results"][0]["dimension"] = 1
        if change == "missing_field":
            del data["results"][0]["latency_mean_ms"]
        if change == "version":
            data["benchmark_schema_version"] = 999
        if change == "nonfinite":
            data["results"][0]["qps"] = float("nan")
        path.write_text(json.dumps(data))
    listing = client.get("/api/v1/benchmarks/runs").json()
    assert len(listing["runs"]) == 1 and len(listing["issues"]) == 1
    identifier = listing["issues"][0]["id"]
    assert client.get(f"/api/v1/benchmarks/runs/{identifier}/results").status_code == 400


def test_boundaries(api_settings):
    with pytest.raises(AppError):
        confined(api_settings.data_dir, api_settings.data_dir / ".." / "outside")


@pytest.mark.parametrize(
    "body",
    [
        {"preset": "full"},
        {"size": 1000000},
        {"output": "../escape"},
        {"seed": True},
        {"metric": "bad"},
        {"command": "whoami"},
    ],
)
def test_unsafe_job_requests(client, body):
    assert client.post("/api/v1/benchmarks/runs", json=body).status_code == 422


def test_job_completion_reuses_cli_and_safe_output(client, monkeypatch, api_settings):
    monkeypatch.setattr(module, "missing_dependencies", lambda: [])
    commands = []

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append(command)
            output = Path(command[-1])
            assert output.resolve().is_relative_to(api_settings.results_dir.resolve())
            shutil.copytree(SOURCE, output / "benchmark_test")

        def communicate(self, timeout):
            return "benchmark complete", None

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    response = client.post("/api/v1/benchmarks/runs", json={"metric": "cosine", "seed": 42})
    assert response.status_code == 202
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "completed" and len(job["run_ids"]) == 1
    assert commands[0][1:3] == ["-m", "vectorforge.benchmarks.cli"]
    assert "--size" in commands[0] and "1000" in commands[0]
    assert client.get(f"/api/v1/benchmarks/runs/{job['run_ids'][0]}/results").status_code == 200


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_output"])
def test_job_failure(client, monkeypatch, failure):
    monkeypatch.setattr(module, "missing_dependencies", lambda: [])

    class Process:
        returncode = 3 if failure == "exit" else 0

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, timeout):
            if failure == "timeout":
                raise subprocess.TimeoutExpired("benchmark", timeout)
            return "test failure", None

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    job = wait_job(client, client.post("/api/v1/benchmarks/runs", json={}).json()["id"])
    assert job["status"] == "failed" and job["error"]


def test_concurrency_limit_queue_and_snapshot(client, monkeypatch):
    monkeypatch.setattr(module, "missing_dependencies", lambda: [])
    started, release = Event(), Event()

    class Process:
        returncode = 1

        def __init__(self, *a, **kw):
            pass

        def communicate(self, timeout):
            started.set()
            release.wait(5)
            return "stopped", None

        def poll(self):
            return 1

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    try:
        first = client.post("/api/v1/benchmarks/runs", json={})
        assert first.status_code == 202 and started.wait(2)
        second = client.post("/api/v1/benchmarks/runs", json={})
        assert second.status_code == 202
        assert (
            client.get("/api/v1/benchmarks/jobs/" + second.json()["id"]).json()["status"]
            == "queued"
        )
        assert client.post("/api/v1/benchmarks/runs", json={}).status_code == 409
    finally:
        release.set()
    assert len(client.get("/api/v1/benchmarks/jobs").json()) == 2


def test_no_process_on_application_construction(api_settings, monkeypatch):
    from backend.main import create_app

    def forbidden(*a, **kw):
        raise AssertionError("Unexpected subprocess")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    create_app(api_settings)
