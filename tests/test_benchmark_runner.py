import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vectorforge.benchmarks import _worker, runner
from vectorforge.benchmarks.config import THREAD_VARIABLES, BenchmarkConfig
from vectorforge.benchmarks.datasets import Dataset


def test_warmup_excluded_and_repetitions_use_actual_workload_clock(monkeypatch):
    state = {"clock": 0, "calls": []}

    class Adapter:
        def search(self, query, k, nprobe=None):
            state["calls"].append((float(query[0]), k, nprobe))
            state["clock"] += 100
            return []

    monkeypatch.setattr(_worker, "perf_counter_ns", lambda: state["clock"])
    config = BenchmarkConfig(query_count=3, repetitions=2, warmup=4)
    samples, elapsed = _worker.measure_search(Adapter(), np.array([[0.0], [1], [2]]), config, 4)
    assert len(state["calls"]) == 4 + 3 * 2
    assert len(samples) == 6 and elapsed == 600
    assert all(sample["latency_ns"] == 100 for sample in samples)
    for repetition in (0, 1):
        assert sorted(s["query_index"] for s in samples if s["repetition"] == repetition) == [
            0,
            1,
            2,
        ]
    assert all(k == config.k and probe == 4 for _, k, probe in state["calls"])


def fake_subprocess(monkeypatch, *, fail_engine=None, corrupt_hash=False):
    seen = []
    # Pure runner protocol tests do not require optional threadpoolctl or native engines.
    monkeypatch.setitem(
        __import__("sys").modules,
        "threadpoolctl",
        SimpleNamespace(threadpool_limits=lambda **kwargs: nullcontext()),
    )

    def execute(command, **kwargs):
        config_path, data_path, truth_path, engine, result_path = command[-5:]
        config = BenchmarkConfig.from_dict(json.loads(runner.Path(config_path).read_text()))
        with np.load(data_path, allow_pickle=False) as data:
            dataset = Dataset(data["training"], data["vectors"], data["queries"], data["ids"])
        truth = json.loads(runner.Path(truth_path).read_text())
        assert len(truth) == config.query_count
        assert len(truth[0]) == min(config.k, config.size)
        for variable in THREAD_VARIABLES:
            assert kwargs["env"][variable] == str(config.threads)
        seen.append((engine, config, dataset.fingerprints()))
        probes = config.nprobes if engine.endswith("ivf") else (None,)
        result = (
            {"status": "failed", "error": "FAISS unavailable in test"}
            if engine == fail_engine
            else {
                "status": "ok",
                "metadata": {"dataset_sha256": {} if corrupt_hash else dataset.fingerprints()},
                "rows": [
                    {**runner.result_identity(engine, config, probe), "status": "ok"}
                    for probe in probes
                ],
                "samples": [],
            }
        )
        runner.Path(result_path).write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    # Keep platform.processor() and other stdlib subprocess users outside this mock.
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=execute))
    return seen


def test_same_data_config_queries_and_seed_across_engines(monkeypatch):
    seen = fake_subprocess(monkeypatch)
    config = BenchmarkConfig(
        size=20,
        training_size=40,
        query_count=4,
        dimension=3,
        nlist=2,
        nprobes=(1, 2),
        engines=("vectorforge_exact", "faiss_ivf"),
    )
    run = runner.run_benchmark(config)
    assert run.complete and len(run.rows) == 3
    assert len(seen) == 2
    assert all(c == config and hashes == run.metadata["dataset_sha256"] for _, c, hashes in seen)
    assert {e for e, _, _ in seen} == set(config.engines)
    assert run.metadata["ground_truth"].startswith("VectorForge Exact")


def test_failed_engine_retains_every_probe_configuration(monkeypatch):
    fake_subprocess(monkeypatch, fail_engine="faiss_ivf")
    config = BenchmarkConfig(size=10, query_count=2, engines=("vectorforge_exact", "faiss_ivf"))
    run = runner.run_benchmark(config)
    assert not run.complete
    failures = [r for r in run.rows if r["status"] == "failed"]
    assert {r["nprobe"] for r in failures} == set(config.nprobes)
    assert all("FAISS unavailable" in r["error"] for r in failures)


def test_dataset_mismatch_is_explicit_failure(monkeypatch):
    fake_subprocess(monkeypatch, corrupt_hash=True)
    run = runner.run_benchmark(
        BenchmarkConfig(size=10, query_count=2, engines=("vectorforge_exact",))
    )
    assert not run.complete
    assert "fingerprint mismatch" in run.rows[0]["error"]


def test_processor_subprocess_is_not_intercepted_as_worker(monkeypatch):
    def processor():
        # Linux discovers the processor via a subprocess; exercise that path on Windows too.
        return subprocess.check_output(
            [sys.executable, "-c", "print('test CPU')"], text=True
        ).strip()

    monkeypatch.setattr(runner.platform, "processor", processor)
    seen = fake_subprocess(monkeypatch)
    run = runner.run_benchmark(
        BenchmarkConfig(size=10, query_count=2, engines=("vectorforge_exact",))
    )
    assert run.complete, run.rows
    assert run.metadata["cpu"] == "test CPU"
    assert [engine for engine, _, _ in seen] == ["vectorforge_exact"]


def test_native_worker_exit_is_reported(monkeypatch):
    fake_subprocess(monkeypatch)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=7, stderr="native crash", stdout=""),
    )
    run = runner.run_benchmark(BenchmarkConfig(size=10, query_count=2, engines=("faiss_flat",)))
    assert not run.complete
    assert "worker exit 7" in run.rows[0]["error"]


def test_worker_launch_failure_is_explicit(monkeypatch):
    fake_subprocess(monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("worker could not start")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    run = runner.run_benchmark(BenchmarkConfig(size=10, query_count=2, engines=("faiss_flat",)))
    assert not run.complete
    assert "worker could not start" in run.rows[0]["error"]


def test_real_four_engine_process_isolation_and_recall(tmp_path):
    pytest.importorskip("faiss", reason="optional four-engine FAISS integration")
    pytest.importorskip("psutil", reason="optional benchmark RSS integration")
    pytest.importorskip("threadpoolctl", reason="optional benchmark thread integration")
    from vectorforge.benchmarks.reporting import write_reports

    config = BenchmarkConfig(
        size=40,
        dimension=3,
        training_size=80,
        query_count=5,
        k=4,
        nlist=2,
        nprobes=(1, 2),
        repetitions=2,
        warmup=1,
        max_iter=4,
    )
    run = runner.run_benchmark(config)
    assert run.complete, run.rows
    assert len(run.rows) == 6
    assert len(run.samples) == 60
    workers = run.metadata["workers"]
    assert len({worker["pid"] for worker in workers.values()}) == 4
    for worker in workers.values():
        assert worker["dataset_sha256"] == run.metadata["dataset_sha256"]
        assert all(pool["num_threads"] == 1 for pool in worker["threadpools"])
    for row in run.rows:
        assert row["completed_queries"] == 10
        assert row["qps"] > 0
        assert 0 <= row["recall_at_k"] <= 1
        assert row["rss_after_insertion_bytes"] > 0
        assert row["training_ms"] is None if row["nlist"] is None else row["training_ms"] >= 0
        if row["engine"] == "vectorforge_ivf" and row["nprobe"] == 2:
            assert row["recall_at_k"] == 1 and row["full_probe_direct_path"]
        if row["engine"] == "faiss_flat":
            assert row["recall_at_k"] == 1
    directory = write_reports(run, tmp_path)
    assert json.loads((directory / "metadata.json").read_text())["complete"]
