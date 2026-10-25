import csv
import json

import pytest

from vectorforge.benchmarks import cli
from vectorforge.benchmarks.config import BenchmarkConfig
from vectorforge.benchmarks.plotting import regenerate_charts
from vectorforge.benchmarks.reporting import SUMMARY_FIELDS, summary_table, write_reports
from vectorforge.benchmarks.runner import BenchmarkRun, result_identity


@pytest.fixture
def run():
    config = BenchmarkConfig(size=100, dimension=4, nlist=2, nprobes=(1, 2))
    row = {
        **result_identity("vectorforge_exact", config, None),
        "status": "ok",
        "error": None,
        "latency_mean_ms": 1.0,
        "latency_p50_ms": 0.9,
        "latency_p95_ms": 2.0,
        "latency_p99_ms": 3.0,
        "qps": 900.0,
        "recall_at_k": 1.0,
        "construction_ms": 0.1,
        "training_ms": None,
        "insertion_ms": 0.2,
        "build_total_ms": 0.3,
        "rss_baseline_bytes": 1000000,
        "rss_after_construction_bytes": 1001000,
        "rss_after_training_bytes": None,
        "rss_after_insertion_bytes": 1200000,
        "retained_rss_delta_bytes": 200000,
    }
    return BenchmarkRun(
        {"benchmark_schema_version": 1, "complete": True},
        [row],
        [
            {
                "engine": "vectorforge_exact",
                "nprobe": None,
                "repetition": 0,
                "query_index": 0,
                "latency_ns": 1000000,
            }
        ],
    )


def test_csv_json_metadata_nulls_and_collision_safety(tmp_path, run):
    directory = write_reports(run, tmp_path, run_id="test")
    assert json.loads((directory / "metadata.json").read_text())["benchmark_schema_version"] == 1
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["results"][0]["training_ms"] is None
    with (directory / "summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == SUMMARY_FIELDS
        row = next(reader)
        assert row["training_ms"] == ""
        assert float(row["latency_mean_ms"]) == 1
    with (directory / "latency_samples.csv").open(newline="") as handle:
        assert next(csv.DictReader(handle))["latency_ns"] == "1000000"
    assert "vectorforge_exact" in summary_table(run.rows)
    with pytest.raises(FileExistsError):
        write_reports(run, tmp_path, run_id="test")
    assert write_reports(run, tmp_path) != directory


@pytest.mark.parametrize("run_id", ["../escape", "..", "/absolute", "a\\b", "C:bad"])
def test_invalid_run_directory_names(tmp_path, run, run_id):
    with pytest.raises(ValueError):
        write_reports(run, tmp_path, run_id=run_id)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_json_rejected_before_creating_directory(tmp_path, run, bad):
    run.rows[0]["qps"] = bad
    with pytest.raises(ValueError):
        write_reports(run, tmp_path, run_id="bad")
    assert not (tmp_path / "bad").exists()


def test_failure_rows_remain_visible(tmp_path, run):
    run.rows.append(
        {
            **result_identity("faiss_flat", BenchmarkConfig(), None),
            "status": "failed",
            "error": "FAISS unavailable",
        }
    )
    directory = write_reports(run, tmp_path)
    assert "FAILED: FAISS unavailable" in (directory / "summary.txt").read_text()
    assert len(json.loads((directory / "summary.json").read_text())["results"]) == 2


def test_chart_generation_regeneration_and_workload_mixing(tmp_path, run):
    pytest.importorskip("matplotlib", reason="optional Matplotlib chart integration")
    directory = write_reports(run, tmp_path)
    paths = regenerate_charts(directory)
    assert len(paths) == 6
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    assert cli.main(["--plot", str(directory)]) == 0
    run.rows.append({**run.rows[0], "dimension": 128})
    other = write_reports(run, tmp_path)
    with pytest.raises(ValueError, match="mix"):
        regenerate_charts(other)


def test_cli_help_dry_run_and_invalid_configuration(capsys):
    assert cli.main(["--dry-run", "--size", "20", "--dimension", "3"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["size"] == 20
    assert cli.main(["--nprobe", "1000", "--dry-run"]) == 2
    assert "Benchmark failed" in capsys.readouterr().err


def test_cli_failed_run_writes_reports_and_returns_nonzero(tmp_path, run, monkeypatch):
    run.rows[0].update(status="failed", error="requested engine unavailable")
    monkeypatch.setattr(cli, "run_benchmark", lambda config: run)
    assert cli.main(["--output", str(tmp_path), "--no-charts"]) == 1
    assert len(list(tmp_path.glob("*/summary.json"))) == 1
