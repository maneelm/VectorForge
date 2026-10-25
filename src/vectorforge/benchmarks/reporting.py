"""Strict versioned JSON, stable CSV columns, and collision-safe run directories."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from vectorforge.benchmarks.runner import SCHEMA_VERSION, BenchmarkRun

SUMMARY_FIELDS = (
    "engine",
    "status",
    "error",
    "metric",
    "dataset_size",
    "dimension",
    "distribution",
    "k",
    "nlist",
    "nprobe",
    "seed",
    "query_count",
    "repetitions",
    "training_size",
    "threads",
    "precision",
    "full_probe_direct_path",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "qps",
    "workload_elapsed_ns",
    "completed_queries",
    "recall_at_k",
    "construction_ms",
    "training_ms",
    "insertion_ms",
    "build_total_ms",
    "rss_baseline_bytes",
    "rss_after_construction_bytes",
    "rss_after_training_bytes",
    "rss_after_insertion_bytes",
    "retained_rss_delta_bytes",
)
SAMPLE_FIELDS = ("engine", "nprobe", "repetition", "query_index", "latency_ns")


def write_reports(run: BenchmarkRun, output_dir: str | Path, *, run_id: str | None = None) -> Path:
    """Never reuse a run directory; failed engine rows are exported too."""
    name = run_id or f"benchmark_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    if not name or Path(name).name != name or name in (".", "..") or ":" in name or "\\" in name:
        raise ValueError("run_id must be a single directory name")
    # Validate strict JSON before publishing any files.
    metadata_text = json.dumps(run.metadata, indent=2, allow_nan=False)
    summary_text = json.dumps(
        {"benchmark_schema_version": SCHEMA_VERSION, "results": run.rows}, indent=2, allow_nan=False
    )
    directory = Path(output_dir) / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "metadata.json").write_text(metadata_text, encoding="utf-8")
    (directory / "summary.json").write_text(summary_text, encoding="utf-8")
    for filename, fields, rows in (
        ("summary.csv", SUMMARY_FIELDS, run.rows),
        ("latency_samples.csv", SAMPLE_FIELDS, run.samples),
    ):
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    (directory / "summary.txt").write_text(summary_table(run.rows), encoding="utf-8")
    return directory


def summary_table(rows: list[dict]) -> str:
    lines = ["Engine             probe  mean(ms)   p95(ms)        QPS  Recall@k  status"]
    for row in rows:
        probe = "-" if row["nprobe"] is None else str(row["nprobe"])
        if row["status"] != "ok":
            lines.append(f"{row['engine']:<19} {probe:>4}  FAILED: {row['error']}")
        else:
            lines.append(
                f"{row['engine']:<19} {probe:>4} {row['latency_mean_ms']:>9.4f} "
                f"{row['latency_p95_ms']:>9.4f} {row['qps']:>10.1f} "
                f"{row['recall_at_k']:>9.4f}  ok"
            )
    return "\n".join(lines)
