"""Sequential process-isolated engine runs with explicit failures and provenance."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from vectorforge import ExactVectorIndex
from vectorforge.benchmarks.config import THREAD_VARIABLES, BenchmarkConfig
from vectorforge.benchmarks.datasets import generate_dataset

SCHEMA_VERSION = 1


@dataclass(slots=True)
class BenchmarkRun:
    metadata: dict
    rows: list[dict]
    samples: list[dict]

    @property
    def complete(self) -> bool:
        return bool(self.rows) and all(row["status"] == "ok" for row in self.rows)


def result_identity(engine: str, config: BenchmarkConfig, probe: int | None) -> dict:
    return {
        "engine": engine,
        "metric": config.metric,
        "dataset_size": config.size,
        "dimension": config.dimension,
        "distribution": config.distribution,
        "k": config.k,
        "nlist": config.nlist if engine.endswith("ivf") else None,
        "nprobe": probe,
        "seed": config.seed,
        "query_count": config.query_count,
        "repetitions": config.repetitions,
        "training_size": config.training_size,
        "threads": config.threads,
        "precision": "float32" if engine.startswith("faiss") else "float64",
        "full_probe_direct_path": engine == "vectorforge_ivf" and probe == config.nlist,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    """Generate once, compute ground truth, then launch one fresh process per engine.

    Failures produce rows for every affected probe; callers must check complete.
    No output files are published by this function. Reporting is a separate stage.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError(
            "install benchmark dependencies: uv sync --locked --extra benchmark"
        ) from exc
    dataset = generate_dataset(config)
    with threadpool_limits(limits=config.threads):
        exact = ExactVectorIndex(config.dimension, config.metric)
        exact.add(dataset.vectors, dataset.ids)
        truth = [[r.id for r in exact.search(query, config.k)] for query in dataset.queries]
    del exact
    metadata = {
        "benchmark_schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "faiss_version": package_version("faiss-cpu"),
        "operating_system": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpu_count": os.cpu_count(),
        "config": config.to_dict(),
        "dataset_sha256": dataset.fingerprints(),
        "workers": {},
        "thread_environment": {key: str(config.threads) for key in THREAD_VARIABLES},
        "ground_truth": "VectorForge Exact float64; canonical score/ascending-ID ties",
        "timing_scope": "single-query end-to-end adapter including validation/conversion/results",
        "qps_scope": "actual repeated single-query workload, including loop/timer overhead",
        "memory_scope": "fresh worker RSS snapshots, datasets/libraries in baseline; not peak",
        "warnings": [
            "FAISS float32 and native boundary ties can differ from float64 ground truth",
            "VectorForge IVF full probing uses the direct full-dataset ranking path",
        ],
    }
    if config.training_size < 39 * config.nlist:
        metadata["warnings"].append(
            "training_size is below FAISS's recommended 39 samples per centroid"
        )
    rows, samples = [], []
    with TemporaryDirectory(prefix="vectorforge-benchmark-") as temporary:
        directory = Path(temporary)
        config_path, data_path, truth_path = (
            directory / name for name in ("config.json", "dataset.npz", "truth.json")
        )
        config_path.write_text(json.dumps(config.to_dict(), allow_nan=False), encoding="utf-8")
        np.savez(
            data_path,
            training=dataset.training,
            vectors=dataset.vectors,
            queries=dataset.queries,
            ids=dataset.ids,
        )
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        environment = {
            **os.environ,
            **metadata["thread_environment"],
            "PYTHONHASHSEED": str(config.seed),
        }
        # Stable seed determines engine order to reduce a permanently fixed order bias.
        order = np.random.default_rng(config.seed).permutation(config.engines).tolist()
        metadata["execution_order"] = order
        for engine in order:
            result_path = directory / f"{engine}.json"
            command = [
                sys.executable,
                "-m",
                "vectorforge.benchmarks._worker",
                str(config_path),
                str(data_path),
                str(truth_path),
                engine,
                str(result_path),
            ]
            try:
                process = subprocess.run(
                    command, env=environment, capture_output=True, text=True, check=False
                )
                if process.returncode:
                    raise RuntimeError(
                        f"worker exit {process.returncode}: {process.stderr.strip()}"
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result["status"] != "ok":
                    raise RuntimeError(result["error"])
                metadata["workers"][engine] = result["metadata"]
                metadata["workers"][engine]["stderr"] = process.stderr.strip()
                if result["metadata"]["dataset_sha256"] != metadata["dataset_sha256"]:
                    raise ValueError("worker dataset fingerprint mismatch")
                rows.extend(result["rows"])
                samples.extend(result["samples"])
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                for probe in config.nprobes if engine.endswith("ivf") else (None,):
                    rows.append(
                        {
                            **result_identity(engine, config, probe),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
    metadata["complete"] = all(row["status"] == "ok" for row in rows)
    return BenchmarkRun(metadata, rows, samples)
