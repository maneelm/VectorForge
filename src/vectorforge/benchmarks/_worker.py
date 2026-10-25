"""Private subprocess entry point. Thread environment is set by the parent first."""

import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from vectorforge import SearchResult
from vectorforge.benchmarks.adapters import Adapter, make_adapter
from vectorforge.benchmarks.config import THREAD_VARIABLES, BenchmarkConfig
from vectorforge.benchmarks.datasets import Dataset
from vectorforge.benchmarks.metrics import latency_statistics, queries_per_second
from vectorforge.benchmarks.runner import result_identity
from vectorforge.evaluation import recall_at_k


def measure_search(
    adapter: Adapter,
    queries: np.ndarray,
    config: BenchmarkConfig,
    probe: int | None,
) -> tuple[list[dict], int]:
    """Warmups and deterministic repetition orders are prepared outside timers."""
    for i in range(config.warmup):
        adapter.search(queries[i % len(queries)], config.k, nprobe=probe)
    rng = np.random.default_rng(config.seed)
    orders = [rng.permutation(len(queries)) for _ in range(config.repetitions)]
    samples = []
    elapsed = 0
    for repetition, order in enumerate(orders):
        workload_start = perf_counter_ns()
        for query_index in order:
            start = perf_counter_ns()
            matches = adapter.search(queries[query_index], config.k, nprobe=probe)
            duration = perf_counter_ns() - start
            samples.append(
                {"repetition": repetition, "query_index": int(query_index), "latency_ns": duration}
            )
            del matches
        elapsed += perf_counter_ns() - workload_start
    return samples, elapsed


def execute(engine: str, config: BenchmarkConfig, dataset: Dataset, truth: list[list[int]]) -> dict:
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits

    faiss = None
    if engine.startswith("faiss"):
        from vectorforge.benchmarks.faiss_adapter import require_faiss

        faiss = require_faiss()
    with threadpool_limits(limits=config.threads):
        if faiss is not None:
            faiss.omp_set_num_threads(config.threads)
        process = psutil.Process()

        def rss() -> int:
            gc.collect()
            return process.memory_info().rss

        baseline = rss()
        start = perf_counter_ns()
        adapter = make_adapter(engine, config)
        construction_ms = (perf_counter_ns() - start) / 1_000_000
        after_construction = rss()
        training_ms = None
        if engine.endswith("ivf"):
            start = perf_counter_ns()
            adapter.train(dataset.training)
            training_ms = (perf_counter_ns() - start) / 1_000_000
        after_training = rss() if training_ms is not None else None
        start = perf_counter_ns()
        adapter.add(dataset.vectors, dataset.ids)
        insertion_ms = (perf_counter_ns() - start) / 1_000_000
        after_insertion = rss()
        rows, all_samples = [], []
        for probe in config.nprobes if engine.endswith("ivf") else (None,):
            samples, elapsed = measure_search(adapter, dataset.queries, config, probe)
            # Quality evaluation is outside all timing sections and done once per query.
            recalls = [
                recall_at_k(
                    adapter.search(query, config.k, nprobe=probe),
                    [SearchResult(i, 0.0) for i in expected],
                    config.k,
                )
                for query, expected in zip(dataset.queries, truth, strict=True)
            ]
            row = {
                **result_identity(engine, config, probe),
                "status": "ok",
                "error": None,
                **latency_statistics([sample["latency_ns"] for sample in samples]),
                "qps": queries_per_second(len(samples), elapsed),
                "workload_elapsed_ns": elapsed,
                "completed_queries": len(samples),
                "recall_at_k": float(np.mean(recalls)),
                "recall_per_query": recalls,
                "construction_ms": construction_ms,
                "training_ms": training_ms,
                "insertion_ms": insertion_ms,
                "build_total_ms": construction_ms + (training_ms or 0.0) + insertion_ms,
                "rss_baseline_bytes": baseline,
                "rss_after_construction_bytes": after_construction,
                "rss_after_training_bytes": after_training,
                "rss_after_insertion_bytes": after_insertion,
                "retained_rss_delta_bytes": after_insertion - baseline,
            }
            rows.append(row)
            all_samples.extend({"engine": engine, "nprobe": probe, **sample} for sample in samples)
        return {
            "status": "ok",
            "rows": rows,
            "samples": all_samples,
            "metadata": {
                "pid": os.getpid(),
                "threadpools": threadpool_info(),
                "thread_environment": {key: os.environ.get(key) for key in THREAD_VARIABLES},
                "faiss_omp_threads": faiss.omp_get_max_threads() if faiss is not None else None,
                "dataset_sha256": dataset.fingerprints(),
            },
        }


def main() -> None:
    config_file, dataset_file, truth_file, engine, result_file = sys.argv[1:]
    try:
        config = BenchmarkConfig.from_dict(
            json.loads(Path(config_file).read_text(encoding="utf-8"))
        )
        with np.load(dataset_file, allow_pickle=False) as data:
            dataset = Dataset(data["training"], data["vectors"], data["queries"], data["ids"])
        truth = json.loads(Path(truth_file).read_text(encoding="utf-8"))
        result = execute(engine, config, dataset, truth)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    Path(result_file).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
