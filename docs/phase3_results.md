# Phase 3 measured quick-run results

These tables were exported from the completed quick benchmark reports. Results are observations on this machine, not general performance guarantees.

The [published representative run](benchmark_results/README.md) contains the
10,000-vector, 128-dimensional workload's reports, raw latency samples, and six
charts. `docs/benchmark_results/` holds representative results included in Git;
`results/` remains ignored for locally generated outputs. The other workloads'
tables are retained below, while their raw artifacts remain local-only.

Environment: Python 3.12.4; NumPy 2.5.4; FAISS 1.15.1; Windows-11-10.0.26200-SP0.

CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel; 16 logical CPUs; one benchmark thread. Cosine, normal distribution, k=10, seed=42, nlist=16, training_size=2000, query_count=50, repetitions=3, warmup=5, max_iter=25.

All four workloads completed all four engines: 48 result rows, 7200 measured query samples, and 24 charts. Worker dataset fingerprints and effective thread limits were verified. Percentiles and QPS were recomputed from the saved measurements; CSV/JSON scalar results agree. All exact, FAISS Flat, and full-probe rows have Recall@10=1.0 on these datasets.

Primary timing includes validation, normalization/conversion, search, and Python results. VectorForge uses float64; FAISS uses float32. An asterisk marks VectorForge IVF direct full-dataset ranking. RSS deltas are allocator-sensitive estimates, not peak memory or exact index sizes.

## Validation

- Original Phase 1-2 files: 404 tests passed.
- Core-only environment after `uv sync --locked`: 478 passed, 9 explicitly skipped optional integrations.
- Optional benchmark environment: `uv run --extra benchmark pytest -v` reported 487 passed in 12.51s, no skips.
- Ruff: all checks passed; formatting check reported all files already formatted.
- Both original examples and the new benchmark demo executed successfully.
- Chart regeneration from saved reports executed successfully.

## Commands executed

```powershell
uv pip install --dry-run faiss-cpu
uv lock
uv sync --locked
uv run pytest -v
uv run pytest tests/test_distance.py tests/test_exact.py tests/test_storage.py tests/test_kmeans.py tests/test_ivf.py tests/test_ivf_storage.py tests/test_recall.py -q
uv run ruff check .
uv run ruff format --check .
uv run python examples/basic_search.py
uv run python examples/ivf_search.py
uv sync --locked --extra benchmark
uv run --extra benchmark pytest -v
uv run --extra benchmark python -m vectorforge.benchmarks.cli --help
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset quick --output results/quick
uv run --extra benchmark python -m vectorforge.benchmarks.cli --plot results/quick/benchmark_20261025T025027Z_4e5164f0
uv run --extra benchmark python examples/benchmark_demo.py
```

Standard and full performance grids were not run; their configuration grids were validated in tests. No Phase 3 acceptance criterion was blocked. The commands and test counts above are historical Phase 3 validation records, before the API, dashboard, Docker, and CI phases. See the current [README](../README.md) for the complete application and current validation commands.

## New and modified files

Added `src/vectorforge/benchmarks/` with `__init__.py`, `config.py`, `datasets.py`,
`adapters.py`, `faiss_adapter.py`, `metrics.py`, `runner.py`, `_worker.py`,
`reporting.py`, `plotting.py`, and `cli.py`.

Added `tests/test_benchmark_datasets.py`, `tests/test_benchmark_metrics.py`,
`tests/test_benchmark_adapters.py`, `tests/test_benchmark_runner.py`, and
`tests/test_benchmark_reporting.py`, plus `examples/benchmark_demo.py`,
`docs/benchmarking.md`, and this measured-results document.

Modified `.gitignore`, `pyproject.toml`, `uv.lock`, `README.md`, and
`docs/architecture.md`. Existing engine modules, evaluation utilities, and the
seven original test files were not modified. Representative published artifacts
are in `docs/benchmark_results/`; additional generated reports remain local under
the ignored `results/` directory.

## N=1,000, dimension=32

Recorded UTC: 2026-10-25T02:49:23.626793+00:00


| Engine | nprobe | Mean ms | p95 ms | Measured QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 1 | 0.1618 | 0.1783 | 6116.8 | 0.2120 |
| faiss_ivf | 2 | 0.1562 | 0.1703 | 6346.2 | 0.3700 |
| faiss_ivf | 4 | 0.1624 | 0.1778 | 6103.9 | 0.5560 |
| faiss_ivf | 8 | 0.1620 | 0.1772 | 6121.0 | 0.7840 |
| faiss_ivf | 16 | 0.1680 | 0.1831 | 5904.2 | 1.0000 |
| faiss_flat | - | 0.1589 | 0.1736 | 6232.0 | 1.0000 |
| vectorforge_ivf | 1 | 0.3141 | 0.3394 | 3163.6 | 0.2300 |
| vectorforge_ivf | 2 | 0.3429 | 0.3742 | 2900.2 | 0.3640 |
| vectorforge_ivf | 4 | 0.4089 | 0.4366 | 2434.3 | 0.5840 |
| vectorforge_ivf | 8 | 0.5373 | 0.5810 | 1854.1 | 0.8460 |
| vectorforge_ivf | 16* | 1.0647 | 1.3300 | 937.0 | 1.0000 |
| vectorforge_exact | - | 1.0425 | 1.3004 | 956.8 | 1.0000 |

| Engine | Construction ms | Training ms | Insertion ms | Total build ms | RSS baseline MiB | RSS after insertion MiB | Retained RSS delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 0.182 | 13.127 | 13.288 | 26.598 | 48.559 | 50.496 | 1.938 |
| faiss_flat | 0.107 | N/A | 13.056 | 13.164 | 48.363 | 50.219 | 1.855 |
| vectorforge_ivf | 0.105 | 103.509 | 14.354 | 117.968 | 36.516 | 38.031 | 1.516 |
| vectorforge_exact | 0.038 | N/A | 11.785 | 11.823 | 36.625 | 37.730 | 1.105 |

## N=1,000, dimension=128

Recorded UTC: 2026-10-25T02:49:30.427280+00:00


| Engine | nprobe | Mean ms | p95 ms | Measured QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 1 | 0.1652 | 0.1793 | 5992.2 | 0.1380 |
| faiss_ivf | 2 | 0.1625 | 0.1808 | 6094.9 | 0.2340 |
| faiss_ivf | 4 | 0.1740 | 0.1988 | 5694.5 | 0.4100 |
| faiss_ivf | 8 | 0.1757 | 0.1952 | 5643.7 | 0.6900 |
| faiss_ivf | 16 | 0.1917 | 0.2083 | 5172.9 | 1.0000 |
| faiss_flat | - | 0.1843 | 0.2013 | 5376.5 | 1.0000 |
| vectorforge_ivf | 1 | 0.4740 | 0.7460 | 2097.9 | 0.1280 |
| vectorforge_ivf | 2 | 0.5065 | 0.7493 | 1965.6 | 0.2140 |
| vectorforge_ivf | 4 | 1.2175 | 1.6042 | 819.4 | 0.3780 |
| vectorforge_ivf | 8 | 2.0849 | 2.6109 | 478.8 | 0.6800 |
| vectorforge_ivf | 16* | 2.9248 | 3.5644 | 341.5 | 1.0000 |
| vectorforge_exact | - | 4.6356 | 5.1154 | 215.5 | 1.0000 |

| Engine | Construction ms | Training ms | Insertion ms | Total build ms | RSS baseline MiB | RSS after insertion MiB | Retained RSS delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 0.165 | 20.393 | 16.918 | 37.476 | 50.773 | 53.551 | 2.777 |
| faiss_flat | 0.102 | N/A | 16.977 | 17.079 | 50.836 | 53.367 | 2.531 |
| vectorforge_ivf | 0.078 | 496.630 | 33.783 | 530.491 | 38.906 | 41.027 | 2.121 |
| vectorforge_exact | 0.040 | N/A | 12.707 | 12.747 | 38.969 | 40.590 | 1.621 |

## N=10,000, dimension=32

Recorded UTC: 2026-10-25T02:49:38.751300+00:00


| Engine | nprobe | Mean ms | p95 ms | Measured QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 1 | 0.2256 | 0.3546 | 4390.5 | 0.2620 |
| faiss_ivf | 2 | 0.5708 | 0.7996 | 1743.4 | 0.3940 |
| faiss_ivf | 4 | 0.2048 | 0.3589 | 4837.8 | 0.6280 |
| faiss_ivf | 8 | 0.2307 | 0.3799 | 4303.0 | 0.8440 |
| faiss_ivf | 16 | 0.2846 | 0.4536 | 3490.8 | 1.0000 |
| faiss_flat | - | 0.2542 | 0.2858 | 3905.2 | 1.0000 |
| vectorforge_ivf | 1 | 0.9892 | 1.1641 | 1007.6 | 0.2640 |
| vectorforge_ivf | 2 | 1.6436 | 2.0281 | 607.0 | 0.4060 |
| vectorforge_ivf | 4 | 3.2087 | 3.8157 | 311.2 | 0.6280 |
| vectorforge_ivf | 8 | 6.5915 | 7.3640 | 151.6 | 0.8580 |
| vectorforge_ivf | 16* | 10.9696 | 13.0080 | 91.1 | 1.0000 |
| vectorforge_exact | - | 10.1402 | 10.5818 | 98.6 | 1.0000 |

| Engine | Construction ms | Training ms | Insertion ms | Total build ms | RSS baseline MiB | RSS after insertion MiB | Retained RSS delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 0.235 | 16.358 | 31.821 | 48.414 | 50.797 | 54.660 | 3.863 |
| faiss_flat | 0.112 | N/A | 26.086 | 26.197 | 50.777 | 54.785 | 4.008 |
| vectorforge_ivf | 0.079 | 94.962 | 49.440 | 144.481 | 38.727 | 43.012 | 4.285 |
| vectorforge_exact | 0.042 | N/A | 15.492 | 15.534 | 39.074 | 42.117 | 3.043 |

## N=10,000, dimension=128

Recorded UTC: 2026-10-25T02:49:53.182227+00:00

[Summary JSON](benchmark_results/benchmark_20261025T025027Z_4e5164f0/summary.json) | [CSV](benchmark_results/benchmark_20261025T025027Z_4e5164f0/summary.csv) | [Metadata](benchmark_results/benchmark_20261025T025027Z_4e5164f0/metadata.json) | [Raw latency samples](benchmark_results/benchmark_20261025T025027Z_4e5164f0/latency_samples.csv) | [Charts](benchmark_results/benchmark_20261025T025027Z_4e5164f0/charts)

| Engine | nprobe | Mean ms | p95 ms | Measured QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 1 | 0.2046 | 0.2420 | 4844.4 | 0.1460 |
| faiss_ivf | 2 | 0.2176 | 0.2423 | 4558.9 | 0.2540 |
| faiss_ivf | 4 | 0.2707 | 0.3120 | 3668.4 | 0.4220 |
| faiss_ivf | 8 | 0.3544 | 0.3996 | 2803.3 | 0.6900 |
| faiss_ivf | 16 | 0.4780 | 0.5156 | 2083.5 | 1.0000 |
| faiss_flat | - | 0.4689 | 0.7547 | 2123.3 | 1.0000 |
| vectorforge_ivf | 1 | 2.7331 | 3.3788 | 365.4 | 0.1400 |
| vectorforge_ivf | 2 | 5.9497 | 6.5157 | 167.9 | 0.2240 |
| vectorforge_ivf | 4 | 13.6284 | 15.6023 | 73.3 | 0.4020 |
| vectorforge_ivf | 8 | 26.9097 | 31.7387 | 37.1 | 0.7080 |
| vectorforge_ivf | 16* | 51.1446 | 57.6283 | 19.5 | 1.0000 |
| vectorforge_exact | - | 44.9418 | 57.6942 | 22.2 | 1.0000 |

| Engine | Construction ms | Training ms | Insertion ms | Total build ms | RSS baseline MiB | RSS after insertion MiB | Retained RSS delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| faiss_ivf | 0.173 | 20.392 | 55.118 | 75.683 | 59.641 | 69.461 | 9.820 |
| faiss_flat | 0.104 | N/A | 50.107 | 50.211 | 60.027 | 67.043 | 7.016 |
| vectorforge_ivf | 0.083 | 441.540 | 140.745 | 582.368 | 47.766 | 59.668 | 11.902 |
| vectorforge_exact | 0.038 | N/A | 21.996 | 22.034 | 47.891 | 59.047 | 11.156 |

