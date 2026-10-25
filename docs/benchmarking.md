# Reproducible benchmarking (Phase 3)

## What is compared

VectorForge Exact and IVF are compared against CPU FAISS Flat and IVFFlat through
one single-query Python adapter interface. FAISS is a useful optimized native
baseline, not a performance target the educational NumPy engine must beat. Python
validation, temporary arrays, metric normalization, result construction, native
kernels, and precision all influence the comparison. These measurements do not
isolate algorithmic complexity or implementation efficiency alone.

No core search code or archive formats were changed for benchmarking. FAISS is
used only by the optional benchmark adapter, never inside VectorForge's algorithms.

## Installation and compatibility

Core development remains:

```powershell
uv sync --locked
uv run pytest -q
```

Install optional FAISS, Matplotlib, psutil, and threadpoolctl:

```powershell
uv sync --locked --extra benchmark
uv run --extra benchmark python -c "import faiss; print(faiss.__version__)"
uv run --extra benchmark pytest -v
```

`uv.lock` pins the resolved versions. Python 3.12.4, NumPy 2.5.4, and the native
Windows x86-64 `faiss-cpu` 1.15.1 wheel were successfully imported together during
implementation. Python and NumPy were not downgraded. The compatibility check was
`uv pip install --dry-run faiss-cpu` before adapter implementation. See the
[versioned PyPI distribution](https://pypi.org/project/faiss-cpu/1.15.1/) for wheel
platform tags and the [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
for supported build/install options.

FAISS import/native-library failures are explicit. A requested four-engine run
never silently becomes a two-engine run: failed engines receive failed rows for
every requested probe and the CLI exits nonzero. Optional FAISS integration and
Matplotlib chart integration tests report explicit skips when dependencies are
absent; core correctness tests do not depend on them.

If a future Windows/Python combination has no suitable wheel, keep the working
environment intact. A reproducible alternative is a separate WSL2 Ubuntu checkout
of the same project and lockfile. With uv installed inside WSL:

```sh
cd ~/vectorforge
uv python install 3.12
uv sync --locked --extra benchmark
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset quick
```

Do not combine WSL timings with native Windows measurements as one workload.

## Commands and workload sizes

```powershell
uv run --extra benchmark python -m vectorforge.benchmarks.cli --help
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset quick --output results/quick
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset standard --output results/standard
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset full --output results/full
```

| Preset | Indexed sizes | Dimensions | Training rows | Queries | nlist | Repetitions |
| --- | --- | --- | --- | --- | --- | --- |
| quick | 1,000; 10,000 | 32; 128 | 2,000 | 50 | 16 | 3 |
| standard | 10,000; 50,000 | 32; 128 | 4,096 | 100 | 32 | 3 |
| full | 10,000; 50,000; 100,000 | 32; 128 | 8,192 | 200 | 64 | 5 |

Each size/dimension pair is a separate run/report folder. The standard and full
grids are opt-in and can be expensive, especially VectorForge K-Means training and
float64 cosine scoring. With no arguments the CLI runs one 1,000-vector,
32-dimensional workload. Presets never run merely by importing modules or tests.

```powershell
uv run --extra benchmark python -m vectorforge.benchmarks.cli --size 10000 --dimension 128 --metric cosine --k 10 --nlist 32 --nprobe 1,4,8,32 --seed 42
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset full --dry-run
uv run --extra benchmark python examples/benchmark_demo.py
```

Supported options are `--preset`, `--size`, `--dimension`, `--training-size`,
`--query-count`, `--metric`, `--distribution`, `--duplicate-fraction`, `--k`,
`--nlist`, `--nprobe`, `--seed`, `--repetitions`, `--warmup`, `--threads`,
`--max-iter`, `--engines`, `--output`, `--no-charts`, `--dry-run`, and `--plot`.
Size/dimension overrides reduce a preset grid to those requested values. Defaults
are seed 42, cosine, normal distribution, k=10, five warmups, one thread, and
25 clustering iterations. Probe defaults are unique values from 1,2,4,8,16,nlist
that do not exceed nlist. Explicit invalid or duplicate probes are rejected.

`--engines` accepts comma-separated `vectorforge_exact`, `vectorforge_ivf`,
`faiss_flat`, and `faiss_ivf`. Any subset still uses VectorForge Exact ground truth.
All four are selected by default. `--no-charts` explicitly omits figures; chart
dependency errors otherwise fail clearly. `--dry-run` validates configuration
without requiring native dependencies, generating data, or launching workers.

## Dataset reproducibility

The generator uses `SeedSequence(seed).spawn(4)` for population parameters,
training rows, indexed rows, and query rows. Each stream uses its own `default_rng`.
Uniform vectors use [-1,1); normal vectors use mean 0 and standard deviation 1.
Clustered data shares up to 16 seeded population centers, with independent sampled
assignments and noise in each stream. Sharing a population is not sharing samples.

Datasets are contiguous float64 matrices. IDs are contiguous signed int64 values
centered around zero, including negative IDs and -1. Optional duplicate_fraction
replaces that fraction of the final indexed rows with the first indexed vector;
IDs remain distinct. It changes neither training nor query samples. Duplicate
workloads are primarily useful for correctness and tie-policy investigation.

The parent generates once, computes exact ground truth once outside search timing,
and writes temporary, pickle-free arrays for all workers. Every worker's SHA256
fingerprints must match the parent's training/index/query/ID fingerprints. Query
count does not affect training or indexed samples. Data bytes, configuration,
NumPy version, and seed are recorded; timestamps and timing measurements themselves
are naturally not deterministic.

## Fairness and process isolation

Engines run sequentially in fresh Python processes on the same machine. A seeded
permutation selects engine order and is recorded. Each process loads identical
dataset bytes and ground truth. There is one build per engine, reused across its
probe settings. A stable permutation of query order is prepared independently for
each repetition and reused across engines/probes.

The parent sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, and NUMEXPR_NUM_THREADS in the child
environment **before** Python imports numerical libraries. This matters because
`python -m vectorforge...` imports the parent package before the CLI executes.
The worker also uses threadpoolctl and FAISS omp_set_num_threads. Requested
environment values, effective discovered BLAS/OpenMP pools, and FAISS thread count
are saved per worker. The parent's untimed ground truth uses threadpoolctl too.
The default `--threads 1` is the primary configuration.

Processes isolate index lifetime and allocator history from other engines, but
do not pin CPUs, fix CPU frequency, disable turbo, flush OS caches, or prevent
background applications. Repeat runs on an otherwise idle machine and inspect
tail distributions instead of trusting one timing. Do not run competing benchmarks
concurrently. Library initialization and datasets are in the baseline; adapter
construction, training, insertion, and search remain measured separately.

## Metric, precision, and ID translation

| Metric | VectorForge | FAISS |
| --- | --- | --- |
| Euclidean | Unsquared L2, float64 | IndexFlatL2 / IVFFlat L2, float32; sqrt before returning scores |
| Cosine | Zero-safe normalization, float64 | Inner product on normalized, contiguous float32 inputs |
| Dot | Inner product, float64 | IndexFlatIP / IVFFlat inner product on original float32 inputs |

Cosine normalization uses the same zero-safe float64 normalization before FAISS
conversion; zero vectors stay zero. Conversion is validated for finite float32
representability. FAISS Flat is wrapped in IndexIDMap2; IVFFlat uses native
add_with_ids. Both use nonnegative surrogate IDs mapped back to external int64
IDs. Therefore native missing row -1 is filtered without losing valid external
ID -1, and the entire signed int64 ID range is supported.

Final FAISS scores are mapped to native metric units and returned sets sorted by
score then external ID. FAISS's own boundary tie selection is retained: the
adapter does not request all rows to expand ties, which would distort its measured
cost. Exact ground truth uses VectorForge's canonical ID boundary policy. Thus
FAISS Flat recall can be below 1 for ties or float32 near-ties even though it is
exhaustive. Float32 uses fewer coordinate bytes than float64, with different
rounding and memory costs. This is not an equal-precision implementation contest.
See [FAISS metric semantics](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

Both IVF engines receive the same independent training matrix, nlist, seed,
maximum iteration count, inserted rows, IDs, and queries. FAISS has its own
clustering/initialization, metric-specific quantizer and convergence behavior;
partitions need not coincide. In particular FAISS's IP routing differs from
VectorForge's original-vector Euclidean routing heuristic for dot product.
Presets supply at least 39 training samples per centroid to avoid the common
FAISS undertraining warning. Custom smaller training sets remain permitted if
training_size >= nlist; the report warns and captures worker stderr.

VectorForge IVF deliberately switches to direct full-dataset ranking at
nprobe=nlist. It bypasses list routing/gathering and must not be interpreted as
ordinary cluster scanning. `full_probe_direct_path` labels these rows. FAISS IVF
full probing still uses native list scanning. No algorithm is altered for timing.

## Timing and statistics

All durations use monotonic `time.perf_counter_ns()`:

- Construction measures adapter/native index creation after library imports.
- Training measures IVF preprocessing plus clustering. Exact/Flat training is
  not applicable (`null`), never a fabricated zero-duration measurement.
- Insertion includes adapter preprocessing/conversion, validation, and insertion.
- build_total_ms is the sum of measured stage durations, excluding RSS snapshots,
  dataset generation, imports, and process startup. Build stages are measured once,
  not repeated statistical estimates.
- Each probe setting gets warmup single-query searches before timing. Every query
  then executes once in every measured repetition. Latencies cover the end-to-end
  adapter: validation, per-query normalization/conversion, search, and Python result
  creation. They do not represent bare native kernel latency.
- Mean, p50, p95, and p99 use all measured samples, converted from ns to ms.
  Percentiles use NumPy's linear interpolation. Raw samples retain query/repetition
  identifiers for further analysis; repeated samples are not independent queries.
- QPS is completed queries divided by separately measured workload wall time.
  That interval includes the query loop, latency timers, sample recording, and
  returned-result cleanup. It is **not** the reciprocal of mean query latency.

No dataset generation, training, recall evaluation, printing, report writing, or
plotting occurs inside search timers. FAISS batch throughput is not compared to
VectorForge single-query throughput. No batch benchmark is implemented.

## Recall and nprobe

VectorForge Exact float64 supplies primary ground truth for every engine, including
FAISS Flat. Recall is the intersection of first-k external ID sets divided by the
number of exact results; the existing recall_at_k utility is reused. Per-query
recall and its arithmetic mean are exported. Ground truth ties use score then
ascending ID, without expanding the cutoff. k larger than N uses the smaller
ground-truth denominator. Empty exact results conventionally score 1, although
benchmark configs require positive dataset/query counts.

Partial probes have no quality guarantee. Full-probe VectorForge IVF must match
exact IDs and recall 1; this is tested. Increasing probes generally improves
coverage, but measured latency can fluctuate or decrease, especially at the
VectorForge full-probe shortcut. Interpret recall-versus-latency jointly instead
of assuming IVF always wins or latency must increase monotonically.

## RSS memory methodology

psutil records process RSS bytes after loading data/libraries and before index
construction, after construction, after IVF training, and after insertion.
gc.collect runs before each snapshot outside timed stages. Flat/exact post-training
RSS is null because the stage is inapplicable. The reported retained_rss_delta_bytes
is post-insertion RSS minus baseline, an **estimate**, not an exact object size.

RSS includes native allocations and allocator-retained temporary memory. Negative
deltas are retained rather than clamped or hidden. OS paging, native caches, lazy
allocation, and gc/allocator behavior can affect snapshots. Search scratch memory
and peak build memory are not measured. No sys.getsizeof or tracemalloc claim is
made about complete index memory. The figure shows both absolute baseline and
post-insertion RSS to make differing library baselines visible.

## Report schema and chart regeneration

[Published representative results](benchmark_results/README.md) are included in Git
under `docs/benchmark_results/`. The ignored `results/` directory holds locally
generated benchmark outputs. Publishing a representative run copies its existing
artifacts without rerunning the benchmark or changing its recorded metadata.

Each unique UTC timestamp/UUID run directory contains:

```text
metadata.json
summary.json
summary.csv
summary.txt
latency_samples.csv
charts/
  search_latency.png
  throughput.png
  recall_vs_nprobe.png
  recall_vs_latency.png
  build_time.png
  memory.png
```

Benchmark schema version is 1. Metadata records the configuration, Python/NumPy/
FAISS versions, OS, CPU description, logical CPU count, timestamp, data hashes,
execution order, thread settings, per-worker pool information, warnings, and
completion status. summary.json has benchmark_schema_version and results; each row
identifies workload, engine, precision, probe settings, measured stages, latency
statistics, QPS, recall_per_query, and RSS snapshots. CSV uses a stable documented
SUMMARY_FIELDS order in reporting.py, with the same scalar fields. Per-query recall
remains in JSON. latency_samples.csv contains engine,nprobe,repetition,query_index,
latency_ns. nprobe is blank/null for Flat/Exact.

Successful rows use status=ok; failed rows use status=failed with an error and omit
unmeasured JSON metrics (blank CSV cells). JSON rejects NaN/Infinity and uses null
for inapplicable measured fields. Existing run directories are never reused; chart
regeneration intentionally replaces only the figures in the chosen saved run.

```powershell
uv run --extra benchmark python -m vectorforge.benchmarks.cli --plot results/quick/benchmark_YOUR_RUN_ID
```

Pass the actual run folder printed by the CLI. Regeneration reads summary.json,
not the engines or original dataset, and rejects mixing distinct workloads in one
figure set. Failed engine rows are omitted from charts but remain explicit in
summary files and completion status. Build and RSS charts show each engine once,
since the same built index is reused across probes. Charts use actual recorded
values; there are no fabricated missing-FAISS bars.

## Limitations and interpretation

Synthetic data is not a substitute for production embeddings. Independent samples
avoid leakage but do not model every distribution shift. Seeds reproduce data,
not OS scheduling or bit-identical results across library/hardware versions. Tail
percentiles from a small query sample have limited statistical strength. Build
stages have one observation; no confidence intervals or peak memory are claimed.
Adapter and conversion overhead can dominate small datasets, especially FAISS.

Read the observed tables and chart titles with these scopes in mind. An optimized
FAISS implementation can outperform educational NumPy code without invalidating
the project: the value is in correct algorithms, reproducibility, explainable
tradeoffs, and honest measurements. FastAPI, frontend, Docker, CI/CD, and HNSW
remain outside Phase 3.
