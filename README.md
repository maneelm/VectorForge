# VectorForge

VectorForge is a vector similarity search engine built from first principles.
It makes the algorithms and tradeoffs behind vector retrieval inspectable, with
an exact index serving as the correctness baseline for future approximate indexes.

Phases 1 and 2 implement in-memory exact search and IVF approximate search from
first principles. Phase 3 adds an optional reproducible comparison against FAISS;
FAISS is never used inside the VectorForge engine.

## Implemented features

- Abstract index interface and immutable, typed `SearchResult(id, score)` results.
- Single and batch insertion with unique, stable signed 64-bit integer IDs.
- Atomic validation of insertions and deletion with stable remaining IDs.
- Cosine similarity, Euclidean distance, and dot product.
- Exact top-k retrieval with `numpy.argpartition` and deterministic ID tie-breaking,
  including ties spanning the selection boundary.
- Strict input validation, empty-index behavior, and safe zero-vector handling.
- Versioned, validated, pickle-free NPZ persistence with atomic file replacement.
- Automated numerical, randomized reference, mutation, and persistence tests.
- Seeded K-Means++ initialization and Lloyd iteration with deterministic recovery
  of empty training clusters and atomic fitting.
- IVF training, inverted lists, atomic insertion/deletion, configurable probing,
  and a separate versioned persistence schema.
- Per-query and mean ID-based Recall@k evaluation against exact search.
- Optional four-engine benchmarks with isolated processes, explicit thread limits,
  raw latency samples, timed QPS, build-stage timings, RSS snapshots, CSV/JSON
  reports, and six regenerable Matplotlib charts.

## Stack and installation

Python 3.12+, uv, NumPy, Pytest, and Ruff. From the project directory, with uv
installed and on your PATH:

```sh
uv sync --locked
```

uv creates `.venv` and installs the package in editable mode, including development
tools. The committed `uv.lock` pins dependencies. `uv sync` updates the lock when
dependency requirements change. A local Python 3.12 interpreter is preferred;
uv can provision it when necessary.

## Run and validate

```sh
uv run python examples/basic_search.py
uv run python examples/ivf_search.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The example inserts and searches vectors, deletes one, saves to a temporary
archive, restores it, and verifies identical search results. It cleans up its
temporary directory automatically. To format changes, use `uv run ruff format .`.
The IVF example compares probe settings over a reproducible synthetic dataset,
reports measured Recall@5, verifies full probing, and checks persistence.

## Python usage

```python
from vectorforge import ExactVectorIndex

index = ExactVectorIndex(dimension=3, metric="cosine")
index.add([1, 0, 0], ids=101)
index.add([[0.8, 0.2, 0], [0, 1, 0]], ids=[102, 103])

for match in index.search([1, 0.1, 0], k=2):
    print(match.id, match.score)

index.delete([101, 999])  # Missing IDs are ignored.
index.save("my-index.npz")
restored = ExactVectorIndex.load("my-index.npz")
assert restored.search([1, 0.1, 0], 2) == index.search([1, 0.1, 0], 2)
```

`load` is a class method that returns a new index; it never replaces an existing
instance's state. Read-only properties expose `dimension`, `metric`, and `size`.

### Input and mutation contracts

- Dimension is a positive integer, fixed at construction along with the metric.
- A single vector has shape `(d,)`, a batch `(N, d)`, and a query exactly `(d,)`.
  An empty batch must have explicit shape `(0, d)` and empty IDs.
- Vectors contain real numbers, stored as copied `float64` arrays. NaN, infinity,
  complex, string, and object arrays are rejected. Nonfinite computed scores
  (including arithmetic overflow) raise `ValueError` rather than returning infinity or NaN.
- Each vector needs one integer ID in `[-2**63, 2**63 - 1]`. Booleans and floating
  IDs are rejected. IDs can be negative and need not be contiguous.
- Duplicate IDs within a batch or already present in the index reject the entire
  insertion. Caller-owned arrays cannot mutate stored data after insertion.
- Deletion ignores missing and repeated IDs. Deleted IDs can be reused. Invalid
  deletion inputs raise before changing any state.
- `k` must be a positive integer (booleans are rejected). A valid search returns
  `min(k, size)` results, or `[]` on an empty index. Queries and `k` are validated
  even when the index is empty.
- Equal computed scores sort by ascending ID. Ties mean exact equality of the
  computed float64 scores; no approximate-equality tolerance is applied.

### Metrics and score interpretation

| Metric argument | Returned score | Best match |
| --- | --- | --- |
| `"cosine"` | `dot(a, b) / (norm(a) * norm(b))`, in `[-1, 1]` | Largest score |
| `"euclidean"` | `sqrt(sum((a - b)**2))`, the unsquared L2 distance | Smallest score |
| `"dot"` | `sum(a * b)`, without normalization | Largest score |

Cosine comparisons involving any zero vector score **0**, including zero versus
zero. Consequently, zero vectors rank above negative cosine similarities. Cosine
is invariant to positive rescaling; dot product also reflects magnitude.

Standalone functions in `vectorforge.engine.distance` accept a 1-D vector as the
first argument and either a matching vector or an `(N, d)` matrix as the second.
They return a Python float or an `(N,)` NumPy array, respectively. Cosine scales
vectors before normalization to avoid norm overflow; L2 uses `hypot` reduction to
avoid overflow from intermediate squares. Finite precision still applies.

### Persistence

The NPZ schema contains `vectors`, `ids`, `dimension`, `metric`, and
`format_version` (currently `1`). Loading always uses `allow_pickle=False` and
validates all fields before constructing an index. No Python objects are serialized.
The exact supplied filename is used, without appending an extension.

Saving can replace a valid archive with the same dimension and metric. It refuses
to overwrite corrupt archives, unsupported formats, or incompatible metadata.
Use a separate path for a different index configuration. The parent directory
must exist. File access failures retain their `OSError` type; malformed archives
raise `ValueError`. Archive loading is intended for bounded, local files: this
version does not impose compressed-archive resource limits.

## IVF workflow (Phase 2)

IVF partitions routing vectors around K-Means centroids. Each inverted list
references rows belonging to one centroid. A query probes a subset of lists and
scores only their candidates using the original metric. `nlist` controls the
number of centroids; `nprobe` controls how many lists a query searches.

```python
from vectorforge import ExactVectorIndex, IVFVectorIndex
from vectorforge.evaluation import evaluate_recall

vectors = [[1, 0], [0.9, 0.1], [0, 1], [-1, 0]]
ids = [101, 102, 103, 104]
ivf = IVFVectorIndex(2, nlist=2, nprobe=1, metric="cosine", random_state=42)
ivf.train(vectors)  # Learn centroids; no vectors have been inserted yet.
ivf.add(vectors, ids)
matches = ivf.search([1, 0], k=3)
full_matches = ivf.search([1, 0], k=3, nprobe=ivf.nlist)
ivf.nprobe = 2  # Validated default for subsequent searches.
ivf.delete(104)
ivf.save("ivf-index.npz")
restored = IVFVectorIndex.load("ivf-index.npz")
assert restored.search([1, 0], 3) == ivf.search([1, 0], 3)

exact = ExactVectorIndex(2, "cosine")
exact.add(vectors[:3], ids[:3])  # Same surviving dataset as IVF.
report = evaluate_recall(ivf, exact, [[1, 0], [0, 1]], k=3)
print(report.per_query, report.mean)
```

Both the default property and a per-search override require `1 <= nprobe <= nlist`.
If using fewer than four clusters, explicitly set `nprobe` (the constructor
defaults remain `nlist=16`, `nprobe=4`). A search can return fewer than k candidates,
including none when all selected lists are empty. All-list probing matches exact
search IDs, order, and equivalent scores. Partial probing has no recall guarantee.

An untrained index rejects add, search, centroid access, and save. Trained-empty
indexes support search (`[]`) and persistence. Training requires an `(N, d)` matrix
with `N >= nlist`; it never inserts samples. Empty trained indexes may be retrained;
populated indexes reject retraining. Deleting every vector leaves the index trained.

### K-Means and routing

`vectorforge.engine.kmeans.KMeans` implements K-Means++ and Lloyd iteration, exposes
defensive centroid/label copies and read-only fitted statistics, and predicts using
squared Euclidean distance. A seeded `default_rng` makes runs reproducible without
changing global random state. Empty clusters receive the highest-loss eligible
sample from a cluster with more than one member, ties by sample row.

Training stops on stable repaired assignments, maximum centroid movement <= `tol`,
or `max_iter`. Final means and inertia describe `labels_`, the final nonempty
training partition. Predictions use nearest centroids with lowest-ID ties; at a
stopping limit or duplicate centroid locations, predictions may differ from training
labels. Failed fitting preserves previous fitted state.

| Search metric | Training and insertion representation | Query routing |
| --- | --- | --- |
| Euclidean | Original vectors | Nearest L2 centroids |
| Cosine | Unit-normalized vectors; zeros stay zero | Normalize query, nearest L2 centroids |
| Dot | Original vectors | Nearest L2 centroids, a heuristic for dot-product retrieval |

Centroids are arithmetic means, including for cosine; they are not renormalized.
Original vectors are retained for final scores, preserving Phase 1 semantics.
Euclidean routing is not mathematically optimal for maximum-inner-product search.

### Recall@k

`recall_at_k(approximate_results, exact_results, k)` compares stable IDs in the
first k results: intersection size divided by the number of exact results.
`evaluate_recall(ivf, exact, queries, k, nprobe=...)` returns a frozen report with
`per_query` and `mean`. Exact ties use ascending IDs at the cutoff, without expanding
the ground-truth set. Empty exact results have recall 1 by convention; an empty
query matrix is rejected. Callers must evaluate indexes with the same dataset and
IDs; dimensions and metrics are checked automatically. This is a retrieval-quality
utility; timing, throughput, and FAISS comparisons remain Phase 3 work.

### IVF persistence

IVF has a separate strict schema with `index_type="ivf"` and independent
`ivf_format_version=1`. It stores original vectors, stable IDs, centroids,
assignments, dimension, metric, nlist, nprobe, trained status, and K-Means settings.
The loader rejects assignments inconsistent with nearest-centroid routing and
reconstructs inverted lists. Only trained indexes can be saved. Atomic writes
refuse corrupt or incompatible targets; IVF compatibility requires matching
dimension, metric, and nlist. Exact and IVF archives cannot be interchanged, and
the original five-field exact schema remains unchanged.

## Benchmarking (Phase 3)

The benchmark compares VectorForge Exact/IVF (float64) against FAISS Flat/IVFFlat
(float32), using the same independent training, indexed, and query datasets. FAISS,
Matplotlib, psutil, and threadpoolctl are optional; core installation stays lightweight.

```powershell
uv sync --locked --extra benchmark
uv run --extra benchmark python -m vectorforge.benchmarks.cli --help
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset quick --output results/quick
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset standard --output results/standard
uv run --extra benchmark python -m vectorforge.benchmarks.cli --preset full --dry-run
uv run --extra benchmark python examples/benchmark_demo.py
```

Quick runs 1,000/10,000 vectors at 32/128 dimensions with 50 independent queries
and three search repetitions. Standard and full workloads are opt-in. All runs
default to one thread. Missing requested engines produce explicit failed report
rows and a nonzero CLI exit, rather than a claimed four-engine success.

Each run creates a new folder containing metadata, summary CSV/JSON/text, raw
latency samples, and charts. Regenerate figures without rerunning indexes:

```powershell
uv run --extra benchmark python -m vectorforge.benchmarks.cli --plot results/quick/benchmark_YOUR_RUN_ID
```

Use the actual folder printed by the benchmark. The primary timing includes
single-query validation, conversion, retrieval, and Python result creation.
QPS uses an actual timed workload, not inverse mean latency. RSS deltas are
estimates, not peak or exact index memory. VectorForge IVF's full-probe shortcut
and FAISS precision/boundary-tie differences are explicitly labeled. See
[benchmark methodology](docs/benchmarking.md) for supported arguments, schema,
Windows/WSL compatibility, reproducibility, and interpretation limits.
The [recorded quick-run results](docs/phase3_results.md) contain actual measurements
from the implementation environment, validation results, and links to local artifacts.

## Exact search complexity and limitations

For `N` vectors of dimension `d`, scoring costs **O(Nd)**. Partial selection costs
O(N), including finding the full boundary tie and selecting its lowest IDs; final
sorting costs O(k' log k'), where `k' = min(k, N)`. Total search time is
O(Nd + N + k' log k'). Storage is O(Nd + N); search may allocate O(Nd) temporary
arrays during validation and scoring, as well as O(N) scores and candidates.

Brute-force search examines every vector on every query. Latency and memory grow
with dataset size and dimension. Insertion and deletion copy arrays; this is a
correctness-oriented baseline, not a high-throughput storage system. There is no
concurrency control, batch-query API, metadata filtering, or streaming loader.
Callers must synchronize concurrent mutations and writes. No measured performance
claims are made in Phase 1.

See [architecture](docs/architecture.md) for module boundaries and design details.
See [IVF design](docs/ivf_design.md) for algorithms, numerical policies, and complexity.

IVF may reduce the number of scored vectors but adds routing and training costs;
its benefit depends on the measured workload. Unbalanced lists, poor training
coverage, or duplicate centroid locations can reduce recall or leave lists empty.
Mutation rebuilds row references and copies arrays. Training uses an `(N, nlist)`
distance matrix, so large training sets require care. There are no locks, archive
resource limits, automatic retraining, or support for distribution drift.

## Local web application (Phase 4)

The local-first FastAPI backend and React/TypeScript dashboard support index
management, IVF training, vector insertion/search/deletion, safe saved archives,
six interactive benchmark charts, and bounded background benchmark jobs.

```powershell
uv sync --locked --extra web --extra benchmark
uv run --extra web --extra benchmark uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>; API documentation is at
<http://127.0.0.1:8000/docs>. Use one backend worker: active indexes are in memory,
while explicitly saved archives and benchmark reports survive restarts. Request
bodies use integer or quoted decimal vector IDs; match IDs are strings to preserve
int64 precision in browsers. The web application adds locks and resource limits
around the unchanged engine. It does not change NPZ formats or benchmark methodology.

See [web application documentation](docs/phase4_webapp.md) for endpoints, examples,
configuration, persistence, limits, tests, browser smoke workflows, and troubleshooting.

## Roadmap

1. **Phase 1 — complete:** project foundation and exact search.
2. **Phase 2 — complete:** k-means and IVF approximate nearest-neighbor search.
3. **Phase 3 — complete:** reproducible FAISS comparisons, reports, and charts.
4. **Phase 4 — complete:** local FastAPI endpoints and a React/TypeScript benchmarking dashboard.
5. **Phase 5:** Docker, CI/CD, profiling, expanded documentation, and optional HNSW.

Phase 5 is not implemented in this repository.
