# Python engine API

These examples and contracts describe the direct Python engine. The
[web application](phase4_webapp.md) adds API validation, workload limits,
per-index locks, and archive manifests around the same engine.

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

## Input and mutation contracts

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

## Metrics and score interpretation

| Metric argument | Returned score                                     | Best match     |
| --------------- | -------------------------------------------------- | -------------- |
| `"cosine"`      | `dot(a, b) / (norm(a) * norm(b))`, in `[-1, 1]`    | Largest score  |
| `"euclidean"`   | `sqrt(sum((a - b)**2))`, the unsquared L2 distance | Smallest score |
| `"dot"`         | `sum(a * b)`, without normalization                | Largest score  |

Cosine comparisons involving any zero vector score **0**, including zero versus
zero. Consequently, zero vectors rank above negative cosine similarities. Cosine
is invariant to positive rescaling; dot product also reflects magnitude.

Standalone functions in `vectorforge.engine.distance` accept a 1-D vector as the
first argument and either a matching vector or an `(N, d)` matrix as the second.
They return a Python float or an `(N,)` NumPy array, respectively. Cosine scales
vectors before normalization to avoid norm overflow; L2 uses `hypot` reduction to
avoid overflow from intermediate squares. Finite precision still applies.

## Persistence

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

## IVF workflow

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

| Search metric | Training and insertion representation    | Query routing                                               |
| ------------- | ---------------------------------------- | ----------------------------------------------------------- |
| Euclidean     | Original vectors                         | Nearest L2 centroids                                        |
| Cosine        | Unit-normalized vectors; zeros stay zero | Normalize query, nearest L2 centroids                       |
| Dot           | Original vectors                         | Nearest L2 centroids, a heuristic for dot-product retrieval |

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
utility; timing, throughput, and FAISS comparisons live in the separate
[benchmark package](benchmarking.md).

### IVF persistence

IVF has a separate strict schema with `index_type="ivf"` and independent
`ivf_format_version=1`. It stores original vectors, stable IDs, centroids,
assignments, dimension, metric, nlist, nprobe, trained status, and K-Means settings.
The loader rejects assignments inconsistent with nearest-centroid routing and
reconstructs inverted lists. Only trained indexes can be saved. Atomic writes
refuse corrupt or incompatible targets; IVF compatibility requires matching
dimension, metric, and nlist. Exact and IVF archives cannot be interchanged, and
the original five-field exact schema remains unchanged.

See [architecture](architecture.md) for exact-search complexity and numerical
policies, and [IVF design](ivf_design.md) for clustering and routing details.
