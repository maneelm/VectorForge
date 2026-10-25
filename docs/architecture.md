# VectorForge architecture

## Phase 2 extension

The exact index and original five-field NPZ format remain supported. IVF implements
the same base interface and extends search with an optional keyword-only nprobe.

```text
engine/kmeans.py       Reusable seeded K-Means++ and Lloyd clustering
engine/ivf.py          Training lifecycle, row-reference inverted lists, probing
engine/ivf_storage.py  Independent strict IVF archive schema and atomic writes
engine/_routing.py     Consistent training/insertion/query representations
engine/_ranking.py     Shared metric scoring and deterministic partial selection
evaluation.py          ID-based per-query and mean Recall@k
```

Exact search delegates its scoring/selection to `_ranking.py`; this extracts the
existing algorithm so both indexes share the complete top-k boundary tie policy.
The base contract, score semantics, validation, exact archive schema, and original
tests remain unchanged. IVF stores original vectors once and references their rows
through lists; it learns centroids before users explicitly insert data.

See [IVF design](ivf_design.md) for the conceptual diagram, clustering semantics,
state transitions, numerical policies, complexity, and archive fields. The remaining
sections document the preserved exact-index foundation.

## Module boundaries

```text
Public API: vectorforge.{BaseVectorIndex, ExactVectorIndex, IVFVectorIndex, SearchResult}
  engine/base.py          Abstract contract and frozen result record
  engine/exact.py         Mutable index, ID lifecycle, ranking
  engine/distance.py      Stateless vectorized scoring
  engine/storage.py       NPZ schema, validation, atomic persistence
  engine/_validation.py   Shared array, ID, dimension, and metric validation
```

The shared internal validation module is a small addition to the requested layout.
It keeps insertion and loading aligned on numeric and ID invariants. `utils` is
an importable namespace for future shared utilities and contains no Phase 1 logic.

## Index contract

`BaseVectorIndex` specifies add, search, delete, save, and class-level load methods,
plus dimension, metric, and size properties. IVF implements this contract; future
HNSW implementations can use it with their own persistence formats without
changing the result type. Approximate search relaxes retrieval guarantees while
retaining score semantics and result ordering.

`SearchResult` is a frozen dataclass with an integer `id` and a float `score`.
Scores retain their native metric meaning: cosine and dot are maximized; L2 is
minimized. There is no implicit conversion between similarity and distance.
Search returns a list sorted by best score then ascending stable ID. Empty
indexes return an empty list after validating the query and k.

## Representation and mutation

An `(N, d)` float64 NumPy array and a parallel `(N,)` int64 array hold the state.
Dimension and metric are configured at construction and exposed as read-only
properties. IDs are independent of row positions, so deleting a row cannot
renumber remaining vectors. Negative IDs are valid.

Insertion validates shapes, finiteness, ID count, ID range, batch uniqueness,
and collisions with existing IDs. It allocates both replacement arrays before
assigning either. A validation or allocation failure leaves existing state intact.
Copying inputs prevents later caller mutations from changing the index. Deletion
validates IDs, creates a row mask, allocates both filtered arrays, and then commits.
Missing and repeated deletion IDs are harmless; deleted IDs may be inserted again.

This simple representation favors clarity and correctness. Append and delete
operations copy data. The index provides no locks or concurrent snapshot semantics.

## Scoring and selection

Distance functions require a nonempty 1-D left vector and a matching vector or
matrix on the right. Numeric conversion and shape checks are shared; nonfinite
values and non-real types fail before scoring. Computation uses NumPy operations
over the batch rather than Python loops.

Cosine first divides by each vector's largest absolute coordinate, then normalizes
the scaled vector. This avoids squaring extremely large or small raw coordinates.
Zero vectors stay zero; their cosine with every vector is defined to be zero.
Roundoff is clipped to `[-1, 1]`. Euclidean uses `hypot.reduce` to avoid unnecessary
square overflow. Dot uses matrix-vector multiplication. When the computed score is
nonfinite, including arithmetic overflow, scoring raises `ValueError`.

Search converts similarity to a lower-is-better cost by negation. It uses
`argpartition` to locate the kth cost, collects every strictly better row, and
examines **all** rows at the cutoff. If a boundary tie exceeds remaining slots,
a second partition chooses its smallest IDs. A lexicographic sort by cost and ID
then orders only the selected results. Sorting only an arbitrary initial partition
would fail the ID tie rule at the boundary, so that approach is avoided.

Scoring is O(Nd), boundary handling O(N), and final sorting O(k' log k'). Score
equality is exact floating-point equality. Platform-dependent rounding of nearly
equal scores can affect ranking; mathematically equivalent expressions need not
produce bit-identical floating-point values.

## Persistence and failure behavior

Version 1 is a NumPy NPZ archive with exactly five fields:

| Field | Shape / representation |
| --- | --- |
| `vectors` | `(N, d)` finite real numeric array, saved as float64 |
| `ids` | `(N,)` unique integers within signed int64 range, saved as int64 |
| `dimension` | positive integer scalar |
| `metric` | Unicode scalar: cosine, euclidean, or dot |
| `format_version` | integer scalar equal to 1 |

The reader uses `allow_pickle=False`, rejects missing, unexpected, or duplicate
archive fields, and validates metadata and arrays together. Object arrays are
never deserialized. Empty indexes retain `(0, d)` vector shape and integer IDs.
Unknown versions are rejected; there is no automatic migration or reinterpretation.
`load` constructs a fresh index after validation, avoiding partial replacement of
an existing object's dimension, metric, or vectors.

The writer validates its payload and any existing target before writing. A target
must have the same dimension and metric; a corrupt or incompatible target remains
untouched. Data is written to a temporary sibling file, flushed and fsynced, then
published with `os.replace`. Failed writes remove the temporary file while
preserving the prior target. This provides atomic visibility on the local
filesystem, not a cross-process lock or a guarantee against every power-loss
scenario. There are no archive size or decompression limits, so callers should
bound input files and avoid accepting arbitrary remote archives without such limits.

Invalid inputs and malformed data raise meaningful `ValueError`s. Filesystem
failures propagate as `OSError` subclasses, including `FileNotFoundError`.

## Verification strategy

Metric tests cover analytic values, shape rejection, nonfinite inputs, zero-vector
policy, batch behavior, and extreme numeric scales. Index tests exercise lifecycle
and atomicity as well as deterministic ties in different insertion orders. A
seeded scalar reference independently computes metrics and fully sorts results;
randomized tests compare IDs and scores across every metric after insertion and
deletion, including zero queries. Persistence tests cover round trips, IDs, metrics,
malformed archives, incompatible overwrites, and simulated write failures.

Both examples are executable integration checks. Pytest and Ruff checks are the
acceptance gates for Phases 1 and 2. No Phase 3–5 algorithms, APIs, frontend, or
benchmark infrastructure are included.
