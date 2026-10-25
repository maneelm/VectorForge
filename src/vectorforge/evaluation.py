"""Small ID-based Recall@k evaluation; no timing or benchmark framework."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vectorforge.engine._validation import positive_integer, real_array
from vectorforge.engine.base import SearchResult
from vectorforge.engine.exact import ExactVectorIndex
from vectorforge.engine.ivf import IVFVectorIndex


@dataclass(frozen=True, slots=True)
class RecallReport:
    per_query: tuple[float, ...]
    mean: float


def recall_at_k(
    approximate: Sequence[SearchResult], exact: Sequence[SearchResult], k: int
) -> float:
    """Compare first k IDs; empty exact ground truth has recall 1 by convention.

    Exact ties use the index's deterministic score/ascending-ID order, without
    expanding the boundary to include other equal-scoring IDs.
    """
    k = positive_integer(k, "k")
    truth = {item.id for item in exact[:k]}
    found = {item.id for item in approximate[:k]}
    return len(truth & found) / len(truth) if truth else 1.0


def evaluate_recall(
    approximate: IVFVectorIndex,
    exact: ExactVectorIndex,
    queries: ArrayLike,
    k: int,
    *,
    nprobe: int | None = None,
) -> RecallReport:
    """Evaluate a nonempty query matrix against indexes with the same metric/dimension.

    Callers must ensure both indexes contain the same dataset and IDs. An empty
    ground-truth index scores 1 per query. An empty query matrix is rejected.
    """
    k = positive_integer(k, "k")
    if (approximate.dimension, approximate.metric) != (exact.dimension, exact.metric):
        raise ValueError("recall evaluation requires matching dimensions and metrics")
    values = real_array(queries, "queries")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != exact.dimension:
        raise ValueError("queries must have nonempty shape (N, index dimension)")
    recalls = tuple(
        recall_at_k(approximate.search(query, k, nprobe=nprobe), exact.search(query, k), k)
        for query in values
    )
    return RecallReport(recalls, float(np.mean(recalls)))
