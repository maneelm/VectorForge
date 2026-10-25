"""Shared partial selection and metric ranking for exact and IVF search."""

import numpy as np
from numpy.typing import NDArray

from vectorforge.engine import distance
from vectorforge.engine._validation import Metric
from vectorforge.engine.base import SearchResult


def smallest(costs: NDArray[np.float64], ids: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    """Select smallest costs, resolving all boundary ties by ascending ID."""
    count = min(k, len(ids))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    partition = np.argpartition(costs, count - 1)
    cutoff = costs[partition[count - 1]]
    better = np.flatnonzero(costs < cutoff)
    tied = np.flatnonzero(costs == cutoff)
    needed = count - len(better)
    if needed < len(tied):
        tied = tied[np.argpartition(ids[tied], needed - 1)[:needed]]
    candidates = np.concatenate((better, tied))
    return candidates[np.lexsort((ids[candidates], costs[candidates]))]


def rank(
    query: NDArray[np.float64],
    vectors: NDArray[np.float64],
    ids: NDArray[np.int64],
    metric: Metric,
    k: int,
) -> list[SearchResult]:
    """Score original vectors and return best-first typed results."""
    if len(ids) == 0:
        return []
    scorer = {
        "cosine": distance.cosine_similarity,
        "euclidean": distance.euclidean_distance,
        "dot": distance.dot_product,
    }[metric]
    scores = np.asarray(scorer(query, vectors))
    costs = scores if metric == "euclidean" else -scores
    return [SearchResult(int(ids[i]), float(scores[i])) for i in smallest(costs, ids, k)]
