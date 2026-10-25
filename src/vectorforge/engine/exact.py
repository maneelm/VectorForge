"""Exact exhaustive search with partial selection and deterministic ties."""

from os import PathLike
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from vectorforge.engine import storage
from vectorforge.engine._ranking import rank
from vectorforge.engine._validation import (
    Ids,
    Metric,
    integer_ids,
    positive_integer,
    real_array,
    validate_metric,
    vector_batch,
)
from vectorforge.engine.base import BaseVectorIndex, SearchResult


class ExactVectorIndex(BaseVectorIndex):
    """In-memory float64 vectors with unique signed int64 IDs.

    The dimension and metric are fixed at construction. Input arrays are copied.
    This mutable index is not thread-safe; callers must synchronize mutations.
    """

    def __init__(self, dimension: int, metric: Metric = "cosine") -> None:
        self._dimension = positive_integer(dimension, "dimension")
        self._metric = validate_metric(metric)
        self._vectors = np.empty((0, self.dimension), dtype=np.float64)
        self._ids = np.empty(0, dtype=np.int64)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, vectors: ArrayLike, ids: Ids) -> None:
        batch = vector_batch(vectors, self.dimension)
        new_ids = integer_ids(ids)
        if len(batch) != len(new_ids):
            raise ValueError("each vector must have exactly one ID")
        if len(np.unique(new_ids)) != len(new_ids) or np.isin(new_ids, self._ids).any():
            raise ValueError("duplicate IDs are not allowed")
        # Allocate both arrays before committing any state.
        all_vectors = np.concatenate((self._vectors, batch))
        all_ids = np.concatenate((self._ids, new_ids))
        self._vectors, self._ids = all_vectors, all_ids

    def search(self, query: ArrayLike, k: int) -> list[SearchResult]:
        k = positive_integer(k, "k")
        query_array = real_array(query, "query")
        if query_array.shape != (self.dimension,):
            raise ValueError(f"query must have shape ({self.dimension},)")
        if self.size == 0:
            return []
        return rank(query_array, self._vectors, self._ids, self.metric, k)

    def delete(self, ids: Ids) -> None:
        requested = integer_ids(ids)
        keep = ~np.isin(self._ids, requested)
        vectors, remaining_ids = self._vectors[keep], self._ids[keep]
        self._vectors, self._ids = vectors, remaining_ids

    def save(self, path: str | PathLike[str]) -> None:
        storage.save_index(path, self._vectors, self._ids, self.dimension, self.metric)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        state = storage.load_index(path)
        index = cls(state.dimension, state.metric)
        index.add(state.vectors, state.ids)
        return index
