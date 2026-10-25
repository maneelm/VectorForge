"""Inverted-file approximate search over trained Euclidean K-Means partitions."""

from os import PathLike
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorforge.engine import ivf_storage
from vectorforge.engine._ranking import rank, smallest
from vectorforge.engine._routing import assign, representation
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
from vectorforge.engine.kmeans import KMeans, configuration, squared_distances


def _lists(assignments: NDArray[np.int64], nlist: int) -> tuple[NDArray[np.int64], ...]:
    """Lists contain row references, not copies of the original vectors."""
    return tuple(np.flatnonzero(assignments == i) for i in range(nlist))


class IVFVectorIndex(BaseVectorIndex):
    """Train, then explicitly add vectors; training alone leaves the index empty.

    Cosine uses normalized vectors for routing. Dot uses original-vector L2
    routing as a heuristic. Candidate scores always retain Phase 1 semantics.
    All mutations stage replacement state; callers must synchronize concurrent use.
    """

    def __init__(
        self,
        dimension: int,
        nlist: int = 16,
        nprobe: int = 4,
        metric: Metric = "cosine",
        random_state: int = 42,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        self._dimension = positive_integer(dimension, "dimension")
        self._metric = validate_metric(metric)
        self._nlist = positive_integer(nlist, "nlist")
        self.nprobe = nprobe
        self._max_iter, self._tol, self._random_state = configuration(max_iter, tol, random_state)
        self._centroids: NDArray[np.float64] | None = None
        self._vectors = np.empty((0, self.dimension), dtype=np.float64)
        self._ids = np.empty(0, dtype=np.int64)
        self._assignments = np.empty(0, dtype=np.int64)
        self._inverted_lists: tuple[NDArray[np.int64], ...] = ()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def nlist(self) -> int:
        return self._nlist

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._nprobe = self._validate_nprobe(value)

    def _validate_nprobe(self, value: int) -> int:
        value = positive_integer(value, "nprobe")
        if value > self.nlist:
            raise ValueError("nprobe must be between 1 and nlist")
        return value

    @property
    def is_trained(self) -> bool:
        return self._centroids is not None

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise ValueError("IVF index must be trained before this operation")

    @property
    def centroids(self) -> NDArray[np.float64]:
        self._require_trained()
        assert self._centroids is not None
        return self._centroids.copy()

    def train(self, training_vectors: ArrayLike) -> None:
        if self.size:
            raise ValueError("cannot retrain a populated IVF index; delete all vectors first")
        values = real_array(training_vectors, "training_vectors")
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"training_vectors must have shape (N, {self.dimension})")
        model = KMeans(self.nlist, self._max_iter, self._tol, self._random_state)
        model.fit(representation(values, self.metric))
        centers = model.cluster_centers_
        lists = _lists(self._assignments, self.nlist)
        self._centroids, self._inverted_lists = centers, lists

    def add(self, vectors: ArrayLike, ids: Ids) -> None:
        self._require_trained()
        assert self._centroids is not None
        batch = vector_batch(vectors, self.dimension)
        identifiers = integer_ids(ids)
        if len(batch) != len(identifiers):
            raise ValueError("each vector must have exactly one ID")
        if len(np.unique(identifiers)) != len(identifiers) or np.isin(identifiers, self._ids).any():
            raise ValueError("duplicate IDs are not allowed")
        assignments = np.concatenate(
            (self._assignments, assign(batch, self._centroids, self.metric))
        )
        all_vectors = np.concatenate((self._vectors, batch))
        all_ids = np.concatenate((self._ids, identifiers))
        lists = _lists(assignments, self.nlist)
        self._vectors, self._ids, self._assignments, self._inverted_lists = (
            all_vectors,
            all_ids,
            assignments,
            lists,
        )

    def search(self, query: ArrayLike, k: int, *, nprobe: int | None = None) -> list[SearchResult]:
        k = positive_integer(k, "k")
        values = real_array(query, "query")
        if values.shape != (self.dimension,):
            raise ValueError(f"query must have shape ({self.dimension},)")
        probes = self.nprobe if nprobe is None else self._validate_nprobe(nprobe)
        self._require_trained()
        assert self._centroids is not None
        if self.size == 0:
            return []
        if probes == self.nlist:
            # Keep original row order, including floating-point reduction behavior.
            return rank(values, self._vectors, self._ids, self.metric, k)
        costs = squared_distances(
            representation(values.reshape(1, -1), self.metric), self._centroids
        )[0]
        clusters = smallest(costs, np.arange(self.nlist, dtype=np.int64), probes)
        rows = np.concatenate([self._inverted_lists[i] for i in clusters])
        return rank(values, self._vectors[rows], self._ids[rows], self.metric, k)

    def delete(self, ids: Ids) -> None:
        requested = integer_ids(ids)
        keep = ~np.isin(self._ids, requested)
        vectors, remaining_ids = self._vectors[keep], self._ids[keep]
        assignments = self._assignments[keep]
        lists = _lists(assignments, self.nlist) if self.is_trained else ()
        self._vectors, self._ids, self._assignments, self._inverted_lists = (
            vectors,
            remaining_ids,
            assignments,
            lists,
        )

    def save(self, path: str | PathLike[str]) -> None:
        self._require_trained()
        assert self._centroids is not None
        ivf_storage.save_index(
            path,
            {
                "index_type": np.asarray("ivf"),
                "ivf_format_version": np.asarray(ivf_storage.FORMAT_VERSION),
                "dimension": np.asarray(self.dimension),
                "metric": np.asarray(self.metric),
                "vectors": self._vectors,
                "ids": self._ids,
                "nlist": np.asarray(self.nlist),
                "nprobe": np.asarray(self.nprobe),
                "centroids": self._centroids,
                "assignments": self._assignments,
                "max_iter": np.asarray(self._max_iter),
                "tol": np.asarray(self._tol),
                "random_state": np.asarray(self._random_state),
                "is_trained": np.asarray(True),
            },
        )

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        state = ivf_storage.load_index(path)
        index = cls(
            dimension=int(state["dimension"]),
            nlist=int(state["nlist"]),
            nprobe=int(state["nprobe"]),
            metric=validate_metric(str(state["metric"].item())),
            random_state=int(state["random_state"]),
            max_iter=int(state["max_iter"]),
            tol=float(state["tol"]),
        )
        lists = _lists(state["assignments"], index.nlist)
        index._centroids = state["centroids"]
        index._vectors, index._ids = state["vectors"], state["ids"]
        index._assignments, index._inverted_lists = state["assignments"], lists
        return index
