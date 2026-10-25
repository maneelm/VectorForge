"""Seeded K-Means++ and Lloyd iteration, implemented with NumPy."""

from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorforge.engine._validation import positive_integer, real_array


def configuration(max_iter: int, tol: float, random_state: int) -> tuple[int, float, int]:
    """Validate clustering configuration without touching a global RNG."""
    iterations = positive_integer(max_iter, "max_iter")
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real):
        raise ValueError("tol must be a finite nonnegative number")
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tol must be a finite nonnegative number")
    if (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
        or not 0 <= random_state <= 2**63 - 1
    ):
        raise ValueError("random_state must be an integer in [0, 2**63 - 1]")
    return iterations, tolerance, int(random_state)


def squared_distances(
    vectors: NDArray[np.float64], centers: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute (N, K) distances with O(NK + ND) memory, avoiding cancellation.

    Loop over centroids, never individual samples. Direct differences avoid the
    catastrophic cancellation of ||x||² + ||c||² - 2 x.c for large offsets.
    Unrepresentable squared distances raise instead of entering the algorithm.
    """
    result = np.empty((len(vectors), len(centers)), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for i, center in enumerate(centers):
            delta = vectors - center
            result[:, i] = np.einsum("ij,ij->i", delta, delta)
    if not np.isfinite(result).all():
        raise ValueError("squared Euclidean distances exceed the finite float64 range")
    return result


def _training_array(vectors: ArrayLike) -> NDArray[np.float64]:
    values = real_array(vectors, "vectors")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("training vectors must have nonempty shape (N, dimension)")
    return values


def _initialize(
    vectors: NDArray[np.float64], n_clusters: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    chosen = np.zeros(len(vectors), dtype=bool)
    first = int(rng.integers(len(vectors)))
    centers = [vectors[first].copy()]
    chosen[first] = True
    nearest = squared_distances(vectors, np.asarray(centers))[:, 0]
    for _ in range(1, n_clusters):
        maximum = float(nearest.max())
        if maximum == 0:
            # All remaining locations coincide; choose the first unused sample.
            selected = int(np.flatnonzero(~chosen)[0])
        else:
            weights = nearest / maximum
            selected = int(rng.choice(len(vectors), p=weights / weights.sum()))
        centers.append(vectors[selected].copy())
        chosen[selected] = True
        nearest = np.minimum(
            nearest, squared_distances(vectors, vectors[selected : selected + 1])[:, 0]
        )
    return np.asarray(centers)


def _partition(distances: NDArray[np.float64]) -> NDArray[np.int64]:
    labels = np.argmin(distances, axis=1).astype(np.int64)
    counts = np.bincount(labels, minlength=distances.shape[1])
    empty_clusters = np.flatnonzero(counts == 0)
    if len(empty_clusters) == 0:
        return labels
    losses = distances[np.arange(len(labels)), labels]
    # Highest loss first, then lowest sample row; each donor must retain a member.
    order = np.lexsort((np.arange(len(labels)), -losses))
    for empty in empty_clusters:
        donor_row = int(order[counts[labels[order]] > 1][0])
        counts[labels[donor_row]] -= 1
        labels[donor_row] = empty
        counts[empty] += 1
    return labels


def _mean(members: NDArray[np.float64]) -> NDArray[np.float64]:
    # Center first: identical samples keep exactly identical means, and summation
    # does not overflow merely because all coordinates share a large offset.
    anchor = members[0]
    return anchor + np.sum((members - anchor) / len(members), axis=0)


class KMeans:
    """K-Means with nonempty training partitions, even for identical samples.

    Training labels describe the partition used to compute the final means;
    inertia is the sum of squared distances to those assigned means. At the
    iteration/tolerance limit these labels need not equal a fresh prediction.
    Prediction always chooses the lowest centroid ID on an exact distance tie.
    """

    def __init__(
        self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42
    ) -> None:
        self.n_clusters = positive_integer(n_clusters, "n_clusters")
        self.max_iter, self.tol, self.random_state = configuration(max_iter, tol, random_state)
        self._centers: NDArray[np.float64] | None = None
        self._labels: NDArray[np.int64] | None = None
        self._n_iter = 0
        self._inertia: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._centers is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("KMeans must be fitted before accessing results or predicting")

    @property
    def cluster_centers_(self) -> NDArray[np.float64]:
        self._require_fitted()
        assert self._centers is not None
        return self._centers.copy()

    @property
    def labels_(self) -> NDArray[np.int64]:
        self._require_fitted()
        assert self._labels is not None
        return self._labels.copy()

    @property
    def n_iter_(self) -> int:
        self._require_fitted()
        return self._n_iter

    @property
    def inertia_(self) -> float:
        self._require_fitted()
        assert self._inertia is not None
        return self._inertia

    def fit(self, vectors: ArrayLike) -> Self:
        values = _training_array(vectors)
        clusters = positive_integer(self.n_clusters, "n_clusters")
        max_iter, tol, seed = configuration(self.max_iter, self.tol, self.random_state)
        if clusters > len(values):
            raise ValueError("n_clusters cannot exceed the number of training samples")
        centers = _initialize(values, clusters, np.random.default_rng(seed))
        previous: NDArray[np.int64] | None = None
        for _iteration in range(1, max_iter + 1):
            labels = _partition(squared_distances(values, centers))
            with np.errstate(over="ignore", invalid="ignore"):
                updated = np.stack([_mean(values[labels == i]) for i in range(clusters)])
            if not np.isfinite(updated).all():
                raise ValueError("centroid means exceed the finite float64 range")
            with np.errstate(over="ignore", invalid="ignore"):
                movement = float(np.max(np.hypot.reduce(centers - updated, axis=1)))
            if not np.isfinite(movement):
                raise ValueError("centroid movement exceeds the finite float64 range")
            centers = updated
            if (previous is not None and np.array_equal(labels, previous)) or movement <= tol:
                break
            previous = labels
        distances = squared_distances(values, centers)
        with np.errstate(over="ignore"):
            inertia = float(np.sum(distances[np.arange(len(values)), labels]))
        if not np.isfinite(inertia):
            raise ValueError("inertia exceeds the finite float64 range")
        self._centers, self._labels, self._n_iter, self._inertia = (
            centers,
            labels,
            _iteration,
            inertia,
        )
        return self

    def predict(self, vectors: ArrayLike) -> NDArray[np.int64]:
        self._require_fitted()
        assert self._centers is not None
        values = real_array(vectors, "vectors")
        if values.ndim != 2 or values.shape[1] != self._centers.shape[1]:
            raise ValueError("prediction vectors must have shape (N, fitted dimension)")
        return np.argmin(squared_distances(values, self._centers), axis=1).astype(np.int64)
