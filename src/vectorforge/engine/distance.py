"""Vectorized one-to-one and one-to-many metrics.

The first argument is a nonempty 1-D vector; the second is a matching vector
or an (N, dimension) matrix. Return a float or an (N,) float64 array respectively.
Cosine involving a zero vector is defined as 0, including zero versus zero.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorforge.engine._validation import real_array

type Score = float | NDArray[np.float64]


def _pair(a: ArrayLike, b: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left, right = real_array(a, "a"), real_array(b, "b")
    if left.ndim != 1 or left.size == 0:
        raise ValueError("a must be a nonempty one-dimensional vector")
    if right.ndim not in (1, 2) or right.shape[-1] != left.size:
        raise ValueError("b must be a vector or matrix with matching dimension")
    return left, right


def _result(value: np.float64 | NDArray[np.float64]) -> Score:
    if not np.isfinite(value).all():
        raise ValueError("metric computation produced a nonfinite float64 score")
    return float(value) if np.ndim(value) == 0 else value


def _unit(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    # Scale before taking norms to avoid overflow/underflow for finite inputs.
    scale = np.max(np.abs(vectors), axis=-1, keepdims=True)
    scaled = np.divide(vectors, scale, out=np.zeros_like(vectors), where=scale != 0)
    norms = np.sqrt(np.sum(scaled * scaled, axis=-1, keepdims=True))
    return np.divide(scaled, norms, out=np.zeros_like(scaled), where=norms != 0)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> Score:
    """Cosine in [-1, 1]; zero-vector comparisons score 0. Higher is better."""
    left, right = _pair(a, b)
    return _result(np.clip(np.sum(_unit(left) * _unit(right), axis=-1), -1.0, 1.0))


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> Score:
    """Unsquared L2 distance. Lower is better."""
    left, right = _pair(a, b)
    with np.errstate(over="ignore", invalid="ignore"):
        # hypot avoids overflow from squaring large, representable distances.
        result = np.hypot.reduce(left - right, axis=-1)
    return _result(result)


def dot_product(a: ArrayLike, b: ArrayLike) -> Score:
    """Unnormalized inner product. Higher is better; scores are unbounded."""
    left, right = _pair(a, b)
    with np.errstate(over="ignore", invalid="ignore"):
        result = right @ left
    return _result(result)
