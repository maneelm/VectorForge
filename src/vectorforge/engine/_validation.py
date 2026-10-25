"""Shared validation; all stored vectors use finite float64 values and int64 IDs."""

from numbers import Integral
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Metric = Literal["cosine", "euclidean", "dot"]
type Ids = int | list[int] | tuple[int, ...] | NDArray[np.integer]


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_metric(metric: str) -> Metric:
    if not isinstance(metric, str) or metric not in ("cosine", "euclidean", "dot"):
        raise ValueError("metric must be 'cosine', 'euclidean', or 'dot'")
    return cast(Metric, metric)


def real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numbers")
        with np.errstate(over="ignore", invalid="ignore"):
            result = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a rectangular array of real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def vector_batch(vectors: ArrayLike, dimension: int) -> NDArray[np.float64]:
    result = real_array(vectors, "vectors")
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if result.ndim != 2 or result.shape[1] != dimension:
        raise ValueError(f"vectors must have shape ({dimension},) or (N, {dimension})")
    return result


def integer_ids(ids: Ids) -> NDArray[np.int64]:
    if isinstance(ids, (list, tuple)) and any(isinstance(i, (bool, np.bool_)) for i in ids):
        raise ValueError("booleans are not IDs")
    try:
        raw = np.asarray(ids)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("IDs must be signed 64-bit integers") from exc
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim != 1:
        raise ValueError("IDs must be an integer or a one-dimensional sequence")
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in "iu" or (raw.dtype.kind == "u" and np.any(raw > 2**63 - 1)):
        raise ValueError("IDs must be signed 64-bit integers (booleans are not IDs)")
    return raw.astype(np.int64, copy=True)
