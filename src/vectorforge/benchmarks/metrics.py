"""Measured latency distributions and wall-clock workload throughput."""

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from vectorforge.engine._validation import positive_integer, real_array


def latency_statistics(samples_ns: ArrayLike) -> dict[str, float]:
    values = real_array(samples_ns, "latency samples")
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("latency samples must be a nonempty nonnegative vector")
    values = values / 1_000_000
    return {
        "latency_mean_ms": float(np.mean(values)),
        "latency_p50_ms": float(np.percentile(values, 50, method="linear")),
        "latency_p95_ms": float(np.percentile(values, 95, method="linear")),
        "latency_p99_ms": float(np.percentile(values, 99, method="linear")),
    }


def queries_per_second(completed: int, elapsed_ns: int) -> float:
    completed = positive_integer(completed, "completed")
    if (
        isinstance(elapsed_ns, (bool, np.bool_))
        or not isinstance(elapsed_ns, Real)
        or not np.isfinite(elapsed_ns)
        or elapsed_ns <= 0
    ):
        raise ValueError("elapsed_ns must be finite and positive")
    value = completed / (float(elapsed_ns) / 1_000_000_000)
    if not np.isfinite(value):
        raise ValueError("QPS exceeds the finite range")
    return value
