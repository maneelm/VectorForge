import numpy as np
import pytest

from vectorforge.benchmarks.metrics import latency_statistics, queries_per_second


def test_latency_statistics_units_and_linear_percentiles():
    result = latency_statistics([1e6, 2e6, 3e6, 4e6, 5e6])
    assert result == pytest.approx(
        {"latency_mean_ms": 3, "latency_p50_ms": 3, "latency_p95_ms": 4.8, "latency_p99_ms": 4.96}
    )
    assert latency_statistics([0])["latency_mean_ms"] == 0


@pytest.mark.parametrize("values", [[], [[1, 2]], [np.nan], [np.inf], [-1], ["1"]])
def test_invalid_latency_samples(values):
    with pytest.raises(ValueError):
        latency_statistics(values)


def test_qps_from_actual_elapsed_workload():
    assert queries_per_second(100, 2_000_000_000) == 50


@pytest.mark.parametrize(
    ("count", "elapsed"),
    [(0, 1), (True, 1), (1, 0), (1, -1), (1, np.nan), (1, np.inf), (1, True), (1, "1")],
)
def test_invalid_qps(count, elapsed):
    with pytest.raises(ValueError):
        queries_per_second(count, elapsed)
