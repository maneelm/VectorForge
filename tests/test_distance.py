import math

import numpy as np
import pytest

from vectorforge.engine.distance import cosine_similarity, dot_product, euclidean_distance

METRICS = [cosine_similarity, euclidean_distance, dot_product]


@pytest.mark.parametrize(
    ("b", "cosine", "euclidean", "dot"),
    [
        ([1, 0], 1, 0, 1),
        ([0, 1], 0, math.sqrt(2), 0),
        ([-1, 0], -1, 2, -1),
        ([0, 0], 0, 1, 0),
        ([3, 4], 0.6, math.sqrt(20), 3),
    ],
)
def test_known_values(b, cosine, euclidean, dot):
    assert cosine_similarity([1, 0], b) == pytest.approx(cosine)
    assert euclidean_distance([1, 0], b) == pytest.approx(euclidean)
    assert dot_product([1, 0], b) == pytest.approx(dot)


@pytest.mark.parametrize("metric", METRICS)
def test_zero_vectors(metric):
    assert metric([0, 0], [0, 0]) == 0
    assert np.array_equal(
        metric([0, 0], [[0, 0], [1, 0]]), [0, 0] if metric != euclidean_distance else [0, 1]
    )


@pytest.mark.parametrize("metric", METRICS)
def test_one_to_many(metric):
    vectors = [[3, 4], [1, -2], [0, 0]]
    result = metric([1, 2], vectors)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [metric([1, 2], v) for v in vectors])
    assert metric([1, 2], np.empty((0, 2))).shape == (0,)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([], []),
        ([[1, 2]], [1, 2]),
        ([1, 2], [1]),
        ([1, 2], [[[1, 2]]]),
        ([1, 2], 1),
        ([1, 2], [[1], [2]]),
        ([1, 2], [[1, 2], [3]]),
        (["1", "2"], [1, 2]),
        ([1j, 2], [1, 2]),
        ([True, False], [1, 2]),
    ],
)
def test_invalid_shape_or_type(metric, a, b):
    with pytest.raises(ValueError):
        metric(a, b)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite(metric, bad):
    with pytest.raises(ValueError):
        metric([bad, 1], [1, 2])
    with pytest.raises(ValueError):
        metric([1, 2], [[1, bad]])


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_cosine_scaling_stability(scale):
    assert cosine_similarity([scale, scale], [scale, scale]) == pytest.approx(1)
    assert cosine_similarity([0, 0], [scale, scale]) == 0


def test_large_euclidean_distance():
    assert euclidean_distance([0, 0], [3e200, 4e200]) == pytest.approx(5e200)


@pytest.mark.parametrize("metric", [dot_product, euclidean_distance])
def test_unrepresentable_score(metric):
    with pytest.raises(ValueError, match="float64"):
        metric([1e308, 1e308], [-1e308, -1e308])
