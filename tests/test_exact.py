import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vectorforge import BaseVectorIndex, ExactVectorIndex, SearchResult

METRICS = ["cosine", "euclidean", "dot"]


def reference(vectors, ids, query, metric, k):
    """Independent scalar reference: no production metrics or selection code."""
    results = []
    for vector, identifier in zip(vectors, ids, strict=True):
        dot = sum(float(a) * float(b) for a, b in zip(vector, query, strict=True))
        if metric == "euclidean":
            score = math.sqrt(
                sum((float(a) - float(b)) ** 2 for a, b in zip(vector, query, strict=True))
            )
        elif metric == "cosine":
            norm = math.sqrt(sum(float(a) ** 2 for a in vector)) * math.sqrt(
                sum(float(b) ** 2 for b in query)
            )
            score = dot / norm if norm else 0.0
        else:
            score = dot
        results.append((int(identifier), score))
    return sorted(results, key=lambda r: (r[1] if metric == "euclidean" else -r[1], r[0]))[:k]


def test_interface_and_empty():
    index = ExactVectorIndex(3)
    assert isinstance(index, BaseVectorIndex)
    assert (index.dimension, index.metric, index.size) == (3, "cosine", 0)
    assert index.search([1, 2, 3], 2) == []
    index.delete([99])
    index.add(np.empty((0, 3)), [])
    assert index.size == 0


@pytest.mark.parametrize("dimension", [0, -1, 2.0, True, "2", None])
def test_invalid_dimension(dimension):
    with pytest.raises(ValueError):
        ExactVectorIndex(dimension)


@pytest.mark.parametrize("metric", ["l2", "COSINE", "", None, []])
def test_invalid_metric(metric):
    with pytest.raises(ValueError):
        ExactVectorIndex(2, metric)


def test_single_and_batch_copy_and_result():
    index = ExactVectorIndex(2, "dot")
    vector = np.array([1.0, 2.0])
    ids = np.array([7])
    index.add(vector, ids)
    vector[:] = 999
    ids[:] = 999
    index.add([[2, 3], [0, 1]], [8, -1])
    assert index.size == 3
    assert index.search([1, 0], 1) == [SearchResult(8, 2.0)]
    result = index.search([1, 0], 99)
    assert [r.id for r in result] == [8, 7, -1]
    assert isinstance(result[0].id, int)
    assert isinstance(result[0].score, float)
    with pytest.raises(FrozenInstanceError):
        result[0].score = 5


@pytest.mark.parametrize(
    ("vectors", "ids"),
    [
        ([[1, 0], [2, 0]], [2, 2]),
        ([[1, 0], [2, 0]], [2, 1]),
        ([[1, 0], [2, 0]], [2]),
        ([[1, 0], [np.nan, 0]], [2, 3]),
        ([[1, 0], [np.inf, 0]], [2, 3]),
        ([[1, 0, 1]], [2]),
        ([[1, 0], [2]], [2, 3]),
        ([1, 0], [2.0]),
        ([1, 0], True),
        ([[1, 0], [2, 0]], [True, 2]),
        ([1, 0], [2**63]),
        ([1, 0], [-(2**63) - 1]),
        ([1, 0], [[2]]),
        ([1, 0], ["2"]),
        ([1j, 0], [2]),
    ],
)
def test_failed_add_is_atomic(vectors, ids):
    index = ExactVectorIndex(2)
    index.add([1, 0], 1)
    before = index.search([1, 0], 10)
    with pytest.raises(ValueError):
        index.add(vectors, ids)
    assert index.size == 1
    assert index.search([1, 0], 10) == before


@pytest.mark.parametrize("metric", METRICS)
def test_top_k(metric):
    vectors = [[1, 0], [0, 2], [-1, 0], [2, 0]]
    ids = [4, 2, 8, 1]
    index = ExactVectorIndex(2, metric)
    index.add(vectors, ids)
    for k in (1, 2, 4, 10):
        expected = reference(vectors, ids, [1, 0], metric, k)
        result = index.search([1, 0], k)
        assert [r.id for r in result] == [r[0] for r in expected]
        assert [r.score for r in result] == pytest.approx([r[1] for r in expected])


@pytest.mark.parametrize("metric", METRICS)
def test_boundary_ties_are_insertion_order_independent(metric):
    rng = np.random.default_rng(42)
    for _ in range(8):
        ids = rng.permutation(np.arange(-25, 25))
        index = ExactVectorIndex(2, metric)
        index.add(np.ones((50, 2)), ids)
        for k in (1, 7, 49, 50, 60):
            assert [r.id for r in index.search([1, 1], k)] == list(range(-25, 25))[:k]


def test_boundary_tie_with_strictly_better_candidates():
    index = ExactVectorIndex(1, "dot")
    index.add([[2], [1], [1], [1], [0]], [100, 3, 2, 1, -100])
    assert [r.id for r in index.search([1], 3)] == [100, 1, 2]


@pytest.mark.parametrize("k", [0, -1, 1.0, True, np.bool_(True), "1", None])
@pytest.mark.parametrize("populated", [False, True])
def test_invalid_k(k, populated):
    index = ExactVectorIndex(2)
    if populated:
        index.add([1, 0], 1)
    with pytest.raises(ValueError, match="k"):
        index.search([1, 0], k)


@pytest.mark.parametrize("query", [[1], [[1, 0]], [np.nan, 0], [np.inf, 0], [1j, 0], []])
@pytest.mark.parametrize("populated", [False, True])
def test_invalid_query(query, populated):
    index = ExactVectorIndex(2)
    if populated:
        index.add([1, 0], 1)
    with pytest.raises(ValueError):
        index.search(query, 1)


def test_delete_stable_ids_and_reinsert():
    index = ExactVectorIndex(2, "dot")
    index.add([[1, 0], [2, 0], [3, 0]], [10, -8, 100])
    index.delete([-8, -8, 999])
    assert index.search([1, 0], 10) == [SearchResult(100, 3), SearchResult(10, 1)]
    index.add([4, 0], -8)
    assert index.search([1, 0], 1)[0].id == -8
    before = index.search([1, 0], 10)
    with pytest.raises(ValueError):
        index.delete([10, 2.5])
    assert index.search([1, 0], 10) == before
    index.delete([10, -8, 100])
    assert index.size == 0
    assert index.search([1, 0], 1) == []


def test_extreme_ids_and_numpy_integer_parameters():
    index = ExactVectorIndex(np.int64(1), "dot")
    index.add([[1], [1]], [-(2**63), 2**63 - 1])
    assert index.search([1], np.int64(1))[0].id == -(2**63)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("seed", range(5))
def test_randomized_reference(metric, seed):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(83, 7))
    vectors[0] = 0
    ids = rng.choice(np.arange(-1000, 1000), size=len(vectors), replace=False)
    index = ExactVectorIndex(7, metric)
    index.add(vectors[:30], ids[:30])
    index.add(vectors[30:], ids[30:])
    index.delete(ids[::5])
    keep = np.arange(len(ids)) % 5 != 0
    for query in [np.zeros(7), *rng.normal(size=(4, 7))]:
        for k in (1, 9, 100):
            expected = reference(vectors[keep], ids[keep], query, metric, k)
            actual = index.search(query, k)
            assert [r.id for r in actual] == [r[0] for r in expected]
            np.testing.assert_allclose(
                [r.score for r in actual], [r[1] for r in expected], atol=1e-14
            )
