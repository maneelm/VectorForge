import numpy as np
import pytest

from vectorforge import ExactVectorIndex, IVFVectorIndex, SearchResult
from vectorforge.evaluation import evaluate_recall, recall_at_k


def results(ids):
    return [SearchResult(i, 0) for i in ids]


@pytest.mark.parametrize(
    ("approximate", "exact", "k", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 3, 1),
        ([3, 2, 1], [1, 2, 3], 3, 1),
        ([1, 9, 8], [1, 2, 3], 3, 1 / 3),
        ([9, 8], [1, 2], 2, 0),
        ([1, 1, 2], [1, 2], 3, 1),
        ([1], [1, 2], 10, 0.5),
        ([1, 2, 3], [1, 2, 3], 1, 1),
        ([9, 1], [1, 2], 1, 0),
        ([], [], 3, 1),
        ([1], [], 3, 1),
        ([], [1], 3, 0),
    ],
)
def test_recall_known_results(approximate, exact, k, expected):
    assert recall_at_k(results(approximate), results(exact), k) == pytest.approx(expected)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "1", None])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k([], [], k)
    with pytest.raises(ValueError):
        evaluate_recall(IVFVectorIndex(1, 1, 1), ExactVectorIndex(1), [[0]], k)


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot"])
def test_multiquery_recall_and_full_probing(metric):
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(90, 5))
    queries = rng.normal(size=(12, 5))
    ids = np.arange(-45, 45)
    exact = ExactVectorIndex(5, metric)
    approximate = IVFVectorIndex(5, nlist=8, nprobe=1, metric=metric)
    approximate.train(vectors)
    exact.add(vectors, ids)
    approximate.add(vectors, ids)
    report = evaluate_recall(approximate, exact, queries, 7)
    expected = [
        len({r.id for r in approximate.search(q, 7)} & {r.id for r in exact.search(q, 7)}) / 7
        for q in queries
    ]
    assert report.per_query == tuple(expected)
    assert report.mean == pytest.approx(sum(expected) / len(expected))
    assert report == evaluate_recall(approximate, exact, queries, 7)
    full = evaluate_recall(approximate, exact, queries, 7, nprobe=8)
    assert full.per_query == (1.0,) * len(queries)
    assert full.mean == 1


def test_empty_ground_truth_and_exact_tie_policy():
    approximate = IVFVectorIndex(1, nlist=1, nprobe=1)
    approximate.train([[1]])
    exact = ExactVectorIndex(1)
    assert evaluate_recall(approximate, exact, [[0], [1]], 10).per_query == (1, 1)
    exact.add([[1], [1], [1]], [-3, -2, -1])
    approximate.add([[1], [1]], [-2, -1])
    # The equally good ID -1 does not replace -3 in the canonical exact top two.
    assert evaluate_recall(approximate, exact, [[1]], 2).mean == 0.5


@pytest.mark.parametrize(
    "queries", [[], np.empty((0, 2)), [1, 0], [[1]], [[[1, 0]]], [[np.nan, 0]], [[np.inf, 0]]]
)
def test_invalid_queries(queries):
    approximate = IVFVectorIndex(2, 1, 1)
    approximate.train([[1, 0]])
    with pytest.raises(ValueError):
        evaluate_recall(approximate, ExactVectorIndex(2), queries, 1)


@pytest.mark.parametrize("exact", [ExactVectorIndex(1), ExactVectorIndex(2, "dot")])
def test_incompatible_indexes(exact):
    with pytest.raises(ValueError, match="matching"):
        evaluate_recall(IVFVectorIndex(2, 1, 1), exact, [[1, 0]], 1)
