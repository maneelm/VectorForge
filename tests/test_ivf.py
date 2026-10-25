import numpy as np
import pytest

from vectorforge import BaseVectorIndex, ExactVectorIndex, IVFVectorIndex
from vectorforge.engine import ivf as ivf_module


def trained(metric="euclidean"):
    index = IVFVectorIndex(2, nlist=2, nprobe=1, metric=metric)
    index.train([[-10, 0], [-9, 0], [9, 0], [10, 0]])
    return index


def test_states_and_untrained_operations(tmp_path):
    index = IVFVectorIndex(2, nlist=2, nprobe=1)
    assert isinstance(index, BaseVectorIndex)
    assert not index.is_trained
    assert index.size == 0
    for operation in (
        lambda: index.add([1, 0], 1),
        lambda: index.search([1, 0], 1),
        lambda: index.save(tmp_path / "untrained.npz"),
        lambda: index.centroids,
    ):
        with pytest.raises(ValueError, match="trained"):
            operation()
    index.delete([1, 1])
    index.train([[1, 0], [0, 1]])
    assert index.is_trained and index.size == 0
    assert index.search([1, 0], 1) == []
    index.add([1, 0], -1)
    assert index.size == 1
    index.delete(-1)
    assert index.size == 0 and index.is_trained


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimension": 0},
        {"dimension": True},
        {"metric": "bad"},
        {"nlist": 0},
        {"nlist": 2.0},
        {"nprobe": 0},
        {"nprobe": 3},
        {"nprobe": True},
        {"nprobe": 1.0},
        {"max_iter": 0},
        {"tol": -1},
        {"random_state": -1},
    ],
)
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        IVFVectorIndex(**({"dimension": 2, "nlist": 2, "nprobe": 1} | kwargs))


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2],
        [[1]],
        [[1, 2]],
        np.empty((0, 2)),
        [[np.nan, 0], [1, 0]],
        [[np.inf, 0], [1, 0]],
        [[[1, 0], [0, 1]]],
        [[1e308, 0], [-1e308, 0]],
    ],
)
def test_failed_training_is_atomic(data):
    index = trained()
    centers = index.centroids
    with pytest.raises(ValueError):
        index.train(data)
    np.testing.assert_array_equal(index.centroids, centers)
    assert index.is_trained and index.size == 0


def test_retraining_and_centroid_copy():
    index = trained()
    centers = index.centroids
    centers[:] = 999
    assert not np.any(index.centroids == 999)
    index.train([[-100, 0], [100, 0]])
    assert set(index.centroids[:, 0]) == {-100, 100}
    index.add([1, 0], 3)
    before = index.centroids
    with pytest.raises(ValueError, match="populated"):
        index.train([[0, 1], [0, -1]])
    np.testing.assert_array_equal(index.centroids, before)
    index.delete(3)
    index.train([[0, 1], [0, -1]])
    assert set(index.centroids[:, 1]) == {-1, 1}


def test_single_batch_empty_and_copy():
    index = trained()
    vector = np.array([-9.0, 0])
    ids = np.array([-7])
    index.add(vector, ids)
    vector[:] = 100
    ids[:] = 100
    index.add([[9, 0], [10, 0]], [42, 1000])
    index.add(np.empty((0, 2)), [])
    assert index.size == 3
    result = index.search([-9, 0], 10)
    assert [(r.id, r.score) for r in result] == [(-7, 0)]
    assert sorted(np.concatenate(index._inverted_lists).tolist()) == [0, 1, 2]
    assert index._assignments.tolist() == [
        int(np.argmin(np.sum((v - index.centroids) ** 2, axis=1))) for v in index._vectors
    ]


@pytest.mark.parametrize(
    ("vectors", "ids"),
    [
        ([[1, 0], [2, 0]], [2, 2]),
        ([[1, 0], [2, 0]], [2, 1]),
        ([[1, 0], [2, 0]], [2]),
        ([[1, 0], [np.nan, 0]], [2, 3]),
        ([np.inf, 0], 2),
        ([1, 0, 1], 2),
        ([[1], [2, 0]], [2, 3]),
        ([1, 0], 2.0),
        ([1, 0], True),
        ([1, 0], 2**63),
        ([1, 0], -(2**63) - 1),
        ([1j, 0], 2),
        ([[1e308, 0], [-1e308, 0]], [2, 3]),
    ],
)
def test_failed_insert_is_atomic(vectors, ids):
    index = trained()
    index.add([1, 0], 1)
    before = index.search([1, 0], 10, nprobe=2)
    assignments = index._assignments.copy()
    with pytest.raises(ValueError):
        index.add(vectors, ids)
    assert index.size == 1
    assert index.search([1, 0], 10, nprobe=2) == before
    np.testing.assert_array_equal(index._assignments, assignments)


def test_allocation_failure_is_atomic(monkeypatch):
    index = trained()
    index.add([1, 0], 1)
    before = index.search([1, 0], 10, nprobe=2)

    def fail(*args):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(ivf_module, "_lists", fail)
    for operation in (lambda: index.add([2, 0], 2), lambda: index.delete(1)):
        with pytest.raises(MemoryError):
            operation()
        assert index.search([1, 0], 10, nprobe=2) == before


@pytest.mark.parametrize("nprobe", [0, -1, 3, True, 1.0, "1"])
def test_invalid_nprobe_override_and_property(nprobe):
    index = trained()
    with pytest.raises(ValueError):
        index.search([1, 0], 1, nprobe=nprobe)
    with pytest.raises(ValueError):
        index.nprobe = nprobe
    assert index.nprobe == 1


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize(
    ("query", "k"),
    [
        ([1, 0], 0),
        ([1, 0], True),
        ([1, 0], 1.0),
        ([1], 1),
        ([[1, 0]], 1),
        ([np.nan, 0], 1),
        ([np.inf, 0], 1),
        ([1j, 0], 1),
    ],
)
def test_invalid_search(query, k, populated):
    index = trained()
    if populated:
        index.add([1, 0], 1)
    with pytest.raises(ValueError):
        index.search(query, k)


def test_probe_subset_empty_lists_and_centroid_ties():
    index = trained()
    first = index.centroids[0]
    second = index.centroids[1]
    index.add(first, 100)
    assert index.search(second, 10) == []
    index.add(second, -100)
    # Query midpoint is equidistant; lower centroid ID wins, not lower vector ID.
    query = (first + second) / 2
    assert [r.id for r in index.search(query, 10)] == [100]
    assert [r.id for r in index.search(query, 10, nprobe=2)] == [-100, 100]
    assert index.nprobe == 1
    index.nprobe = 2
    assert len(index.search(query, 10)) == 2


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot"])
@pytest.mark.parametrize("seed", range(4))
def test_full_probe_matches_exact_randomized_lifecycle(metric, seed):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(70, 6))
    vectors[0] = 0
    vectors[1:5] = vectors[5]
    ids = rng.permutation(np.arange(-35, 35))
    exact = ExactVectorIndex(6, metric)
    index = IVFVectorIndex(6, nlist=7, nprobe=2, metric=metric, random_state=seed)
    index.train(vectors)
    for engine in (exact, index):
        engine.add(vectors[:30], ids[:30])
        engine.add(vectors[30:], ids[30:])
        engine.delete([*ids[::6], 999])
        engine.add(vectors[::6], ids[::6])
    for query in [np.zeros(6), vectors[5], *rng.normal(size=(5, 6))]:
        for k in (1, 3, 20, 100):
            actual = index.search(query, k, nprobe=index.nlist)
            expected = exact.search(query, k)
            assert [r.id for r in actual] == [r.id for r in expected]
            np.testing.assert_allclose(
                [r.score for r in actual], [r.score for r in expected], atol=1e-14
            )
            partial = index.search(query, k)
            assert len({r.id for r in partial}) == len(partial) <= k
            scored = {r.id: r.score for r in exact.search(query, 100)}
            assert [r.score for r in partial] == pytest.approx([scored[r.id] for r in partial])
            costs = [(r.score if metric == "euclidean" else -r.score, r.id) for r in partial]
            assert costs == sorted(costs)


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot"])
def test_boundary_ties_across_lists_and_zero_vectors(metric):
    index = IVFVectorIndex(2, nlist=3, nprobe=3, metric=metric)
    vectors = [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]
    ids = [10, -5, -10, 5, -20]
    index.train(vectors)
    index.add(vectors, ids)
    exact = ExactVectorIndex(2, metric)
    exact.add(vectors, ids)
    for k in (1, 2, 4, 8):
        assert index.search([0, 0], k) == exact.search([0, 0], k)


def test_cosine_routes_by_direction_and_preserves_original_vectors():
    index = IVFVectorIndex(2, nlist=2, nprobe=1, metric="cosine")
    index.train([[1000, 0], [1, 0], [0, 1], [0, 1000]])
    index.add([[1e300, 0], [1e-300, 0], [0, 1]], [3, -1, 8])
    assert [r.id for r in index.search([10, 0], 10)] == [-1, 3]
    assert [r.score for r in index.search([10, 0], 10)] == [1, 1]
    assert np.max(index._vectors) == 1e300


def test_delete_no_stale_rows_and_reuse_ids():
    index = trained()
    index.add([[-10, 0], [10, 0], [11, 0]], [-1, 9, 100])
    centers = index.centroids
    index.delete([9, 9, 999])
    assert {r.id for r in index.search([10, 0], 100, nprobe=2)} == {-1, 100}
    with pytest.raises(ValueError):
        index.delete([-1, 2.5])
    assert index.size == 2
    index.add([9, 0], 9)
    assert index.search([9, 0], 1)[0].id == 9
    index.delete([-1, 9, 100])
    assert index.is_trained and index.size == 0
    np.testing.assert_array_equal(index.centroids, centers)
    assert all(len(rows) == 0 for rows in index._inverted_lists)


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot"])
def test_identical_centroids_empty_lists_and_partial_boundary_ties(metric):
    index = IVFVectorIndex(2, nlist=3, nprobe=1, metric=metric)
    index.train(np.zeros((7, 2)))
    index.add(np.zeros((7, 2)), [8, 6, 4, 2, 0, -2, -4])
    assert [len(rows) for rows in index._inverted_lists] == [7, 0, 0]
    for probes in (1, 2, 3):
        assert [r.id for r in index.search([1, 0], 3, nprobe=probes)] == [-4, -2, 0]


def test_failed_first_training_and_retraining_allocation_are_atomic(monkeypatch):
    untrained = IVFVectorIndex(2, nlist=2, nprobe=1)
    with pytest.raises(ValueError):
        untrained.train([[np.nan, 1], [0, 0]])
    assert not untrained.is_trained
    index = trained()
    centers = index.centroids

    def fail(*args):
        raise MemoryError("simulated list allocation failure")

    monkeypatch.setattr(ivf_module, "_lists", fail)
    with pytest.raises(MemoryError):
        index.train([[-100, 0], [100, 0]])
    assert index.is_trained and index.size == 0
    np.testing.assert_array_equal(index.centroids, centers)
