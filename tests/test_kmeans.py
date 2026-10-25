import numpy as np
import pytest

from vectorforge.engine import kmeans as kmeans_module
from vectorforge.engine.kmeans import KMeans, _initialize, _partition, squared_distances


def assert_consistent(model, vectors):
    centers, labels = model.cluster_centers_, model.labels_
    assert centers.shape == (model.n_clusters, vectors.shape[1])
    assert labels.shape == (len(vectors),)
    assert labels.dtype == np.int64
    assert np.all(np.bincount(labels, minlength=model.n_clusters) > 0)
    for i in range(model.n_clusters):
        np.testing.assert_allclose(centers[i], vectors[labels == i].mean(axis=0))
    assert model.inertia_ == pytest.approx(float(np.sum((vectors - centers[labels]) ** 2)))


def test_separated_clusters_predict_and_converge():
    vectors = np.array([[-11.0, 0], [-10, 0], [-9, 0], [9, 0], [10, 0], [11, 0]])
    model = KMeans(2).fit(vectors)
    assert model.is_fitted
    assert model.n_iter_ < model.max_iter
    np.testing.assert_allclose(np.sort(model.cluster_centers_[:, 0]), [-10, 10])
    assert len(set(model.labels_[:3])) == 1
    assert len(set(model.labels_[3:])) == 1
    assert model.labels_[0] != model.labels_[3]
    np.testing.assert_array_equal(model.labels_, model.predict(vectors))
    assert model.predict(np.empty((0, 2))).shape == (0,)
    assert_consistent(model, vectors)


def test_reproducibility_and_global_rng_independence():
    vectors = np.random.default_rng(1).normal(size=(75, 4))
    np.random.seed(100)
    before = np.random.get_state()
    first = KMeans(5, random_state=17).fit(vectors)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    np.random.seed(900)
    second = KMeans(5, random_state=17).fit(vectors)
    np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    assert (first.n_iter_, first.inertia_) == (second.n_iter_, second.inertia_)


@pytest.mark.parametrize("clusters", [1, 3, 8])
def test_identical_samples_nonempty_and_prediction_ties(clusters):
    vectors = np.ones((8, 3))
    model = KMeans(clusters).fit(vectors)
    assert_consistent(model, vectors)
    assert model.inertia_ == pytest.approx(0)
    np.testing.assert_array_equal(model.predict(vectors), np.zeros(8))
    assert model.n_iter_ <= 2


def test_duplicate_locations_and_empty_cluster_recovery():
    vectors = np.array([[0.0], [0], [0], [10], [10], [10]])
    model = KMeans(5).fit(vectors)
    assert_consistent(model, vectors)
    assert model.inertia_ == 0
    repeated = KMeans(5).fit(vectors)
    np.testing.assert_array_equal(model.labels_, repeated.labels_)


def test_kmeans_plus_plus_weighted_initialization():
    vectors = np.array([[0.0], [1], [10], [50], [100]])
    seed = 12
    rng = np.random.default_rng(seed)
    first = rng.integers(len(vectors))
    losses = (vectors[:, 0] - vectors[first, 0]) ** 2
    expected_second = rng.choice(len(vectors), p=losses / losses.sum())
    centers = _initialize(vectors, 2, np.random.default_rng(seed))
    np.testing.assert_array_equal(centers, vectors[[first, expected_second]])


@pytest.mark.parametrize("iterations", [1, 2, 100])
@pytest.mark.parametrize("tol", [0, 1e-4, 1e6])
def test_iteration_limit_and_final_partition_consistency(iterations, tol):
    vectors = np.random.default_rng(2).normal(size=(50, 3))
    model = KMeans(4, max_iter=iterations, tol=tol).fit(vectors)
    assert 1 <= model.n_iter_ <= iterations
    if tol == 1e6:
        assert model.n_iter_ == 1
    assert_consistent(model, vectors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_clusters": 0},
        {"n_clusters": -1},
        {"n_clusters": True},
        {"n_clusters": 1.5},
        {"max_iter": 0},
        {"max_iter": True},
        {"max_iter": 1.0},
        {"tol": -1},
        {"tol": np.nan},
        {"tol": np.inf},
        {"tol": True},
        {"tol": "0"},
        {"random_state": -1},
        {"random_state": 1.2},
        {"random_state": True},
        {"random_state": 2**63},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        KMeans(**({"n_clusters": 2} | kwargs))


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [1, 2],
        np.empty((0, 2)),
        np.empty((2, 0)),
        [[[1, 2]]],
        [[np.nan, 1], [1, 2]],
        [[np.inf, 1], [1, 2]],
        [[1], [2, 3]],
        [[1j], [2j]],
        [["a"], ["b"]],
        [[1, 2]],
        [[1e308], [-1e308]],
    ],
)
def test_failed_fit_preserves_state(vectors):
    model = KMeans(2).fit([[0, 0], [1, 1], [2, 2]])
    centers, labels, iterations, inertia = (
        model.cluster_centers_,
        model.labels_,
        model.n_iter_,
        model.inertia_,
    )
    with pytest.raises(ValueError):
        model.fit(vectors)
    np.testing.assert_array_equal(model.cluster_centers_, centers)
    np.testing.assert_array_equal(model.labels_, labels)
    assert (model.n_iter_, model.inertia_) == (iterations, inertia)


def test_unfitted_properties_and_prediction():
    model = KMeans(1)
    assert not model.is_fitted
    for name in ("cluster_centers_", "labels_", "n_iter_", "inertia_"):
        with pytest.raises(ValueError, match="fitted"):
            getattr(model, name)
    with pytest.raises(ValueError, match="fitted"):
        model.predict([[1]])
    with pytest.raises(ValueError):
        model.fit([])
    assert not model.is_fitted


@pytest.mark.parametrize("vectors", [[1, 2], [[1]], [[np.nan, 1]], [[[1, 2]]]])
def test_prediction_validation(vectors):
    model = KMeans(1).fit([[1, 2]])
    with pytest.raises(ValueError):
        model.predict(vectors)


def test_defensive_outputs_and_caller_inputs():
    vectors = np.array([[0.0, 0], [1, 1], [10, 10]])
    original = vectors.copy()
    model = KMeans(2).fit(vectors)
    np.testing.assert_array_equal(vectors, original)
    centers, labels = model.cluster_centers_, model.labels_
    centers[:] = 1000
    labels[:] = 999
    vectors[:] = 100
    assert not np.any(model.cluster_centers_ == 1000)
    assert not np.any(model.labels_ == 999)


def test_large_offset_distances_avoid_cancellation():
    vectors = np.array([[1e12, 1e12], [1e12 + 1, 1e12 - 2]])
    np.testing.assert_array_equal(squared_distances(vectors, vectors), [[0, 5], [5, 0]])
    model = KMeans(2).fit(vectors)
    assert model.inertia_ == 0


def test_empty_recovery_uses_highest_losses_then_sample_row():
    distances = np.array([[0.0, 10, 10], [9, 20, 20], [9, 20, 20], [0, 10, 10]])
    np.testing.assert_array_equal(_partition(distances), [0, 1, 2, 0])


def test_failed_allocation_during_refit_preserves_fitted_state(monkeypatch):
    model = KMeans(2).fit([[0.0], [2], [10]])
    centers, labels = model.cluster_centers_, model.labels_

    def fail(*args):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(kmeans_module, "_mean", fail)
    with pytest.raises(MemoryError):
        model.fit([[0.0], [3], [20]])
    np.testing.assert_array_equal(model.cluster_centers_, centers)
    np.testing.assert_array_equal(model.labels_, labels)


def test_inertia_overflow_is_rejected_atomically():
    model = KMeans(1).fit([[1.0]])
    with pytest.raises(ValueError, match="inertia"):
        model.fit([[5e153], [-5e153]] * 4)
    np.testing.assert_array_equal(model.cluster_centers_, [[1.0]])


def test_identical_extreme_finite_samples_are_safe():
    model = KMeans(2).fit([[1e308], [1e308], [1e308]])
    np.testing.assert_array_equal(model.cluster_centers_, [[1e308], [1e308]])
    assert model.inertia_ == 0
