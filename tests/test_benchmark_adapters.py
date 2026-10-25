import numpy as np
import pytest

from vectorforge.benchmarks.adapters import make_adapter
from vectorforge.benchmarks.config import BenchmarkConfig
from vectorforge.benchmarks.faiss_adapter import FaissAdapter, require_faiss


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "dot"])
def test_vectorforge_adapter_full_probe(metric):
    config = BenchmarkConfig(dimension=2, metric=metric, nlist=2, nprobes=(1, 2))
    exact, ivf = make_adapter("vectorforge_exact", config), make_adapter("vectorforge_ivf", config)
    vectors = [[1, 0], [0, 1], [0, 0], [-1, 0]]
    ivf.train(vectors)
    exact.add(vectors, [-1, -9, 5, 2])
    ivf.add(vectors, [-1, -9, 5, 2])
    assert ivf.search([1, 0], 9, nprobe=2) == exact.search([1, 0], 9)
    with pytest.raises(ValueError):
        exact.search([1, 0], 1, nprobe=1)


def test_unknown_adapter_and_explicit_missing_dependency(monkeypatch):
    config = BenchmarkConfig()
    with pytest.raises(ValueError):
        make_adapter("unknown", config)

    def missing(*args):
        raise ImportError("test missing library")

    monkeypatch.setattr("vectorforge.benchmarks.faiss_adapter.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="FAISS unavailable"):
        require_faiss()


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "dot"])
@pytest.mark.parametrize("ivf", [False, True])
def test_faiss_integration_metric_scores_ids_and_missing_rows(metric, ivf):
    faiss = pytest.importorskip(
        "faiss", reason="optional FAISS integration; install benchmark extra"
    )
    faiss.omp_set_num_threads(1)
    config = BenchmarkConfig(dimension=2, metric=metric, nlist=2, nprobes=(1, 2), max_iter=4)
    adapter = FaissAdapter(config, ivf=ivf)
    vectors = np.array([[3.0, 4], [0, 0], [-3, -4], [1, 0]])
    ids = [-1, -(2**63), 2**63 - 1, 17]
    if ivf:
        adapter.train(np.tile(vectors, (20, 1)))
    adapter.add(vectors, ids)
    exact = make_adapter("vectorforge_exact", config)
    exact.add(vectors, ids)
    probe = 2 if ivf else None
    for query in ([1, 2], [0, 0]):
        found, truth = adapter.search(query, 100, nprobe=probe), exact.search(query, 100)
        assert {r.id for r in found} == set(ids)
        scores = {r.id: r.score for r in truth}
        assert [r.score for r in found] == pytest.approx([scores[r.id] for r in found], abs=1e-5)
    prepared = adapter._prepare(vectors[:, ::-1])
    assert prepared.dtype == np.float32 and prepared.flags.c_contiguous
    if metric == "cosine":
        np.testing.assert_allclose(np.linalg.norm(prepared, axis=1), [1, 0, 1, 1], atol=1e-6)
    with pytest.raises(ValueError):
        adapter.add([[1, 0], [2, 0]], [99, -1])
    with pytest.raises(ValueError):
        adapter.search([1], 1)
    with pytest.raises(ValueError):
        adapter.search([1, np.inf], 1)
    if ivf:
        with pytest.raises(ValueError):
            adapter.search([1, 0], 1, nprobe=3)
        sparse = adapter.search([1, 0], 100, nprobe=1)
        assert all(r.id in ids for r in sparse)
    elif metric == "dot":
        with pytest.raises(ValueError, match="float32"):
            adapter._prepare([[1e308, 0]])


def test_faiss_missing_sentinel_preserves_external_minus_one(monkeypatch):
    pytest.importorskip("faiss", reason="optional FAISS integration")
    adapter = FaissAdapter(BenchmarkConfig(dimension=1))
    adapter.add([[1]], [-1])

    class MissingNativeResult:
        is_trained = True

        def search(self, values, k):
            return np.array([[1.0, np.inf]]), np.array([[0, -1]])

    monkeypatch.setattr(adapter, "index", MissingNativeResult())
    found = adapter.search([1], 2)
    assert len(found) == 1 and found[0].id == -1
