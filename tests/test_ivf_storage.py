import io
import zipfile

import numpy as np
import pytest

from vectorforge import ExactVectorIndex, IVFVectorIndex
from vectorforge.engine import ivf_storage


def make_index(metric="cosine", empty=False):
    index = IVFVectorIndex(
        2, nlist=3, nprobe=1, metric=metric, random_state=17, max_iter=12, tol=1e-5
    )
    vectors = [[-2, 0], [2, 0], [0, 2], [0, 0], [2, 0]]
    index.train(vectors)
    if not empty:
        index.add(vectors, [-(2**63), 2**63 - 1, -5, 9, 10])
    return index


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot"])
@pytest.mark.parametrize("empty", [False, True])
def test_round_trip_all_properties_and_queries(tmp_path, metric, empty):
    index = make_index(metric, empty)
    path = tmp_path / "index.data"
    index.save(path)
    assert path.exists()
    restored = IVFVectorIndex.load(path)
    assert (
        restored.dimension,
        restored.metric,
        restored.size,
        restored.nlist,
        restored.nprobe,
        restored.is_trained,
    ) == (2, metric, index.size, 3, 1, True)
    assert (restored._max_iter, restored._tol, restored._random_state) == (12, 1e-5, 17)
    np.testing.assert_array_equal(restored.centroids, index.centroids)
    np.testing.assert_array_equal(restored._ids, index._ids)
    np.testing.assert_array_equal(restored._assignments, index._assignments)
    for actual, expected in zip(restored._inverted_lists, index._inverted_lists, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for query in [[0, 0], [1, 0], [0, 1], *np.random.default_rng(3).normal(size=(8, 2))]:
        for probe in (1, 2, 3):
            assert restored.search(query, 99, nprobe=probe) == index.search(query, 99, nprobe=probe)
    restored.add([1, 0], 123)
    assert index.size == (0 if empty else 5)
    restored.delete(123)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == ivf_storage._KEYS
        assert all(archive[key].dtype.kind != "O" for key in archive.files)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "source.npz"
    make_index().save(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ivf_format_version", np.array(99)),
        ("ivf_format_version", np.array(1.0)),
        ("index_type", np.array("exact")),
        ("index_type", np.array(["ivf"])),
        ("dimension", np.array(3)),
        ("dimension", np.array(0)),
        ("dimension", np.array(True)),
        ("metric", np.array("bad")),
        ("metric", np.array(b"cosine")),
        ("nlist", np.array(0)),
        ("nlist", np.array(2)),
        ("nlist", np.array([3])),
        ("nprobe", np.array(0)),
        ("nprobe", np.array(4)),
        ("nprobe", np.array(True)),
        ("max_iter", np.array(0)),
        ("tol", np.array(-1.0)),
        ("tol", np.array(np.nan)),
        ("tol", np.array(np.inf)),
        ("tol", np.array([0.1])),
        ("tol", np.array(True)),
        ("random_state", np.array(-1)),
        ("random_state", np.array(1.5)),
        ("is_trained", np.array(False)),
        ("is_trained", np.array(1)),
        ("centroids", np.zeros((2, 2))),
        ("centroids", np.full((3, 2), np.inf)),
        ("centroids", np.full((3, 2), 100.0)),
        ("centroids", np.zeros((3, 2), dtype=object)),
        ("vectors", np.zeros((5, 3))),
        ("vectors", np.zeros(5)),
        ("vectors", np.full((5, 2), np.nan)),
        ("ids", np.array([1, 1, 2, 3, 4])),
        ("ids", np.arange(4)),
        ("ids", np.arange(5, dtype=float)),
        ("ids", np.zeros((1, 5), dtype=int)),
        ("ids", np.array([2**63, 1, 2, 3, 4], dtype=np.uint64)),
        ("assignments", np.full(5, -1)),
        ("assignments", np.full(5, 3)),
        ("assignments", np.zeros(4, dtype=int)),
        ("assignments", np.zeros(5)),
        ("assignments", np.zeros((1, 5), dtype=int)),
    ],
)
def test_invalid_schema(tmp_path, data, key, value):
    data[key] = value
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="invalid IVF"):
        IVFVectorIndex.load(path)


def test_in_range_but_inconsistent_assignments(tmp_path, data):
    data["assignments"] = (data["assignments"] + 1) % 3
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="routing"):
        IVFVectorIndex.load(path)


@pytest.mark.parametrize("extra", [False, True])
def test_missing_and_extra_fields(tmp_path, data, extra):
    if extra:
        data["unexpected"] = np.array(1)
    else:
        del data["centroids"]
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="fields"):
        IVFVectorIndex.load(path)


@pytest.mark.parametrize("content", [b"", b"not a zip", b"PK\x03\x04corrupt"])
def test_corrupt_archives(tmp_path, content):
    path = tmp_path / "bad.npz"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="invalid IVF"):
        IVFVectorIndex.load(path)


def test_truncated_and_wrong_format(tmp_path, data):
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(ValueError):
        IVFVectorIndex.load(path)
    np.save(tmp_path / "array.npy", np.array([1]))
    with pytest.raises(ValueError):
        IVFVectorIndex.load(tmp_path / "array.npy")
    with pytest.raises(FileNotFoundError):
        IVFVectorIndex.load(tmp_path / "missing.npz")


def test_duplicate_archive_field(tmp_path, data):
    path = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in data.items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            archive.writestr(f"{key}.npy", buffer.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("metric.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="fields"):
        IVFVectorIndex.load(path)


def test_exact_archives_unchanged_and_not_interchangeable(tmp_path):
    exact_path, ivf_path = tmp_path / "exact.npz", tmp_path / "ivf.npz"
    exact = ExactVectorIndex(2)
    exact.add([1, 0], -8)
    exact.save(exact_path)
    assert ExactVectorIndex.load(exact_path).search([1, 0], 1) == exact.search([1, 0], 1)
    with np.load(exact_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"vectors", "ids", "dimension", "metric", "format_version"}
    index = make_index()
    index.save(ivf_path)
    for path, loader, writer in [
        (exact_path, IVFVectorIndex.load, index.save),
        (ivf_path, ExactVectorIndex.load, exact.save),
    ]:
        before = path.read_bytes()
        with pytest.raises(ValueError):
            loader(path)
        with pytest.raises(ValueError):
            writer(path)
        assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["dimension", "metric", "nlist"])
def test_incompatible_overwrite(tmp_path, change):
    path = tmp_path / "index.npz"
    make_index().save(path)
    before = path.read_bytes()
    kwargs = {"dimension": 2, "metric": "cosine", "nlist": 3, "nprobe": 1}
    kwargs[change] = {"dimension": 3, "metric": "dot", "nlist": 2}[change]
    index = IVFVectorIndex(**kwargs)
    index.train(np.eye(3, kwargs["dimension"]))
    with pytest.raises(ValueError, match="incompatible"):
        index.save(path)
    assert path.read_bytes() == before


def test_compatible_overwrite_and_corrupt_target(tmp_path):
    path = tmp_path / "index.npz"
    index = make_index(empty=True)
    index.save(path)
    index.add([1, 0], 123)
    index.nprobe = 3
    index.save(path)
    assert IVFVectorIndex.load(path).size == 1
    assert IVFVectorIndex.load(path).nprobe == 3
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        index.save(path)
    assert path.read_bytes() == b"corrupt"


@pytest.mark.parametrize("step", ["savez", "fsync", "replace"])
def test_failure_preserves_target_and_cleans_temporary_file(tmp_path, monkeypatch, step):
    path = tmp_path / "index.npz"
    index = make_index()
    index.save(path)
    before = path.read_bytes()
    index.add([1, 0], 100)

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    owner = ivf_storage.np if step == "savez" else ivf_storage.os
    monkeypatch.setattr(owner, step, fail)
    with pytest.raises(OSError, match="write failure"):
        index.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
