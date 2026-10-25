import io
import zipfile

import numpy as np
import pytest

from vectorforge import ExactVectorIndex
from vectorforge.engine import storage


def payload():
    return dict(
        vectors=np.array([[1.0, 2.0], [3.0, 4.0]]),
        ids=np.array([91, -4]),
        dimension=np.array(2),
        metric=np.array("cosine"),
        format_version=np.array(1),
    )


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot"])
@pytest.mark.parametrize("empty", [False, True])
def test_round_trip(tmp_path, metric, empty):
    index = ExactVectorIndex(2, metric)
    if not empty:
        index.add([[1, 2], [3, 4], [0, 0]], [91, -4, 2**63 - 1])
    path = tmp_path / "index.data"
    index.save(path)
    assert path.exists()
    loaded = ExactVectorIndex.load(path)
    assert (loaded.dimension, loaded.metric, loaded.size) == (2, metric, index.size)
    for query in ([1, 2], [0, 0], [-3, 1]):
        assert loaded.search(query, 99) == index.search(query, 99)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
    loaded.add([5, 6], 5)
    assert index.size == (0 if empty else 3)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dimension", np.array(0)),
        ("dimension", np.array(3)),
        ("dimension", np.array(2.0)),
        ("dimension", np.array([2])),
        ("dimension", np.array(True)),
        ("metric", np.array("unsupported")),
        ("metric", np.array(["cosine"])),
        ("metric", np.array(b"cosine")),
        ("format_version", np.array(2)),
        ("format_version", np.array(1.0)),
        ("ids", np.array([1, 1])),
        ("ids", np.array([1])),
        ("ids", np.array([1.0, 2.0])),
        ("ids", np.array([[1, 2]])),
        ("ids", np.array([1, 2**63], dtype=np.uint64)),
        ("vectors", np.array([1, 2])),
        ("vectors", np.array([[1, np.nan], [3, 4]])),
        ("vectors", np.array([[1, np.inf], [3, 4]])),
        ("vectors", np.array([[1, 2], [3, 4]], dtype=object)),
    ],
)
def test_invalid_metadata_and_arrays(tmp_path, key, value):
    data = payload()
    data[key] = value
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="invalid index"):
        ExactVectorIndex.load(path)


@pytest.mark.parametrize("extra", [False, True])
def test_invalid_archive_fields(tmp_path, extra):
    data = payload()
    if extra:
        data["unknown"] = np.array(1)
    else:
        del data["metric"]
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="fields"):
        ExactVectorIndex.load(path)


@pytest.mark.parametrize("content", [b"", b"not an archive", b"PK\x03\x04corrupt"])
def test_corrupt_file(tmp_path, content):
    path = tmp_path / "bad.npz"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="invalid index"):
        ExactVectorIndex.load(path)


def test_truncated_and_wrong_format(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, **payload())
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(ValueError):
        ExactVectorIndex.load(path)
    np.save(tmp_path / "array.npy", np.array([1, 2]))
    with pytest.raises(ValueError):
        ExactVectorIndex.load(tmp_path / "array.npy")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExactVectorIndex.load(tmp_path / "missing.npz")


def test_compatible_overwrite(tmp_path):
    path = tmp_path / "index.npz"
    index = ExactVectorIndex(2)
    index.save(path)
    index.add([1, 0], 10)
    index.save(path)
    assert ExactVectorIndex.load(path).size == 1


@pytest.mark.parametrize(("dimension", "metric"), [(3, "cosine"), (2, "dot")])
def test_incompatible_overwrite_preserves_file(tmp_path, dimension, metric):
    path = tmp_path / "index.npz"
    ExactVectorIndex(2).save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        ExactVectorIndex(dimension, metric).save(path)
    assert path.read_bytes() == before


def test_corrupt_target_not_overwritten(tmp_path):
    path = tmp_path / "index.npz"
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        ExactVectorIndex(2).save(path)
    assert path.read_bytes() == b"corrupted"


def test_failed_save_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "index.npz"
    index = ExactVectorIndex(2)
    index.save(path)
    before = path.read_bytes()
    index.add([1, 0], 10)

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(storage.np, "savez", fail)
    with pytest.raises(OSError, match="disk failure"):
        index.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_duplicate_archive_field(tmp_path):
    path = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in payload().items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            archive.writestr(f"{key}.npy", buffer.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("metric.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="fields"):
        ExactVectorIndex.load(path)


def test_damaged_compressed_archive(tmp_path):
    path = tmp_path / "damaged.npz"
    np.savez_compressed(path, **payload())
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("vectors.npy")
        offset = member.header_offset
    content = bytearray(path.read_bytes())
    # Locate compressed data using ZIP local-header name and extra-field lengths.
    name_length = int.from_bytes(content[offset + 26 : offset + 28], "little")
    extra_length = int.from_bytes(content[offset + 28 : offset + 30], "little")
    data_offset = offset + 30 + name_length + extra_length
    content[data_offset] ^= 0xFF
    path.write_bytes(content)
    with pytest.raises(ValueError, match="invalid index"):
        ExactVectorIndex.load(path)


def test_failed_replacement_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "index.npz"
    index = ExactVectorIndex(2)
    index.save(path)
    before = path.read_bytes()
    index.add([1, 0], 10)

    def fail(*args):
        raise PermissionError("simulated replacement failure")

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(PermissionError, match="replacement failure"):
        index.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
