"""Versioned, pickle-free NPZ persistence with strict schema validation."""

import os
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vectorforge.engine._validation import (
    Metric,
    integer_ids,
    positive_integer,
    validate_metric,
    vector_batch,
)

FORMAT_VERSION = 1
_KEYS = {"vectors", "ids", "dimension", "metric", "format_version"}


@dataclass(frozen=True, slots=True)
class IndexState:
    vectors: NDArray[np.float64]
    ids: NDArray[np.int64]
    dimension: int
    metric: Metric


def _validate(data: dict[str, np.ndarray]) -> IndexState:
    if set(data) != _KEYS:
        raise ValueError(f"index archive must contain exactly {sorted(_KEYS)}")
    for name in ("format_version", "dimension"):
        if data[name].shape != () or data[name].dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer scalar")
    if int(data["format_version"]) != FORMAT_VERSION:
        raise ValueError("unsupported index format version")
    dimension = positive_integer(int(data["dimension"]), "dimension")
    if data["metric"].shape != () or data["metric"].dtype.kind != "U":
        raise ValueError("metric must be a Unicode string scalar")
    metric = validate_metric(str(data["metric"].item()))
    if data["vectors"].ndim != 2 or data["ids"].ndim != 1:
        raise ValueError("persisted vectors must be 2-D and IDs must be 1-D")
    vectors = vector_batch(data["vectors"], dimension)
    ids = integer_ids(data["ids"])
    if data["ids"].dtype.kind not in "iu":
        raise ValueError("persisted IDs must have integer dtype")
    if len(ids) != len(vectors) or len(np.unique(ids)) != len(ids):
        raise ValueError("persisted IDs must be unique and match the vector count")
    return IndexState(vectors, ids, dimension, metric)


def load_index(path: str | os.PathLike[str]) -> IndexState:
    """Read and validate an archive. File access errors retain their OSError type."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError("expected an NPZ index archive")
            if len(archive.files) != len(_KEYS) or set(archive.files) != _KEYS:
                raise ValueError("invalid index archive fields")
            return _validate({key: archive[key] for key in _KEYS})
    except (
        ValueError,
        TypeError,
        EOFError,
        zipfile.BadZipFile,
        zlib.error,
        AttributeError,
        KeyError,
    ) as exc:
        raise ValueError(f"invalid index file: {exc}") from exc


def save_index(
    path: str | os.PathLike[str],
    vectors: NDArray[np.float64],
    ids: NDArray[np.int64],
    dimension: int,
    metric: Metric,
) -> None:
    """Save atomically; existing files must be valid and dimension/metric compatible.

    A sibling temporary file plus os.replace keeps the previous archive intact on
    write failure. Concurrent writers require external synchronization.
    """
    payload = {
        "vectors": vectors,
        "ids": ids,
        "dimension": np.asarray(dimension),
        "metric": np.asarray(metric),
        "format_version": np.asarray(FORMAT_VERSION),
    }
    state = _validate(payload)
    target = Path(path)
    if target.exists():
        previous = load_index(target)
        if (previous.dimension, previous.metric) != (state.dimension, state.metric):
            raise ValueError("refusing to overwrite an index with incompatible dimension or metric")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
            temporary = handle.name
            # Passing a handle preserves the exact path; np.savez writes only numeric/Unicode data.
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
