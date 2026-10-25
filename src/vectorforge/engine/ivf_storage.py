"""Independent, strictly validated IVF NPZ schema; exact v1 remains unchanged."""

import os
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np

from vectorforge.engine._routing import assign
from vectorforge.engine._validation import (
    integer_ids,
    positive_integer,
    real_array,
    validate_metric,
)
from vectorforge.engine.kmeans import configuration

FORMAT_VERSION = 1
_KEYS = {
    "index_type",
    "ivf_format_version",
    "dimension",
    "metric",
    "vectors",
    "ids",
    "nlist",
    "nprobe",
    "centroids",
    "assignments",
    "max_iter",
    "tol",
    "random_state",
    "is_trained",
}


def _validate(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(data) != _KEYS:
        raise ValueError("invalid IVF archive fields")
    integers = {}
    for key in ("ivf_format_version", "dimension", "nlist", "nprobe", "max_iter", "random_state"):
        if data[key].shape != () or data[key].dtype.kind not in "iu":
            raise ValueError(f"{key} must be an integer scalar")
        integers[key] = int(data[key])
    if integers["ivf_format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported IVF format version")
    for key in ("index_type", "metric"):
        if data[key].shape != () or data[key].dtype.kind != "U":
            raise ValueError(f"{key} must be a Unicode scalar")
    if data["index_type"].item() != "ivf":
        raise ValueError("expected IVF index type")
    if (
        data["is_trained"].shape != ()
        or data["is_trained"].dtype.kind != "b"
        or not bool(data["is_trained"])
    ):
        raise ValueError("only trained IVF indexes can be persisted")
    dimension = positive_integer(integers["dimension"], "dimension")
    nlist = positive_integer(integers["nlist"], "nlist")
    nprobe = positive_integer(integers["nprobe"], "nprobe")
    if nprobe > nlist:
        raise ValueError("nprobe cannot exceed nlist")
    if data["tol"].shape != () or data["tol"].dtype.kind not in "iuf":
        raise ValueError("tol must be a numeric scalar")
    configuration(integers["max_iter"], float(data["tol"]), integers["random_state"])
    metric = validate_metric(str(data["metric"].item()))
    vectors = real_array(data["vectors"], "vectors")
    centers = real_array(data["centroids"], "centroids")
    if vectors.ndim != 2 or vectors.shape[1] != dimension:
        raise ValueError("invalid persisted vector shape")
    if centers.shape != (nlist, dimension):
        raise ValueError("invalid centroid shape")
    if metric == "cosine" and np.any(np.hypot.reduce(centers, axis=1) > 1 + 1e-12):
        raise ValueError("cosine centroids must lie within the unit ball")
    for key in ("ids", "assignments"):
        if data[key].ndim != 1 or data[key].dtype.kind not in "iu":
            raise ValueError(f"{key} must be a one-dimensional integer array")
    ids = integer_ids(data["ids"])
    assignments = integer_ids(data["assignments"])
    if len(ids) != len(vectors) or len(np.unique(ids)) != len(ids):
        raise ValueError("persisted IDs must be unique and match vectors")
    if len(assignments) != len(vectors) or np.any((assignments < 0) | (assignments >= nlist)):
        raise ValueError("invalid persisted assignments")
    if not np.array_equal(assignments, assign(vectors, centers, metric)):
        raise ValueError("persisted assignments do not match nearest-centroid routing")
    return {
        **data,
        "vectors": vectors,
        "centroids": centers,
        "ids": ids,
        "assignments": assignments,
    }


def load_index(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Validate before restoration; filesystem errors retain their OSError type."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(_KEYS) or set(archive.files) != _KEYS:
                raise ValueError("invalid IVF archive fields")
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
        raise ValueError(f"invalid IVF index file: {exc}") from exc


def save_index(path: str | os.PathLike[str], payload: dict[str, np.ndarray]) -> None:
    """Atomically replace only valid, compatible IVF targets (dimension/metric/nlist)."""
    state = _validate(payload)
    target = Path(path)
    if target.exists():
        previous = load_index(target)
        if any(
            previous[key].item() != state[key].item() for key in ("dimension", "metric", "nlist")
        ):
            raise ValueError("refusing to overwrite incompatible IVF index")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
            temporary = handle.name
            np.savez(handle, **state)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
