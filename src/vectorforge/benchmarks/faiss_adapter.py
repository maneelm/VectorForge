"""FAISS CPU metric translation and lossless external signed-ID mapping."""

import importlib
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorforge.benchmarks.config import BenchmarkConfig
from vectorforge.engine._validation import (
    Ids,
    integer_ids,
    positive_integer,
    real_array,
    vector_batch,
)
from vectorforge.engine.base import SearchResult
from vectorforge.engine.distance import _unit


def require_faiss():
    try:
        return importlib.import_module("faiss")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "FAISS unavailable; install with: uv sync --locked --extra benchmark"
        ) from exc


class FaissAdapter:
    """Map all external IDs (including -1) to nonnegative native row IDs.

    FAISS boundary ties retain FAISS's selection policy. We sort the returned set
    by native score and external ID, but do not expand ties by searching all rows.
    """

    def __init__(self, config: BenchmarkConfig, *, ivf: bool = False) -> None:
        faiss = require_faiss()
        self.config, self.is_ivf = config, ivf
        metric = faiss.METRIC_L2 if config.metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        flat = (
            faiss.IndexFlatL2(config.dimension)
            if metric == faiss.METRIC_L2
            else faiss.IndexFlatIP(config.dimension)
        )
        if ivf:
            self.index = faiss.IndexIVFFlat(flat, config.dimension, config.nlist, metric)
            self.index.cp.seed = config.seed
            self.index.cp.niter = config.max_iter
            self.index.cp.nredo = 1
            self.index.nprobe = config.nprobes[0]
        else:
            self.index = faiss.IndexIDMap2(flat)
        self._ids = np.empty(0, dtype=np.int64)

    def _prepare(self, vectors: ArrayLike) -> NDArray[np.float32]:
        values = vector_batch(vectors, self.config.dimension)
        if self.config.metric == "cosine":
            values = _unit(values)
        with np.errstate(over="ignore", invalid="ignore"):
            prepared = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(prepared).all():
            raise ValueError("FAISS inputs exceed the finite float32 range")
        return prepared

    def train(self, vectors: ArrayLike) -> None:
        values = self._prepare(vectors)
        if self.is_ivf:
            if len(values) < self.config.nlist:
                raise ValueError("FAISS training requires at least nlist samples")
            if self.index.ntotal:
                raise ValueError("cannot retrain a populated FAISS adapter")
            self.index.train(values)

    def add(self, vectors: ArrayLike, ids: Ids) -> None:
        values, identifiers = self._prepare(vectors), integer_ids(ids)
        if len(values) != len(identifiers):
            raise ValueError("each vector must have exactly one ID")
        if len(np.unique(identifiers)) != len(identifiers) or np.isin(identifiers, self._ids).any():
            raise ValueError("duplicate IDs are not allowed")
        if not self.index.is_trained:
            raise ValueError("FAISS IVF must be trained before insertion")
        mapped = np.concatenate((self._ids, identifiers))
        self.index.add_with_ids(values, np.arange(len(self._ids), len(mapped), dtype=np.int64))
        self._ids = mapped

    def search(self, query: ArrayLike, k: int, nprobe: int | None = None) -> list[SearchResult]:
        k = positive_integer(k, "k")
        values = real_array(query, "query")
        if values.shape != (self.config.dimension,):
            raise ValueError("query must have shape (dimension,)")
        prepared = self._prepare(values)
        if self.is_ivf:
            probe = self.config.nprobes[0] if nprobe is None else positive_integer(nprobe, "nprobe")
            if probe > self.config.nlist:
                raise ValueError("nprobe cannot exceed nlist")
            self.index.nprobe = probe
        elif nprobe is not None:
            raise ValueError("nprobe does not apply to FAISS Flat")
        if not self.index.is_trained:
            raise ValueError("FAISS IVF must be trained before search")
        if not len(self._ids):
            return []
        scores, rows = self.index.search(prepared, min(k, len(self._ids)))
        matches = []
        for row, raw_score in zip(rows[0], scores[0], strict=True):
            if row == -1:
                continue
            if row < 0 or row >= len(self._ids) or not np.isfinite(raw_score):
                raise ValueError("FAISS returned an invalid row or nonfinite score")
            score = float(raw_score)
            if self.config.metric == "euclidean":
                score = math.sqrt(max(0.0, score))
            elif self.config.metric == "cosine":
                score = float(np.clip(score, -1, 1))
            matches.append(SearchResult(int(self._ids[row]), score))
        return sorted(
            matches,
            key=lambda r: (r.score if self.config.metric == "euclidean" else -r.score, r.id),
        )
