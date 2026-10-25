"""Single-query adapters; FAISS is imported only when explicitly selected."""

from typing import Protocol

from numpy.typing import ArrayLike

from vectorforge import ExactVectorIndex, IVFVectorIndex, SearchResult
from vectorforge.benchmarks.config import BenchmarkConfig
from vectorforge.engine._validation import Ids


class Adapter(Protocol):
    def train(self, vectors: ArrayLike) -> None: ...
    def add(self, vectors: ArrayLike, ids: Ids) -> None: ...
    def search(self, query: ArrayLike, k: int, nprobe: int | None = None) -> list[SearchResult]: ...


class VectorForgeAdapter:
    def __init__(self, config: BenchmarkConfig, *, ivf: bool = False) -> None:
        self.index = (
            IVFVectorIndex(
                config.dimension,
                nlist=config.nlist,
                nprobe=config.nprobes[0],
                metric=config.metric,
                random_state=config.seed,
                max_iter=config.max_iter,
            )
            if ivf
            else ExactVectorIndex(config.dimension, config.metric)
        )

    def train(self, vectors: ArrayLike) -> None:
        if isinstance(self.index, IVFVectorIndex):
            self.index.train(vectors)

    def add(self, vectors: ArrayLike, ids: Ids) -> None:
        self.index.add(vectors, ids)

    def search(self, query: ArrayLike, k: int, nprobe: int | None = None) -> list[SearchResult]:
        if isinstance(self.index, IVFVectorIndex):
            return self.index.search(query, k, nprobe=nprobe)
        if nprobe is not None:
            raise ValueError("nprobe does not apply to exact indexes")
        return self.index.search(query, k)


def make_adapter(engine: str, config: BenchmarkConfig) -> Adapter:
    if engine in ("vectorforge_exact", "vectorforge_ivf"):
        return VectorForgeAdapter(config, ivf=engine == "vectorforge_ivf")
    if engine in ("faiss_flat", "faiss_ivf"):
        from vectorforge.benchmarks.faiss_adapter import FaissAdapter

        return FaissAdapter(config, ivf=engine == "faiss_ivf")
    raise ValueError(f"unknown engine: {engine}")
