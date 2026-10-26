"""Catalog locking is short; each index independently serializes operations."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter_ns
from uuid import UUID, uuid4

from backend.core.config import Settings
from backend.core.errors import AppError
from backend.schemas import IndexCreate, IndexInfo, Match, Search, SearchResponse
from vectorforge import BaseVectorIndex, ExactVectorIndex, IVFVectorIndex


@dataclass
class Entry:
    id: UUID
    config: IndexCreate
    engine: BaseVectorIndex
    lock: RLock = field(default_factory=RLock)
    deleted: bool = False

    def info(self) -> IndexInfo:
        trained = not isinstance(self.engine, IVFVectorIndex) or self.engine.is_trained
        config = self.config.model_dump(exclude={"name", "type"})
        if self.config.type == "exact":
            config = {k: config[k] for k in ("dimension", "metric")}
        return IndexInfo(
            id=self.id,
            name=self.config.name,
            type=self.config.type,
            dimension=self.engine.dimension,
            metric=self.engine.metric,
            size=self.engine.size,
            is_trained=trained,
            status="ready" if trained else "untrained",
            configuration=config,
        )


class IndexManager:
    def __init__(self, settings: Settings):
        self.settings, self._catalog, self._lock = settings, {}, RLock()

    def register(self, config: IndexCreate, engine: BaseVectorIndex) -> IndexInfo:
        if engine.size > 50000 or engine.size * engine.dimension > 2000000:
            raise AppError(
                422, "capacity", "Index exceeds the 50,000-vector / 2,000,000-coordinate limit."
            )
        entry = Entry(uuid4(), config, engine)
        with self._lock:
            if len(self._catalog) >= self.settings.max_indexes:
                raise AppError(409, "capacity", "Maximum active index count reached.")
            self._catalog[entry.id] = entry
        return entry.info()

    def create(self, config: IndexCreate) -> IndexInfo:
        if config.type == "exact":
            engine = ExactVectorIndex(config.dimension, config.metric)
        else:
            engine = IVFVectorIndex(**config.model_dump(exclude={"name", "type"}))
        return self.register(config, engine)

    @contextmanager
    def access(self, identifier: UUID):
        with self._lock:
            entry = self._catalog.get(identifier)
        if entry is None:
            raise AppError(404, "index_not_found", "Index not found.")
        with entry.lock:
            if entry.deleted:
                raise AppError(404, "index_not_found", "Index was deleted.")
            try:
                yield entry
            except ValueError as exc:
                message = str(exc)
                conflict = any(word in message for word in ("duplicate", "trained", "populated"))
                raise AppError(409 if conflict else 422, "index_operation", message) from exc

    def list(self) -> list[IndexInfo]:
        with self._lock:
            entries = list(self._catalog.values())
        result = []
        for entry in entries:
            with entry.lock:
                if not entry.deleted:
                    result.append(entry.info())
        return result

    def delete(self, identifier: UUID) -> None:
        with self.access(identifier) as entry:
            with self._lock:
                entry.deleted = True
                del self._catalog[identifier]

    def search(self, identifier: UUID, request: Search) -> SearchResponse:
        with self.access(identifier) as entry:
            if entry.config.type == "exact" and request.nprobe is not None:
                raise AppError(422, "unsupported_parameter", "Exact search does not accept nprobe.")
            kwargs = {"nprobe": request.nprobe} if entry.config.type == "ivf" else {}
            start = perf_counter_ns()
            matches = entry.engine.search(request.query, request.k, **kwargs)
            elapsed = (perf_counter_ns() - start) / 1e6
            return SearchResponse(
                matches=[
                    Match(id=str(m.id), score=m.score, rank=i + 1) for i, m in enumerate(matches)
                ],
                latency_ms=elapsed,
                metric=entry.engine.metric,
                index_type=entry.config.type,
                nprobe=(request.nprobe or entry.engine.nprobe)
                if entry.config.type == "ivf"
                else None,
            )
