"""Application manifests accompany unchanged engine NPZ archives."""

import json
import zipfile
from uuid import UUID, uuid4

import numpy as np

from backend.core.config import Settings, confined
from backend.core.errors import AppError
from backend.schemas import IndexCreate, SavedIndex
from backend.services.index_manager import IndexManager
from vectorforge import ExactVectorIndex, IVFVectorIndex


class IndexPersistence:
    def __init__(self, settings: Settings, indexes: IndexManager):
        self.settings, self.indexes = settings, indexes
        self.root = settings.data_dir / "indexes"

    def paths(self, identifier: UUID):
        return tuple(
            confined(self.root, self.root / f"{identifier}.{ext}") for ext in ("npz", "json")
        )

    def save(self, identifier: UUID) -> SavedIndex:
        archive_id = uuid4()
        archive, manifest = self.paths(archive_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.indexes.access(identifier) as entry:
            if archive.exists() or manifest.exists():
                raise AppError(409, "archive_exists", "Archive already exists.")
            entry.engine.save(archive)
            config = (
                entry.config.model_dump()
                if entry.config.type == "ivf"
                else entry.config.model_dump(include={"name", "type", "dimension", "metric"})
            )
            try:
                with manifest.open("x", encoding="utf-8") as handle:
                    json.dump(config, handle, allow_nan=False)
            except OSError:
                archive.unlink(missing_ok=True)
                raise
            return SavedIndex(
                archive_id=archive_id,
                name=entry.config.name,
                type=entry.config.type,
                size_bytes=archive.stat().st_size,
            )

    def configuration(self, identifier: UUID) -> IndexCreate:
        archive, manifest = self.paths(identifier)
        if not archive.is_file() or not manifest.is_file():
            raise AppError(404, "archive_not_found", "Saved index not found.")
        if manifest.stat().st_size > 16384 or archive.stat().st_size > self.settings.archive_bytes:
            raise AppError(413, "archive_too_large", "Saved archive exceeds application limits.")
        return IndexCreate.model_validate_json(manifest.read_bytes())

    def list(self) -> list[SavedIndex]:
        result = []
        if not self.root.exists():
            return result
        for path in list(self.root.glob("*.json"))[:1000]:
            try:
                identifier = UUID(path.stem)
                config = self.configuration(identifier)
                archive, _ = self.paths(identifier)
                result.append(
                    SavedIndex(
                        archive_id=identifier,
                        name=config.name,
                        type=config.type,
                        size_bytes=archive.stat().st_size,
                    )
                )
            except (AppError, ValueError, OSError):
                continue
        return result

    def load(self, identifier: UUID):
        try:
            config = self.configuration(identifier)
            archive, _ = self.paths(identifier)
            with zipfile.ZipFile(archive) as data:
                if (
                    sum(m.file_size for m in data.infolist()) > self.settings.archive_bytes
                    or len(data.infolist()) > 20
                ):
                    raise ValueError("Uncompressed archive exceeds application limits")
            engine = (IVFVectorIndex if config.type == "ivf" else ExactVectorIndex).load(archive)
            if engine.dimension != config.dimension or engine.metric != config.metric:
                raise ValueError("Archive and manifest configuration disagree")
            if config.type == "ivf" and (engine.nlist, engine.nprobe) != (
                config.nlist,
                config.nprobe,
            ):
                raise ValueError("IVF archive and manifest configuration disagree")
            if config.type == "ivf":
                with np.load(archive, allow_pickle=False) as stored:
                    if any(
                        stored[key].item() != getattr(config, key)
                        for key in ("random_state", "max_iter", "tol")
                    ):
                        raise ValueError("IVF training configuration disagrees with manifest")
            return self.indexes.register(config, engine)
        except (ValueError, zipfile.BadZipFile) as exc:
            raise AppError(400, "invalid_archive", str(exc)) from exc
