"""Explicit local filesystem roots; no browser-supplied paths."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VECTORFORGE_DATA_DIR", ROOT / ".vectorforge"))
    )
    results_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VECTORFORGE_RESULTS_DIR", ROOT / "results"))
    )
    published_dir: Path = ROOT / "docs" / "benchmark_results"
    origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    request_bytes: int = 4 * 1024 * 1024
    report_bytes: int = 5 * 1024 * 1024
    archive_bytes: int = 64 * 1024 * 1024
    max_indexes: int = 16
    job_timeout: int = 180


def confined(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        from backend.core.errors import AppError

        raise AppError(400, "unsafe_path", "File is outside the application directory.")
    return resolved
