"""Explicit local filesystem roots; no browser-supplied paths."""

import ipaddress
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value.lower() == "true"


def env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    return default if value is None else tuple(x.strip() for x in value.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    environment: str = field(default_factory=lambda: os.getenv("VECTORFORGE_ENV", "development"))
    api_host: str = field(default_factory=lambda: os.getenv("VECTORFORGE_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("VECTORFORGE_API_PORT", "8000")))
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VECTORFORGE_DATA_DIR", ROOT / ".vectorforge"))
    )
    results_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VECTORFORGE_RESULTS_DIR", ROOT / "results"))
    )
    published_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("VECTORFORGE_PUBLISHED_DIR", ROOT / "docs/benchmark_results")
        )
    )
    allowed_hosts: tuple[str, ...] | None = None
    origins: tuple[str, ...] | None = None
    enable_docs: bool = field(default_factory=lambda: env_bool("VECTORFORGE_ENABLE_DOCS", True))
    enable_benchmarks: bool = field(
        default_factory=lambda: env_bool("VECTORFORGE_ENABLE_BENCHMARKS", True)
    )
    request_bytes: int = 4 * 1024 * 1024
    report_bytes: int = 5 * 1024 * 1024
    archive_bytes: int = 64 * 1024 * 1024
    max_indexes: int = field(
        default_factory=lambda: int(os.getenv("VECTORFORGE_MAX_INDEXES", "16"))
    )
    job_timeout: int = field(
        default_factory=lambda: int(os.getenv("VECTORFORGE_JOB_TIMEOUT", "180"))
    )
    max_pending_jobs: int = field(
        default_factory=lambda: int(os.getenv("VECTORFORGE_MAX_PENDING_JOBS", "2"))
    )

    def __post_init__(self):
        if self.environment not in ("development", "production"):
            raise ValueError("VECTORFORGE_ENV must be development or production")
        if not isinstance(self.api_host, str):
            raise ValueError("API host must be an explicit bind IP or localhost")
        if self.api_host != "localhost":
            ipaddress.ip_address(self.api_host)
        development = self.environment == "development"
        if self.allowed_hosts is None:
            object.__setattr__(
                self,
                "allowed_hosts",
                env_list(
                    "VECTORFORGE_ALLOWED_HOSTS",
                    ("localhost", "127.0.0.1", "testserver")
                    if development
                    else ("localhost", "127.0.0.1"),
                ),
            )
        if self.origins is None:
            object.__setattr__(
                self,
                "origins",
                env_list(
                    "VECTORFORGE_CORS_ORIGINS",
                    ("http://localhost:5173", "http://127.0.0.1:5173") if development else (),
                ),
            )
        for name, maximum in (
            ("api_port", 65535),
            ("max_indexes", 16),
            ("job_timeout", 180),
            ("max_pending_jobs", 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if not development and self.api_port < 1024:
            raise ValueError("Production API port must be at least 1024 for the non-root user")
        if not self.allowed_hosts or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host) for host in self.allowed_hosts
        ):
            raise ValueError(
                "Allowed hosts must be explicit hostnames/IPs without ports or wildcards"
            )
        for origin in self.origins:
            parsed = urlsplit(origin)
            _ = parsed.port  # Validate malformed numeric ports too.
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or "*" in origin
                or parsed.username
                or parsed.password
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "CORS origins must be explicit HTTP(S) origins, without paths or wildcards"
                )
        for name in ("data_dir", "results_dir", "published_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())

    def prepare_storage(self) -> None:
        """Fail at production startup if persistent mounts cannot be written."""
        for path in (self.data_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path) as handle:
                handle.write(b"storage check")


def confined(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        from backend.core.errors import AppError

        raise AppError(400, "unsafe_path", "File is outside the application directory.")
    return resolved
