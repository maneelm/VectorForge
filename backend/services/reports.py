"""Read bounded Phase 3 reports without importing FAISS or launching a benchmark."""

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.core.config import Settings, confined
from backend.core.errors import AppError
from backend.schemas import BenchmarkResult, RunInfo, RunList, RunResults
from vectorforge.benchmarks.config import BenchmarkConfig


def invalid_constant(value):
    raise ValueError("Nonfinite JSON is not supported")


class ReportService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.roots = {"published": settings.published_dir, "local": settings.results_dir}

    def discover(self):
        found = {}
        for source, root in self.roots.items():
            visited = 0
            for directory, dirs, files in os.walk(root, followlinks=False):
                visited += 1
                if visited > 2000 or len(found) >= 200:
                    break
                dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
                if "summary.json" in files:
                    path = confined(root, Path(directory))
                    identifier = str(
                        uuid5(
                            NAMESPACE_URL,
                            source + ":" + path.relative_to(root.resolve()).as_posix(),
                        )
                    )
                    found[identifier] = (source, root, path)
        return found

    def read(self, identifier: str, location) -> RunResults:
        source, root, directory = location
        documents = []
        for filename in ("summary.json", "metadata.json"):
            path = confined(root, directory / filename)
            if path.stat().st_size > self.settings.report_bytes:
                raise ValueError("Report file exceeds 5 MiB")
            documents.append(
                json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
            )
        summary, metadata = documents
        if (
            summary.get("benchmark_schema_version") != 1
            or metadata.get("benchmark_schema_version") != 1
        ):
            raise ValueError("Unsupported benchmark schema")
        config = BenchmarkConfig.from_dict(metadata["config"])
        if not isinstance(summary["results"], list) or not 1 <= len(summary["results"]) <= 512:
            raise ValueError("Report needs 1–512 result rows")
        rows = [BenchmarkResult.model_validate(row) for row in summary["results"]]
        identities = set()
        for row in rows:
            for key, expected in (
                ("dataset_size", config.size),
                ("dimension", config.dimension),
                ("metric", config.metric),
                ("distribution", config.distribution),
                ("seed", config.seed),
                ("k", config.k),
                ("query_count", config.query_count),
                ("repetitions", config.repetitions),
                ("training_size", config.training_size),
                ("threads", config.threads),
            ):
                if getattr(row, key) != expected:
                    raise ValueError("Mixed or inconsistent workload metadata")
            identity = (row.engine, row.nprobe)
            if identity in identities or row.engine not in config.engines:
                raise ValueError("Duplicate or unexpected engine configuration")
            if row.nprobe is not None and (
                row.nlist != config.nlist or row.nprobe not in config.nprobes
            ):
                raise ValueError("Inconsistent IVF configuration")
            identities.add(identity)
        expected = {
            (e, p)
            for e in config.engines
            for p in (config.nprobes if e.endswith("ivf") else (None,))
        }
        if identities != expected or metadata["complete"] != all(r.status == "ok" for r in rows):
            raise ValueError("Incomplete or inconsistent result configurations")
        datetime.fromisoformat(metadata["timestamp_utc"])
        info = RunInfo(
            id=identifier,
            name=directory.name,
            source=source,
            timestamp=metadata["timestamp_utc"],
            complete=metadata["complete"],
            size=config.size,
            dimension=config.dimension,
            metric=config.metric,
            distribution=config.distribution,
            result_count=len(rows),
        )
        public = {
            key: metadata.get(key)
            for key in (
                "python_version",
                "numpy_version",
                "faiss_version",
                "operating_system",
                "cpu",
                "thread_environment",
                "warnings",
                "timing_scope",
                "qps_scope",
                "memory_scope",
                "config",
            )
        }
        return RunResults(run=info, metadata=public, results=rows)

    def list(self) -> RunList:
        runs, issues = [], []
        for identifier, location in self.discover().items():
            try:
                runs.append(self.read(identifier, location).run)
            except (ValueError, OSError, KeyError, TypeError, AppError) as exc:
                error = exc.message if isinstance(exc, AppError) else str(exc)
                issues.append({"id": identifier, "name": location[2].name, "error": error[:300]})
        return RunList(runs=sorted(runs, key=lambda r: r.timestamp, reverse=True), issues=issues)

    def get(self, identifier: UUID) -> RunResults:
        location = self.discover().get(str(identifier))
        if location is None:
            raise AppError(404, "report_not_found", "Benchmark run not found.")
        try:
            return self.read(str(identifier), location)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise AppError(400, "invalid_report", str(exc)[:300]) from exc
