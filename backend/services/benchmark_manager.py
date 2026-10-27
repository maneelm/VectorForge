"""One bounded CLI job at a time; no process starts at import or construction."""

import importlib.util
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import RLock
from uuid import UUID, uuid4

from backend.core.config import ROOT, Settings, confined
from backend.core.errors import AppError
from backend.schemas import JobInfo, QuickRun
from backend.services.reports import ReportService

logger = logging.getLogger(__name__)


def missing_dependencies() -> list[str]:
    return [
        name
        for name in ("faiss", "matplotlib", "psutil", "threadpoolctl")
        if importlib.util.find_spec(name) is None
    ]


def stop_tree(process):
    try:
        import psutil

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs(children, timeout=3)
    except (ImportError, OSError):
        process.kill()
    except Exception:
        process.kill()


class BenchmarkManager:
    def __init__(self, settings: Settings, reports: ReportService):
        self.settings, self.reports = settings, reports
        self.lock, self.jobs = RLock(), {}
        self.executor = None
        self.process = None
        self.closed = False

    def public_job(self, job: JobInfo) -> JobInfo:
        result = job.model_copy(deep=True)
        if self.settings.environment == "production":
            result.log = ""
            if result.error:
                result.error = "Benchmark failed. Inspect the server logs using the job ID."
        return result

    def list(self) -> list[JobInfo]:
        with self.lock:
            return [self.public_job(job) for job in reversed(list(self.jobs.values()))]

    def get(self, identifier: UUID) -> JobInfo:
        with self.lock:
            if identifier not in self.jobs:
                raise AppError(404, "job_not_found", "Benchmark job not found.")
            return self.public_job(self.jobs[identifier])

    def submit(self, request: QuickRun) -> JobInfo:
        if not self.settings.enable_benchmarks:
            raise AppError(
                503, "benchmark_disabled", "Benchmark execution is disabled by the server."
            )
        missing = missing_dependencies()
        if missing:
            raise AppError(
                503,
                "benchmark_unavailable",
                "Missing "
                + ", ".join(missing)
                + ". Run: uv sync --locked --extra web --extra benchmark",
            )
        web_root = confined(self.settings.results_dir, self.settings.results_dir / "web")
        retained_bytes = 0
        for path in web_root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                retained_bytes += path.stat().st_size
            if retained_bytes > 128 * 1024 * 1024:
                raise AppError(
                    409,
                    "benchmark_storage_full",
                    "Web benchmark outputs exceed 128 MiB. Archive or remove old local runs first.",
                )
        with self.lock:
            if (
                self.closed
                or sum(j.status in ("queued", "running") for j in self.jobs.values())
                >= self.settings.max_pending_jobs
            ):
                raise AppError(409, "benchmark_busy", "Benchmark running/queued capacity reached.")
            if len(self.jobs) >= 50:
                oldest = next(
                    (i for i, j in self.jobs.items() if j.status in ("completed", "failed")), None
                )
                if oldest is not None:
                    del self.jobs[oldest]
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="benchmark-job"
                )
            job = JobInfo(id=uuid4(), status="queued", created_at=datetime.now(UTC).isoformat())
            self.jobs[job.id] = job
            self.executor.submit(self.execute, job.id, request)
            return self.public_job(job)

    def execute(self, identifier: UUID, request: QuickRun):
        process = None
        try:
            with self.lock:
                if self.closed:
                    return
                self.jobs[identifier].status = "running"
            output = confined(
                self.settings.results_dir, self.settings.results_dir / "web" / str(identifier)
            )
            output.mkdir(parents=True, exist_ok=False)
            command = [
                sys.executable,
                "-m",
                "vectorforge.benchmarks.cli",
                "--preset",
                "quick",
                "--size",
                "1000",
                "--dimension",
                "32",
                "--nprobe",
                "1,4,16",
                "--metric",
                request.metric,
                "--seed",
                str(request.seed),
                "--threads",
                "1",
                "--output",
                str(output),
            ]
            options = (
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if sys.platform == "win32"
                else {"start_new_session": True}
            )
            with self.lock:
                if self.closed:
                    return
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    **options,
                )
                self.process = process
            log, _ = process.communicate(timeout=self.settings.job_timeout)
            with self.lock:
                self.jobs[identifier].log = log[-8000:]
            if process.returncode:
                raise RuntimeError(f"Benchmark CLI exited {process.returncode}. " + log[-2000:])
            found = []
            for run_id, location in self.reports.discover().items():
                if location[2].is_relative_to(output):
                    report = self.reports.read(run_id, location)
                    if not report.run.complete:
                        raise RuntimeError("Benchmark report contains failed configurations")
                    found.append(run_id)
            if not found:
                raise RuntimeError("CLI produced no valid complete report")
            with self.lock:
                self.jobs[identifier].run_ids = found
                self.jobs[identifier].status = "completed"
        except Exception as exc:
            logger.error("Benchmark job %s failed: %s", identifier, exc)
            if process is not None and process.poll() is None:
                try:
                    stop_tree(process)
                    process.communicate(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            with self.lock:
                self.jobs[identifier].status = "failed"
                self.jobs[identifier].error = (
                    "Benchmark timed out."
                    if isinstance(exc, subprocess.TimeoutExpired)
                    else str(exc)
                )[:2500]
        finally:
            with self.lock:
                self.process = None

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
            for job in self.jobs.values():
                if job.status in ("queued", "running"):
                    job.status, job.error = "failed", "Application shut down."
        if process is not None and process.poll() is None:
            try:
                stop_tree(process)
            except OSError:
                logger.exception("Could not terminate benchmark process")
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
