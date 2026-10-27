"""Run from the checkout: uv run --extra web uvicorn backend.main:app --port 8000."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.api import benchmarks, indexes
from backend.core.config import Settings
from backend.core.errors import AppError, app_error, validation_error
from backend.services.benchmark_manager import BenchmarkManager, missing_dependencies
from backend.services.index_manager import IndexManager
from backend.services.index_persistence import IndexPersistence
from backend.services.reports import ReportService


class BodyLimit:
    def __init__(self, app, maximum):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, length = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            length += len(message.get("body", b""))
            if length > self.maximum:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "body_too_large",
                            "message": "Request body exceeds 4 MiB.",
                        }
                    },
                )
                return await response(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            return chunks.pop(0) if chunks else await receive()

        await self.app(scope, replay, send)


class Health(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    can_run_benchmarks: bool
    benchmark_enabled: bool = True
    missing_dependencies: list[str]
    benchmark_preset: dict
    benchmark_limits: dict


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        if settings.environment == "production":
            settings.prepare_storage()
        app.state.indexes = IndexManager(settings)
        app.state.persistence = IndexPersistence(settings, app.state.indexes)
        app.state.reports = ReportService(settings)
        app.state.jobs = BenchmarkManager(settings, app.state.reports)
        yield
        app.state.jobs.close()

    app = FastAPI(
        title="VectorForge API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.add_exception_handler(AppError, app_error)
    app.add_exception_handler(RequestValidationError, validation_error)

    @app.exception_handler(OSError)
    async def storage_error(request: Request, exc: OSError):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "storage_error",
                    "message": "Storage unavailable. Check server directory permissions.",
                }
            },
        )

    app.add_middleware(BodyLimit, maximum=settings.request_bytes)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.include_router(indexes.router, prefix="/api/v1")
    app.include_router(benchmarks.router, prefix="/api/v1")

    @app.get("/api/v1/health", response_model=Health, tags=["health"])
    def health():
        missing = missing_dependencies()
        return Health(
            can_run_benchmarks=settings.enable_benchmarks and not missing,
            benchmark_enabled=settings.enable_benchmarks,
            benchmark_limits={
                "max_pending_jobs": settings.max_pending_jobs,
                "job_timeout_seconds": settings.job_timeout,
            },
            missing_dependencies=missing,
            benchmark_preset={
                "size": 1000,
                "dimension": 32,
                "query_count": 50,
                "nlist": 16,
                "nprobes": [1, 4, 16],
                "repetitions": 3,
                "threads": 1,
            },
        )

    return app


app = create_app()
