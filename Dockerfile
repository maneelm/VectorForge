# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.28 AS uv
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra web --extra benchmark --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra web --extra benchmark --no-editable

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 vectorforge \
    && useradd --uid 10001 --gid 10001 --create-home vectorforge \
    && install -d -o 10001 -g 10001 -m 0750 /var/lib/vectorforge/data /var/lib/vectorforge/results
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY backend ./backend
COPY docs/benchmark_results ./docs/benchmark_results
ARG API_PORT=8000
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VECTORFORGE_ENV=production VECTORFORGE_API_HOST=0.0.0.0 VECTORFORGE_API_PORT=${API_PORT} \
    VECTORFORGE_DATA_DIR=/var/lib/vectorforge/data VECTORFORGE_RESULTS_DIR=/var/lib/vectorforge/results \
    MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/cache \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER 10001:10001
EXPOSE ${API_PORT}
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=4 \
    CMD ["python", "-m", "backend.healthcheck"]
CMD ["python", "-m", "backend.serve"]
