"""Environment-aware, single-worker entry point for source and container use."""

import uvicorn

from backend.core.config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,
        proxy_headers=False,
        timeout_graceful_shutdown=15,
    )


if __name__ == "__main__":
    main()
