"""Dependency-free container liveness/readiness probe."""

import json
import os
from urllib.request import urlopen


def main() -> None:
    port = int(os.getenv("VECTORFORGE_API_PORT", "8000"))
    with urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=3) as response:
        if response.status != 200 or json.load(response)["status"] != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
