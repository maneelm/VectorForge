"""Real Compose HTTP/persistence smoke workflow. No engine imports or fake measurements.

Run 'before', recreate containers with their volumes retained, then run 'after'.
The optional auth phase uses ephemeral CI credentials from environment variables.
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after", "auth"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--state-file", type=Path, default=Path(".vectorforge/container-smoke.json")
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    def http(path, method="GET", body=None, authenticated=True):
        headers = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        user, password = (
            os.getenv("VECTORFORGE_SMOKE_USER"),
            os.getenv("VECTORFORGE_SMOKE_PASSWORD"),
        )
        if authenticated and user and password:
            headers["Authorization"] = (
                "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
            )
        request = Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=15) as response:
            payload = response.read()
            if "application/json" in response.headers.get("Content-Type", ""):
                return json.loads(payload)
            return payload

    if args.phase == "auth":
        for path in ("/", "/api/v1/health", "/docs", "/openapi.json"):
            try:
                http(path, authenticated=False)
            except HTTPError as exc:
                assert exc.code == 401, (path, exc.code)
            else:
                raise AssertionError(f"Unauthenticated access allowed: {path}")
        assert http("/api/v1/health")["status"] == "ok"
        assert b'<div id="root">' in http("/benchmarks")
        print("PASS: unauthenticated UI/API/docs denied; authenticated health and SPA available")
        return

    assert http("/api/v1/health")["status"] == "ok"
    assert b'<div id="root">' in http("/")
    assert b'<div id="root">' in http("/workspace")
    assert b"swagger" in http("/docs").lower()
    assert http("/openapi.json")["info"]["title"] == "VectorForge API"
    if args.phase == "before":
        state = {"archives": []}
        for kind in ("exact", "ivf"):
            config = {
                "name": "Container smoke " + kind,
                "type": kind,
                "dimension": 2,
                "metric": "cosine",
            }
            if kind == "ivf":
                config.update(nlist=2, nprobe=1)
            index = http("/api/v1/indexes", "POST", config)
            path = "/api/v1/indexes/" + index["id"]
            if kind == "ivf":
                assert http(path + "/train", "POST", {"vectors": [[1, 0], [-1, 0]]})["is_trained"]
            http(
                path + "/vectors",
                "POST",
                {
                    "vectors": [[1, 0], [-1, 0]],
                    "ids": ["-9223372036854775808", "9223372036854775807"],
                },
            )
            query = {"query": [1, 0], "k": 5}
            if kind == "ivf":
                query["nprobe"] = 2
            matches = http(path + "/search", "POST", query)["matches"]
            assert [m["id"] for m in matches] == ["-9223372036854775808", "9223372036854775807"]
            saved = http(path + "/save", "POST")
            state["archives"].append(
                {"archive_id": saved["archive_id"], "query": query, "matches": matches}
            )
        runs = http("/api/v1/benchmarks/runs")["runs"]
        assert any(run["source"] == "published" for run in runs)
        job = http("/api/v1/benchmarks/runs", "POST", {"preset": "quick"})
        deadline = time.monotonic() + 200
        while job["status"] in ("running", "queued") and time.monotonic() < deadline:
            time.sleep(1)
            job = http("/api/v1/benchmarks/jobs/" + job["id"])
        assert job["status"] == "completed", job
        assert len(job["run_ids"]) == 1
        state["run_id"] = job["run_ids"][0]
        results = http("/api/v1/benchmarks/runs/" + state["run_id"] + "/results")
        assert len(results["results"]) == 8
        assert all(row["status"] == "ok" for row in results["results"])
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(json.dumps(state), encoding="utf-8")
        print(
            "PASS: health, SPA refresh, Swagger, Exact, IVF training, int64 IDs, "
            "saves, published reports, real four-engine quick job"
        )
    else:
        state = json.loads(args.state_file.read_text(encoding="utf-8"))
        assert http("/api/v1/indexes") == [], (
            "Run after recreating the backend, before manually loading indexes"
        )
        for saved in state["archives"]:
            loaded = http("/api/v1/indexes/load", "POST", {"archive_id": saved["archive_id"]})
            matches = http("/api/v1/indexes/" + loaded["id"] + "/search", "POST", saved["query"])[
                "matches"
            ]
            assert matches == saved["matches"]
        report = http("/api/v1/benchmarks/runs/" + state["run_id"] + "/results")
        assert report["run"]["complete"] and len(report["results"]) == 8
        assert any(r["source"] == "published" for r in http("/api/v1/benchmarks/runs")["runs"])
        print(
            "PASS: ephemeral active catalog, persistent Exact/IVF archives, "
            "identical restored scores, "
            "persistent generated and published reports"
        )


if __name__ == "__main__":
    main()
