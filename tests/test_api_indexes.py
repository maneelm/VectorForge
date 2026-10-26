from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import UUID, uuid4

import pytest

from backend.schemas import IndexCreate
from backend.services.index_manager import IndexManager


def test_health_docs_catalog_lifecycle(client, index):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    UUID(index)
    info = client.get(f"/api/v1/indexes/{index}").json()
    assert info["is_trained"] and info["size"] == 0
    assert client.get("/api/v1/indexes").json() == [info]
    assert client.delete(f"/api/v1/indexes/{index}").status_code == 204
    assert client.get(f"/api/v1/indexes/{index}").status_code == 404
    assert client.get(f"/api/v1/indexes/{uuid4()}").json()["error"]["code"] == "index_not_found"


@pytest.mark.parametrize(
    "values",
    [
        {"dimension": 0},
        {"dimension": True},
        {"dimension": 2049},
        {"metric": "bad"},
        {"type": "ivf", "nlist": 2, "nprobe": 3},
        {"nprobe": 1},
        {"type": "ivf", "max_iter": 101},
        {"type": "ivf", "tol": -1},
        {"name": " "},
        {"type": "ivf", "nlist": 129},
        {"path": "../escape"},
    ],
)
def test_invalid_configuration(client, values):
    response = client.post("/api/v1/indexes", json={"name": "test", "dimension": 2} | values)
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


def test_ivf_lifecycle_and_override(client):
    info = client.post(
        "/api/v1/indexes",
        json={"name": "ivf", "type": "ivf", "dimension": 2, "nlist": 2, "nprobe": 1},
    ).json()
    assert not info["is_trained"]
    path = f"/api/v1/indexes/{info['id']}"
    assert client.post(path + "/search", json={"query": [1, 0]}).status_code == 409
    assert client.post(path + "/vectors", json={"vectors": [[1, 0]], "ids": [1]}).status_code == 409
    assert client.post(path + "/train", json={"vectors": [[1, 0]]}).status_code == 422
    assert client.post(
        path + "/train", json={"vectors": [[-10, 0], [-9, 0], [9, 0], [10, 0]]}
    ).json()["is_trained"]
    assert (
        client.post(
            path + "/vectors", json={"vectors": [[-10, 0], [10, 0]], "ids": [1, 2]}
        ).status_code
        == 200
    )
    assert client.post(path + "/train", json={"vectors": [[1, 0], [0, 1]]}).status_code == 409
    assert (
        len(
            client.post(path + "/search", json={"query": [1, 0], "k": 10, "nprobe": 2}).json()[
                "matches"
            ]
        )
        == 2
    )
    assert client.get(path).json()["configuration"]["nprobe"] == 1
    assert client.post(path + "/search", json={"query": [1, 0], "nprobe": 3}).status_code == 422
    client.request("DELETE", path + "/vectors", json={"ids": [1, 2]})
    assert client.get(path).json()["is_trained"]


def test_independent_locks_and_concurrent_mutation(api_settings):
    manager = IndexManager(api_settings)
    first = manager.create(IndexCreate(name="a", dimension=2)).id
    second = manager.create(IndexCreate(name="b", dimension=2)).id
    held, release = Event(), Event()

    def hold():
        with manager.access(first):
            held.set()
            assert release.wait(5)

    def independent():
        with manager.access(second) as entry:
            entry.engine.add([1, 0], 9)

    with ThreadPoolExecutor(max_workers=3) as pool:
        blocked = pool.submit(hold)
        assert held.wait(2)
        try:
            pool.submit(independent).result(timeout=2)
        finally:
            release.set()
        blocked.result()

        def add(i):
            with manager.access(first) as entry:
                entry.engine.add([i, 0], i)

        list(pool.map(add, range(30)))
    with manager.access(first) as entry:
        assert entry.engine.size == 30


def test_request_limits_cors_and_host(client):
    assert client.post("/api/v1/indexes", content=b" " * (4 * 1024 * 1024 + 1)).status_code == 413
    response = client.options(
        "/api/v1/indexes",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert (
        client.options(
            "/api/v1/indexes",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        ).status_code
        == 400
    )
    assert client.get("/api/v1/health", headers={"Host": "untrusted.example"}).status_code == 400
