import json
import zipfile
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.mark.parametrize("kind", ["exact", "ivf"])
def test_round_trip_restart(client, api_settings, kind):
    body = {"name": "saved", "dimension": 2, "type": kind}
    if kind == "ivf":
        body.update(nlist=2, nprobe=1)
    identifier = client.post("/api/v1/indexes", json=body).json()["id"]
    path = f"/api/v1/indexes/{identifier}"
    if kind == "ivf":
        client.post(path + "/train", json={"vectors": [[1, 0], [-1, 0]]})
    client.post(
        path + "/vectors",
        json={"vectors": [[1, 0], [-1, 0]], "ids": [str(-(2**63)), str(2**63 - 1)]},
    )
    query = {"query": [1, 0], "k": 5}
    if kind == "ivf":
        query["nprobe"] = 2
    expected = client.post(path + "/search", json=query).json()["matches"]
    saved = client.post(path + "/save").json()
    assert client.get("/api/v1/indexes/saved").json() == [saved]
    with TestClient(create_app(api_settings)) as restarted:
        assert restarted.get("/api/v1/indexes").json() == []
        loaded = restarted.post("/api/v1/indexes/load", json={"archive_id": saved["archive_id"]})
        assert loaded.status_code == 201 and loaded.json()["id"] != identifier
        assert (
            restarted.post(f"/api/v1/indexes/{loaded.json()['id']}/search", json=query).json()[
                "matches"
            ]
            == expected
        )


@pytest.mark.parametrize(
    "body",
    [{"path": "../../escape.npz"}, {"archive_id": "../escape"}, {"archive_id": str(uuid4())}],
)
def test_invalid_persistence_request(client, body):
    assert client.post("/api/v1/indexes/load", json=body).status_code in (404, 422)


def test_corrupt_archive_unique_names(client, index, api_settings):
    first = client.post(f"/api/v1/indexes/{index}/save").json()["archive_id"]
    second = client.post(f"/api/v1/indexes/{index}/save").json()["archive_id"]
    assert first != second
    (api_settings.data_dir / "indexes" / f"{first}.npz").write_bytes(b"corrupt")
    assert client.post("/api/v1/indexes/load", json={"archive_id": first}).status_code == 400


def test_untrained_save(client):
    identifier = client.post(
        "/api/v1/indexes", json={"name": "untrained", "dimension": 2, "type": "ivf"}
    ).json()["id"]
    assert client.post(f"/api/v1/indexes/{identifier}/save").status_code == 409


def test_archive_manifest_mismatch(client, index, api_settings):
    archive = client.post(f"/api/v1/indexes/{index}/save").json()["archive_id"]
    path = api_settings.data_dir / "indexes" / f"{archive}.json"
    payload = json.loads(path.read_text())
    payload["dimension"] = 3
    path.write_text(json.dumps(payload))
    assert client.post("/api/v1/indexes/load", json={"archive_id": archive}).status_code == 400


def test_expansion_limit(api_settings):
    settings = replace(api_settings, archive_bytes=4096)
    with TestClient(create_app(settings)) as client:
        identifier = client.post("/api/v1/indexes", json={"name": "test", "dimension": 1}).json()[
            "id"
        ]
        archive = client.post(f"/api/v1/indexes/{identifier}/save").json()["archive_id"]
        with zipfile.ZipFile(
            settings.data_dir / "indexes" / f"{archive}.npz", "w", compression=zipfile.ZIP_DEFLATED
        ) as handle:
            handle.writestr("vectors.npy", b"0" * 10000)
        response = client.post("/api/v1/indexes/load", json={"archive_id": archive})
        assert response.status_code == 400 and "Uncompressed" in response.json()["error"]["message"]
