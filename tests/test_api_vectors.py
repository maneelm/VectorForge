import pytest


@pytest.mark.parametrize(
    "metric,scores", [("cosine", [1, 0, -1]), ("euclidean", [0, 1, 2]), ("dot", [1, 0, -1])]
)
def test_raw_scores_and_order(client, metric, scores):
    identifier = client.post(
        "/api/v1/indexes", json={"name": "scores", "dimension": 2, "metric": metric}
    ).json()["id"]
    path = f"/api/v1/indexes/{identifier}"
    assert (
        client.post(
            path + "/vectors", json={"vectors": [[1, 0], [0, 0], [-1, 0]], "ids": [-1, "5", 9]}
        ).status_code
        == 200
    )
    result = client.post(path + "/search", json={"query": [1, 0], "k": 10}).json()
    assert [m["id"] for m in result["matches"]] == ["-1", "5", "9"]
    assert [m["score"] for m in result["matches"]] == pytest.approx(scores)
    assert [m["rank"] for m in result["matches"]] == [1, 2, 3]
    assert result["latency_ms"] >= 0 and "API-measured" in result["latency_scope"]
    assert result["nprobe"] is None
    assert client.post(path + "/search", json={"query": [1, 0], "nprobe": 1}).status_code == 422
    assert client.post(path + "/train", json={"vectors": [[1, 0]]}).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"vectors": [], "ids": []},
        {"vectors": [[1]], "ids": [1]},
        {"vectors": [[1, 0], [0, 1]], "ids": [1, 1]},
        {"vectors": [[1, 0]], "ids": [True]},
        {"vectors": [[1, 0]], "ids": [1.2]},
        {"vectors": [[1, 0]], "ids": [str(2**63)]},
        {"vectors": [[1, 0]], "ids": ["01"]},
        {"vectors": [[1, 0]], "ids": []},
        {"vectors": [[True, 0]], "ids": [1]},
    ],
)
def test_invalid_insert_atomic(client, index, body):
    path = f"/api/v1/indexes/{index}"
    assert client.post(path + "/vectors", json=body).status_code in (409, 422)
    assert client.get(path).json()["size"] == 0


def test_atomic_duplicates_delete_reuse_extreme_ids(client, index):
    path = f"/api/v1/indexes/{index}"
    ids = [str(-(2**63)), str(2**63 - 1)]
    assert (
        client.post(path + "/vectors", json={"vectors": [[1, 0], [1, 0]], "ids": ids}).status_code
        == 200
    )
    assert (
        client.post(
            path + "/vectors", json={"vectors": [[1, 0], [0, 1]], "ids": ["9", ids[0]]}
        ).status_code
        == 409
    )
    assert (
        client.post(path + "/search", json={"query": [1, 0], "k": 1}).json()["matches"][0]["id"]
        == ids[0]
    )
    assert (
        client.request("DELETE", path + "/vectors", json={"ids": [ids[0], ids[0], 999]}).json()[
            "size"
        ]
        == 1
    )
    assert (
        client.post(path + "/vectors", json={"vectors": [[1, 0]], "ids": [ids[0]]}).status_code
        == 200
    )


@pytest.mark.parametrize("query,k", [([1], 1), ([1, 0], 0), ([1, 0], True), ([[1, 0]], 1)])
def test_invalid_search(client, index, query, k):
    assert (
        client.post(f"/api/v1/indexes/{index}/search", json={"query": query, "k": k}).status_code
        == 422
    )


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_rejection(client, index, bad):
    response = client.post(
        f"/api/v1/indexes/{index}/vectors",
        content='{"vectors":[[' + bad + ',0]],"ids":[1]}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"
