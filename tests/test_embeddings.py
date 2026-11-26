"""Image vectors for 이미지 탐색, made with a training on the server it trains on (``model.embed``): only the images
without one yet (a version's new ones), none when all have one, never on a server without the 임베딩용 role.

Uses the ``dummy`` embedder (a few numbers of the pixels, the node's own Python) and a real Node Agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sample_data import make_yolo
from test_training import _logs, _wait, hub, release  # noqa: F401 (fixtures)
from test_transfer import two_nodes  # noqa: F401 (fixture)

from badamoa.hub import embeddings


@pytest.fixture(autouse=True)
def _dummy_model(monkeypatch):
    monkeypatch.setattr(embeddings, "DEFAULT_MODEL", "dummy")


def _coverage(client, release_id):
    return client.get(f"/api/releases/{release_id}/embeddings", params={"model": "dummy"}).json()


def _train(client, release_id, **extra):
    body = {"release_id": release_id, "template": "dummy", "params": {"epochs": 1}, "gpus": 0, **extra}
    resp = client.post("/api/runs", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _embedded(client, run):
    made = (client.get(f"/api/runs/{run['id']}").json().get("config") or {}).get("embed") or {}
    if made.get("job_id"):
        job = _wait(client, made["job_id"])
        assert job["status"] == "succeeded", _logs(client, job["id"])
        return made, job
    return made, None


def test_vectors_come_with_a_training_once_per_image(hub, release, tmp_path):  # noqa: F811
    assert _coverage(hub, release["id"]) == {
        **_coverage(hub, release["id"]),
        "total": 8,
        "done": 0,
        "todo": 8,
    }
    # a training that also makes them: on the server it trains on, from the images it has
    made, job = _embedded(hub, _train(hub, release["id"], embed=True))
    assert made["model"] == "dummy" and job is not None
    assert job["type"] == "model.embed" and job["result"]["embedded"] == 8
    assert _coverage(hub, release["id"])["done"] == 8
    # again: every image has one, nothing to do
    again, none = _embedded(hub, _train(hub, release["id"], embed=True))
    assert again["skipped"] and none is None

    # the next version: only the images it added (the first six are the same pictures)
    folder = make_yolo(tmp_path / "more", train=10)
    body = {"node_id": hub.node_id, "path": str(folder), "dataset": "street", "version": "v2"}
    imported = _wait(hub, hub.post(f"/api/projects/{release['project_id']}/imports", json=body).json()["id"])
    assert imported["status"] == "succeeded", _logs(hub, imported["id"])
    v2 = imported["result"]["release_id"]
    assert {k: _coverage(hub, v2)[k] for k in ("total", "done", "todo")} == {"total": 12, "done": 6, "todo": 6}
    _, job = _embedded(hub, _train(hub, v2, embed=True))
    assert job["result"]["embedded"] == 6 and _coverage(hub, v2)["todo"] == 0
    # each is a record of the version, made alongside its training
    records = hub.get("/api/embeddings/jobs", params={"release_id": v2}).json()
    assert [(r["id"], r["via"], r["images"], r["made"]) for r in records] == [(job["id"], "training", 6, 6)]
    assert records[0]["run_id"] and records[0]["release"]["version"] == "v2"


def test_a_server_without_the_role_makes_none(hub, release):  # noqa: F811
    try:
        assert hub.patch(f"/api/nodes/{hub.node_id}", json={"roles": ["compute", "storage"]}).status_code == 200
        refused = hub.post(
            "/api/runs",
            json={"release_id": release["id"], "template": "dummy", "params": {"epochs": 1}, "gpus": 0, "embed": True},
        )
        assert refused.status_code == 400 and "임베딩용" in refused.json()["detail"]
        # the training alone still starts
        run = _train(hub, release["id"])
        assert "embed" not in run["config"]
    finally:
        hub.patch(f"/api/nodes/{hub.node_id}", json={"roles": ["compute", "autolabel", "inference", "storage", "embedding"]})


def test_explore_finds_the_nearest_images(hub, release):  # noqa: F811
    _embedded(hub, _train(hub, release["id"], embed=True))
    project = release["project_id"]
    status = hub.get(f"/api/projects/{project}/explore").json()
    assert status["images"] == 8 and status["embedded"] == 8
    assert not status["text"]["offered"] and not status["text"]["ready"]  # the dummy has no text encoder
    first = hub.get(f"/api/releases/{release['id']}/items").json()["items"][0]["id"]
    found = hub.get(f"/api/projects/{project}/explore/search", params={"like": first}).json()
    assert found["searched"] == 8 and len(found["items"]) == 7 and first not in [i["id"] for i in found["items"]]
    scores = [i["score"] for i in found["items"]]
    assert scores == sorted(scores, reverse=True) and all(-1.01 <= s <= 1.01 for s in scores)
    assert found["items"][0]["release"]["id"] == release["id"] and found["items"][0]["thumb_url"]
    # one version only, and words before the text encoder came
    one = hub.get(f"/api/projects/{project}/explore/search", params={"like": first, "release_id": release["id"]})
    assert one.status_code == 200 and len(one.json()["items"]) == 7
    words = hub.get(f"/api/projects/{project}/explore/search", params={"text": "a cat"})
    assert words.status_code == 409 and "글 검색" in words.json()["detail"]
    # opened from the version screen: the image says it has a vector, and the search names the image looked like
    detail = hub.get(f"/api/releases/{release['id']}/items/{first}").json()
    assert detail["embedded"] and detail["project_id"] == project
    assert found["anchor"]["id"] == first and found["anchor"]["thumb_url"]
    # only the images at least this near, and whether more are left beyond the limit
    middle = scores[3]
    some = hub.get(f"/api/projects/{project}/explore/search", params={"like": first, "min_score": middle}).json()
    assert [i["score"] for i in some["items"]] == [s for s in scores if s >= middle] and not some["more"]
    assert some["min_score"] == middle
    two = hub.get(f"/api/projects/{project}/explore/search", params={"like": first, "limit": 2}).json()
    assert len(two["items"]) == 2 and two["more"]
    # a result opens as on the version screen: the version's list of just the results, in their order
    order = [found["items"][2]["id"], found["items"][0]["id"], found["items"][4]["id"]]
    url = f"/api/releases/{release['id']}/items"
    page = hub.get(url, params={"ids": ",".join(map(str, order)), "offset": 1, "limit": 1}).json()
    assert page["total"] == 3 and [i["id"] for i in page["items"]] == [order[1]]
    every = hub.get(url, params={"ids": ",".join(map(str, order))}).json()["items"]
    assert [i["id"] for i in every] == order
    # the results as a file: the nearest first, with their scores and where each image is
    listed = hub.get(f"/api/projects/{project}/explore/search/list", params={"like": first, "format": "csv"})
    assert listed.status_code == 200 and listed.headers["x-images"] == "7"
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(listed.content.decode("utf-8-sig"))))
    assert [int(r["item_id"]) for r in rows] == [i["id"] for i in found["items"]]
    assert [r["rank"] for r in rows] == [str(n) for n in range(1, 8)] and float(rows[0]["score"]) == scores[0]
    assert rows[0]["version"] == found["items"][0]["release"]["version"] and rows[0]["split"] and rows[0]["path"]
    paths = hub.get(f"/api/projects/{project}/explore/search/list", params={"like": first, "format": "txt", "limit": 2})
    assert paths.text.splitlines() == [r["path"] for r in rows[:2]]

    # an image put in: kept by its content, found like any image (the same picture as one of the dataset's: it first)
    assert status["image"] == {"offered": True, "ready": True, "problem": ""}  # the dummy: the Hub makes it itself
    picture = Path(rows[3]["path"]).read_bytes()
    put = hub.post(f"/api/projects/{project}/explore/query-image", content=picture)
    assert put.status_code == 200, put.text
    image = put.json()["id"]
    assert put.json()["width"] > 0 and len(image) == 64
    by_image = hub.get(f"/api/projects/{project}/explore/search", params={"image": image}).json()
    assert by_image["items"][0]["id"] == int(rows[3]["item_id"]) and by_image["items"][0]["score"] > 0.999
    assert len(by_image["items"]) == 8 and by_image["anchor"] is None  # nothing left out: it is none of them
    listed = hub.get(f"/api/projects/{project}/explore/search/list", params={"image": image, "format": "txt"})
    assert listed.text.splitlines()[0] == rows[3]["path"]
    assert hub.post(f"/api/projects/{project}/explore/query-image", content=b"not an image").status_code == 400
    missing = hub.get(f"/api/projects/{project}/explore/search", params={"image": "0" * 64})
    assert missing.status_code == 404 and "다시 올려" in missing.json()["detail"]


def test_vectors_without_a_training_and_as_a_pipeline_step(hub, release, tmp_path):  # noqa: F811
    import time

    places = hub.get(f"/api/releases/{release['id']}/embeddings/places").json()
    here = next(p for p in places["nodes"] if p["id"] == hub.node_id)
    assert here["fits"] and not here["transfer"] and places["default"] == hub.node_id
    made = hub.post(f"/api/releases/{release['id']}/embeddings", json={"gpus": 0})
    assert made.status_code == 200, made.text
    job = _wait(hub, made.json()["job_id"])
    assert job["status"] == "succeeded" and job["result"]["embedded"] == 8, _logs(hub, job["id"])
    again = hub.post(f"/api/releases/{release['id']}/embeddings", json={})
    assert again.status_code == 409 and "모두 벡터가 있습니다" in again.json()["detail"]
    rows = hub.get(f"/api/projects/{release['project_id']}/explore/versions").json()
    assert [(r["release"]["id"], r["done"], r["todo"]) for r in rows] == [(release["id"], 8, 0)]
    # 임베딩 작업 기록: the project's and the version's, its page, and 이미지 탐색's board
    records = hub.get("/api/embeddings/jobs", params={"project_id": release["project_id"]}).json()
    assert [r["id"] for r in records] == [job["id"]]
    record = records[0]
    assert record["status"] == "succeeded" and record["via"] == "ui" and record["gpus"] == 0
    assert record["images"] == 8 and record["made"] == 8 and record["node_name"] and record["by"]
    assert hub.get("/api/embeddings/jobs", params={"release_id": release["id"]}).json()[0]["id"] == job["id"]
    page = hub.get(f"/api/embeddings/jobs/{job['id']}").json()
    assert page["coverage"] == {"total": 8, "done": 8, "todo": 0} and page["project"]["id"] == release["project_id"]
    assert hub.get("/api/embeddings/jobs/nope").status_code == 404
    board = hub.get("/api/explore/board").json()
    card = next(c for c in board["projects"] if c["project"]["id"] == release["project_id"])
    assert (card["images"], card["embedded"], card["jobs"], card["going"]) == (8, 8, 1, 0)
    assert card["recent"][0]["id"] == job["id"] and board["going"] == [] and board["totals"]["embedded"] >= 8

    # a pipeline: 가져오기 → 임베딩, only the images the new version added
    folder = make_yolo(tmp_path / "more", train=10)
    steps = [
        {"kind": "import", "config": {"node_id": hub.node_id, "path": str(folder)}},
        {"kind": "embed", "config": {"gpus": 0}},
    ]
    started = hub.post("/api/pipeline-runs", json={"release_id": release["id"], "steps": steps})
    assert started.status_code == 200, started.text
    deadline = time.monotonic() + 120
    view = started.json()
    while time.monotonic() < deadline and view["status"] == "running":
        time.sleep(0.3)
        view = hub.get(f"/api/pipeline-runs/{view['id']}").json()
    assert view["status"] == "succeeded", [(s["kind"], s["status"], s["message"]) for s in view["steps"]]
    assert view["steps"][1]["label"] == "임베딩" and view["steps"][1]["message"] == "벡터 6개 만듦"
    # a server without the role is refused for the step, as everywhere
    try:
        hub.patch(f"/api/nodes/{hub.node_id}", json={"roles": ["compute", "storage"]})
        step = {"kind": "embed", "config": {"node_id": hub.node_id}}
        refused = hub.post("/api/pipelines", json={"release_id": release["id"], "steps": [step]})
        assert refused.status_code == 400 and "임베딩용" in refused.json()["detail"]
    finally:
        hub.patch(f"/api/nodes/{hub.node_id}", json={"roles": ["compute", "autolabel", "inference", "storage", "embedding"]})


def test_the_vector_db_moves_to_another_folder(hub, release, tmp_path):  # noqa: F811
    import time

    _embedded(hub, _train(hub, release["id"], embed=True))
    before = hub.get("/api/vector-store").json()
    assert before["path"] == before["default"] and sum(t["rows"] for t in before["tables"]) == 8
    target = tmp_path / "vectors-elsewhere"
    assert hub.post("/api/vector-store/move", json={"path": "relative/folder"}).status_code == 400
    moved = hub.post("/api/vector-store/move", json={"path": str(target)})
    assert moved.status_code == 200, moved.text
    deadline = time.monotonic() + 60
    view = moved.json()
    while time.monotonic() < deadline and view["moving"]["state"] == "moving":
        time.sleep(0.2)
        view = hub.get("/api/vector-store").json()
    assert view["moving"]["state"] == "done" and view["path"] == str(target), view
    assert sum(t["rows"] for t in view["tables"]) == 8 and any(target.iterdir())
    # found there as before
    first = hub.get(f"/api/releases/{release['id']}/items").json()["items"][0]["id"]
    found = hub.get(f"/api/projects/{release['project_id']}/explore/search", params={"like": first}).json()
    assert len(found["items"]) == 7


def test_an_embedding_on_another_server_removes_the_copy_it_received_if_asked(two_nodes, tmp_path):  # noqa: F811
    """As 새 학습: the server picked receives the version's images first, and once the vectors are made the copy goes
    (지우기) or stays (남겨 두기); the images where they were registered always stay."""
    import time

    client = two_nodes
    data, worker = client.data.config.node_id, client.train.config.node_id
    client.patch(f"/api/nodes/{worker}", json={"roles": ["compute", "embedding"]})
    project = client.post("/api/projects", json={"name": "어디서 임베딩"}).json()
    body = {"node_id": data, "path": str(make_yolo(tmp_path / "ds")), "dataset": "street", "version": "v1"}
    imported = _wait(client, client.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    assert imported["status"] == "succeeded", _logs(client, imported["id"])
    version = imported["result"]["release_id"]

    places = client.get(f"/api/releases/{version}/embeddings/places").json()
    by = {p["id"]: p for p in places["nodes"]}
    assert by[worker]["fits"] and by[worker]["transfer"] and by[worker]["images"] == ""
    assert set(by[worker]["cache"]) == {"dir", "free_gb", "auto"} and places["items"] == 8 and places["bytes"] > 0
    url = f"/api/releases/{version}/embeddings"
    assert client.post(url, json={"node_id": worker, "gpus": 0, "data_after": "later"}).status_code == 400
    refused = client.post(url, json={"node_id": worker, "gpu_indices": [9]})
    assert refused.status_code == 400 and "GPU 9" in refused.json()["detail"]

    made = client.post(url, json={"node_id": worker, "gpus": 0, "data_after": "delete"})
    assert made.status_code == 200, made.text
    job = _wait(client, made.json()["job_id"])
    assert job["status"] == "succeeded" and job["node_id"] == worker, _logs(client, job["id"])
    record = client.get(f"/api/embeddings/jobs/{job['id']}").json()
    assert record["data_after"] == "delete" and record["sent"]["status"] == "succeeded"
    deadline = time.monotonic() + 60
    copies = client.get(f"/api/releases/{version}/copies").json()
    while time.monotonic() < deadline and copies["copies"]:
        time.sleep(0.3)
        copies = client.get(f"/api/releases/{version}/copies").json()
    assert copies["copies"] == [] and copies["origin"]["node_id"] == data  # the registered images stay
