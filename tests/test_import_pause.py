"""A dataset import paused keeps what it registered; going on with it reads only the files not in it yet, into the
same version; a paused import thrown away goes as a stopped one does."""

from __future__ import annotations

INFO = {"hostname": "data-pc", "os": "linux", "cpu_count": 4, "memory_mb": 8192, "gpus": []}


def _node(client, admin):
    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    reg = client.post("/api/agent/register", json={"join_token": token, "agent_version": "0.93.0", "info": INFO}).json()
    node = {"Authorization": f"Bearer {reg['node_token']}"}
    client.post("/api/agent/heartbeat", json={"agent_version": "0.93.0", "metrics": {}}, headers=node)
    return reg["node_id"], node


def _item(n):
    return {
        "sha256": f"{n:064x}",
        "file_name": f"{n}.jpg",
        "rel_path": f"images/train/{n}.jpg",
        "width": 64,
        "height": 48,
        "bytes": 100 + n,
        "split": "train",
        "state": "labelled",
        "boxes": [[0, 0.5, 0.5, 0.2, 0.2]],
    }


def _claim(client, node):
    job = client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).json()["job"]
    assert job is not None
    return job["id"]


def test_a_paused_import_goes_on_from_where_it_stopped(client, admin):
    node_id, node = _node(client, admin)
    project = client.post("/api/projects", json={"name": "등록"}, headers=admin).json()
    body = {"node_id": node_id, "path": "/data/big", "dataset": "big", "version": "v1"}
    started = client.post(f"/api/projects/{project['id']}/imports", json=body, headers=admin)
    assert started.status_code == 200, started.text
    first = _claim(client, node)
    base = f"/api/agent/jobs/{first}/import"
    begin = {"classes": ["car"], "total": 4, "format": "yolo"}
    release_id = client.post(base + "/begin", json=begin, headers=node).json()["release_id"]
    assert client.post(base + "/items", json={"items": [_item(1), _item(2)]}, headers=node).status_code == 200

    # ---- paused: the job stops on its server, the version and its two images stay
    assert client.post(f"/api/imports/{first}/pause", headers=admin).json() == {"job_id": first, "kept": True}
    assert client.post(f"/api/agent/jobs/{first}/logs", json={"lines": []}, headers=node).json()["cancel_requested"]
    client.post(f"/api/agent/jobs/{first}/complete", json={"status": "canceled", "result": {}}, headers=node)
    listed = {i["job_id"]: i for i in client.get("/api/imports", headers=admin).json()}
    assert (listed[first]["status"], listed[first]["done"], listed[first]["total"]) == ("paused", 2, 4)
    assert listed[first]["release_id"] == release_id

    # ---- gone on with: a new job on the same server, into the same version; it is told what it holds
    resumed = client.post(f"/api/imports/{first}/resume", headers=admin)
    assert resumed.status_code == 200, resumed.text
    again = resumed.json()["job_id"]
    assert client.post(f"/api/imports/{first}/resume", headers=admin).status_code == 409  # once
    # another import ending before the new job takes the version: the version stays (it used to be dropped)
    from badamoa.hub import catalog

    with client.app.state.hub.db.session() as db:
        assert catalog.drop_unfinished_imports(db, client.app.state.hub.settings) == 0
    assert _claim(client, node) == again
    base = f"/api/agent/jobs/{again}/import"
    began = client.post(base + "/begin", json={"classes": ["car"], "total": 4, "format": "yolo"}, headers=node).json()
    assert began == {"release_id": release_id, "resume": 2}
    known = client.get(base + "/known", headers=node).json()
    assert known == {"paths": ["images/train/1.jpg", "images/train/2.jpg"], "next": None}
    paged = client.get(base + "/known", params={"limit": 1}, headers=node).json()
    assert paged == {"paths": ["images/train/1.jpg"], "next": "images/train/1.jpg"}
    # the rest (and one it holds, from an older node that reads everything again: counted once)
    assert client.post(base + "/items", json={"items": [_item(2), _item(3), _item(4)]}, headers=node).status_code == 200
    assert client.post(base + "/finish", json={}, headers=node).status_code == 200
    client.post(f"/api/agent/jobs/{again}/complete", json={"status": "succeeded", "result": {}}, headers=node)
    release = client.get(f"/api/releases/{release_id}", headers=admin).json()
    assert release["status"] == "ready" and release["stats"]["items"] == 4
    assert first not in {i["job_id"] for i in client.get("/api/imports", headers=admin).json() if i["status"] == "paused"}


def test_a_paused_import_thrown_away_goes(client, admin):
    node_id, node = _node(client, admin)
    project = client.post("/api/projects", json={"name": "버림"}, headers=admin).json()
    body = {"node_id": node_id, "path": "/data/gone", "dataset": "gone", "version": "v1"}
    client.post(f"/api/projects/{project['id']}/imports", json=body, headers=admin)
    job = _claim(client, node)
    base = f"/api/agent/jobs/{job}/import"
    begin = {"classes": ["car"], "total": 2, "format": "yolo"}
    release_id = client.post(base + "/begin", json=begin, headers=node).json()["release_id"]
    client.post(base + "/items", json={"items": [_item(7)]}, headers=node)
    client.post(f"/api/imports/{job}/pause", headers=admin)
    client.post(f"/api/agent/jobs/{job}/complete", json={"status": "canceled", "result": {}}, headers=node)
    assert client.get(f"/api/releases/{release_id}", headers=admin).status_code == 200  # kept while paused
    assert client.post(f"/api/imports/{job}/discard", headers=admin).json()["dropped"] == 1
    assert client.get(f"/api/releases/{release_id}", headers=admin).status_code == 404
    assert client.post(f"/api/imports/{job}/resume", headers=admin).status_code == 409
    assert job not in {i["job_id"] for i in client.get("/api/imports", headers=admin).json() if i["status"] == "paused"}


def test_a_resumed_import_on_the_node_skips_the_files_it_holds(tmp_path):
    """The node reads the files the version holds page by page and leaves them out (paths as the Hub keeps them)."""
    import httpx

    from badamoa.node import importer

    pages = {
        "": {"paths": ["images/train/a.jpg"], "next": "images/train/a.jpg"},
        "images/train/a.jpg": {"paths": ["images/val/c.jpg"], "next": None},
    }

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[request.url.params.get("after", "")])

    with httpx.Client(base_url="http://hub", transport=httpx.MockTransport(answer)) as client:
        known = importer._known_paths(client, "/api/agent/jobs/j/import")
    assert known == {"images/train/a.jpg", "images/val/c.jpg"}
    root = tmp_path / "ds"
    files = [root / "images" / "train" / "a.jpg", root / "images" / "train" / "b.jpg", root / "images" / "val" / "c.jpg"]
    assert [importer._relative(f, root) for f in files if importer._relative(f, root) not in known] == ["images/train/b.jpg"]
