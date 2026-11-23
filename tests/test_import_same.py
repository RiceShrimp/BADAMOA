"""A new version of a dataset whose files the project holds already: the node asks which are unchanged (the same
path in the folder, size and modification time as when an earlier import read them) and does not read those again;
"모든 파일 다시 읽기" (full_read) reads every file all the same."""

from __future__ import annotations

INFO = {"hostname": "data-pc", "os": "linux", "cpu_count": 4, "memory_mb": 8192, "gpus": []}
T = 1_700_000_000_000_000_000  # a modification time, ns


def _node(client, admin):
    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    reg = client.post("/api/agent/register", json={"join_token": token, "agent_version": "0.95.0", "info": INFO}).json()
    node = {"Authorization": f"Bearer {reg['node_token']}"}
    client.post("/api/agent/heartbeat", json={"agent_version": "0.95.0", "metrics": {}}, headers=node)
    return reg["node_id"], node


def _item(n, mtime_ns=0):
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
        "mtime_ns": mtime_ns,
    }


def _start(client, admin, node, project_id, node_id, version, **extra):
    body = {"node_id": node_id, "path": "/data/street", "dataset": "street", "version": version, **extra}
    assert client.post(f"/api/projects/{project_id}/imports", json=body, headers=admin).status_code == 200
    job = client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).json()["job"]
    return job["id"], f"/api/agent/jobs/{job['id']}/import"


def _end(client, node, job_id, base):
    assert client.post(base + "/finish", json={}, headers=node).status_code == 200
    client.post(f"/api/agent/jobs/{job_id}/complete", json={"status": "succeeded", "result": {}}, headers=node)


def _same(client, node, base, files):
    resp = client.post(base + "/same", json={"files": files}, headers=node)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_unchanged_files_are_found_by_path_size_and_time(client, admin):
    node_id, node = _node(client, admin)
    project = client.post("/api/projects", json={"name": "같은 파일"}, headers=admin).json()

    # ---- the first version: nothing to know yet, every file is read (two with their time, one from an older node)
    job, base = _start(client, admin, node, project["id"], node_id, "v1")
    assert _same(client, node, base, []) == {"found": {}, "may_know": False}
    client.post(base + "/begin", json={"classes": ["car"], "total": 3, "format": "yolo"}, headers=node)
    items = [_item(1, T + 1), _item(2, T + 2), _item(3)]
    assert client.post(base + "/items", json={"items": items}, headers=node).status_code == 200
    _end(client, node, job, base)

    # ---- the next: only the file with the same path, size and time is known unchanged
    job, base = _start(client, admin, node, project["id"], node_id, "v2")
    asked = [
        ["images/train/1.jpg", 101, T + 1],  # unchanged
        ["images/train/2.jpg", 102, T + 99],  # touched since
        ["images/train/3.jpg", 103, T + 3],  # registered without its time
        ["images/train/9.jpg", 109, T + 9],  # new
        ["images/val/1.jpg", 101, T + 1],  # the same file somewhere else in the folder
    ]
    assert _same(client, node, base, asked) == {"found": {"images/train/1.jpg": [f"{1:064x}", 64, 48]}, "may_know": True}
    client.post(base + "/begin", json={"classes": ["car"], "total": 3, "format": "yolo"}, headers=node)
    # the unchanged one sent as the Hub has it (no thumbnail), the one without a time read again: it learns it
    sent = [{**_item(1, T + 1), "thumb_b64": None}, _item(3, T + 3)]
    assert client.post(base + "/items", json={"items": sent}, headers=node).status_code == 200
    _end(client, node, job, base)
    releases = client.get(f"/api/projects/{project['id']}/datasets", headers=admin).json()[0]["releases"]
    counts = {r["version"]: client.get(f"/api/releases/{r['id']}", headers=admin).json()["stats"]["items"] for r in releases}
    assert counts == {"v1": 3, "v2": 2}

    job, base = _start(client, admin, node, project["id"], node_id, "v3")
    assert set(_same(client, node, base, asked)["found"]) == {"images/train/1.jpg", "images/train/3.jpg"}
    client.post(f"/api/agent/jobs/{job}/complete", json={"status": "failed", "result": {}}, headers=node)

    # ---- "모든 파일 다시 읽기": the node is told it may know none
    job, base = _start(client, admin, node, project["id"], node_id, "v4", full_read=True)
    assert _same(client, node, base, asked) == {"found": {}, "may_know": False}


def test_only_the_importing_node_may_ask(client, admin):
    node_id, node = _node(client, admin)
    _, other = _node(client, admin)
    project = client.post("/api/projects", json={"name": "남의 작업"}, headers=admin).json()
    _, base = _start(client, admin, node, project["id"], node_id, "v1")
    assert client.post(base + "/same", json={"files": []}, headers=other).status_code == 404
