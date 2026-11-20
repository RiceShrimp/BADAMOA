"""Training on a node that does not hold the images: they go straight from node to node, never through the Hub."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_yolo

from badamoa.hub import transfers
from badamoa.node import agent as node_agent
from badamoa.node import cache


def _wait(hub, job_id, timeout=180):
    deadline = time.monotonic() + timeout
    job = hub.get(f"/api/jobs/{job_id}").json()
    while time.monotonic() < deadline and job["status"] in ("queued", "running"):
        time.sleep(0.3)
        job = hub.get(f"/api/jobs/{job_id}").json()
    return job


def _logs(hub, job_id):
    return "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job_id}/logs").json())


@pytest.fixture
def two_nodes(live_hub, tmp_path, monkeypatch):
    """A node holding the images and another to train on, both with their data port on loopback."""
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    # Both run on this machine: other nodes reach a node at loopback here (a real one reports its LAN address).
    monkeypatch.setattr(transfers, "addresses", lambda node: ["127.0.0.1"])
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    runners = []
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        for name in ("data-node", "train-node"):
            join = client.post("/api/nodes/join-tokens", json={}).json()
            home = tmp_path / name
            home.mkdir()
            runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name=name), home)
            threading.Thread(target=runner.run_forever, daemon=True).start()
            runners.append(runner)
        client.data, client.train = runners
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:  # both have told the Hub their data port
            nodes = {n["id"]: n for n in client.get("/api/nodes").json()}
            if all((nodes.get(r.config.node_id, {}).get("metrics") or {}).get("serve", {}).get("port") for r in runners):
                break
            time.sleep(0.2)
        yield client
        for runner in runners:
            runner.stop()


def test_a_run_trains_on_a_node_that_receives_the_images_first(two_nodes, tmp_path, settings):
    hub = two_nodes
    data, train = hub.data.config.node_id, hub.train.config.node_id
    project = hub.post("/api/projects", json={"name": "옮겨서 학습"}).json()
    folder = make_yolo(tmp_path / "ds")
    body = {"node_id": data, "path": str(folder), "dataset": "street", "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    release = job["result"]["release_id"]

    # ---- what the new-run dialog offers: where the images are, and what has to move
    choices = hub.get(f"/api/releases/{release}/nodes").json()
    assert choices["items"] == 8 and choices["bytes"] > 0 and choices["source"] == data
    images = {n["id"]: n["images"] for n in choices["nodes"]}
    assert images == {data: "registered", train: ""}
    assert all(n["can_receive"] for n in choices["nodes"])

    # ---- a run allowed only on the other node: the images go there first, without a GPU held
    run = hub.post(
        "/api/runs",
        json={"release_id": release, "template": "dummy", "params": {"epochs": 1}, "gpus": 0, "nodes": [train]},
    ).json()
    assert run["config"]["nodes"] == [train]
    syncs = [j for j in hub.get("/api/jobs?limit=20").json() if j["type"] == "data.sync"]
    assert len(syncs) == 1 and syncs[0]["params"]["release_id"] == release and syncs[0]["gpus_requested"] == 0
    sync = _wait(hub, syncs[0]["id"])
    assert sync["status"] == "succeeded", _logs(hub, sync["id"])
    assert (sync["result"]["items"], sync["result"]["fetched"], sync["result"]["changed_count"]) == (8, 8, 0)
    assert "에 있는 데이터셋을 이 서버로 받습니다" in _logs(hub, sync["id"])
    done = _wait(hub, run["job_id"])
    assert done["status"] == "succeeded", _logs(hub, run["job_id"])
    view = hub.get(f"/api/runs/{run['id']}").json()
    assert view["node_name"] == "train-node" and view["export"]["items"]["train"] + view["export"]["items"]["val"] > 0
    root = cache.root(hub.train.home)
    kept = sorted(p.name for p in (root / "images").rglob("*.jpg"))
    assert len(kept) == 8  # by content: <sha256>.jpg
    # the Hub stored none of the images (they went node to node): no blob holds one
    stored = {p.name for p in settings.storage_dir.rglob("*") if p.is_file()}
    assert not stored & {Path(name).stem for name in kept}

    # ---- the node has them now: the next run starts there right away
    assert {n["id"]: n["images"] for n in hub.get(f"/api/releases/{release}/nodes").json()["nodes"]}[train] == "received"
    again = hub.post(
        "/api/runs",
        json={"release_id": release, "template": "dummy", "params": {"epochs": 1}, "gpus": 0, "nodes": [train]},
    ).json()
    assert len([j for j in hub.get("/api/jobs?limit=20").json() if j["type"] == "data.sync"]) == 1
    assert _wait(hub, again["job_id"])["status"] == "succeeded"

    # ---- an image gone from the cache and changed at the source: fetched again, refused, left out
    (lost,) = [p for p in (root / "images").rglob("*.jpg")][:1]
    lost.unlink()
    original = next(p for p in (folder / "images").rglob("*.jpg") if _sha(p) == lost.stem)
    original.write_bytes(original.read_bytes() + b"changed")
    third = hub.post(
        "/api/runs",
        json={"release_id": release, "template": "dummy", "params": {"epochs": 1}, "gpus": 0, "nodes": [train]},
    ).json()
    finished = _wait(hub, third["job_id"])
    assert finished["status"] == "succeeded", _logs(hub, third["job_id"])
    assert "바뀌었거나 없어진 이미지 1장" in _logs(hub, third["job_id"])
    assert hub.get(f"/api/runs/{third['id']}").json()["export"]["excluded"]["missing"] == 1


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_data_port_serves_nothing_without_a_grant(two_nodes):
    port = two_nodes.data._data.port
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/badamoa/ping", timeout=5) as resp:
        assert json.loads(resp.read())["node_id"] == two_nodes.data.config.node_id
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/badamoa/files",
        data=json.dumps({"grant": "made-up", "ids": [1, 2]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=10)
    assert refused.value.code == 403


def test_a_node_too_old_to_receive_is_refused_up_front(two_nodes, tmp_path, monkeypatch):
    hub = two_nodes
    data, train = hub.data.config.node_id, hub.train.config.node_id
    project = hub.post("/api/projects", json={"name": "옛 노드"}).json()
    body = {"node_id": data, "path": str(make_yolo(tmp_path / "ds")), "dataset": "street", "version": "v1"}
    release = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])["result"]["release_id"]
    monkeypatch.setattr(transfers, "SYNC_SINCE", (99, 0, 0))  # as if both nodes were from before transfers
    refused = hub.post(
        "/api/runs", json={"release_id": release, "template": "dummy", "params": {"epochs": 1}, "gpus": 0, "nodes": [train]}
    )
    assert refused.status_code == 400 and "99.0.0 이상" in refused.json()["detail"]
    assert not [j for j in hub.get("/api/jobs?limit=20").json() if j["type"] == "data.sync"]
    assert not hub.get(f"/api/releases/{release}/nodes").json()["nodes"][1]["can_receive"]


def test_a_model_is_evaluated_and_exported_on_a_server_picked(two_nodes, tmp_path):
    hub = two_nodes
    data, train = hub.data.config.node_id, hub.train.config.node_id
    project = hub.post("/api/projects", json={"name": "어디서 평가"}).json()
    body = {"node_id": data, "path": str(make_yolo(tmp_path / "ds")), "dataset": "street", "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    release = job["result"]["release_id"]
    run = hub.post("/api/runs", json={"release_id": release, "template": "dummy", "params": {"epochs": 1}, "gpus": 0}).json()
    assert _wait(hub, run["job_id"])["status"] == "succeeded", _logs(hub, run["job_id"])
    model = hub.get(f"/api/runs/{run['id']}").json()["model"]

    # ---- where an evaluation can run: where the images are (the default), or the other server, which receives them
    places = hub.get(f"/api/models/{model['id']}/places", params={"kind": "eval", "release_id": release}).json()
    assert places["default"] == data
    by = {p["id"]: p for p in places["nodes"]}
    assert by[data]["holds_images"] and not by[data]["transfer"] and by[data]["fits"]
    assert by[train]["transfer"] and by[train]["fits"], by[train]["why"]
    op = hub.post(
        f"/api/models/{model['id']}/eval", json={"release_id": release, "split": "val", "gpus": 0, "node_id": train}
    ).json()
    done = _wait(hub, op["job_id"])
    assert done["status"] == "succeeded", _logs(hub, op["job_id"])
    assert hub.get(f"/api/jobs/{op['job_id']}").json()["node_id"] == train
    syncs = [j for j in hub.get("/api/jobs?limit=20").json() if j["type"] == "data.sync"]
    assert len(syncs) == 1 and syncs[0]["status"] == "succeeded"

    # ---- export and test: the server that trained it by default, another one when picked
    places = hub.get(f"/api/models/{model['id']}/places", params={"kind": "export"}).json()
    assert places["default"] == data and {p["id"] for p in places["nodes"] if p["fits"]} == {data, train}
    assert next(p for p in places["nodes"] if p["id"] == data)["trained_here"]
    op = hub.post(f"/api/models/{model['id']}/export", json={"format": "onnx", "node_id": train}).json()
    assert _wait(hub, op["job_id"])["status"] == "succeeded", _logs(hub, op["job_id"])
    assert hub.get(f"/api/jobs/{op['job_id']}").json()["node_id"] == train
    assert hub.post(f"/api/models/{model['id']}/export", json={"format": "onnx", "node_id": "nope"}).status_code == 404
