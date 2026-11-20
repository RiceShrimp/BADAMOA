"""What BADAMOA keeps on a server: measured by the node, shown by the Hub with what each item is for,
removed on request, never what a queued or running job may use."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from conftest import ADMIN

from badamoa.hub import node_storage
from badamoa.hub.db import Database
from badamoa.hub.models import Job
from badamoa.node import agent as node_agent
from badamoa.node import containers, storage

ENV = {"requirements": ["ultralytics>=8.3.0"], "python": "3.12", "torch": True}
OTHER_ENV = {"requirements": ["timm"], "python": "3.11", "torch": False}


def _home(tmp_path):
    """A node home with one of everything BADAMOA keeps."""
    home = tmp_path / "node"
    for name, spec in (("env-a", ENV), ("env-b", OTHER_ENV)):
        folder = home / "envs" / name
        folder.mkdir(parents=True)
        (folder / "badamoa-env.json").write_text(json.dumps(spec), encoding="utf-8")
        (folder / "lib.bin").write_bytes(b"x" * 4000)
    (home / "cache" / "models").mkdir(parents=True)
    (home / "cache" / "models" / "blob1.pt").write_bytes(b"w" * 3000)
    (home / "cache" / "models" / "blob2.pt").write_bytes(b"w" * 1000)
    (home / "cache" / "templates" / "sha-a").mkdir(parents=True)
    (home / "cache" / "templates" / "sha-a" / "badamoa.toml").write_text('name = "a"', encoding="utf-8")
    (home / "cache" / "template-data" / "yolo").mkdir(parents=True)
    (home / "cache" / "template-data" / "yolo" / "yolo11n.pt").write_bytes(b"p" * 2000)
    for job in ("job-old", "job-now"):
        (home / "runs" / job / "out").mkdir(parents=True)
        (home / "runs" / job / "out" / "last.pt").write_bytes(b"c" * 500)
    return home


@pytest.fixture
def no_uv_no_docker(monkeypatch):
    monkeypatch.setattr(storage, "uv_cache_dir", lambda: None)
    monkeypatch.setattr(containers, "docker", lambda: None)


def test_a_node_measures_what_it_keeps_and_removes_items(tmp_path, no_uv_no_docker):
    home = _home(tmp_path)
    report = storage.measure(home, home / "cache")
    kinds = report["kinds"]
    assert [e["id"] for e in kinds["envs"]["items"]] == ["env-a", "env-b"]
    first = kinds["envs"]["items"][0]
    assert first["requirements"] == ["ultralytics>=8.3.0"] and first["torch"] is True and first["bytes"] > 4000
    assert [(m["id"], m["blob"], m["bytes"]) for m in kinds["models"]["items"]] == [
        ("blob1.pt", "blob1", 3000),
        ("blob2.pt", "blob2", 1000),
    ]
    assert [t["id"] for t in kinds["templates"]["items"]] == ["sha-a"]
    assert [t["id"] for t in kinds["template_data"]["items"]] == ["yolo"]
    assert sorted(r["id"] for r in kinds["runs"]["items"]) == ["job-now", "job-old"]
    assert "uv" not in kinds and kinds["containers"]["items"] == []
    assert all(d["free"] for d in report["disks"].values())
    # every folder's disk: the mount point it is on (shown with the folder on the node page)
    assert str(home / "envs") in report["disks"] and all(d["mount"] for d in report["disks"].values())

    # an environment being built stays; a running job's folder stays; a name is never a path
    (home / "envs" / "env-b.lock").mkdir()
    done = storage.clean(home, "envs", ["env-a", "env-b", "../runs"], log=lambda _: None)
    assert done["removed"] == 1 and not (home / "envs" / "env-a").exists() and (home / "envs" / "env-b").exists()
    assert (home / "runs").is_dir()
    done = storage.clean(home, "runs", [], everything=True, log=lambda _: None, running={"job-now"})
    assert done["removed"] == 1 and (home / "runs" / "job-now").exists() and not (home / "runs" / "job-old").exists()
    with pytest.raises(RuntimeError):
        storage.clean(home, "images", ["x"], log=lambda _: None)


def _join(client, admin, version="0.50.0"):
    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    reg = client.post("/api/agent/register", json={"join_token": token, "agent_version": version, "info": {}}).json()
    return reg["node_id"], {"Authorization": f"Bearer {reg['node_token']}"}


def _beat(client, node, report, version="0.50.0"):
    body = {"agent_version": version, "metrics": {}, "running_job_ids": [], "storage": report}
    assert client.post("/api/agent/heartbeat", json=body, headers=node).status_code == 200


def test_the_hub_shows_what_jobs_use_and_never_removes_it(client, admin, settings, tmp_path, no_uv_no_docker):
    node_id, node = _join(client, admin)
    home = _home(tmp_path)
    _beat(client, node, storage.measure(home, home / "cache"))

    # a training queued for this server: its environment, code, weights to start from and model type's cache
    database = Database(settings.database_url)
    with database.session() as db:
        job = Job(
            type="train.run",
            params={
                "environment": {"kind": "auto", "name": "자동 설치", **ENV},
                "template": {"name": "yolo", "sha256": "sha-a"},
                "init_weights": {"blob": "blob1", "file_name": "best.pt"},
                "resume": {"job_id": "job-old"},
            },
            target_node_id=node_id,
        )
        db.add(job)
        db.flush()
        queued = job.id
    view = client.get(f"/api/nodes/{node_id}/storage", headers=admin).json()
    assert view["measured_at"] and view["can_clean"] is True and view["bytes"] > 0
    kinds = {k["kind"]: k for k in view["kinds"]}
    # where each kind is kept: its folder and disk; received images in a folder the node picked (no admin setting)
    assert kinds["envs"]["path"].endswith("envs") and kinds["envs"]["mount"] and view["home_mount"]
    assert kinds["images"]["auto"] is True
    used = {k: {(i.get("image") or i["id"]): i["in_use"] for i in kinds[k]["items"]} for k in kinds if "items" in kinds[k]}
    assert used["envs"] == {"env-a": queued, "env-b": ""}
    assert used["models"] == {"blob1.pt": queued, "blob2.pt": ""}
    assert used["templates"] == {"sha-a": queued} and used["template_data"] == {"yolo": queued}
    assert used["runs"] == {"job-old": queued, "job-now": ""}
    assert next(i for i in kinds["envs"]["items"] if i["id"] == "env-a")["label"] == "Python 3.12 · PyTorch"
    assert kinds["models"]["items"][0]["label"] == "Hub에 없는 가중치"

    def clean(body):
        return client.post(f"/api/nodes/{node_id}/storage/clean", json=body, headers=admin)

    assert clean({"kind": "envs", "items": ["env-a"]}).status_code == 409  # the queued job uses it
    assert clean({"kind": "images", "all": True}).status_code == 400  # version by version, elsewhere
    assert clean({"kind": "envs", "items": ["nowhere"]}).status_code == 400  # not in the last report
    resp = clean({"kind": "envs", "all": True})  # every one no job uses
    assert resp.status_code == 200, resp.text
    first = resp.json()
    with database.session() as db:
        made = db.get(Job, first["job_id"])
        assert made.type == "node.clean" and made.params == {"kind": "envs", "items": ["env-b"]}
        assert made.target_node_id == node_id
    assert first["cleaning"] == [{"job_id": first["job_id"], "kind": "envs"}]
    assert clean({"kind": "runs", "items": ["job-now"]}).status_code == 409  # one clean at a time

    # it ran: the Hub notes what it removed, the node's next report shows the rest
    with database.session() as db:
        made = db.get(Job, first["job_id"])
        made.status, made.node_id, made.result = "succeeded", node_id, {"removed": 1, "bytes": 4096}
        node_storage.on_job_finished(db, made)
    view = client.get(f"/api/nodes/{node_id}/storage", headers=admin).json()
    assert view["last_clean"]["ok"] and view["last_clean"]["removed"] == 1 and view["cleaning"] == []
    _beat(client, node, storage.measure(home, home / "cache"))
    assert client.get(f"/api/nodes/{node_id}/storage", headers=admin).json()["last_clean"]["kind"] == "envs"

    # members see it but do not remove; an older node cannot clean
    old_id, old = _join(client, admin, version="0.47.0")
    _beat(client, old, storage.measure(home, home / "cache"), version="0.47.0")
    assert client.get(f"/api/nodes/{old_id}/storage", headers=admin).json()["can_clean"] is False
    assert (
        client.post(f"/api/nodes/{old_id}/storage/clean", json={"kind": "runs", "all": True}, headers=admin).status_code
        == 400
    )
    database.engine.dispose()


def test_a_real_node_reports_and_cleans(live_hub, tmp_path, monkeypatch, no_uv_no_docker):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as hub:
        join = hub.post("/api/nodes/join-tokens", json={}).json()
        home = _home(tmp_path)
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="gpu-1"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        node_id = runner.config.node_id
        try:

            def view():
                return hub.get(f"/api/nodes/{node_id}/storage").json()

            def until(check, timeout=60):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    found = check()
                    if found:
                        return found
                    time.sleep(0.3)
                return check()

            first = until(lambda: (v := view())["measured_at"] and v)
            runs = next(k for k in first["kinds"] if k["kind"] == "runs")
            assert sorted(i["id"] for i in runs["items"]) == ["job-now", "job-old"]

            resp = hub.post(f"/api/nodes/{node_id}/storage/clean", json={"kind": "runs", "items": ["job-old"]})
            assert resp.status_code == 200, resp.text
            done = until(lambda: (v := view())["last_clean"] and not v["cleaning"] and v)
            assert done["last_clean"]["ok"] and done["last_clean"]["removed"] == 1, done["last_clean"]
            assert not (home / "runs" / "job-old").exists() and (home / "runs" / "job-now").exists()

            def remeasured():
                v = view()
                items = next(k for k in v["kinds"] if k["kind"] == "runs")["items"]
                return [i["id"] for i in items] == ["job-now"] and v

            assert until(remeasured)
        finally:
            runner.stop()

