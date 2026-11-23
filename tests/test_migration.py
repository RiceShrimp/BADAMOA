"""Moving the Hub to another PC with one file: export (the Hub turns read-only), import on the new PC, hand the
nodes over; a node follows the new address once that Hub knows it."""

from __future__ import annotations

import time

import httpx
import pytest
from conftest import ADMIN
from fastapi.testclient import TestClient

from badamoa.hub import migration, services
from badamoa.hub.app import create_app
from badamoa.hub.config import HubSettings
from badamoa.hub.models import HubSetting, Node


def _wait_exported(client, admin, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = client.get("/api/migration", headers=admin).json()
        if state["phase"] in ("exported", "failed"):
            return state
        time.sleep(0.2)
    raise AssertionError("export did not finish")


def _node(client, admin) -> tuple[str, str]:
    join = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    made = client.post(
        "/api/agent/register",
        json={"join_token": join, "agent_version": "0.78.0", "info": {"hostname": "box", "os": "linux"}},
    ).json()
    return made["node_id"], made["node_token"]


def test_a_hub_moves_to_another_pc_with_one_file(client, admin, settings, tmp_path, monkeypatch):
    hub = client.app.state.hub
    node_id, node_token = _node(client, admin)
    node = {"Authorization": f"Bearer {node_token}"}
    project = client.post("/api/projects", json={"name": "옮길 현장"}, headers=admin).json()
    blob = settings.storage_dir / "blobs" / "ab" / "cd"
    blob.mkdir(parents=True)
    (blob / "abcd1234").write_bytes(b"a run file")
    (settings.home / "media.key").write_bytes(b"k" * 32)
    with hub.db.session() as db:  # this PC's own node, and settings that belong to this PC
        services.ensure_local_node(db, {"hostname": "OLDPC"}, "0.78.0")
        db.add(HubSetting(key="hub.address", value={"url": "http://10.0.0.1:8470"}))

    # ---- what an export takes, and the export: the Hub reads but no longer changes
    seen = client.get("/api/migration", headers=admin).json()
    assert seen["phase"] == "" and seen["files"] >= 1 and seen["disks"] and seen["stays"]["local_node"] == "OLDPC (hub)"
    out = tmp_path / "out"
    assert client.post("/api/migration/export", json={"folder": "relative"}, headers=admin).status_code == 400
    started = client.post("/api/migration/export", json={"folder": str(out)}, headers=admin).json()
    state = _wait_exported(client, admin)
    assert state["phase"] == "exported", state.get("error")
    assert state["download"].startswith("/media/hub-export/") and state["file_bytes"] > 0
    file = out / started["file"].replace("\\", "/").rsplit("/", 1)[-1]
    assert client.get(state["download"]).content[:512].strip(b"\0")  # the signed link serves it
    blocked = client.post("/api/projects", json={"name": "늦게 만든 것"}, headers=admin)
    assert blocked.status_code == 409 and "읽기만" in blocked.json()["detail"]
    assert client.get("/api/projects", headers=admin).status_code == 200
    assert client.post("/api/auth/login", json=ADMIN).status_code == 200  # people still sign in
    assert client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).json()["job"] is None
    assert client.get("/api/setup/status").json()["migration"]["read_only"] is True

    # ---- the new PC: import, then its Hub is the same Hub (people, nodes, files) with its own address and node
    new_home = tmp_path / "newpc"
    done = migration.import_file(file, new_home, log_line=lambda line: None)
    assert done["files"] >= 3 and (new_home / "storage" / "blobs" / "ab" / "cd" / "abcd1234").read_bytes() == b"a run file"
    assert (new_home / "media.key").read_bytes() == b"k" * 32
    new_settings = HubSettings(home=new_home, web_dir=tmp_path / "no-web")
    with TestClient(create_app(new_settings)) as fresh:
        assert fresh.get("/api/health").json()["hub_id"] == client.get("/api/health").json()["hub_id"]
        signed_in = fresh.post("/api/auth/login", json=ADMIN).json()
        new_admin = {"Authorization": f"Bearer {signed_in['token']}"}
        assert [p["name"] for p in fresh.get("/api/projects", headers=new_admin).json()] == [project["name"]]
        assert fresh.get("/api/setup/status").json()["migration"]["read_only"] is False  # writable there
        assert fresh.get("/api/setup/status").json()["migration"]["imported"]["file"] == file.name
        with fresh.app.state.hub.db.session() as db:
            old_pc = db.query(Node).filter(Node.name == "OLDPC (예전 Hub)").one()
            assert old_pc.is_local is False and db.get(HubSetting, "hub.address") is None
        # the node is known there with the same token
        assert (
            fresh.post("/api/agent/heartbeat", json={"agent_version": "0.78.0", "metrics": {}}, headers=node).status_code
            == 200
        )

        # ---- hand over: only to a Hub that answers as this one's import
        monkeypatch.setattr(migration.httpx, "get", lambda url, timeout: fresh.get(url.replace("http://newpc:8470", "")))
        moved = client.post("/api/migration/hand-over", json={"url": "http://newpc:8470/"}, headers=admin).json()
        assert moved["phase"] == "moved" and moved["url"] == "http://newpc:8470"
        reply = client.post("/api/agent/heartbeat", json={"agent_version": "0.78.0", "metrics": {}}, headers=node).json()
        assert reply["hub_url"] == "http://newpc:8470"

        # ---- a node follows once the new Hub knows it, and keeps the address
        from badamoa.node.agent import NodeAgent, NodeConfig

        agent_home = tmp_path / "node"
        agent = NodeAgent(NodeConfig(hub_url="http://testserver", node_id=node_id, node_token=node_token), agent_home)
        monkeypatch.setattr(agent, "_hub_client", lambda url: TestClient(fresh.app, base_url=url, headers=node))
        agent._follow(reply["hub_url"])
        assert agent.config.hub_url == "http://newpc:8470"
        assert NodeConfig.load(agent_home).hub_url == "http://newpc:8470"

    # ---- an unknown Hub at the address is refused; cancel makes this Hub writable again
    monkeypatch.setattr(migration.httpx, "get", lambda url, timeout: httpx.Response(200, json={"hub_id": "other"}))
    assert client.post("/api/migration/hand-over", json={"url": "http://elsewhere:8470"}, headers=admin).status_code == 400
    assert client.post("/api/migration/cancel", headers=admin).json()["ok"] is True
    assert client.post("/api/projects", json={"name": "다시 씀"}, headers=admin).status_code == 200


def test_an_import_does_not_overwrite_a_hub_in_use(client, admin, tmp_path):
    started = client.post("/api/migration/export", json={"folder": str(tmp_path / "out")}, headers=admin).json()
    assert _wait_exported(client, admin)["phase"] == "exported"
    file = tmp_path / "out" / started["file"].replace("\\", "/").rsplit("/", 1)[-1]
    client.post("/api/migration/cancel", headers=admin)

    used = tmp_path / "used"  # another PC's Hub that people already use (stopped)
    app = create_app(HubSettings(home=used, web_dir=tmp_path / "no-web"))
    with TestClient(app) as other:
        other.post("/api/setup", json={"username": "lee", "password": "lee-used-pw-1"})
    app.state.hub.db.engine.dispose()
    with pytest.raises(services.ServiceError, match="이미 쓰던 Hub"):
        migration.import_file(file, used)
    said = []
    migration.import_file(file, used, force=True, log_line=said.append)
    assert any("before-import-" in line for line in said) and list(used.glob("before-import-*/hub.db"))


def test_a_new_hubs_first_requests_agree_on_its_id(tmp_path):
    """A new Hub's first requests ask its id at once (health checks, its own node): one id, no unique-key error."""
    import threading

    from badamoa.hub.db import Database

    database = Database(f"sqlite:///{(tmp_path / 'hub.db').as_posix()}")
    database.create_all()
    start = threading.Barrier(8)
    ids: list[str] = []
    errors: list[Exception] = []

    def ask() -> None:
        try:
            with database.session() as db:
                start.wait()
                found = migration.hub_id(db)
                time.sleep(0.05)  # still in the transaction while the others insert
            ids.append(found)
        except Exception as e:  # noqa: BLE001 - collected for the assert
            errors.append(e)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(ids)) == 1
    with database.session() as db:
        assert migration.hub_id(db) == ids[0]
