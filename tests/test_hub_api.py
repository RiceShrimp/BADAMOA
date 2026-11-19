"""Hub REST API and the Node protocol, driven through the in-process test client."""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import ADMIN

INFO = {"hostname": "gpu-01", "os": "linux", "gpus": [{"index": 0, "name": "RTX", "memory_total_mb": 24000}]}


def _join(client, admin, roles=None) -> tuple[str, dict[str, str]]:
    body = client.post("/api/nodes/join-tokens", json={"roles": roles}, headers=admin).json()
    reg = client.post(
        "/api/agent/register", json={"join_token": body["token"], "agent_version": "0.1.0", "info": INFO}
    )
    assert reg.status_code == 200, reg.text
    data = reg.json()
    return data["node_id"], {"Authorization": f"Bearer {data['node_token']}"}


def test_setup_runs_once_and_login_works(client):
    assert client.get("/api/health").json()["needs_setup"] is True
    assert client.post("/api/setup", json=ADMIN).status_code == 200
    assert client.post("/api/setup", json=ADMIN).status_code == 409
    assert client.get("/api/health").json()["needs_setup"] is False

    assert client.post("/api/auth/login", json={**ADMIN, "password": "wrong-password"}).status_code == 401
    token = client.post("/api/auth/login", json=ADMIN).json()["token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["username"] == "admin" and me["role"] == "admin"


def test_api_requires_auth(client):
    assert client.get("/api/nodes").status_code == 401
    assert client.get("/api/jobs").status_code == 401
    assert client.post("/api/agent/heartbeat", json={"agent_version": "x"}).status_code == 401


def test_weak_setup_password_rejected(client):
    assert client.post("/api/setup", json={"username": "admin", "password": "short"}).status_code == 400


def test_members_cannot_add_nodes(client, admin):
    client.post("/api/users", json={"username": "alice", "password": "alice-password"}, headers=admin)
    login = client.post("/api/auth/login", json={"username": "alice", "password": "alice-password"})
    member = {"Authorization": f"Bearer {login.json()['token']}"}
    assert client.post("/api/nodes/join-tokens", json={}, headers=member).status_code == 403
    assert client.get("/api/nodes", headers=member).status_code == 200


def test_join_token_is_single_use_and_commands_point_at_hub(client, admin):
    body = client.post("/api/nodes/join-tokens", json={"roles": ["compute"]}, headers=admin).json()
    assert body["roles"] == ["compute"]
    assert body["token"] in body["commands"]["linux"] and "/install/node.sh" in body["commands"]["linux"]
    assert "/install/node.ps1" in body["commands"]["windows"]
    # a server where the account has no sudo: the same script, run as that account
    assert body["commands"]["linux_user"] == body["commands"]["linux"].replace("| sudo bash", "| bash")

    payload = {"join_token": body["token"], "agent_version": "0.1.0", "info": INFO}
    assert client.post("/api/agent/register", json=payload).status_code == 200
    assert client.post("/api/agent/register", json=payload).status_code == 401
    assert client.post("/api/nodes/join-tokens", json={"roles": ["gpu"]}, headers=admin).status_code == 400


def test_roles_default_to_all_and_training_and_autolabel_split_once(client, admin, settings):
    """학습용 · 오토라벨용 · 추론용 · 데이터 서버용. Before 0.54.0 "학습·오토라벨" was one role (compute): a
    server with it keeps doing both, once — switching 오토라벨용 off afterwards stays off."""
    from badamoa.hub.db import Database
    from badamoa.hub.models import HubSetting, Node

    body = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()
    assert body["roles"] == ["compute", "autolabel", "inference", "storage"]
    bad = client.post("/api/nodes/join-tokens", json={"roles": ["gpu"]}, headers=admin)
    assert "compute(학습용)" in bad.json()["detail"] and "storage(데이터 서버용)" in bad.json()["detail"]

    db = Database(settings.database_url or "")
    with db.session() as s:  # as an older Hub left them
        s.add(Node(name="old", token_hash="h-old", roles=["compute", "storage"]))
        s.add(Node(name="data", token_hash="h-data", roles=["storage"]))
        s.delete(s.get(HubSetting, "migration.autolabel_role"))
    db.create_all()
    with db.session() as s:
        roles = {n.name: n.roles for n in s.query(Node).all()}
        assert roles["old"] == ["compute", "autolabel", "storage"] and roles["data"] == ["storage"]
        s.query(Node).filter(Node.name == "old").one().roles = ["compute"]
    db.create_all()
    with db.session() as s:
        assert s.query(Node).filter(Node.name == "old").one().roles == ["compute"]


def test_node_lifecycle_and_job_flow(client, admin):
    node_id, node = _join(client, admin)

    hb = client.post(
        "/api/agent/heartbeat", json={"agent_version": "0.1.0", "metrics": {"cpu_pct": 3}}, headers=node
    )
    assert hb.status_code == 200
    listed = client.get("/api/nodes", headers=admin).json()
    assert [(n["id"], n["status"], n["os"]) for n in listed] == [(node_id, "online", "linux")]
    assert listed[0]["info"]["gpus"][0]["name"] == "RTX"

    # unknown type / bad params are rejected up front
    assert client.post("/api/jobs", json={"type": "nope"}, headers=admin).status_code == 400
    bad = client.post("/api/jobs", json={"type": "debug.sleep", "params": {"seconds": 0}}, headers=admin)
    assert bad.status_code == 400

    job = client.post("/api/jobs", json={"type": "debug.sleep", "params": {"seconds": 2}}, headers=admin).json()
    assert job["status"] == "queued" and job["params"] == {"seconds": 2, "message": "tick", "fail": False}

    claimed = client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).json()["job"]
    assert claimed["id"] == job["id"]
    assert client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).json()["job"] is None

    now = datetime.now(UTC).isoformat()
    lines = [{"ts": now, "stream": "stdout", "text": f"tick {i}"} for i in (1, 2)]
    ack = client.post(f"/api/agent/jobs/{job['id']}/logs", json={"lines": lines}, headers=node).json()
    assert ack == {"cancel_requested": False}

    done = client.post(
        f"/api/agent/jobs/{job['id']}/complete",
        json={"status": "succeeded", "result": {"slept_seconds": 2}},
        headers=node,
    )
    assert done.json() == {"status": "succeeded"}

    final = client.get(f"/api/jobs/{job['id']}", headers=admin).json()
    assert final["status"] == "succeeded" and final["result"] == {"slept_seconds": 2}
    assert final["node_name"] == "gpu-01" and final["log_count"] == 2
    logs = client.get(f"/api/jobs/{job['id']}/logs", headers=admin).json()
    assert [(line["seq"], line["text"]) for line in logs] == [(1, "tick 1"), (2, "tick 2")]
    assert client.get(f"/api/jobs/{job['id']}/logs?after=1", headers=admin).json()[0]["seq"] == 2


def test_activity_feed_lists_recent_actions_newest_first(client, admin):
    assert client.get("/api/activity").status_code == 401
    node_id, _ = _join(client, admin)
    job = client.post("/api/jobs", json={"type": "system.info"}, headers=admin).json()
    feed = client.get("/api/activity?limit=3", headers=admin).json()
    assert [(e["action"], e["target"]) for e in feed] == [
        ("job.create", job["id"]),
        ("node.register", node_id),
        ("node.join_token", feed[2]["target"]),
    ]
    assert feed[0]["username"] == "admin" and feed[0]["detail"]["type"] == "system.info"


def test_job_targeted_at_one_node_is_not_claimed_by_another(client, admin):
    node_a, auth_a = _join(client, admin)
    _, auth_b = _join(client, admin)
    job = client.post("/api/jobs", json={"type": "system.info", "node_id": node_a}, headers=admin).json()
    assert client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=auth_b).json()["job"] is None
    claimed = client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=auth_a).json()["job"]
    assert claimed["id"] == job["id"]


def test_cancel_queued_and_running_jobs(client, admin):
    _, node = _join(client, admin)
    queued = client.post("/api/jobs", json={"type": "system.info"}, headers=admin).json()
    assert client.post(f"/api/jobs/{queued['id']}/cancel", headers=admin).json()["status"] == "canceled"
    assert client.post(f"/api/jobs/{queued['id']}/cancel", headers=admin).status_code == 409

    running = client.post("/api/jobs", json={"type": "debug.sleep"}, headers=admin).json()
    client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node)
    assert client.post(f"/api/jobs/{running['id']}/cancel", headers=admin).json()["cancel_requested"] is True
    hb = client.post(
        "/api/agent/heartbeat",
        json={"agent_version": "0.1.0", "running_job_ids": [running["id"]]},
        headers=node,
    ).json()
    assert hb["cancel_job_ids"] == [running["id"]]


def test_disabled_node_gets_no_jobs_and_deleted_node_is_rejected(client, admin):
    node_id, node = _join(client, admin)
    client.post("/api/jobs", json={"type": "system.info"}, headers=admin)
    patched = client.patch(
        f"/api/nodes/{node_id}", json={"enabled": False, "roles": ["compute"]}, headers=admin
    ).json()
    assert patched["status"] == "disabled" and patched["roles"] == ["compute"]
    assert client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).json()["job"] is None

    assert client.delete(f"/api/nodes/{node_id}", headers=admin).status_code == 200
    assert client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).status_code == 401


def test_install_scripts_and_wheel_index(client, settings):
    sh = client.get("/install/node.sh").text
    assert 'HUB_URL="http://testserver"' in sh and "{{" not in sh
    assert 'ROOT="$HOME/.badamoa"' in sh and "badamoa node install-service" in sh  # also without sudo
    gone = client.get("/install/node-remove.sh").text
    assert "node uninstall-service" in gone and "rm -f /etc/systemd/system/badamoa-node.service" in gone
    assert "{{" not in client.get("/install/node.ps1").text

    (settings.wheels_dir / "badamoa-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    assert '<a href="badamoa/">' in client.get("/install/simple/").text
    assert "badamoa-0.1.0-py3-none-any.whl" in client.get("/install/simple/badamoa/").text
    assert client.get("/install/wheels/badamoa-0.1.0-py3-none-any.whl").content == b"wheel"
    assert client.get("/install/simple/other/").status_code == 404
    assert client.get("/install/wheels/..%2Fhub.db").status_code == 404


def test_template_environments_ready_and_prepared_ahead(client, admin, tmp_path):
    import json

    from badamoa.node import envs

    # what a node reports: the environments it has built (requirements, Python, torch)
    marker = tmp_path / "envs" / "abc123" / "badamoa-env.json"
    marker.parent.mkdir(parents=True)
    yolo_needs = ["ultralytics>=8.3,<9", "onnx>=1.12", "onnxslim", "onnxruntime"]
    marker.write_text(json.dumps({"requirements": list(reversed(yolo_needs)), "python": "3.12", "torch": True}))
    (tmp_path / "envs" / "broken").mkdir()
    reported = envs.listing(tmp_path)
    assert reported == [{"id": "abc123", "requirements": sorted(yolo_needs), "python": "3.12", "torch": True}]

    ready_id, ready_auth = _join(client, admin)
    other_id, _ = _join(client, admin)
    beat = {"agent_version": "0.1.0", "info": {**INFO, "envs": reported}}
    assert client.post("/api/agent/heartbeat", json=beat, headers=ready_auth).status_code == 200
    names = {n["id"]: n["name"] for n in client.get("/api/nodes", headers=admin).json()}

    def yolo():
        return next(t for t in client.get("/api/templates", headers=admin).json() if t["name"] == "yolo")

    env = yolo()["environment"]
    assert env["installs"] and env["ready"] == [names[ready_id]] and env["preparing"] == [] and env["nodes"] == 2

    # prepared ahead on the other server: a job there, shown as under way
    started = client.post(f"/api/templates/{yolo()['id']}/prepare", json={"node_ids": [other_id]}, headers=admin)
    assert started.status_code == 200, started.text
    job = client.get(f"/api/jobs/{started.json()['jobs'][0]}", headers=admin).json()
    assert job["type"] == "env.prepare" and job["target_node_id"] == other_id
    assert job["params"]["environment"]["requirements"] == yolo_needs and job["params"]["environment"]["torch"] is True
    assert yolo()["environment"]["preparing"] == [names[other_id]]

    # nothing to prepare for code that needs no packages
    refused = client.post("/api/templates/dummy/prepare", json={}, headers=admin)
    assert refused.status_code == 400 and "설치할 환경이 없습니다" in refused.json()["detail"]
