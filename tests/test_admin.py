"""The settings screen's admin parts: people's accounts and when a GPU counts as busy."""

from __future__ import annotations

from conftest import ADMIN
from fastapi.testclient import TestClient
from test_resources import _beat, _gpu, _join

from badamoa.hub.app import create_app


def _login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def _headers(resp):
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def test_accounts_roles_stopping_and_passwords(client, admin):
    made = client.post("/api/users", json={"username": "kim", "password": "kim-first-pw", "role": "member"}, headers=admin)
    assert made.status_code == 200 and made.json()["disabled"] is False
    kim = made.json()["id"]
    kim_auth = _headers(_login(client, "kim", "kim-first-pw"))
    assert client.get("/api/users", headers=kim_auth).status_code == 403  # members do not see the list
    assert client.get("/api/auth/me", headers=kim_auth).json()["username"] == "kim"

    people = {u["username"]: u for u in client.get("/api/users", headers=admin).json()}
    assert set(people) == {"admin", "kim"} and people["kim"]["last_seen_at"] and people["kim"]["api_tokens"] == 0

    # ---- a new password from the admin: the person's sessions end, the new password works
    assert client.patch(f"/api/users/{kim}", json={"password": "short"}, headers=admin).status_code == 400
    assert client.patch(f"/api/users/{kim}", json={"password": "kim-second-pw"}, headers=admin).status_code == 200
    assert client.get("/api/auth/me", headers=kim_auth).status_code == 401
    assert _login(client, "kim", "kim-first-pw").status_code == 401
    kim_auth = _headers(_login(client, "kim", "kim-second-pw"))

    # ---- your own password: the current one first; this session stays, the others end
    other = _headers(_login(client, "kim", "kim-second-pw"))
    wrong = client.post("/api/auth/password", json={"current": "nope", "new": "kim-third-pw"}, headers=kim_auth)
    assert wrong.status_code == 400
    ok = client.post("/api/auth/password", json={"current": "kim-second-pw", "new": "kim-third-pw"}, headers=kim_auth)
    assert ok.status_code == 200
    assert client.get("/api/auth/me", headers=kim_auth).status_code == 200
    assert client.get("/api/auth/me", headers=other).status_code == 401

    # ---- an API token, then the account stopped: no sign-in, tokens stop working; restarted: they work again
    token = client.post("/api/tokens", json={"name": "mcp"}, headers=kim_auth).json()["token"]
    api = {"Authorization": f"Bearer {token}"}
    stopped = client.patch(f"/api/users/{kim}", json={"disabled": True}, headers=admin)
    assert stopped.status_code == 200 and stopped.json()["disabled"] and stopped.json()["api_tokens"] == 1
    assert client.get("/api/auth/me", headers=api).status_code == 401
    refused = _login(client, "kim", "kim-third-pw")
    assert refused.status_code == 403 and "중지" in refused.json()["detail"]
    assert client.patch(f"/api/users/{kim}", json={"disabled": False}, headers=admin).status_code == 200
    assert client.get("/api/auth/me", headers=api).status_code == 200

    # ---- there is always an admin who can sign in
    me = next(u["id"] for u in client.get("/api/users", headers=admin).json() if u["username"] == ADMIN["username"])
    assert client.patch(f"/api/users/{me}", json={"role": "member"}, headers=admin).status_code == 409
    assert client.patch(f"/api/users/{me}", json={"disabled": True}, headers=admin).status_code == 409
    assert client.patch(f"/api/users/{kim}", json={"role": "admin"}, headers=admin).json()["role"] == "admin"
    assert client.patch(f"/api/users/{me}", json={"role": "member"}, headers=admin).json()["role"] == "member"
    assert client.patch(f"/api/users/{kim}", json={"role": "boss"}, headers=api).status_code == 400
    assert client.patch("/api/users/nobody", json={"role": "member"}, headers=api).status_code == 404

    actions = [a["action"] for a in client.get("/api/activity?limit=20", headers=api).json()]
    assert "user.update" in actions and "user.password" in actions


def test_gpu_busy_thresholds_set_on_the_screen(client, admin, settings):
    _, node = _join(client, admin)
    _beat(client, node, [_gpu(0, 6000, util=5), _gpu(1, 300)])  # GPU 0: a quarter of its memory in use

    view = client.get("/api/settings/gpu-busy", headers=admin).json()
    assert (view["memory_pct"], view["util_pct"], view["source"]) == (10.0, 50.0, "launch")
    assert [(g["index"], g["state"], g["memory_pct"]) for g in view["gpus"]] == [(0, "busy", 25.0), (1, "free", 1.2)]

    # ---- raised: GPU 0 is free for jobs now, and the value is kept
    saved = client.put("/api/settings/gpu-busy", json={"memory_pct": 30, "util_pct": 60}, headers=admin).json()
    assert (saved["memory_pct"], saved["source"]) == (30.0, "screen") and saved["gpus"][0]["state"] == "free"
    assert client.get("/api/nodes", headers=admin).json()[0]["gpu_states"][0]["state"] == "free"
    for wrong in ({"memory_pct": 0, "util_pct": 50}, {"memory_pct": 50}, {"memory_pct": 50, "util_pct": 200}):
        assert client.put("/api/settings/gpu-busy", json=wrong, headers=admin).status_code == 400

    # ---- a restarted Hub keeps the screen's values over its launch options
    with TestClient(create_app(settings)) as again:
        token = again.post("/api/auth/login", json=ADMIN).json()["token"]
        view = again.get("/api/settings/gpu-busy", headers={"Authorization": f"Bearer {token}"}).json()
        assert (view["memory_pct"], view["util_pct"], view["source"]) == (30.0, 60.0, "screen")

    # ---- back to the launch options; members may look but not change
    reset = client.put("/api/settings/gpu-busy", json={}, headers=admin).json()
    assert (reset["memory_pct"], reset["util_pct"], reset["source"]) == (10.0, 50.0, "launch")
    client.post("/api/users", json={"username": "lee", "password": "lee-first-pw"}, headers=admin)
    lee = _headers(_login(client, "lee", "lee-first-pw"))
    assert client.get("/api/settings/gpu-busy", headers=lee).status_code == 200
    assert client.put("/api/settings/gpu-busy", json={"memory_pct": 90, "util_pct": 90}, headers=lee).status_code == 403


def test_heartbeat_tells_nodes_how_long_work_folders_stay(client, admin):
    _, node = _join(client, admin)
    body = {"agent_version": "0.3.0", "metrics": {"gpus": []}, "running_job_ids": []}
    assert client.post("/api/agent/heartbeat", json=body, headers=node).json()["workdir_days"] == 30
    client.put("/api/storage/policy", json={"workdir_days": 0}, headers=admin)
    assert client.post("/api/agent/heartbeat", json=body, headers=node).json()["workdir_days"] == 0


def test_admins_delete_accounts_and_what_they_made_stays(client, admin):
    client.post("/api/users", json={"username": "kim", "password": "kim-delete-pw"}, headers=admin)
    kim = _headers(_login(client, "kim", "kim-delete-pw"))
    made = client.post("/api/projects", json={"name": "김의 프로젝트"}, headers=kim).json()
    users = {u["username"]: u["id"] for u in client.get("/api/users", headers=admin).json()}

    assert client.delete(f"/api/users/{users['kim']}", headers=kim).status_code == 403  # admins only
    assert client.delete(f"/api/users/{users['admin']}", headers=admin).status_code == 409  # not one's own
    done = client.delete(f"/api/users/{users['kim']}", headers=admin).json()
    assert done["ok"] is True and done["username"] == "kim"
    assert "kim" not in {u["username"] for u in client.get("/api/users", headers=admin).json()}
    assert client.get("/api/auth/me", headers=kim).status_code == 401  # signed out at once
    assert [p["id"] for p in client.get("/api/projects", headers=admin).json()] == [made["id"]]  # what kim made stays
    assert client.post("/api/users", json={"username": "kim", "password": "kim-again-pw1"}, headers=admin).status_code == 200

    # the last admin who can sign in stays: another admin may be deleted, then nobody can delete the one left
    client.post("/api/users", json={"username": "lee", "password": "lee-admin-pw1", "role": "admin"}, headers=admin)
    lee = _headers(_login(client, "lee", "lee-admin-pw1"))
    users = {u["username"]: u["id"] for u in client.get("/api/users", headers=admin).json()}
    assert client.delete(f"/api/users/{users['admin']}", headers=lee).status_code == 200
    assert client.delete(f"/api/users/{users['lee']}", headers=lee).status_code == 409
