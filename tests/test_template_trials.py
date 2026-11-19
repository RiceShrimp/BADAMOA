"""A template version is tried on a server before anyone is offered it (EXTENSION_PLAN 8).

The trials here use a copy of the built-in ``dummy`` template (standard library only), so they go
through the whole path — upload, the node's own sample data, environment, train, predict, eval,
export — without torch.
"""

from __future__ import annotations

import re
import shutil
import threading
import time

import httpx
import pytest
from conftest import ADMIN

from badamoa import templates as shipped
from badamoa.core import templates as template_format
from badamoa.node import agent as node_agent
from badamoa.node import exporter, samples


def _bundle(tmp_path, name="trial", version="1", extra="", fail=False):
    """The dummy template as a team would upload it, under another name, shown in the dialog."""
    folder = tmp_path / f"{name}-{version}"
    shutil.copytree(shipped.builtin()["dummy"], folder, ignore=shutil.ignore_patterns("__pycache__"))
    text = (folder / "badamoa.toml").read_text(encoding="utf-8")
    text = text.replace('name = "dummy"', f'name = "{name}"').replace('version = "3"', f'version = "{version}"')
    text = text.replace("hidden = true", "hidden = false")
    if fail:
        text = re.sub(r"(\[params\.fail\][^\[]*default = )false", r"\1true", text)
    (folder / "badamoa.toml").write_text(text + extra, encoding="utf-8")
    return template_format.bundle(folder)


@pytest.fixture
def hub(live_hub, tmp_path, monkeypatch):
    """A Hub with one real node (it trains) and a member besides the admin."""
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        join = client.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="gpu-1"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:  # online: it has said hello at least once
            node = client.get(f"/api/nodes/{runner.config.node_id}").json()
            if node.get("status") == "online":
                break
            time.sleep(0.2)
        member = client.post("/api/users", json={"username": "kim", "password": "kim-pass-123", "role": "member"})
        assert member.status_code == 200, member.text
        login = httpx.post(f"{live_hub}/api/auth/login", json={"username": "kim", "password": "kim-pass-123"}).json()
        client.member = httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {login['token']}"}, timeout=60)
        client.node_id = runner.config.node_id
        yield client
        client.member.close()
        runner.stop()


def _upload(hub, data):
    resp = hub.post("/api/templates", content=data, headers={"Content-Type": "application/zip"})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _settled(hub, template_id, timeout=120):
    deadline = time.monotonic() + timeout
    view = hub.get(f"/api/templates/{template_id}").json()
    while view["check"]["state"] == "checking" and time.monotonic() < deadline:
        time.sleep(0.4)
        view = hub.get(f"/api/templates/{template_id}").json()
    return view


def _names(client):
    return {t["name"]: t["version"] for t in client.get("/api/templates").json()}


def test_a_new_version_is_tried_before_members_see_it(hub, tmp_path):
    first = _upload(hub, _bundle(tmp_path, version="1"))
    assert first["check"]["state"] == "checking"
    assert "trial" not in _names(hub.member)  # not offered while it is being tried
    assert _names(hub)["trial"] == "1"  # admins see it, with its state

    done = _settled(hub, first["id"])
    check = done["check"]
    assert check["state"] == "ok" and check["offered"], check
    assert [s["step"] for s in check["steps"]] == ["data", "env", "train", "predict", "eval", "export"]
    assert check["node"] == "gpu-1" and check["metrics"]
    assert _names(hub.member)["trial"] == "1"

    # a newer version that fails: members keep the one that passed, admins see why
    second = _upload(hub, _bundle(tmp_path, version="2", fail=True))
    failed = _settled(hub, second["id"])["check"]
    assert failed["state"] == "failed" and failed["failed"] == "train" and "종료 코드" in failed["reason"]
    assert _names(hub.member)["trial"] == "1"
    admin_view = next(t for t in hub.get("/api/templates").json() if t["name"] == "trial")
    assert admin_view["version"] == "2" and admin_view["offered_version"] == "1"
    logs = hub.get(f"/api/jobs/{failed['job_id']}/logs").json()
    assert any("의도적 실패" in line["text"] for line in logs)

    # members cannot start it by its id; an admin lets it through with a reason
    assert hub.member.post(f"/api/templates/{second['id']}/allow", json={"reason": "x"}).status_code == 403
    assert hub.post(f"/api/templates/{second['id']}/allow", json={"reason": " "}).status_code == 400
    allowed = hub.post(f"/api/templates/{second['id']}/allow", json={"reason": "사내 서버에서 따로 확인함"})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["check"]["state"] == "allowed" and allowed.json()["check"]["by"] == "admin"
    assert _names(hub.member)["trial"] == "2"


def test_check_settings_skip_and_steps_that_say_what_is_wrong(hub, tmp_path):
    # export left out on purpose; the trial uses its own short settings
    extra = '\n[check]\nskip = ["export"]\nparams = { sleep = 0.0 }\ntimeout_min = 5\n'
    view = _settled(hub, _upload(hub, _bundle(tmp_path, name="skips", extra=extra))["id"])
    assert view["check"]["state"] == "ok", view["check"]
    assert [s["step"] for s in view["check"]["steps"]] == ["data", "env", "train", "predict", "eval"]

    # trying again on demand: a new job, the same verdict
    again = hub.post(f"/api/templates/{view['id']}/check")
    assert again.status_code == 200 and again.json()["check"]["state"] == "checking"
    assert _settled(hub, view["id"])["check"]["state"] == "ok"


def test_check_declarations_are_checked_on_upload(tmp_path):
    bad = [
        ('\n[check]\nskip = ["train"]\n', "train은 넣을 수 없습니다"),
        ('\n[check]\nskip = ["serve"]\n', "없는 진입점"),
        ("\n[check]\ntimeout_min = 0\n", "1~180분"),
        ("\n[check]\nparams = { nope = 1 }\n", "없는 설정"),
        ('\n[check]\nparams = { epochs = "many" }\n', "integer"),
    ]
    for n, (extra, said) in enumerate(bad):
        with pytest.raises(template_format.TemplateError, match=said):
            template_format.read_bundle(_bundle(tmp_path, name="bad", version=str(n), extra=extra))
    manifest = template_format.read_bundle(_bundle(tmp_path, name="good", extra="\n[check]\nparams = { imgsz = 64 }\n"))
    # one epoch where there is such a setting, the trial's own values, the defaults for the rest
    assert template_format.check_params(manifest) == {
        **template_format.resolve_params(manifest, {}),
        "epochs": 1,
        "imgsz": 64,
    }


@pytest.mark.parametrize(("data_format", "kind"), [("yolo", "detect"), ("yolo-seg", "segment"), ("imagefolder", "classify")])
def test_the_sample_a_node_draws_writes_in_each_format(tmp_path, data_format, kind):
    assert samples.kind_of(data_format) == kind
    items, classes, locations = samples.make(tmp_path / "sample", kind, classes=3)
    options = exporter.ExportOptions(keep_splits=True)
    summary = exporter.BUILDERS[data_format](items, classes, locations, tmp_path / "data", options)
    assert summary["items"]["train"] >= 8 and summary["items"]["val"] >= 3
    assert summary["classes"] == ["red", "blue", "green"] and not any(summary["excluded"].values())
    if kind == "segment":  # outlines, not just boxes
        label = next((tmp_path / "data" / "labels" / "train").glob("*.txt")).read_text().splitlines()[0]
        assert len(label.split()) == 9


def test_versions_from_before_trials_stay_offered_until_tried(client, admin, settings, tmp_path):
    from badamoa.hub import template_checks
    from badamoa.hub.db import Database
    from badamoa.hub.models import Job
    from badamoa.hub.train_models import Template

    member = client.post("/api/users", json={"username": "lee", "password": "lee-pass-123", "role": "member"}, headers=admin)
    assert member.status_code == 200
    token = client.post("/api/auth/login", json={"username": "lee", "password": "lee-pass-123"}).json()["token"]
    lee = {"Authorization": f"Bearer {token}"}

    # no server can try it yet: not offered, and the admin sees why
    up = client.post("/api/templates", content=_bundle(tmp_path, name="old"), headers=admin)
    assert up.status_code == 200, up.text
    assert up.json()["check"]["state"] == "unchecked" and "학습 서버" in up.json()["check"]["reason"]
    assert "old" not in {t["name"] for t in client.get("/api/templates", headers=lee).json()}

    # as if registered before trials existed: offered, until a server can try it
    database = Database(settings.database_url)
    with database.session() as db:
        db.get(Template, up.json()["id"]).trial = {}
    with database.session() as db:
        template_checks.mark_legacy(db)
    assert "old" in {t["name"] for t in client.get("/api/templates", headers=lee).json()}

    # a server too old to try it says nothing; one that can gets the trial, and it stays offered meanwhile
    def join(version):
        body = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()
        info = {"agent_version": version, "info": {}, "join_token": body["token"]}
        reg = client.post("/api/agent/register", json=info).json()
        node = {"Authorization": f"Bearer {reg['node_token']}"}
        beat = {"agent_version": version, "metrics": {}, "running_job_ids": []}
        assert client.post("/api/agent/heartbeat", json=beat, headers=node).status_code == 200
        return reg["node_id"]

    join("0.51.0")
    assert client.get(f"/api/templates/{up.json()['id']}", headers=admin).json()["check"]["state"] == "legacy"
    new = join(".".join(str(n) for n in template_checks.CHECK_SINCE))
    check = client.get(f"/api/templates/{up.json()['id']}", headers=admin).json()["check"]
    assert check["state"] == "checking" and check["legacy"] and check["offered"]
    assert "old" in {t["name"] for t in client.get("/api/templates", headers=lee).json()}

    # the server went away mid-trial: no verdict on the code, still offered
    with database.session() as db:
        job = db.get(Job, check["job_id"])
        assert job.target_node_id == new and job.type == "template.check"
        job.status, job.node_id, job.error = "failed", new, "노드가 더 이상 이 작업을 실행하고 있지 않습니다"
        template_checks.on_job_finished(db, job)
    check = client.get(f"/api/templates/{up.json()['id']}", headers=admin).json()["check"]
    assert check["state"] == "legacy" and check["tried"] and "노드" in check["reason"]

    # a real training that succeeded with a version nobody could try counts as passing
    with database.session() as db:
        row = db.get(Template, up.json()["id"])
        row.trial = {"state": "unchecked", "reason": "x"}
        run = Job(type="train.run", status="succeeded", params={"template": {"id": row.id}})
        db.add(run)
        db.flush()
        template_checks.on_job_finished(db, run)
        assert row.trial["state"] == "ok" and row.trial["via"] == "run"
    database.engine.dispose()
