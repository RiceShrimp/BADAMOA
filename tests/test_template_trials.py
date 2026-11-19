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


def _bundle(tmp_path, name="trial", version="1", extra="", fail=False, autolabel=True, labels_logged=True):
    """The dummy template as a team would upload it, under another name, shown in the dialog."""
    folder = tmp_path / f"{name}-{version}"
    shutil.copytree(shipped.builtin()["dummy"], folder, ignore=shutil.ignore_patterns("__pycache__"))
    text = (folder / "badamoa.toml").read_text(encoding="utf-8")
    text = text.replace('name = "dummy"', f'name = "{name}"').replace('version = "3"', f'version = "{version}"')
    text = text.replace("hidden = true", "hidden = false")
    if fail:
        text = re.sub(r"(\[params\.fail\][^\[]*default = )false", r"\1true", text)
    if not autolabel:
        text = text.replace("autolabel = true\n", "")
    if not labels_logged:  # its predict code ignores autolabel mode: nothing per image
        ops = (folder / "ops.py").read_text(encoding="utf-8").replace("    bm.log_predictions(predicted)\n", "")
        (folder / "ops.py").write_text(ops, encoding="utf-8")
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


def _try(hub, template_id):
    """An admin presses "시험 실행" and waits for the verdict."""
    started = hub.post(f"/api/templates/{template_id}/check")
    assert started.status_code == 200, started.text
    assert started.json()["check"]["state"] == "checking"
    return _settled(hub, template_id)["check"]


def test_an_admin_tries_a_new_version_and_offers_it_once_it_passed(hub, tmp_path):
    first = _upload(hub, _bundle(tmp_path, version="1"))
    assert first["check"]["state"] == "pending"  # nothing starts on its own
    time.sleep(1.5)
    assert hub.get(f"/api/templates/{first['id']}").json()["check"]["state"] == "pending"
    assert "trial" not in _names(hub.member)
    assert _names(hub)["trial"] == "1"  # admins see it, with its state
    assert hub.member.post(f"/api/templates/{first['id']}/check").status_code == 403

    check = _try(hub, first["id"])
    assert check["state"] == "passed" and not check["offered"], check  # a pass waits for the admin
    assert [s["step"] for s in check["steps"]] == ["data", "env", "train", "predict", "autolabel", "eval", "export"]
    assert check["node"] == "gpu-1" and check["metrics"] and check["by"] == "admin"
    assert "trial" not in _names(hub.member)
    offered = hub.post(f"/api/templates/{first['id']}/allow", json={})
    assert offered.status_code == 200, offered.text
    assert offered.json()["check"]["state"] == "ok" and offered.json()["check"]["approved_by"] == "admin"
    assert offered.json()["check"]["steps"]  # the trial's record stays with it
    assert _names(hub.member)["trial"] == "1"

    # a newer version that fails: members keep the one offered, admins see why
    second = _upload(hub, _bundle(tmp_path, version="2", fail=True))
    failed = _try(hub, second["id"])
    assert failed["state"] == "failed" and failed["failed"] == "train" and "종료 코드" in failed["reason"]
    assert _names(hub.member)["trial"] == "1"
    admin_view = next(t for t in hub.get("/api/templates").json() if t["name"] == "trial")
    assert admin_view["version"] == "2" and admin_view["offered_version"] == "1"
    logs = hub.get(f"/api/jobs/{failed['job_id']}/logs").json()
    assert any("의도적 실패" in line["text"] for line in logs)

    # without a pass, only with a reason (the admin's escape hatch); members cannot
    assert hub.member.post(f"/api/templates/{second['id']}/allow", json={"reason": "x"}).status_code == 403
    assert hub.post(f"/api/templates/{second['id']}/allow", json={"reason": " "}).status_code == 400
    allowed = hub.post(f"/api/templates/{second['id']}/allow", json={"reason": "사내 서버에서 따로 확인함"})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["check"]["state"] == "allowed" and allowed.json()["check"]["approved_by"] == "admin"
    assert allowed.json()["check"]["was"] == "failed"
    assert _names(hub.member)["trial"] == "2"


def test_check_settings_skip_and_steps_that_say_what_is_wrong(hub, tmp_path):
    # export left out on purpose; the trial uses its own short settings
    extra = '\n[check]\nskip = ["export"]\nparams = { sleep = 0.0 }\ntimeout_min = 5\n'
    view = _upload(hub, _bundle(tmp_path, name="skips", extra=extra))
    check = _try(hub, view["id"])
    assert check["state"] == "passed", check
    assert [s["step"] for s in check["steps"]] == ["data", "env", "train", "predict", "autolabel", "eval"]
    assert hub.post(f"/api/templates/{view['id']}/check").status_code == 200  # again, on demand
    assert hub.post(f"/api/templates/{view['id']}/check").status_code == 409  # not twice at once
    assert _settled(hub, view["id"])["check"]["state"] == "passed"


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


def test_versions_from_before_trials_stay_offered_and_admins_try_them_by_hand(client, admin, settings, tmp_path):
    from badamoa.hub import template_checks
    from badamoa.hub.db import Database
    from badamoa.hub.models import Job
    from badamoa.hub.train_models import Template

    member = client.post("/api/users", json={"username": "lee", "password": "lee-pass-123", "role": "member"}, headers=admin)
    assert member.status_code == 200
    token = client.post("/api/auth/login", json={"username": "lee", "password": "lee-pass-123"}).json()["token"]
    lee = {"Authorization": f"Bearer {token}"}

    up = client.post("/api/templates", content=_bundle(tmp_path, name="old"), headers=admin)
    assert up.status_code == 200, up.text
    assert up.json()["check"]["state"] == "pending"
    assert "old" not in {t["name"] for t in client.get("/api/templates", headers=lee).json()}
    # no server can try it: the admin is told why, nothing changes
    nobody = client.post(f"/api/templates/{up.json()['id']}/check", headers=admin)
    assert nobody.status_code == 409 and "학습 서버" in nobody.json()["detail"]

    # as if registered before trials existed: offered as before
    database = Database(settings.database_url)
    with database.session() as db:
        db.get(Template, up.json()["id"]).trial = {}
    with database.session() as db:
        template_checks.mark_legacy(db)
    assert "old" in {t["name"] for t in client.get("/api/templates", headers=lee).json()}

    # a server that can try it reports in: still nothing starts on its own
    body = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()
    version = ".".join(str(n) for n in template_checks.CHECK_SINCE)
    reg = client.post("/api/agent/register", json={"agent_version": version, "info": {}, "join_token": body["token"]}).json()
    node = {"Authorization": f"Bearer {reg['node_token']}"}
    beat = {"agent_version": version, "metrics": {}, "running_job_ids": []}
    assert client.post("/api/agent/heartbeat", json=beat, headers=node).status_code == 200
    assert client.get(f"/api/templates/{up.json()['id']}", headers=admin).json()["check"]["state"] == "legacy"

    # an admin tries it by hand; it stays offered while it runs
    started = client.post(f"/api/templates/{up.json()['id']}/check", headers=admin)
    assert started.status_code == 200, started.text
    check = started.json()["check"]
    assert check["state"] == "checking" and check["legacy"] and check["offered"]
    assert "old" in {t["name"] for t in client.get("/api/templates", headers=lee).json()}

    # the server went away mid-trial: no verdict on the code, back to how it was
    with database.session() as db:
        job = db.get(Job, check["job_id"])
        assert job.target_node_id == reg["node_id"] and job.type == "template.check"
        job.status, job.node_id, job.error = "failed", reg["node_id"], "노드가 더 이상 이 작업을 실행하고 있지 않습니다"
        template_checks.on_job_finished(db, job)
    check = client.get(f"/api/templates/{up.json()['id']}", headers=admin).json()["check"]
    assert check["state"] == "legacy" and "노드" in check["reason"] and check["offered"]

    # a real training no longer changes a version's state by itself
    with database.session() as db:
        row = db.get(Template, up.json()["id"])
        row.trial = {"state": "pending"}
        run = Job(type="train.run", status="succeeded", params={"template": {"id": row.id}})
        db.add(run)
        db.flush()
        template_checks.on_job_finished(db, run)
        assert row.trial == {"state": "pending"}
    database.engine.dispose()


# ---- the same packages on every server (EXTENSION_PLAN 7.2 · 7.3) --------------------------------


def test_pins_keep_versions_not_builds_and_leave_out_what_has_its_own_source():
    installed = ["torch==2.5.1+cu121", "numpy==2.1.0", "nvidia-cudnn-cu12==9.1.0.70", "My_Lib==0.3.0", "plain"]
    sources = {"my-lib": {"kind": "git", "url": "https://git.local/my-lib", "commit": "abc"}}
    assert template_format.pinned_versions(installed, sources) == ["torch==2.5.1", "numpy==2.1.0"]


def test_where_packages_came_from_and_wheels_checked_against_their_hash(tmp_path):
    import hashlib

    from badamoa.node import envs

    git = {"url": "https://git.local/lib.git", "vcs_info": {"vcs": "git", "commit_id": "1a2b3c"}}
    assert envs.describe(git) == {"kind": "git", "url": "https://git.local/lib.git", "commit": "1a2b3c"}
    assert envs.describe({"url": "file:///srv/lib", "dir_info": {"editable": True}})["kind"] == "editable"
    assert envs.describe({"url": "file:///srv/lib-1.0.whl", "archive_info": {}})["kind"] == "local"
    assert envs.describe({"url": "https://x/lib-1.0.whl", "archive_info": {}})["kind"] == "url"

    data = b"PK\x03\x04 a wheel"
    digest = hashlib.sha256(data).hexdigest()
    asked: list[str] = []

    def get(sha):
        asked.append(sha)
        return data

    wheels = [{"filename": "mylib-1.0-py3-none-any.whl", "sha256": digest}]
    folder = envs.fetch_wheels(tmp_path, wheels, get)
    assert (folder / "mylib-1.0-py3-none-any.whl").read_bytes() == data
    envs.fetch_wheels(tmp_path, wheels, get)  # already here with the right hash: not fetched again
    assert asked == [digest]
    with pytest.raises(RuntimeError, match="해시"):
        envs.fetch_wheels(tmp_path / "other", [{"filename": "b-1.0-py3-none-any.whl", "sha256": "0" * 64}], get)


def test_in_house_wheels_and_pinned_versions_reach_the_environment(client, admin, settings, tmp_path):
    from badamoa.hub import registry
    from badamoa.hub.db import Database
    from badamoa.hub.models import Node
    from badamoa.hub.train_models import Template

    wheel = b"PK\x03\x04" + b"0" * 64
    up = client.post(
        "/api/packages",
        params={"filename": "My_Lib-1.2.0-py3-none-any.whl"},
        content=wheel,
        headers={**admin, "Content-Type": "application/octet-stream"},
    )
    assert up.status_code == 200, up.text
    assert up.json()["name"] == "my-lib" and up.json()["version"] == "1.2.0"
    bad = client.post("/api/packages", params={"filename": "notes.txt"}, content=wheel, headers=admin)
    assert bad.status_code == 400 and "wheel" in bad.json()["detail"]
    assert [p["filename"] for p in client.get("/api/packages", headers=admin).json()] == ["My_Lib-1.2.0-py3-none-any.whl"]

    # a server fetches it by its hash (and nothing else)
    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    reg = client.post("/api/agent/register", json={"join_token": token, "agent_version": "0.53.0", "info": {}}).json()
    node = {"Authorization": f"Bearer {reg['node_token']}"}
    got = client.get(f"/api/agent/packages/{up.json()['sha256']}", headers=node)
    assert got.status_code == 200 and got.content == wheel
    assert client.get(f"/api/agent/packages/{'0' * 64}", headers=node).status_code == 404

    # a template that installs packages: the wheels to look in, and once its trial passed, the versions it installed
    extra = '\n[environment]\nrequirements = ["my-lib", "numpy"]\n'
    folder = tmp_path / "needs-env"
    shutil.copytree(shipped.builtin()["dummy"], folder, ignore=shutil.ignore_patterns("__pycache__"))
    text = (folder / "badamoa.toml").read_text(encoding="utf-8").replace('name = "dummy"', 'name = "needs"')
    (folder / "badamoa.toml").write_text(text.replace("hidden = true", "hidden = false") + extra, encoding="utf-8")
    data = template_format.bundle(folder)
    view = client.post("/api/templates", content=data, headers=admin).json()
    database = Database(settings.database_url)
    with database.session() as db:
        row = db.get(Template, view["id"])
        manifest = registry.manifest_of(row)
        server = db.get(Node, reg["node_id"])
        spec = registry.environment_for_run(db, manifest, "", server, row)
        assert spec["requirements"] == ["my-lib", "numpy"] and "pinned" not in spec  # no passed trial yet
        assert spec["wheels"] == [{"filename": "My_Lib-1.2.0-py3-none-any.whl", "sha256": up.json()["sha256"]}]
        row.trial = {"state": "ok", "installed": ["my-lib==1.2.0", "numpy==2.1.0+cpu"], "sources": {}}
        assert registry.environment_for_run(db, manifest, "", server, row)["pinned"] == ["my-lib==1.2.0", "numpy==2.1.0"]
    assert client.get(f"/api/templates/{view['id']}", headers=admin).json()["check"]["pinned"] == 2
    assert view["check"]["state"] == "pending"

    # admins get the version's code as uploaded, by a signed link
    code = client.get(f"/api/templates/{view['id']}", headers=admin).json()["code_url"]
    assert client.get(code).content == data
    assert client.get(code.replace("sig=", "sig=0")).status_code == 403
    assert client.delete(f"/api/packages/{up.json()['id']}", headers=admin).status_code == 200
    assert client.get("/api/packages", headers=admin).json() == []
    database.engine.dispose()


def test_the_trial_runs_on_a_developer_pc_too(tmp_path):
    from typer.testing import CliRunner

    from badamoa.cli import app

    runner = CliRunner()
    for name, fail, code in (("mine", False, 0), ("broken", True, 1)):
        folder = tmp_path / name
        folder.mkdir()
        template_format.unpack(_bundle(tmp_path / "b", name=name, fail=fail), folder)
        done = runner.invoke(app, ["template", "check", str(folder), "--run", "--home", str(tmp_path / "home")])
        assert done.exit_code == code, done.output
        assert ("실패 학습" in done.output) is fail and ("통과 학습" in done.output) is not fail
        assert "통과 시험 데이터" in done.output


def test_a_trial_takes_an_idle_gpu_and_runs_on_the_cpu_when_none_is(client, admin, settings):
    from test_resources import _beat, _claim, _gpu, _join

    from badamoa.hub.db import Database
    from badamoa.hub.models import Job

    node_id, node = _join(client, admin)
    database = Database(settings.database_url)
    wants = {"roles": ["compute"], "gpus": 1, "min_vram_mb": 0, "gpu_if_free": True}

    def queue():
        with database.session() as db:
            db.add(Job(type="template.check", params={}, target_node_id=node_id, requirements=wants))

    _beat(client, node, [_gpu(0, 20000, util=90), _gpu(1, 20000, util=90)])  # both in use by someone
    queue()
    job = _claim(client, node)
    assert job["type"] == "template.check" and job["gpu_indices"] == []  # not waiting: the CPU
    _beat(client, node, [_gpu(0, 20000, util=90), _gpu(1, 300)], running=[job["id"]])
    queue()
    assert _claim(client, node)["gpu_indices"] == [1]  # the idle one
    database.engine.dispose()


def test_autolabel_is_declared_and_tried_only_when_declared(hub, tmp_path):
    folder = tmp_path / "no-predict"
    shutil.copytree(shipped.builtin()["dummy"], folder, ignore=shutil.ignore_patterns("__pycache__"))
    text = (folder / "badamoa.toml").read_text(encoding="utf-8")
    (folder / "badamoa.toml").write_text(text.replace('predict = "python ops.py --config {config}"\n', ""), encoding="utf-8")
    with pytest.raises(template_format.TemplateError, match="predict 진입점"):
        template_format.load_manifest(folder)

    plain = _try(hub, _upload(hub, _bundle(tmp_path, name="plain", autolabel=False))["id"])
    assert plain["state"] == "passed", plain
    skipped = next(s for s in plain["steps"] if s["step"] == "autolabel")
    assert skipped["skipped"] and "하지 않음" in skipped["detail"]

    quiet = _try(hub, _upload(hub, _bundle(tmp_path, name="quiet", labels_logged=False))["id"])
    assert quiet["state"] == "failed" and quiet["failed"] == "autolabel", quiet
    assert "log_predictions" in quiet["reason"]


# ---- members' drafts, with proof they ran on the uploader's PC (decided 2026-11-19) -----------------


def test_members_an_admin_lets_upload_send_drafts_with_their_own_trial(hub, tmp_path):
    from typer.testing import CliRunner

    from badamoa.cli import app

    member_token = hub.member.headers["Authorization"].split()[1]
    folder = tmp_path / "mine"
    folder.mkdir()
    template_format.unpack(_bundle(tmp_path / "b", name="kims-model"), folder)
    runner = CliRunner()

    def push():
        return runner.invoke(app, ["template", "push", str(folder), "--hub", str(hub.base_url), "--token", member_token])

    # not allowed yet
    denied = push()
    assert denied.exit_code == 1 and "권한이 없습니다" in denied.output
    assert hub.member.get("/api/templates/kit").status_code == 403

    kim = next(u for u in hub.get("/api/users").json() if u["username"] == "kim")
    allowed = hub.patch(f"/api/users/{kim['id']}", json={"template_upload": True})
    assert allowed.status_code == 200 and allowed.json()["template_upload"] is True
    assert hub.member.get("/api/auth/me").json()["can_upload_templates"] is True
    kit = hub.member.get("/api/templates/kit").json()
    port = str(hub.base_url.port)
    assert kit["draft"] and f":{port} " in kit["push"] and "/install/simple/" in kit["install_uv"]
    assert "127.0.0.1" not in kit["push"] and "localhost" not in kit["push"]  # an address other PCs can reach

    # without having run it here: refused
    refused = push()
    assert refused.exit_code == 1 and "check . --run" in refused.output

    ran = runner.invoke(app, ["template", "check", str(folder), "--run", "--home", str(tmp_path / "home")])
    assert ran.exit_code == 0, ran.output
    assert (folder / template_format.REPORT).is_file()  # beside the code, never in the bundle
    import io
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(template_format.bundle(folder))).namelist()
    assert template_format.REPORT not in names and not any(n.startswith(".badamoa") for n in names)
    pushed = push()
    assert pushed.exit_code == 0, pushed.output

    mine = next(t for t in hub.member.get("/api/templates").json() if t["name"] == "kims-model")
    assert mine["draft"] and mine["uploaded_by"] == "kim" and mine["check"]["state"] == "pending"
    assert mine["local_check"]["ok"] and mine["local_check"]["sdk"] and mine["local_check"]["python"]
    assert mine["local_check"]["packages"] > 0 and "autolabel" in [s["step"] for s in mine["local_check"]["steps"]]
    admin_view = next(t for t in hub.get("/api/templates").json() if t["name"] == "kims-model")
    assert admin_view["draft"] and admin_view["local_check"]["by"] == "kim"
    assert hub.member.post(f"/api/templates/{mine['id']}/check").status_code == 403  # the trial is the admin's

    # nobody else sees a member's draft
    hub.post("/api/users", json={"username": "park", "password": "park-pass-123", "role": "member"})
    park = httpx.post(f"{hub.base_url}/api/auth/login", json={"username": "park", "password": "park-pass-123"}).json()
    others = httpx.get(f"{hub.base_url}/api/templates", headers={"Authorization": f"Bearer {park['token']}"}).json()
    assert "kims-model" not in {t["name"] for t in others}

    # changed after the local run: the old report is not for this bundle
    (folder / "train.py").write_text((folder / "train.py").read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    toml = folder / "badamoa.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace('version = "1"', 'version = "2"'), encoding="utf-8")
    stale = push()
    assert stale.exit_code == 1 and "바뀌었습니다" in stale.output


def test_a_newer_version_that_reads_older_weights_gives_older_models_what_it_adds(client, admin, settings, tmp_path):
    from sqlalchemy import select

    from badamoa.hub import modelops
    from badamoa.hub.catalog_models import Project
    from badamoa.hub.db import Database
    from badamoa.hub.models import Job, Node, User
    from badamoa.hub.train_models import ModelVersion, Template

    def upload(name, version, entries, reads=""):
        folder = tmp_path / f"{name}-{version}"
        shutil.copytree(shipped.builtin()["dummy"], folder, ignore=shutil.ignore_patterns("__pycache__"))
        text = (folder / "badamoa.toml").read_text(encoding="utf-8")
        text = text.replace('name = "dummy"', f'name = "{name}"').replace('version = "3"', f'version = "{version}"')
        for entry in ("eval", "export"):
            if entry not in entries:
                text = text.replace(f'{entry} = "python ops.py --config {{config}}"\n', "")
        if reads:
            text = text.replace("resume = true\n", f'resume = true\nreads_weights_from = "{reads}"\n')
        (folder / "badamoa.toml").write_text(text, encoding="utf-8")
        resp = client.post("/api/templates", content=template_format.bundle(folder), headers=admin)
        assert resp.status_code == 200, resp.text
        allowed = client.post(f"/api/templates/{resp.json()['id']}/allow", json={"reason": "시험"}, headers=admin)
        assert allowed.status_code == 200, allowed.text
        return resp.json()

    with pytest.raises(template_format.TemplateError, match="버전 범위"):
        bad = tmp_path / "bad"
        shutil.copytree(shipped.builtin()["dummy"], bad, ignore=shutil.ignore_patterns("__pycache__"))
        toml = bad / "badamoa.toml"
        toml.write_text(
            toml.read_text(encoding="utf-8").replace(
                "resume = true\n", 'resume = true\nreads_weights_from = "one or two"\n'
            ),
            encoding="utf-8",
        )
        template_format.load_manifest(bad)

    old = upload("team", "1.0.0", ["predict"])
    upload("team", "1.1.0", ["predict", "eval", "export"], reads="1.*")
    lone = upload("solo", "1.0.0", ["predict"])
    upload("solo", "1.1.0", ["predict", "eval", "export"])  # adds them, says nothing about older weights

    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    reg = client.post("/api/agent/register", json={"join_token": token, "agent_version": "0.56.0", "info": {}}).json()
    database = Database(settings.database_url)
    with database.session() as db:
        project = Project(name="p")
        db.add(project)
        db.flush()

        def model_of(row):
            made = ModelVersion(
                project_id=project.id, name=f"{row['name']}-m", version=1, blob="0" * 64, trainer=row["name"]
            )
            made.template = {"id": row["id"], "name": row["name"], "version": row["version"]}
            db.add(made)
            db.flush()
            return made

        older, alone = model_of(old), model_of(lone)
        assert modelops.actions(db, older) == ["eval", "export", "predict"]
        assert modelops.code_for(db, older, "export").version == "1.1.0"
        assert modelops.code_for(db, older, "predict").version == "1.0.0"  # its own code where it has it
        assert modelops.actions(db, alone) == ["predict"]
        assert modelops.code_for(db, alone, "export") is None

        # the job runs 1.1.0's code with the model's own weights, and says so
        boss = db.scalar(select(User).where(User.username == "admin"))
        op = modelops._start(db, boss, older, "export", db.get(Node, reg["node_id"]), {"format": "onnx"})
        template = db.get(Job, op.job_id).params["template"]
        assert (template["version"], template["trained_with"]) == ("1.1.0", "1.0.0")
        assert modelops._code_of(db.get(Job, op.job_id))["trained_with"] == "1.0.0"
        assert db.get(Template, old["id"]).version == "1.0.0"
    database.engine.dispose()
