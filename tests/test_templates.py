"""Model templates and environments: the declaration format, registering in-house code, running it.

The custom template here is the skeleton ``badamoa template init`` writes, pushed with
``badamoa template push`` to a real Hub and trained by a real Node Agent.
"""

from __future__ import annotations

import sys
import threading
import time

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_yolo
from typer.testing import CliRunner

from badamoa import templates as shipped
from badamoa.cli import app as cli
from badamoa.core import templates as fmt
from badamoa.node import agent as node_agent
from badamoa.node import trainer

MANIFEST = """
name = "inhouse-det"
title = "사내 검출기"
version = "1.0.0"
variant = "backbone"

[environment]
default = "lab-conda"

[entrypoints]
train = "python train.py --config {config}"

[params.backbone]
type = "string"
choices = ["r18", "r50"]
default = "r18"

[params.epochs]
type = "integer"
default = 10
minimum = 1

[params.lr]
type = "number"
default = 0.001
advanced = true
"""


# ---- the declaration -----------------------------------------------------------------


def test_manifest_defaults_and_param_checks():
    manifest = fmt.parse_manifest(MANIFEST)
    assert manifest.title == "사내 검출기" and manifest.task == "detect" and manifest.data_format == "yolo"
    assert fmt.resolve_params(manifest, {}) == {"backbone": "r18", "epochs": 10, "lr": 0.001}
    assert fmt.resolve_params(manifest, {"epochs": "30", "backbone": "r50"})["epochs"] == 30
    for bad, hint in [({"epochs": 0}, "1 이상"), ({"backbone": "vit"}, "r18, r50"), ({"momentum": 0.9}, "momentum")]:
        with pytest.raises(fmt.TemplateError, match=hint):
            fmt.resolve_params(manifest, bad)


@pytest.mark.parametrize(
    ("change", "hint"),
    [
        (('name = "inhouse-det"', 'name = "In House"'), "name은"),
        (('train = "python train.py --config {config}"', 'train = "python train.py"'), "{config}"),
        (('train = "python train.py --config {config}"', 'serve = "python s.py --config {config}"'), "알 수 없는 진입점"),
        (("[params.epochs]", "[params.classes]"), "BADAMOA가 쓰는 이름"),
        (("default = 10", 'default = "many"'), "기본값"),
        (('variant = "backbone"', 'variant = "arch"'), "variant"),
        (('version = "1.0.0"', "version = 1.0"), "형식이 맞지 않습니다"),
    ],
)
def test_manifest_problems_are_reported_in_words(change, hint):
    with pytest.raises(fmt.TemplateError, match=hint):
        fmt.parse_manifest(MANIFEST.replace(*change))


def test_shipped_templates_are_valid_and_the_command_uses_the_environment_python(tmp_path):
    names = set(shipped.builtin())
    assert {"yolo", "dummy"} <= names and "_scaffold" not in names
    for folder in shipped.builtin().values():
        manifest = fmt.load_manifest(folder)
        assert (folder / "train.py").is_file() and "train" in manifest.entrypoints
    python, config = tmp_path / "env" / "python", tmp_path / "run 1" / "config.json"
    command = trainer.command_for("python train.py --config {config}", python, config)
    assert command == [str(python), "train.py", "--config", str(config)]
    assert trainer.command_for("bash run.sh {config}", python, config)[0] == "bash"


def test_bundle_is_stable_and_refuses_unsafe_archives(tmp_path):
    import io
    import zipfile

    folder = tmp_path / "t"
    (folder / "__pycache__").mkdir(parents=True)
    (folder / "badamoa.toml").write_text(MANIFEST, encoding="utf-8")
    (folder / "train.py").write_text("print('hi')\n", encoding="utf-8")
    (folder / "__pycache__" / "train.cpython-312.pyc").write_bytes(b"junk")
    first = fmt.bundle(folder)
    assert first == fmt.bundle(folder) and fmt.read_bundle(first).name == "inhouse-det"
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert sorted(archive.namelist()) == ["badamoa.toml", "train.py"]  # caches are left out

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("badamoa.toml", MANIFEST)
        archive.writestr("../outside.py", "x")
    with pytest.raises(fmt.TemplateError, match="허용되지 않는 경로"):
        fmt.read_bundle(evil.getvalue())
    with pytest.raises(fmt.TemplateError, match="zip 파일이 아닙니다"):
        fmt.read_bundle(b"not a zip")


# ---- registering and running in-house code ---------------------------------------------


@pytest.fixture
def hub(live_hub, tmp_path, monkeypatch):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        join = client.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="train-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        client.url = live_hub
        client.node_id = runner.config.node_id
        client.node_home = home
        client.api_token = client.post("/api/tokens", json={"name": "cli"}).json()["token"]
        yield client
        runner.stop()


def _wait(hub, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    job = hub.get(f"/api/jobs/{job_id}").json()
    while time.monotonic() < deadline and job["status"] in ("queued", "running"):
        time.sleep(0.2)
        job = hub.get(f"/api/jobs/{job_id}").json()
    return job


def _logs(hub, job_id):
    return "\\n".join(line["text"] for line in hub.get(f"/api/jobs/{job_id}/logs").json())


@pytest.fixture
def release(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "주차장"}).json()
    body = {"node_id": hub.node_id, "path": str(make_yolo(tmp_path / "ds")), "dataset": "street", "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    return {"project_id": project["id"], "id": job["result"]["release_id"]}


def _push(hub, folder, token=None):
    return CliRunner().invoke(cli, ["template", "push", str(folder), "--hub", hub.url, "--token", token or hub.api_token])


def test_init_push_and_train_an_in_house_template(hub, release, tmp_path):
    folder = tmp_path / "My Detector"
    cli_run = CliRunner()
    created = cli_run.invoke(cli, ["template", "init", str(folder)])
    assert created.exit_code == 0, created.output
    assert cli_run.invoke(cli, ["template", "check", str(folder)]).exit_code == 0
    assert cli_run.invoke(cli, ["template", "init", str(folder)]).exit_code == 1  # never overwrites

    pushed = _push(hub, folder)
    assert pushed.exit_code == 0 and "my-detector 0.1.0" in pushed.output, pushed.output
    assert _push(hub, folder).exit_code == 0  # the same content again is fine
    listed = {t["name"]: t for t in hub.get("/api/templates").json()}
    assert set(listed) == {"yolo", "my-detector"} and listed["yolo"]["builtin"] is True
    mine = listed["my-detector"]
    assert (mine["title"], mine["versions"], mine["entrypoints"]) == ("My Detector", 1, ["train"])
    assert [p["name"] for p in mine["params"]] == ["model", "epochs", "lr"] and mine["params"][2]["advanced"] is True

    # ---- train with it: metrics arrive through log(), the bar through progress()
    body = {"release_id": release["id"], "template": "my-detector", "params": {"epochs": 3, "model": "large"}, "gpus": 0}
    run = hub.post("/api/runs", json=body).json()
    job = _wait(hub, run["job_id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    done = hub.get(f"/api/runs/{run['id']}").json()
    assert done["template"]["title"] == "My Detector" and done["template"]["version"] == "0.1.0"
    assert done["environment"] == {"kind": "auto", "name": "자동 설치", "requirements": [], "python": "3.12", "torch": False}
    assert (done["epoch"], done["epochs"], done["history"]["steps"]) == (3, 3, [1, 2, 3])
    assert done["history"]["series"]["map50_95"] == [pytest.approx(0.5 * n / 3) for n in (1, 2, 3)]
    assert done["metrics"]["map50_95"] == 0.5 and done["model"]["name"] == "street-large"
    assert {"name": "lr", "label": "학습률", "value": 0.001} in done["settings"]
    assert "학습 이미지 5장" in _logs(hub, job["id"])
    cached = list((hub.node_home / "cache" / "templates").iterdir())
    assert len(cached) == 1 and (cached[0] / "train.py").is_file()

    # ---- changed code needs a new version; the old run can still be repeated with its own version
    script = folder / "train.py"
    script.write_text(script.read_text(encoding="utf-8").replace("0.5 * epoch", "0.25 * epoch"), encoding="utf-8")
    refused = _push(hub, folder)
    assert refused.exit_code == 1 and "version을 올리세요" in refused.output
    toml = folder / "badamoa.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace('"0.1.0"', '"0.2.0"'), encoding="utf-8")
    assert _push(hub, folder).exit_code == 0
    latest = next(t for t in hub.get("/api/templates").json() if t["name"] == "my-detector")
    assert (latest["version"], latest["versions"]) == ("0.2.0", 2)
    again = hub.post(f"/api/runs/{run['id']}/rerun").json()
    assert again["template"]["version"] == "0.1.0" and _wait(hub, again["job_id"])["status"] == "succeeded"
    newer = hub.post("/api/runs", json=body).json()
    assert newer["template"]["version"] == "0.2.0" and _wait(hub, newer["job_id"])["status"] == "succeeded"
    assert hub.get(f"/api/runs/{newer['id']}").json()["metrics"]["map50_95"] == 0.25

    # ---- only admins register templates
    hub.post("/api/users", json={"username": "member", "password": "member-test-pw", "role": "member"})
    login = httpx.post(f"{hub.url}/api/auth/login", json={"username": "member", "password": "member-test-pw"}).json()
    member = {"Authorization": f"Bearer {login['token']}"}
    assert httpx.post(f"{hub.url}/api/templates", content=fmt.bundle(folder), headers=member).status_code == 403
    assert len(httpx.get(f"{hub.url}/api/templates", headers=member).json()) == 2  # but everyone can pick one
    assert hub.delete(f"/api/templates/{listed['yolo']['id']}").status_code == 400
    assert hub.delete(f"/api/templates/{latest['id']}").status_code == 200
    assert [t["name"] for t in hub.get("/api/templates").json()] == ["yolo"]
    assert hub.get(f"/api/runs/{run['id']}").json()["template"]["title"] == "My Detector"  # the record stays


def test_registered_environment_and_project_default(hub, release):
    # An environment that exists on the server already: here, the Python running the tests.
    body = {"name": "lab-conda", "description": "연구실 공용 환경", "paths": {hub.node_id: sys.executable}}
    env = hub.post("/api/environments", json=body).json()
    assert env["nodes"][0]["ok"] is None and env["nodes"][0]["node_name"] == "train-node"
    assert hub.post("/api/environments", json=body).status_code == 409
    for job_id in hub.post(f"/api/environments/{env['id']}/check").json()["jobs"]:
        assert _wait(hub, job_id)["status"] == "succeeded", _logs(hub, job_id)
    checked = hub.get("/api/environments").json()[0]["nodes"][0]
    assert checked["ok"] is True and checked["detail"]["python"] == sys.version.split()[0]

    run = hub.post(
        "/api/runs", json={"release_id": release["id"], "template": "dummy", "environment": "lab-conda", "gpus": 0}
    ).json()
    assert run["environment"] == {"kind": "python", "id": env["id"], "name": "lab-conda"}  # no server path shown
    assert _wait(hub, run["job_id"])["status"] == "succeeded", _logs(hub, run["job_id"])

    # A path that does not work is reported by the check, and a run fails with something to act on.
    broken = hub.put(f"/api/environments/{env['id']}", json={**body, "paths": {hub.node_id: "/no/such/python"}}).json()
    assert broken["nodes"][0]["ok"] is None  # a changed path is unchecked again
    for job_id in hub.post(f"/api/environments/{env['id']}/check").json()["jobs"]:
        assert _wait(hub, job_id)["status"] == "failed"
    failed = hub.get("/api/environments").json()[0]["nodes"][0]
    assert failed["ok"] is False and "종료 코드" in failed["detail"]["error"]
    bad = hub.post(
        "/api/runs", json={"release_id": release["id"], "template": "dummy", "environment": "lab-conda", "gpus": 0}
    ).json()
    assert _wait(hub, bad["job_id"])["status"] == "failed" and "설정 → 환경" in _logs(hub, bad["job_id"])

    # Not set up on this server at all: refused before anything starts.
    hub.put(f"/api/environments/{env['id']}", json={**body, "paths": {}})
    refused = hub.post(
        "/api/runs", json={"release_id": release["id"], "template": "dummy", "environment": "lab-conda", "gpus": 0}
    )
    assert refused.status_code == 400 and "준비되어 있지 않습니다" in refused.json()["detail"]
    assert hub.post("/api/runs", json={"release_id": release["id"], "environment": "nope"}).status_code == 404
    assert hub.delete(f"/api/environments/{env['id']}").status_code == 200

    # ---- the project's default template is what an empty request gets
    assert hub.put(f"/api/projects/{release['project_id']}/settings", json={"template": "nope"}).status_code == 404
    assert hub.put(f"/api/projects/{release['project_id']}/settings", json={"template": "dummy"}).json() == {
        "template": "dummy"
    }
    assert hub.get("/api/projects").json()[0]["settings"] == {"template": "dummy"}
    default = hub.post("/api/runs", json={"release_id": release["id"], "gpus": 0}).json()
    assert default["template"]["name"] == "dummy" and default["environment"]["kind"] == "auto"
    assert _wait(hub, default["job_id"])["status"] == "succeeded"
