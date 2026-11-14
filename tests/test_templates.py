"""Model templates and environments: the declaration format, registering in-house code, running it.

The custom template here is the skeleton ``badamoa template init`` writes, pushed with
``badamoa template push`` to a real Hub and trained by a real Node Agent.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

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
        (("advanced = true", 'advanced = true\nunused_when = { optimizer = ["auto"] }'), "optimizer라는 설정이 없습니다"),
    ],
)
def test_manifest_problems_are_reported_in_words(change, hint):
    with pytest.raises(fmt.TemplateError, match=hint):
        fmt.parse_manifest(MANIFEST.replace(*change))


def test_shipped_templates_are_valid_and_the_command_uses_the_environment_python(tmp_path):
    names = set(shipped.builtin())
    assert names == {"yolo", "yolo-seg", "torchvision", "classify", "dummy"}  # folders starting with _ are not templates
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


CODE = """
import badamoa_sdk as bm

hp = bm.params(
    model=bm.choice(["r18", "r50"], label="모델"),
    epochs=bm.integer(30, min=1, max=1000, label="epoch"),
    lr=bm.number(1e-3, min=0, label="학습률", group="최적화", advanced=True),
    amp=bm.boolean(True, label="혼합 정밀도", group="최적화", advanced=True),
    resume=bm.text(label="이어 학습할 가중치", advanced=True),
)
other = something.params(weights)  # not a declaration: left alone
"""


def test_settings_declared_in_code_are_read_without_running_it():
    declared = fmt.params_from_code(CODE)
    assert list(declared) == ["model", "epochs", "lr", "amp", "resume"]
    model = declared["model"]
    assert (model.choices, model.default, model.type, model.label) == (["r18", "r50"], "r18", "string", "모델")
    assert (declared["epochs"].minimum, declared["epochs"].maximum, declared["epochs"].advanced) == (1, 1000, False)
    assert (declared["lr"].default, declared["lr"].group, declared["lr"].advanced) == (0.001, "최적화", True)
    assert declared["amp"].type == "boolean" and declared["resume"].default == ""

    # the declaration file can add to and override what the code declares
    manifest = fmt.parse_manifest(MANIFEST.replace('variant = "backbone"', 'variant = "model"'), CODE)
    assert list(manifest.params) == ["model", "epochs", "lr", "amp", "resume", "backbone"]
    assert manifest.params["epochs"].default == 10 and manifest.params["lr"].group == ""  # from the file

    with pytest.raises(fmt.TemplateError, match="3행, 설정 lr.*직접 적어야"):
        fmt.params_from_code("import badamoa_sdk as bm\nBASE = 0.1\nhp = bm.params(lr=bm.number(BASE * 2))")
    with pytest.raises(fmt.TemplateError, match="목록이 필요"):
        fmt.params_from_code("hp = bm.params(model=bm.choice([]))")
    with pytest.raises(fmt.TemplateError, match="BADAMOA가 쓰는 이름"):
        fmt.parse_manifest(MANIFEST, "hp = bm.params(classes=bm.integer(3))")


def test_sdk_gives_chosen_values_and_defaults(tmp_path, capsys, monkeypatch):
    import importlib
    import json

    monkeypatch.syspath_prepend(str(shipped.SDK))
    bm = importlib.import_module("badamoa_sdk")
    declare = {"epochs": bm.integer(30), "lr": bm.number(0.001), "model": bm.choice(["a", "b"]), "amp": bm.boolean(True)}

    bm.load_config([])  # run outside BADAMOA: the declared defaults
    assert bm.params(**declare) == {"epochs": 30, "lr": 0.001, "model": "a", "amp": True}

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"params": {"epochs": 5, "lr": 1, "amp": False}, "out": str(tmp_path / "o")}))
    bm.load_config(["train.py", "--config", str(config)])
    hp = bm.params(**declare)
    assert (hp.epochs, hp.lr, hp["model"], hp.amp) == (5, 1.0, "a", False) and isinstance(hp.lr, float)
    with pytest.raises(AttributeError):
        _ = hp.momentum

    bm.init()
    bm.log({"loss": 0.5, "bad": float("nan"), "val/acc": 0.9})
    bm.log_model(tmp_path / "o" / "w.pt")
    bm.finish()
    events = [json.loads(line[10:]) for line in capsys.readouterr().out.splitlines() if line.startswith("@@badamoa ")]
    assert [e["event"] for e in events] == ["info", "log", "result"]
    assert events[1] == {"event": "log", "step": 1, "metrics": {"loss": 0.5, "val_acc": 0.9}}
    assert events[2]["metrics"] == {"loss": 0.5, "val_acc": 0.9} and events[2]["weights"].endswith("w.pt")
    bm.load_config([])


def test_sdk_records_images_and_tables_by_step(tmp_path, capsys, monkeypatch):
    import importlib
    import json

    monkeypatch.syspath_prepend(str(shipped.SDK))
    bm = importlib.import_module("badamoa_sdk")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"params": {}, "out": str(tmp_path / "o")}))
    bm.load_config(["train.py", "--config", str(config)])
    run = bm.init()
    picture = run.out / "pred.jpg"
    picture.write_bytes(b"jpg")
    bm.log({"loss": 1.0}, step=3)
    name = bm.log_image(picture, key="예측/val", caption="고양이")  # the step of the last log
    plain = bm.log_image(picture)  # without a key: an ordinary file
    bm.log_table("val preds", ["image", "score"], [[name, 0.91234567], [None, float("nan")]], step=4)
    events = [json.loads(line[10:]) for line in capsys.readouterr().out.splitlines() if line.startswith("@@badamoa ")]
    image, artifact, table = [e for e in events if e["event"] == "artifact"]
    assert name == "예측-val/000003-pred.jpg" and plain == "pred.jpg"
    assert (image["key"], image["step"], image["caption"], image["name"]) == ("예측-val", 3, "고양이", name)
    assert "key" not in artifact and artifact["name"] == "pred.jpg"
    assert (table["key"], table["step"], table["media"], table["name"]) == (
        "val preds",
        4,
        "table",
        "val preds/000004.table.json",
    )
    stored = json.loads(Path(table["path"]).read_text(encoding="utf-8"))
    assert stored == {"columns": ["image", "score"], "rows": [[name, 0.912346], [None, None]]}
    bm.load_config([])


def test_yolo_hyperparameters_are_detailed_settings():
    manifest = fmt.load_manifest(shipped.builtin()["yolo"])
    basic = [name for name, spec in manifest.params.items() if not spec.advanced]
    assert basic == ["model", "epochs"]
    groups = {spec.group for spec in manifest.params.values() if spec.advanced and spec.type != "object"}
    assert groups == {"학습", "최적화", "증강"}
    chosen = fmt.resolve_params(manifest, {"lr0": "0.002", "optimizer": "AdamW", "mosaic": 0})
    assert (chosen["lr0"], chosen["optimizer"], chosen["mosaic"], chosen["momentum"]) == (0.002, "AdamW", 0.0, 0.937)
    with pytest.raises(fmt.TemplateError, match="옵티마이저"):
        fmt.resolve_params(manifest, {"optimizer": "Lion"})
    assert manifest.params["lr0"].unused_when == {"optimizer": ["auto"]}  # Ultralytics picks it itself then
    classify = fmt.load_manifest(shipped.builtin()["classify"])  # declared in code, read without running it
    assert classify.params["momentum"].unused_when == {"optimizer": ["AdamW"]}

    # the torchvision template declares everything in code
    torchvision = fmt.load_manifest(shipped.builtin()["torchvision"])
    assert "fasterrcnn_resnet50_fpn" in torchvision.params["model"].choices
    assert torchvision.params["lr"].group == "최적화" and torchvision.variant == "model"


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

    (folder / "badamoa_sdk.py").unlink()  # the node provides the SDK; a template need not ship it
    pushed = _push(hub, folder)
    assert pushed.exit_code == 0 and "my-detector 0.1.0" in pushed.output, pushed.output
    assert _push(hub, folder).exit_code == 0  # the same content again is fine
    listed = {t["name"]: t for t in hub.get("/api/templates").json()}
    assert set(listed) == {"yolo", "yolo-seg", "classify", "torchvision", "my-detector"}
    assert listed["yolo"]["builtin"] is True
    mine = listed["my-detector"]
    assert (mine["title"], mine["versions"], mine["entrypoints"]) == ("My Detector", 1, ["train"])
    # the settings come from bm.params(...) in train.py, read without running it
    assert [p["name"] for p in mine["params"]] == ["model", "epochs", "lr", "weight_decay", "batch", "augment"]
    assert [(p["group"], p["advanced"]) for p in mine["params"][1:4]] == [("", False), ("최적화", True), ("최적화", True)]

    # ---- train with it: metrics arrive through log(), the bar through progress()
    chosen = {"epochs": 3, "model": "large", "weight_decay": 0.01}
    body = {"release_id": release["id"], "template": "my-detector", "params": chosen, "gpus": 0}
    run = hub.post("/api/runs", json=body).json()
    job = _wait(hub, run["job_id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    done = hub.get(f"/api/runs/{run['id']}").json()
    assert done["template"]["title"] == "My Detector" and done["template"]["version"] == "0.1.0"
    assert done["environment"] == {"kind": "auto", "name": "자동 설치", "requirements": [], "python": "3.12", "torch": False}
    assert (done["epoch"], done["epochs"], done["history"]["steps"]) == (3, 3, [1, 2, 3])
    assert done["history"]["series"]["map50_95"] == [pytest.approx(0.5 * n / 3) for n in (1, 2, 3)]
    assert done["metrics"]["map50_95"] == 0.5 and done["model"]["name"] == "street-large"
    settings = {s["name"]: s for s in done["settings"]}
    assert settings["lr"] == {
        "name": "lr", "label": "학습률", "value": 0.001, "group": "최적화", "advanced": True, "changed": False
    }  # fmt: skip
    assert settings["weight_decay"]["changed"] is True and settings["augment"]["value"] is True
    assert "모델 large · 학습률 0.001" in _logs(hub, job["id"])  # hp.* in the code got the chosen values
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
    assert len(httpx.get(f"{hub.url}/api/templates", headers=member).json()) == 5  # but everyone can pick one
    assert hub.delete(f"/api/templates/{listed['yolo']['id']}").status_code == 400
    assert hub.delete(f"/api/templates/{latest['id']}").status_code == 200
    assert [t["name"] for t in hub.get("/api/templates").json()] == ["yolo", "yolo-seg", "classify", "torchvision"]
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
