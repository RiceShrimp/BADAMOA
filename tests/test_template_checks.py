"""Checks before anything runs: a template's code and declaration on upload, a version against the model
before a run starts, the node's credentials kept from code it runs, and a named task's label values kept."""

from __future__ import annotations

import io
import threading
import time
import zipfile

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_imagefolder, make_yolo

from badamoa.core import templates as fmt
from badamoa.node import agent as node_agent
from badamoa.node import envs, importer

DECLARATION = """
name = "inhouse"
version = "1.0.0"
task = "{task}"
data_format = "{data_format}"

[entrypoints]
train = "python train.py --config {{config}}"
eval = "python tools/eval.py --config {{config}}"
"""


def _bundle(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def _template(task="detect", data_format="yolo", **extra: str) -> dict[str, str]:
    return {
        "badamoa.toml": DECLARATION.format(task=task, data_format=data_format),
        "train.py": "print('train')\n",
        "tools/eval.py": "print('eval')\n",
        **extra,
    }


# ---- on upload ------------------------------------------------------------------------------


def test_a_bundle_with_everything_in_place_passes():
    manifest = fmt.read_bundle(_bundle(_template()))
    assert manifest.notes == [] and manifest.data.min_classes("detect") == 1


def test_a_script_an_entry_point_starts_must_be_in_the_template():
    files = _template()
    del files["tools/eval.py"]
    with pytest.raises(fmt.TemplateError, match="tools/eval.py"):
        fmt.read_bundle(_bundle(files))


def test_every_python_file_must_parse_without_being_run():
    files = _template(**{"model/blocks.py": "def broken(:\n    pass\n"})
    with pytest.raises(fmt.TemplateError, match="model/blocks.py 1행"):
        fmt.read_bundle(_bundle(files))


@pytest.mark.parametrize(
    ("task", "data_format", "outcome"),
    [
        ("classify", "yolo", r"imagefolder을\(를\) 쓰세요"),
        ("detect", "imagefolder", "풀 수 없습니다"),
        ("pose", "yolo-seg", "풀 수 없습니다"),
        ("OCR", "imagefolder", "풀 수 없습니다"),
    ],
)
def test_a_task_and_format_no_node_can_write_are_refused(task, data_format, outcome):
    with pytest.raises(fmt.TemplateError, match=outcome):
        fmt.read_bundle(_bundle(_template(task, data_format)))


@pytest.mark.parametrize(
    ("task", "note"),
    [("segment", "윤곽은 버리고"), ("pose", "점은 버리고"), ("OCR", "박스 뒤의 값은 버리고")],
)
def test_a_format_that_leaves_values_out_is_allowed_with_a_note(task, note):
    manifest = fmt.read_bundle(_bundle(_template(task, "yolo")))
    assert len(manifest.notes) == 1 and note in manifest.notes[0]


def test_what_data_a_template_takes_is_declared_and_checked():
    head = DECLARATION.format(task="classify", data_format="imagefolder")
    text = head + "\n[data]\nclasses = { max = 10 }\nmin_images = 20\n"
    manifest = fmt.parse_manifest(text)
    assert (manifest.data.min_classes("classify"), manifest.data.classes.max, manifest.data.min_images) == (2, 10, 20)
    with pytest.raises(fmt.TemplateError, match="max가 min보다"):
        fmt.parse_manifest(text.replace("max = 10", "min = 5, max = 3"))


def test_requirements_are_written_as_packages():
    pytest.importorskip("packaging")
    text = DECLARATION.format(task="detect", data_format="yolo") + '\n[environment]\nrequirements = ["REQ"]\n'
    for fine in ("torch>=2.3", "mylib @ git+https://git.example.com/mylib@v1", "./libs/mylib", "--extra-index-url x"):
        fmt.parse_manifest(text.replace("REQ", fine))
    with pytest.raises(fmt.TemplateError, match="torch>="):
        fmt.parse_manifest(text.replace("REQ", "torch>="))


# ---- on the node ----------------------------------------------------------------------------


def test_code_the_node_runs_does_not_get_its_token(monkeypatch):
    monkeypatch.setenv("BADAMOA_NODE_TOKEN", "secret")
    monkeypatch.setenv("BADAMOA_HUB_URL", "http://hub")
    env = envs.code_env({"PYTHONUNBUFFERED": "1"})
    assert "BADAMOA_NODE_TOKEN" not in env and env["BADAMOA_HUB_URL"] == "http://hub" and env["PYTHONUNBUFFERED"] == "1"


def test_a_named_tasks_label_values_are_kept_as_written(tmp_path):
    root = make_yolo(tmp_path / "plates")
    label = root / "labels" / "train" / "train_0000.txt"
    label.write_text("0 0.5 0.5 0.2 0.2 7.5 -3\n1 0.4 0.4 0.1 0.1 0.25\n", encoding="utf-8")
    image = next(e for e in importer.scan(root, task="OCR").entries if e.path.stem == "train_0000")
    assert image.boxes == [[0, 0.5, 0.5, 0.2, 0.2, 7.5, -3.0], [1, 0.4, 0.4, 0.1, 0.1, 0.25]]
    # read as detection data the same lines are a box (the second) and an outline (the first)
    plain = next(e for e in importer.scan(root, task="detect").entries if e.path.stem == "train_0000")
    assert plain.boxes[1] == [1, 0.4, 0.4, 0.1, 0.1] and len(plain.boxes[0]) > 7


# ---- before a run starts --------------------------------------------------------------------


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
        client.node_id = runner.config.node_id
        yield client
        runner.stop()


def _wait(hub, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    job = hub.get(f"/api/jobs/{job_id}").json()
    while time.monotonic() < deadline and job["status"] in ("queued", "running"):
        time.sleep(0.2)
        job = hub.get(f"/api/jobs/{job_id}").json()
    return job


def _import(hub, path, dataset, **extra):
    project = hub.post("/api/projects", json={"name": f"{dataset} 프로젝트"}).json()
    body = {"node_id": hub.node_id, "path": str(path), "dataset": dataset, "version": "v1", **extra}
    job = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", job
    return job["result"]["release_id"]


def test_a_version_that_does_not_fit_the_model_is_refused_before_it_starts(hub, tmp_path):
    # a classifier needs two classes at least: one class is turned away with the reason, before any job
    one = _import(hub, make_imagefolder(tmp_path / "one", classes=("cat",), held_split="test"), "one")
    check = hub.post("/api/runs/check", json={"release_id": one, "template": "classify"}).json()
    assert check["problems"] == ["이 모델은 클래스가 2개 이상이어야 합니다. 지금 1개입니다."]
    assert check["notes"] == ["이 버전에는 val이 없어 test로 검증합니다."]
    refused = hub.post("/api/runs", json={"release_id": one, "template": "classify", "gpus": 0})
    assert refused.status_code == 400 and "2개 이상" in refused.json()["detail"]
    assert not [j for j in hub.get("/api/jobs?limit=20").json() if j["type"] == "train.run"]

    # two classes, but only one picked: the same, with what to do
    two = _import(hub, make_imagefolder(tmp_path / "two", classes=("cat", "dog")), "two")
    picked = hub.post("/api/runs/check", json={"release_id": two, "template": "classify", "classes": [0]}).json()
    assert "학습할 클래스를 더 고르세요" in picked["problems"][0]
    assert hub.post("/api/runs/check", json={"release_id": two, "template": "classify"}).json() == {
        "problems": [],
        "notes": [],
    }

    # a model for another task, and settings it does not have
    street = _import(hub, make_yolo(tmp_path / "street"), "street")
    other = hub.post("/api/runs/check", json={"release_id": two, "template": "dummy"}).json()
    assert other["problems"] == ["테스트 학습은(는) Det용이고 이 데이터셋은 Cls입니다."]
    # classify was merged into torchvision, which trains Det too: a run that names it goes there
    merged = hub.post("/api/runs/check", json={"release_id": street, "template": "classify"}).json()
    assert merged["problems"] == []
    unknown = hub.post("/api/runs/check", json={"release_id": street, "template": "dummy", "params": {"nope": 1}}).json()
    assert "nope" in unknown["problems"][0]


def test_an_evaluation_on_a_split_without_images_is_refused(hub, tmp_path):
    street = _import(hub, make_yolo(tmp_path / "street"), "street")
    run = hub.post("/api/runs", json={"release_id": street, "template": "dummy", "params": {"epochs": 1}, "gpus": 0}).json()
    assert _wait(hub, run["job_id"])["status"] == "succeeded"
    model = hub.get(f"/api/runs/{run['id']}").json()["model"]["id"]
    refused = hub.post(f"/api/models/{model}/eval", json={"release_id": street, "split": "test", "gpus": 0})
    assert refused.status_code == 400 and "test에는 이미지가 없습니다" in refused.json()["detail"]
