"""One model kind for several tasks (0.60.0): the declaration, the pose data format, the Hub's choice of kind,
and a run that tells the code its task."""

from __future__ import annotations

import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_yolo
from typer.testing import CliRunner

from badamoa.cli import app as cli
from badamoa.core.templates import TemplateError, parse_manifest
from badamoa.node import agent as node_agent
from badamoa.node import exporter, samples

ENTRY = '[entrypoints]\ntrain = "python t.py --config {config}"\n'

SEVERAL = """
name = "several"
version = "1"
tasks = ["detect", "segment", "classify", "pose"]
[entrypoints]
train = "python train.py --config {config}"
[params.model]
type = "string"
default = "m.pt"
choices = ["m.pt", "l.pt"]
by_task.segment = { default = "m-seg.pt", choices = ["m-seg.pt"] }
by_task.Cls = { default = "m-cls.pt", choices = ["m-cls.pt"], label = "분류 모델" }
[params.mosaic]
type = "number"
default = 1
tasks = ["detect", "segment", "pose"]
"""


def test_a_kind_of_several_tasks_is_seen_per_task():
    manifest = parse_manifest(SEVERAL)
    assert manifest.task == "detect" and manifest.trains() == ["detect", "segment", "classify", "pose"]
    assert manifest.data_formats == {"detect": "yolo", "segment": "yolo-seg", "classify": "imagefolder", "pose": "yolo-pose"}
    seg, cls, pose = (manifest.for_task(t) for t in ("segment", "classify", "pose"))
    assert (seg.task, seg.data_format, seg.params["model"].default, seg.params["model"].choices) == (
        "segment", "yolo-seg", "m-seg.pt", ["m-seg.pt"],
    )  # fmt: skip
    assert (cls.data_format, cls.params["model"].label, sorted(cls.params)) == ("imagefolder", "분류 모델", ["model"])
    assert (pose.params["model"].default, sorted(pose.params)) == ("m.pt", ["model", "mosaic"])
    assert manifest.supports("pose") and not manifest.supports("ocr") and manifest.for_task("ocr") is manifest

    # one task, as before: nothing changes
    one = parse_manifest('name = "one"\nversion = "1"\ntask = "segment"\n' + ENTRY)
    assert (one.trains(), one.data_format, one.for_task("segment") is one) == (["segment"], "yolo", True)

    # what is turned away, in words
    for change, hint in (
        (('"pose"]', '"pose", "detect"]'), "겹치지 않게"),
        (("by_task.Cls", "by_task.ocr"), "ocr은(는) tasks에 없습니다"),
        (('default = "m-seg.pt"', 'default = "x.pt"'), "(Seg)"),
        (('version = "1"', 'version = "1"\ntask = "ocr"'), "tasks에 없습니다"),
    ):
        with pytest.raises(TemplateError, match=hint.replace("(", r"\(").replace(")", r"\)")):
            parse_manifest(SEVERAL.replace(*change))
    with pytest.raises(TemplateError, match="data_formats는 tasks와 함께"):
        parse_manifest('name = "x1"\nversion = "1"\ndata_formats = { detect = "yolo" }\n' + ENTRY)


def test_pose_data_is_written_with_its_points(tmp_path):
    """yolo-pose: each box with its points (x y visibility), kpt_shape and the left · right pairs in data.yaml."""
    items, classes, locations = samples.make(tmp_path / "s", "pose", classes=2, train=4, val=2)
    assert classes[0]["keypoints"] == samples.CORNERS
    options = exporter.ExportOptions(symlinks=False)
    summary = exporter.BUILDERS["yolo-pose"](items, classes, locations, tmp_path / "d", options)
    yaml = (tmp_path / "d" / "data.yaml").read_text(encoding="utf-8")
    assert "kpt_shape: [4, 3]" in yaml and "flip_idx: [1, 0, 3, 2]" in yaml
    line = next(iter(sorted((tmp_path / "d" / "labels" / "train").iterdir()))).read_text().splitlines()[0].split()
    assert len(line) == 1 + 4 + 4 * 3 and line[7] == "2"  # class, box, then x y v for each point
    assert summary["items"]["train"] == 4
    assert exporter._flip_pairs(["nose", "left_eye", "right_eye", "l_hand", "r_hand"]) == [0, 2, 1, 4, 3]
    assert exporter._flip_pairs(["a", "b"]) is None  # nothing pairs up: no flip_idx, the trainer does not flip


def test_built_in_kinds_by_task(client, admin):
    """Ultralytics YOLO is offered for every task, first for Det · Seg · Kpt; the merged kinds are hidden."""

    def names(task: str) -> list[str]:
        return [t["name"] for t in client.get("/api/templates", params={"task": task}, headers=admin).json()]

    for task in ("detect", "segment", "pose"):
        assert names(task)[0] == "yolo" and "yolo-seg" not in names(task)
    assert "yolo" in names("classify") and "classify" not in names("classify")
    views = {t["name"]: t for t in client.get("/api/templates", params={"task": "classify"}, headers=admin).json()}
    yolo = views["yolo"]
    assert yolo["tasks"] == ["detect", "segment", "classify", "pose"] and yolo["data_format"] == "imagefolder"
    settings = {p["name"]: p for p in yolo["params"]}
    assert settings["model"]["default"] == "yolo11n-cls.pt" and settings["imgsz"]["default"] == 224
    assert "mosaic" not in settings  # classification does not use it
    hidden = {t["name"]: t for t in client.get("/api/templates", params={"hidden": True}, headers=admin).json()}
    assert hidden["yolo-seg"]["replaced_by"] == "yolo" and hidden["yolo-seg"]["hidden"] is True


def test_a_project_that_chose_a_merged_kind_trains_with_its_successor(client, admin, settings):
    from badamoa.hub import registry
    from badamoa.hub.db import Database

    assert client.get("/api/templates", headers=admin).status_code == 200  # the Hub has started: built-ins synced
    database = Database(settings.database_url)
    with database.session() as db:
        assert registry.template_for(db, {"template": "yolo-seg"}, "segment").name == "yolo"
        assert registry.template_for(db, {}, "pose").name == "yolo"
        assert registry.template_for(db, {}, "segment", "yolo-seg").name == "yolo"
        assert registry.manifest_of(registry.get_template(db, "yolo"), "segment").data_format == "yolo-seg"
    database.engine.dispose()


# ---- end to end: the code is told its task and gets that task's data -------------------------

PROBE_TOML = """
name = "task-probe"
title = "Task probe"
version = "1.0.0"
tasks = ["detect", "segment"]

[entrypoints]
train = "python train.py --config {config}"
"""

PROBE_CODE = textwrap.dedent(
    """
    from pathlib import Path

    import badamoa_sdk as bm

    run = bm.init()
    labels = Path(run.data).parent.glob("labels/train/*.txt")
    values = max(len(line.split()) for label in labels for line in label.read_text().splitlines() if line.strip())
    print(f"task={run.task} values={values}", flush=True)
    weights = run.out / "best.pt"
    weights.write_bytes(b"probe")
    bm.log_model(weights)
    bm.finish({"map50_95": 0.1, "mask_map50_95": 0.2})
    """
)


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
        client.api_token = client.post("/api/tokens", json={"name": "cli"}).json()["token"]
        yield client
        runner.stop()


def _wait(hub, job_id: str, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = hub.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "canceled"):
            return job
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def _logs(hub, job_id: str) -> str:
    return "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job_id}/logs").json())


def _make_segments(root: Path) -> Path:
    """A tiny YOLO segmentation folder: every image one square, labelled by its outline."""
    from PIL import Image

    for split, count in (("train", 4), ("val", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (64, 64), (40 + i * 20, 90, 120)).save(root / "images" / split / f"{split}_{i}.png")
            (root / "labels" / split / f"{split}_{i}.txt").write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
    (root / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [square]\n")
    return root


def _import(hub, project_id: str, path: Path, dataset: str, task: str) -> str:
    body = {"node_id": hub.node_id, "path": str(path), "dataset": dataset, "version": "v1", "task": task}
    job = _wait(hub, hub.post(f"/api/projects/{project_id}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    return job["result"]["release_id"]


def test_one_kind_trains_each_dataset_with_its_task(hub, tmp_path):
    folder = tmp_path / "probe"
    folder.mkdir()
    (folder / "badamoa.toml").write_text(PROBE_TOML, encoding="utf-8")
    (folder / "train.py").write_text(PROBE_CODE, encoding="utf-8")
    pushed = CliRunner().invoke(cli, ["template", "push", str(folder), "--hub", hub.url, "--token", hub.api_token])
    assert pushed.exit_code == 0, pushed.output
    probe = next(t for t in hub.get("/api/templates", params={"hidden": True}).json() if t["name"] == "task-probe")
    assert hub.post(f"/api/templates/{probe['id']}/allow", json={"reason": "test"}).status_code == 200

    project = hub.post("/api/projects", json={"name": "여러 태스크"}).json()
    boxes = _import(hub, project["id"], make_yolo(tmp_path / "boxes"), "boxes", "detect")
    outlines = _import(hub, project["id"], _make_segments(tmp_path / "segs"), "outlines", "segment")
    for release, task, values, fmt in ((boxes, "detect", 5, "yolo"), (outlines, "segment", 9, "yolo-seg")):
        body = {"release_id": release, "template": "task-probe", "params": {}, "gpus": 0}
        run = hub.post("/api/runs", json=body)
        assert run.status_code == 200, run.text
        job = _wait(hub, run.json()["job_id"])
        assert job["status"] == "succeeded", _logs(hub, job["id"])
        assert f"task={task} values={values}" in _logs(hub, job["id"])
        sent = job["params"]["template"]
        assert (sent["task"], sent["data_format"]) == (task, fmt)
