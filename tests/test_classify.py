"""Classification data (one folder per class) and what a run can record besides metrics.

The template used here is a small in-house one written with the SDK: it needs no
torch, reads the image-folder export, logs many steps quickly and records files.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_imagefolder, make_yolo

from badamoa.core import templates as fmt
from badamoa.node import agent as node_agent
from badamoa.node import exporter, importer

MANIFEST = """
name = "tiny-classifier"
title = "작은 분류기"
version = "1.0.0"
task = "classify"
data_format = "imagefolder"

[entrypoints]
train = "python train.py --config {config}"
"""

TRAIN = '''
import struct, zlib
import badamoa_sdk as bm

hp = bm.params(steps=bm.integer(40, min=1, label="step"), best_at=bm.integer(25, label="가장 좋은 step", advanced=True))


def png(path):  # a 1x1 picture, written by hand so the template needs nothing installed
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = zlib.compress(b"\\x00\\xff\\x00\\x00")
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    path.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", header) + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


run = bm.init(framework="none")
folders = sorted(p.name for p in (run.data_dir / "train").iterdir())
counts = {split: len(list((run.data_dir / split).rglob("*.jpg"))) for split in ("train", "val")}
print("folders", folders, "counts", counts)
for step in range(1, hp.steps + 1):  # logged as fast as possible: the node sends them in batches
    score = 1 - abs(step - hp.best_at) / hp.steps
    bm.log({"top1": score, "train/loss": 1 / step}, step=step)
bm.progress(hp.steps, hp.steps)
picture, report = run.out / "confusion_matrix.png", run.out / "report.txt"
png(picture)
report.write_text("ok")
bm.log_artifact(picture)
bm.log_artifact(report)
bm.log_artifact(run.out / "missing.txt")  # a mistake in the code must not fail the run
weights = run.out / "model.bin"
weights.write_bytes(b"w")
per_class = [{"id": i, "name": n, "precision": 1, "recall": 0.5, "f1": 0.66} for i, n in enumerate(run.classes)]
bm.log_model(weights, classes=per_class)
bm.finish()
'''


# ---- reading and writing class folders (pure) ---------------------------------------------


def test_scan_reads_class_folders_with_and_without_splits(tmp_path):
    found = importer.scan(make_imagefolder(tmp_path / "a"))
    assert found.format == "imagefolder" and found.classes == ["bird", "cat", "dog"]  # sorted names
    assert len(found.entries) == 18 and {e.split for e in found.entries} == {"train", "val"}
    cat = next(e for e in found.entries if e.path.name == "cat_train_000.jpg")
    assert (cat.state, cat.boxes, cat.split) == ("labelled", [[1]], "train")

    flat = make_imagefolder(tmp_path / "b") / "train"  # just <class>/<image>
    assert {e.split for e in importer.scan(flat).entries} == {""} and len(importer.scan(flat).entries) == 12
    assert importer.scan(make_yolo(tmp_path / "c")).format == "yolo"  # a detection layout is still detection


def _manifest_items(root, entries, classes):
    import hashlib

    items = []
    for n, entry in enumerate(entries):
        raw = entry.path.read_bytes()
        items.append(
            {
                "id": n + 1,
                "sha256": hashlib.sha256(raw + str(n).encode()).hexdigest(),
                "rel_path": entry.path.relative_to(root).as_posix(),
                "bytes": len(raw),
                "location_id": "loc",
                "split": entry.split,
                "state": entry.state,
                "boxes": entry.boxes,
                "labelled_class_ids": list(range(len(classes))),
            }
        )
    return items, [{"id": i, "name": name} for i, name in enumerate(classes)], {"loc": str(root)}


def test_imagefolder_export_orders_classes_and_finds_a_val_split(tmp_path):
    root = make_imagefolder(tmp_path / "ds", held_split="test")
    found = importer.scan(root)
    items, classes, locations = _manifest_items(root, found.entries, found.classes)
    unlabelled = next(i for i in items if i["split"] == "train")  # the first one is a bird
    unlabelled["state"], unlabelled["boxes"] = "unlabelled", []

    out = tmp_path / "out"
    summary = exporter.build_imagefolder(items, classes, locations, out, exporter.ExportOptions())
    assert summary["val_source"] == "test" and summary["items"] == {"train": 11, "val": 6, "test": 0}
    assert summary["excluded"]["unlabelled"] == 1 and summary["classes"] == ["bird", "cat", "dog"]
    assert sorted(p.name for p in (out / "train").iterdir()) == ["0000_bird", "0001_cat", "0002_dog"]
    assert len(list((out / "val" / "0002_dog").iterdir())) == 2

    subset = exporter.build_imagefolder(items, classes, locations, tmp_path / "two", exporter.ExportOptions(classes=[2, 1]))
    assert subset["classes"] == ["cat", "dog"] and subset["items"]["train"] == 8  # label-space order, renumbered
    assert sorted(p.name for p in (tmp_path / "two" / "val").iterdir()) == ["0000_cat", "0001_dog"]

    held = [i for i in items if i["split"] == "train"]
    alone = exporter.build_imagefolder(held, classes, locations, tmp_path / "h", exporter.ExportOptions(val_fraction=0.5))
    assert alone["val_source"] == "holdout" and alone["items"]["val"] > 0
    assert exporter.BUILDERS["imagefolder"] is exporter.build_imagefolder


# ---- a classification project end to end --------------------------------------------------


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
    return "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job_id}/logs").json())


def _import(hub, project_id, path, **extra):
    body = {"node_id": hub.node_id, "path": str(path), "dataset": "pets", "version": "v1", **extra}
    return _wait(hub, hub.post(f"/api/projects/{project_id}/imports", json=body).json()["id"])


def test_classification_project_from_import_to_model_with_files(hub, tmp_path):
    folder = tmp_path / "template"
    folder.mkdir()
    (folder / "badamoa.toml").write_text(MANIFEST, encoding="utf-8")
    (folder / "train.py").write_text(TRAIN, encoding="utf-8")
    assert hub.post("/api/templates", content=fmt.bundle(folder)).status_code == 200

    project = hub.post("/api/projects", json={"name": "반려동물", "task": "classify"}).json()
    data = make_imagefolder(tmp_path / "pets")
    job = _import(hub, project["id"], data)
    assert job["status"] == "succeeded" and job["result"]["format"] == "imagefolder", _logs(hub, job["id"])
    release = hub.get(f"/api/releases/{job['result']['release_id']}").json()
    assert [c["name"] for c in release["label_space"]["classes"]] == ["bird", "cat", "dog"]
    assert release["stats"]["items"] == 18 and release["stats"]["states"]["labelled"] == 18
    assert {k: v["items"] for k, v in release["stats"]["classes"].items()} == {"0": 6, "1": 6, "2": 6}
    cats = hub.get(f"/api/releases/{release['id']}/items", params={"class_id": 1, "split": "train"}).json()
    assert cats["total"] == 4
    detail = hub.get(f"/api/releases/{release['id']}/items/{cats['items'][0]['id']}").json()
    assert detail["revisions"][0]["boxes"] == [[1]]  # a label without geometry

    # Detection data does not go into a classification project, and the other way round.
    wrong = _import(hub, project["id"], make_yolo(tmp_path / "boxes"), dataset="boxes")
    assert wrong["status"] == "failed" and "객체 탐지 데이터인데 프로젝트는 분류용" in _logs(hub, wrong["id"])
    detect = hub.post("/api/projects", json={"name": "검출"}).json()
    assert "분류 데이터인데" in _logs(hub, _import(hub, detect["id"], data)["id"])

    # Templates are offered by task; a detection template is refused here.
    names = lambda task: [t["name"] for t in hub.get("/api/templates", params={"task": task}).json()]  # noqa: E731
    offered = {task: names(task) for task in ("classify", "detect")}
    assert offered == {"classify": ["classify", "tiny-classifier"], "detect": ["yolo", "torchvision"]}
    refused = hub.post("/api/runs", json={"release_id": release["id"], "template": "yolo"})
    assert refused.status_code == 400 and "객체 탐지용이고 이 프로젝트는 분류용" in refused.json()["detail"]

    run = hub.post("/api/runs", json={"release_id": release["id"], "template": "tiny-classifier", "gpus": 0}).json()
    job = _wait(hub, run["job_id"])
    logs = _logs(hub, job["id"])
    assert job["status"] == "succeeded", logs
    assert "folders ['0000_bird', '0001_cat', '0002_dog'] counts {'train': 12, 'val': 6}" in logs
    assert "기록할 파일을 찾을 수 없습니다" in logs  # reported, not fatal

    done = hub.get(f"/api/runs/{run['id']}").json()
    assert done["task"] == "classify" and done["export"]["val_source"] == "release"
    assert done["history"]["steps"] == list(range(1, 41))  # all 40 steps arrived, though sent in batches
    assert done["history"]["series"]["train_loss"][0] == 1.0 and (done["epoch"], done["epochs"]) == (40, 40)
    assert done["metrics"]["top1"] == pytest.approx(1 - 15 / 40)  # the last logged values are the final ones
    assert any(p.startswith("httpx==") for p in done["info"]["packages"]) and done["info"]["framework"] == "none"

    files = {f["name"]: f for f in done["files"]}
    assert {name: f["kind"] for name, f in files.items()} == {"confusion_matrix.png": "image", "report.txt": "file"}
    picture = httpx.get(hub.url + files["confusion_matrix.png"]["url"])
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/png"
    assert picture.headers["content-disposition"].startswith("inline") and picture.content[:4] == b"\x89PNG"
    report = httpx.get(hub.url + files["report.txt"]["url"])
    assert report.content == b"ok" and report.headers["content-disposition"].startswith("attachment")
    assert httpx.get(hub.url + files["report.txt"]["url"].replace("sig=", "sig=0")).status_code == 403

    model = hub.get(f"/api/models/{done['model']['id']}").json()
    assert (model["task"], model["name"], model["classes"]) == ("classify", "pets-tiny-classifier", ["bird", "cat", "dog"])
    assert model["class_metrics"][1] == {"id": 1, "name": "cat", "precision": 1, "recall": 0.5, "f1": 0.66}

    # While it runs, the best step so far is judged by the task's own metric (top-1 here).
    slow = hub.post(
        "/api/runs", json={"release_id": release["id"], "template": "tiny-classifier", "params": {"best_at": 10}, "gpus": 0}
    ).json()
    assert _wait(hub, slow["job_id"])["status"] == "succeeded"
    assert hub.get(f"/api/runs/{slow['id']}").json()["metrics"]["top1"] == pytest.approx(1 - 30 / 40)

    # An empty request uses the built-in template for the project's task.
    default = hub.post("/api/runs", json={"release_id": release["id"], "gpus": 0}).json()
    assert default["template"]["name"] == "classify"
    hub.post(f"/api/jobs/{default['job_id']}/cancel")


# ---- segmentation: outlines travel with the boxes -------------------------------------------


def _make_seg(root):
    """A YOLO dataset whose first training image is labelled with an outline instead of a box."""
    make_yolo(root)
    label = root / "labels" / "train" / "train_0000.txt"
    label.write_text("2 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n1 0.7 0.7 0.2 0.2\n", encoding="utf-8")
    return root


def test_segmentation_labels_keep_their_outline_from_import_to_export(tmp_path):
    root = _make_seg(tmp_path / "ds")
    found = importer.scan(root)
    first = next(e for e in found.entries if e.path.name == "train_0000.jpg")
    assert first.boxes[0] == [2, 0.3, 0.4, 0.4, 0.4, 0.1, 0.2, 0.5, 0.2, 0.5, 0.6, 0.1, 0.6]  # box, then the outline
    assert first.boxes[1] == [1, 0.7, 0.7, 0.2, 0.2]

    items, classes, locations = _manifest_items(root, found.entries, found.classes)
    seg = exporter.BUILDERS["yolo-seg"](items, classes, locations, tmp_path / "seg", exporter.ExportOptions())
    name = next(i["sha256"] for i in items if i["rel_path"].endswith("train_0000.jpg"))
    lines = (tmp_path / "seg" / "labels" / "train" / f"{name}.txt").read_text().splitlines()
    assert lines[0] == "2 0.100000 0.200000 0.500000 0.200000 0.500000 0.600000 0.100000 0.600000"  # the outline
    assert lines[1] == "1 0.700000 0.700000 0.200000 0.200000" and seg["boxes"] > 2
    boxes = exporter.BUILDERS["yolo"](items, classes, locations, tmp_path / "det", exporter.ExportOptions())
    det = (tmp_path / "det" / "labels" / "train" / f"{name}.txt").read_text().splitlines()
    assert det[0] == "2 0.300000 0.400000 0.400000 0.400000" and boxes["boxes"] == seg["boxes"]  # a detector gets the box


def test_outlines_are_kept_in_segmentation_projects_only(hub, tmp_path):
    root = _make_seg(tmp_path / "ds")
    stored = {}
    for name, task in (("분할", "segment"), ("검출", "detect")):
        project = hub.post("/api/projects", json={"name": name, "task": task}).json()
        job = _import(hub, project["id"], root)
        assert job["status"] == "succeeded", _logs(hub, job["id"])
        release = job["result"]["release_id"]
        item = hub.get(f"/api/releases/{release}/items", params={"q": "train_0000"}).json()["items"][0]
        stored[task] = hub.get(f"/api/releases/{release}/items/{item['id']}").json()["revisions"][0]["boxes"]
    assert stored["segment"][0] == [2, 0.3, 0.4, 0.4, 0.4, 0.1, 0.2, 0.5, 0.2, 0.5, 0.6, 0.1, 0.6]
    assert stored["detect"] == [[2, 0.3, 0.4, 0.4, 0.4], [1, 0.7, 0.7, 0.2, 0.2]]
    offered = [t["name"] for t in hub.get("/api/templates", params={"task": "segment"}).json()]
    assert offered == ["yolo-seg"]


def test_coco_outlines_are_read_and_the_largest_polygon_kept(tmp_path):
    import json

    from PIL import Image

    root = tmp_path / "coco"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (200, 100), (30, 60, 90)).save(root / "images" / "a.jpg")
    small, large = [10, 10, 20, 10, 20, 20], [100, 20, 180, 20, 180, 80, 100, 80]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 7, "bbox": [100, 20, 80, 60], "segmentation": [small, large]},
        {"id": 2, "image_id": 1, "category_id": 7, "bbox": [0, 0, 50, 50], "segmentation": {"counts": "rle"}},
    ]
    data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
        "categories": [{"id": 7, "name": "car"}],
        "annotations": annotations,
    }
    (root / "instances_train.json").write_text(json.dumps(data), encoding="utf-8")
    found = importer.scan(root)
    outlined, rle = found.entries[0].boxes
    assert outlined == [0, 0.7, 0.5, 0.4, 0.6, 0.5, 0.2, 0.9, 0.2, 0.9, 0.8, 0.5, 0.8]  # box, then the larger polygon
    assert rle == [0, 0.125, 0.25, 0.25, 0.5]  # an RLE mask is not read: the box only
