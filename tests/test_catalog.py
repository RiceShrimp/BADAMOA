"""Dataset catalog: in-place import by a real Node Agent, releases, label revisions, comparison."""

from __future__ import annotations

import os
import threading
import time

import httpx
import pytest
from conftest import ADMIN
from sample_data import CLASSES, make_coco, make_imagefolder, make_yolo

from badamoa.node import agent as node_agent
from badamoa.node import importer


def test_yolo_yaml_variants():
    block = importer.parse_yolo_yaml("path: ../data\ntrain: images/train  # comment\nnames:\n  0: person\n  1: 'car'\n")
    assert block == {"path": "../data", "train": "images/train", "names": ["person", "car"]}
    assert importer.parse_yolo_yaml("names: [a, 'b', \"c\"]")["names"] == ["a", "b", "c"]
    assert importer.parse_yolo_yaml("names:\n- a\n- b\nval: v")["names"] == ["a", "b"]
    assert importer.parse_yolo_yaml("nc: 2")["names"] == []


def test_scan_yolo_reads_splits_and_label_states(tmp_path):
    found = importer.scan(make_yolo(tmp_path / "ds"))
    assert found.format == "yolo" and found.classes == CLASSES
    assert [e.split for e in found.entries].count("train") == 6 and [e.split for e in found.entries].count("val") == 2
    states = {e.path.name: e.state for e in found.entries}
    assert states["train_0001.jpg"] == "background" and states["train_0002.jpg"] == "background"  # no label file
    assert states["train_0000.jpg"] == "labelled"
    assert all(0 <= v <= 1 for e in found.entries for b in e.boxes for v in b[1:])


def test_scan_yolo_without_yaml_uses_folder_names(tmp_path):
    root = make_yolo(tmp_path / "ds")
    (root / "data.yaml").unlink()
    found = importer.scan(root)
    assert {e.split for e in found.entries} == {"train", "val"}
    assert found.classes[:1] == ["class_0"]  # no names available, ids only


def test_scan_coco_converts_pixel_boxes_and_category_ids(tmp_path):
    found = importer.scan(make_coco(tmp_path / "coco"))
    assert found.format == "coco" and found.classes == CLASSES
    assert len(found.entries) == 4 and {e.split for e in found.entries} == {"train"}
    assert found.entries[0].state == "background" and found.entries[1].state == "labelled"
    box = found.entries[1].boxes[0]
    assert box[0] in (0, 1, 2) and all(0 <= v <= 1 for v in box[1:])


# ---- end to end -------------------------------------------------------------------


@pytest.fixture
def hub(live_hub, tmp_path, monkeypatch):
    """A Hub with one real node running in this process."""
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        join = client.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="data-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        client.node_id = runner.config.node_id
        yield client
        runner.stop()


def _import(hub, project_id, path, version, **extra):
    body = {"node_id": hub.node_id, "path": str(path), "dataset": "street", "version": version, **extra}
    resp = hub.post(f"/api/projects/{project_id}/imports", json=body)
    assert resp.status_code == 200, resp.text
    job = resp.json()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = hub.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in {"succeeded", "failed", "canceled"}:
            break
        time.sleep(0.3)
    logs = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
    assert job["status"] == "succeeded", logs
    return job["result"]["release_id"]


def test_import_browse_and_compare_releases(hub, tmp_path):
    root = make_yolo(tmp_path / "ds")
    project = hub.post("/api/projects", json={"name": "주차장", "task": "detect"}).json()
    assert hub.post("/api/projects", json={"name": "주차장"}).status_code == 409

    v1 = _import(hub, project["id"], root, "v1", site="Hangang", camera="4Kx4", note="첫 릴리스")

    # ---- the release and its numbers
    tree = hub.get(f"/api/projects/{project['id']}/datasets").json()
    assert [(d["name"], [(r["version"], r["status"], r["items"]) for r in d["releases"]]) for d in tree] == [
        ("street", [("v1", "ready", 8)])
    ]
    release = hub.get(f"/api/releases/{v1}").json()
    stats = release["stats"]
    assert stats["items"] == 8 and stats["splits"]["train"]["items"] == 6 and stats["splits"]["val"]["items"] == 2
    assert stats["states"] == {"labelled": 6, "background": 2}
    assert stats["sites"] == ["Hangang"] and stats["bytes"] > 0
    assert [c["name"] for c in release["label_space"]["classes"]] == CLASSES and release["base"] is None
    assert sum(c["boxes"] for c in stats["classes"].values()) == stats["boxes"] > 0

    # ---- filters and paging
    def items(**params):
        return hub.get(f"/api/releases/{v1}/items", params=params).json()

    assert items()["total"] == 8 and items(split="val")["total"] == 2
    background = items(state="background")["items"]
    assert [i["file_name"] for i in background] == ["train_0001.jpg", "train_0002.jpg"]
    assert {i["box_count"] for i in background} == {0}
    assert items(q="val_")["total"] == 2 and items(site="Hangang")["total"] == 8 and items(site="nowhere")["total"] == 0
    class_id = int(next(iter(stats["classes"])))
    assert items(class_id=class_id)["total"] == stats["classes"][str(class_id)]["items"]
    page = items(limit=3, offset=6)
    assert len(page["items"]) == 2 and page["total"] == 8

    # ---- images: signed thumbnail and the original file read in place
    first = items(q="train_0000")["items"][0]
    thumb = httpx.get(str(hub.base_url).rstrip("/") + first["thumb_url"])
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert httpx.get(str(hub.base_url).rstrip("/") + first["thumb_url"].replace("sig=", "sig=0")).status_code == 403
    detail = hub.get(f"/api/releases/{v1}/items/{first['id']}").json()
    assert detail["site"] == "Hangang" and detail["camera"] == "4Kx4" and len(detail["sha256"]) == 64
    assert [(r["rev"], r["source"], r["labelled_classes"]) for r in detail["revisions"]] == [
        (1, "import:street v1", CLASSES)
    ]
    assert detail["revisions"][0]["boxes"] and detail["releases"] == [
        {"id": v1, "dataset": "street", "version": "v1", "split": "train", "rev": 1, "current": True}
    ]
    original = httpx.get(str(hub.base_url).rstrip("/") + detail["image_url"])
    assert original.status_code == 200 and len(original.content) == detail["bytes"]

    # ---- a second release: one label edited, one image removed, one image added
    label = root / "labels" / "train" / "train_0000.txt"
    label.write_text(label.read_text() + "1 0.5 0.5 0.2 0.2\n")
    (root / "images" / "val" / "val_0001.jpg").unlink()
    make_yolo(tmp_path / "extra", train=9, val=0, seed=7)
    (tmp_path / "extra" / "images" / "train" / "train_0008.jpg").rename(root / "images" / "train" / "new_0000.jpg")
    (tmp_path / "extra" / "labels" / "train" / "train_0008.txt").rename(root / "labels" / "train" / "new_0000.txt")

    assert (
        hub.post(
            f"/api/projects/{project['id']}/imports",
            json={"node_id": hub.node_id, "path": str(root), "dataset": "street", "version": "v1"},
        ).status_code
        == 409
    )
    v2 = _import(hub, project["id"], root, "v2")

    assert hub.get(f"/api/releases/{v2}").json()["base"] == {"id": v1, "version": "v1"}
    diff = hub.get(f"/api/releases/{v2}/compare", params={"other": v1}).json()
    assert (diff["added"], diff["removed"], diff["relabelled"], diff["moved_split"]) == (1, 1, 1, 0)
    assert sum(c["delta"] for c in diff["classes"]) == diff["release"]["boxes"] - diff["other"]["boxes"]

    # the edited image now has two revisions; v1 still points at the first
    again = hub.get(f"/api/releases/{v2}/items/{first['id']}").json()
    assert [r["rev"] for r in again["revisions"]] == [1, 2]
    assert again["revisions"][1]["box_count"] == again["revisions"][0]["box_count"] + 1
    assert [(r["version"], r["rev"]) for r in again["releases"]] == [("v1", 1), ("v2", 2)]
    untouched = hub.get(f"/api/releases/{v2}/items", params={"q": "train_0003"}).json()["items"][0]
    assert (
        len(hub.get(f"/api/releases/{v2}/items/{untouched['id']}").json()["revisions"]) == 1
    )  # same labels, no new revision

    # ---- trash and restore
    assert hub.delete(f"/api/releases/{v2}").status_code == 200
    assert hub.get(f"/api/releases/{v2}").status_code == 404
    assert [r["version"] for r in hub.get(f"/api/projects/{project['id']}/datasets").json()[0]["releases"]] == ["v1"]
    assert hub.post(f"/api/releases/{v2}/restore").status_code == 200
    assert hub.get(f"/api/releases/{v2}").json()["stats"]["items"] == 8


def test_import_of_missing_folder_fails_with_a_clear_log(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "빈 프로젝트"}).json()
    job = hub.post(
        f"/api/projects/{project['id']}/imports",
        json={"node_id": hub.node_id, "path": str(tmp_path / "nope"), "dataset": "d", "version": "v1"},
    ).json()
    for _ in range(200):
        job = hub.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "queued" and job["status"] != "running":
            break
        time.sleep(0.2)
    assert job["status"] == "failed"
    logs = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
    assert "폴더를 찾을 수 없습니다" in logs
    assert hub.get(f"/api/projects/{project['id']}/datasets").json() == []
    assert "data.import" not in [t["type"] for t in hub.get("/api/job-types").json()]


def test_new_releases_from_existing_ones(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "새 릴리스", "task": "detect"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1", camera="cam1")
    more = _import(hub, project["id"], make_yolo(tmp_path / "b", train=4, val=0, seed=5), "more", camera="cam2")

    def make(**body):
        return hub.post("/api/releases", json={"sources": [v1], **body})

    def splits_of(release_id):
        items = hub.get(f"/api/releases/{release_id}/items", params={"limit": 200}).json()["items"]
        return {i["id"]: i["split"] for i in items}

    def releases():
        tree = hub.get(f"/api/projects/{project['id']}/datasets").json()
        return {d["name"]: [r["version"] for r in d["releases"]] for d in tree}

    # ---- a dry run counts and makes nothing
    counted = make(dry_run=True, filter={"state": "labelled"}).json()
    assert counted == {"items": 6, "splits": {"train": 4, "val": 2}, "classes": 3}
    stats = hub.get(f"/api/releases/{v1}").json()["stats"]
    assert make(dry_run=True, filter={"class_id": "0"}).json()["items"] == stats["classes"]["0"]["items"]
    assert releases() == {"street": ["more", "v1"]}

    # ---- only what matches a filter, in the splits it had
    labelled = make(version="v1-labelled", filter={"state": "labelled"}, note="라벨 있는 것만").json()
    view = hub.get(f"/api/releases/{labelled['release_id']}").json()
    assert view["stats"]["items"] == 6 and view["stats"]["states"]["labelled"] == 6
    assert view["base"] == {"id": v1, "version": "v1"} and view["note"] == "라벨 있는 것만"
    assert all(splits_of(v1)[k] == v for k, v in splits_of(labelled["release_id"]).items())
    # all but what matches
    assert make(version="no-val", filter={"split": "val"}, exclude=True).json()["splits"] == {"train": 6}

    # ---- dealt out again by a hash: the same item lands in the same split every time
    first = make(version="resplit-a", split={"train": 50, "val": 50}).json()
    second = make(version="resplit-b", split={"train": 50, "val": 50, "test": 0}).json()
    assert first["items"] == 8 and set(first["splits"]) <= {"train", "val"}
    assert splits_of(first["release_id"]) == splits_of(second["release_id"])

    # ---- two releases into a new dataset, split by camera: each camera's images stay together
    merged = hub.post(
        "/api/releases",
        json={
            "sources": [v1, more],
            "version": "v1",
            "dataset": "street-all",
            "split": {"train": 70, "val": 30},
            "group_by": "camera",
        },
    ).json()
    assert merged["items"] == 12 and merged["dataset"] == "street-all"
    together = splits_of(merged["release_id"])
    assert len({together[i] for i in splits_of(v1)}) == 1 and len({together[i] for i in splits_of(more)}) == 1
    assert releases()["street-all"] == ["v1"]

    # ---- refused, with the reason
    assert make(version="v1").status_code == 409  # that version exists
    assert make(version="empty", filter={"q": "no-such-file"}).status_code == 400
    assert make(version="bad", split={"train": -1}).status_code == 400
    assert make(version="bad", group_by="weather").status_code == 400
    assert hub.post("/api/releases", json={"sources": [], "version": "x"}).status_code == 400
    other_root = make_yolo(tmp_path / "c", train=2, val=0, seed=9)
    (other_root / "data.yaml").write_text("path: .\ntrain: images/train\nnames:\n  0: a\n  1: b\n  2: c\n", encoding="utf-8")
    other = _import(hub, project["id"], other_root, "other-classes")
    mixed = hub.post("/api/releases", json={"sources": [v1, other], "version": "mixed"})
    assert mixed.status_code == 400 and "클래스" in mixed.json()["detail"]


def _wait_job(hub, job):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = hub.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in {"succeeded", "failed", "canceled"}:
            return job
        time.sleep(0.3)
    raise AssertionError(job)


def test_download_a_release_as_yolo_or_coco(hub, tmp_path, settings):
    import io
    import json
    import zipfile

    project = hub.post("/api/projects", json={"name": "내려받기", "task": "detect"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")

    def export(**body):
        resp = hub.post(f"/api/releases/{v1}/exports", json=body)
        assert resp.status_code == 200, resp.text
        job = _wait_job(hub, resp.json())
        logs = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
        assert job["status"] == "succeeded", logs
        listed = next(e for e in hub.get(f"/api/releases/{v1}/exports").json() if e["job_id"] == job["id"])
        data = httpx.get(str(hub.base_url).rstrip("/") + listed["download_url"])
        assert data.status_code == 200 and "attachment" in data.headers["content-disposition"]
        return listed, zipfile.ZipFile(io.BytesIO(data.content))

    # ---- YOLO with the images: an image without labels goes in as background, with an empty label file
    listed, archive = export(format="yolo")
    names = archive.namelist()
    assert listed["counts"] == {"train": 6, "val": 2} and listed["skipped"] == {"unlabelled": 0, "missing": 0}
    assert listed["file"] == "street-v1-yolo.zip" and listed["expires_at"]
    assert sum(n.startswith("images/") for n in names) == 8 and sum(n.startswith("labels/") for n in names) == 8
    assert archive.read("labels/train/train_0001.txt") == b""  # background
    assert archive.read("labels/train/train_0002.txt") == b""  # had no label file: background too
    data_yaml = archive.read("data.yaml").decode()
    assert "train: images/train" in data_yaml and "val: images/val" in data_yaml and '0: "person"' in data_yaml
    files = archive.read("files.csv").decode().splitlines()
    assert files[0] == "split,file,sha256,registered_path" and len(files) == 9
    assert "street v1" in archive.read("README.txt").decode()

    # ---- COCO, the labels only: pixel boxes, 1-based categories, no images
    listed, archive = export(format="coco", images=False)
    assert not any(n.startswith("images/") for n in archive.namelist()) and listed["file"] == "street-v1-coco-labels.zip"
    train = json.loads(archive.read("annotations/instances_train.json"))
    assert [c["name"] for c in train["categories"]] == CLASSES and train["categories"][0]["id"] == 1
    assert len(train["images"]) == 6 and train["images"][0]["width"] == 320  # the two without labels as background
    box = train["annotations"][0]["bbox"]
    assert len(box) == 4 and 0 <= box[0] <= 320 and box[2] > 1  # pixels, not fractions

    # ---- refused: a format the task does not have
    assert hub.post(f"/api/releases/{v1}/exports", json={"format": "imagefolder"}).status_code == 400

    # ---- a zip is offered for a week, then the link is gone and the storage cleanup may take it
    from badamoa.hub import exports
    from badamoa.hub.db import Database
    from badamoa.hub.models import Job

    database = Database(settings.database_url)
    with database.session() as db:
        assert len(exports.kept_blobs(db)) == 2
        for job in db.query(Job).filter(Job.type == "data.export"):
            job.finished_at = job.finished_at - exports.timedelta(days=exports.KEEP_DAYS + 1)
    with database.session() as db:
        assert exports.kept_blobs(db) == set()
    database.engine.dispose()
    assert all(e["download_url"] is None for e in hub.get(f"/api/releases/{v1}/exports").json())


def test_download_layouts_for_segments_and_classes(tmp_path):
    import json
    import zipfile

    from badamoa.node import exporter

    image = tmp_path / "pics" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg")
    header = {
        "dataset": "d",
        "version": "v1",
        "classes": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}],
        "locations": {"loc": str(tmp_path / "pics")},
    }

    def item(boxes, split="train", state="labelled"):
        return {
            "sha256": "ab" * 32,
            "rel_path": "a.jpg",
            "bytes": 4,
            "width": 200,
            "height": 100,
            "location_id": "loc",
            "split": split,
            "state": state,
            "boxes": boxes,
        }

    outline = [1, 0.5, 0.5, 0.2, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6]  # a box, then its outline
    rows = [item([outline, [0, 0.5, 0.5, 0.2, 0.2]], split="")]
    seg = exporter.write_archive(rows, header, tmp_path / "seg.zip", "yolo", "segment")
    with zipfile.ZipFile(tmp_path / "seg.zip") as archive:
        lines = archive.read("labels/train/a.txt").decode().splitlines()
    assert lines[0].split() == ["1", "0.400000", "0.400000", "0.600000", "0.400000", "0.600000", "0.600000"]
    assert len(lines[1].split()) == 9  # a box without an outline: its four corners
    assert seg["unsplit"] == 1 and seg["counts"] == {"train": 1}

    exporter.write_archive([item([outline])], header, tmp_path / "seg-coco.zip", "coco", "segment")
    with zipfile.ZipFile(tmp_path / "seg-coco.zip") as archive:
        note = json.loads(archive.read("annotations/instances_train.json"))["annotations"][0]
    assert note["segmentation"] == [[80.0, 40.0, 120.0, 40.0, 120.0, 60.0]] and note["area"] == 400.0
    assert note["category_id"] == 2 and note["bbox"] == [80.0, 40.0, 40.0, 20.0]

    folders = exporter.write_archive([item([[1]], split="val")], header, tmp_path / "cls.zip", "imagefolder", "classify")
    with zipfile.ZipFile(tmp_path / "cls.zip") as archive:
        assert "val/dog/a.jpg" in archive.namelist()
    assert folders["counts"] == {"val": 1}
    with pytest.raises(ValueError, match="담을 항목"):
        exporter.write_archive([item([], state="unlabelled")], header, tmp_path / "none.zip", "yolo")


def test_home_totals(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "홈", "task": "detect"}).json()
    _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")
    home = hub.get("/api/home").json()
    totals = home["totals"]
    assert (totals["projects"], totals["datasets"], totals["releases"], totals["items"], totals["runs"]) == (1, 1, 1, 8, 0)
    assert totals["boxes"] > 0
    assert home["projects"][0]["id"] == project["id"] and home["projects"][0]["collage"]
    assert home["runs"] == []


def test_cards_and_release_tabs(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "카드", "task": "detect", "folder": "감시"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")

    # ---- project cards: counts, a collage of labelled images with their boxes
    card = next(c for c in hub.get("/api/project-cards").json() if c["id"] == project["id"])
    assert (card["folder"], card["datasets"], card["releases"], card["items"], card["runs"], card["models"]) == (
        "감시", 1, 1, 8, 0, 0
    )
    assert 1 <= len(card["collage"]) <= 4
    assert all(c["boxes"] and c["url"].startswith("/media/thumb/") for c in card["collage"])
    assert card["best"] is None and card["updated_at"]
    # the classes that have labels (for the board's class filter), most labels first
    labels = [c["labels"] for c in card["classes"]]
    assert {c["name"] for c in card["classes"]} <= set(CLASSES) and labels == sorted(labels, reverse=True)
    assert sum(labels) == hub.get(f"/api/releases/{v1}").json()["stats"]["boxes"]

    # ---- moved to another folder, renamed
    moved = hub.patch(f"/api/projects/{project['id']}", json={"folder": "공장", "name": "카드-2"})
    assert moved.status_code == 200 and (moved.json()["folder"], moved.json()["name"]) == ("공장", "카드-2")
    assert hub.patch(f"/api/projects/{project['id']}", json={"name": ""}).status_code == 400

    # ---- the project page: datasets as cards
    page = hub.get(f"/api/projects/{project['id']}/overview").json()
    (dataset,) = page["dataset_cards"]
    assert (dataset["name"], dataset["release"]["id"], dataset["releases"], dataset["items"], dataset["classes"]) == (
        "street", v1, 1, 8, 3
    )
    assert dataset["splits"] == {"train": 6, "val": 2} and dataset["states"] == {"labelled": 6, "background": 2}
    assert [c["name"] for c in dataset["top_classes"]] and page["metric"] == "map50_95" and page["top_models"] == []

    # ---- a release's charts: sizes, classes, where the boxes sit
    charts = hub.get(f"/api/releases/{v1}/charts").json()
    assert charts["sizes"]["images"] == 8 and charts["sizes"]["mean_width"] == 320.0
    assert sum(sum(row) for row in charts["sizes"]["grid"]) == 8
    boxes = sum(c["boxes"] for c in charts["classes"])
    assert charts["boxes"]["boxes"] == boxes == sum(sum(row) for row in charts["boxes"]["centres"])
    assert charts["boxes"]["sampled_items"] == 6  # the labelled ones that have boxes
    assert hub.get(f"/api/releases/{v1}/models").json() == []

    # ---- a class's colour, kept with the label space
    assert hub.put(f"/api/releases/{v1}/classes/1", json={"color": "#FF0000"}).json() == {"id": 1, "color": "#ff0000"}
    assert next(c for c in hub.get(f"/api/releases/{v1}/charts").json()["classes"] if c["id"] == 1)["color"] == "#ff0000"
    assert hub.put(f"/api/releases/{v1}/classes/1", json={"color": "red"}).status_code == 400
    assert hub.put(f"/api/releases/{v1}/classes/9", json={"color": "#00ff00"}).status_code == 404
    assert hub.put(f"/api/releases/{v1}/classes/1", json={"color": ""}).json()["color"] is None

    # ---- grid items carry their boxes
    items = hub.get(f"/api/releases/{v1}/items").json()["items"]
    assert any(i["boxes"] for i in items) and all(len(i["boxes"]) == i["box_count"] for i in items if i["box_count"] <= 30)


def test_project_trash_and_restore(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "지울 것", "task": "detect"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")
    empty = {"projects": [], "releases": []}
    assert hub.get("/api/trash").json() == empty

    # ---- a version to the trash, then the whole project (its versions go with it)
    assert hub.delete(f"/api/releases/{v1}").status_code == 200
    trash = hub.get("/api/trash").json()
    assert [r["id"] for r in trash["releases"]] == [v1] and trash["releases"][0]["project"]["name"] == "지울 것"
    assert hub.delete(f"/api/projects/{project['id']}").status_code == 200
    trash = hub.get("/api/trash").json()
    assert [p["id"] for p in trash["projects"]] == [project["id"]] and trash["releases"] == []
    assert (trash["projects"][0]["items"], trash["projects"][0]["releases"]) == (8, 0)
    assert all(c["id"] != project["id"] for c in hub.get("/api/project-cards").json())
    assert hub.get(f"/api/projects/{project['id']}/overview").status_code == 404
    taken = hub.post("/api/projects", json={"name": "지울 것"})
    assert taken.status_code == 409 and "휴지통" in taken.json()["detail"]

    # ---- back again, with everything it held
    assert hub.post(f"/api/projects/{project['id']}/restore").status_code == 200
    assert hub.post(f"/api/releases/{v1}/restore").status_code == 200
    page = hub.get(f"/api/projects/{project['id']}/overview").json()
    assert (page["items"], page["releases"]) == (8, 1)
    assert hub.get("/api/trash").json() == empty
    assert hub.delete("/api/projects/no-such").status_code == 404


def test_site_and_camera_from_file_names(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "이름 규칙", "task": "detect"}).json()
    # read while importing: images/train/train_0000.jpg → site "train"; the fixed camera goes to every image
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1", camera="cam", name_pattern="{site}_*")
    stats = hub.get(f"/api/releases/{v1}").json()["stats"]
    assert (stats["sites"], stats["cameras"]) == (["train", "val"], ["cam"])
    bad_import = {"node_id": hub.node_id, "path": "x", "dataset": "d", "version": "v9", "name_pattern": "{weather}"}
    assert hub.post(f"/api/projects/{project['id']}/imports", json=bad_import).status_code == 400

    def rule(pattern, dry_run=True):
        return hub.post(f"/api/releases/{v1}/names", json={"pattern": pattern, "dry_run": dry_run})

    # ---- a dry run shows what the rule finds, and changes nothing
    preview = rule("{camera}_{site}").json()
    assert (preview["items"], preview["matched"], preview["changed"], preview["fields"]) == (8, 8, 8, ["camera", "site"])
    assert preview["values"]["camera"]["count"] == 2 and preview["values"]["site"]["count"] == 6
    assert preview["examples"][0] == {"path": "images/train/train_0000.jpg", "camera": "train", "site": "0000"}
    assert hub.get(f"/api/releases/{v1}").json()["stats"]["cameras"] == ["cam"]
    folders = rule("{camera}/*").json()  # the folder the file sits in
    assert folders["matched"] == 8 and folders["examples"][0]["camera"] == "train"
    missed = rule("{site}-{camera}").json()
    assert missed["matched"] == 0 and len(missed["misses"]) == 4
    assert rule("{site}-{camera}", dry_run=False).status_code == 400
    for wrong in ("", "*", "{weather}_*", "{site}_{site}"):
        assert rule(wrong).status_code == 400, wrong

    # ---- applied: every version holding the images offers the new values
    assert rule("{camera}_{site}", dry_run=False).json()["changed"] == 8
    stats = hub.get(f"/api/releases/{v1}").json()["stats"]
    assert stats["cameras"] == ["train", "val"] and len(stats["sites"]) == 6
    assert hub.get(f"/api/releases/{v1}/items", params={"camera": "val"}).json()["total"] == 2
    assert rule("{camera}_{site}").json()["changed"] == 0


def test_rename_merge_and_drop_classes_as_a_new_version(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "클래스 바꾸기", "task": "detect"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")
    before = hub.get(f"/api/releases/{v1}").json()
    boxes = {name: before["stats"]["classes"].get(str(i), {}).get("boxes", 0) for i, name in enumerate(CLASSES)}
    merge = {"0": "사람", "1": "사물", "2": "사물"}

    def make(version, classes, dry_run=False):
        body = {"sources": [v1], "version": version, "classes": classes, "dry_run": dry_run}
        return hub.post("/api/releases", json=body)

    # ---- renamed and merged: car and dog become one class
    assert make("x", merge, dry_run=True).json() == {"items": 8, "splits": {"train": 6, "val": 2}, "classes": 2}
    merged = make("v2", merge).json()
    view = hub.get(f"/api/releases/{merged['release_id']}").json()
    assert [c["name"] for c in view["label_space"]["classes"]] == ["사람", "사물"]
    assert view["stats"]["boxes"] == before["stats"]["boxes"] and view["stats"]["items"] == 8
    assert view["stats"]["classes"]["1"]["boxes"] == boxes["car"] + boxes["dog"]
    assert view["stats"]["classes"].get("0", {}).get("boxes", 0) == boxes["person"]
    assert hub.get(f"/api/releases/{v1}").json()["stats"] == before["stats"]  # the old version is as it was
    item = next(i for i in hub.get(f"/api/releases/{merged['release_id']}/items").json()["items"] if i["box_count"])
    detail = hub.get(f"/api/releases/{merged['release_id']}/items/{item['id']}").json()
    assert [r["rev"] for r in detail["revisions"]] == [1, 2] and detail["revisions"][1]["source"] == "classes:street v2"
    assert {r["rev"] for r in detail["releases"]} == {1, 2}
    assert hub.get(f"/api/releases/{merged['release_id']}/compare", params={"other": v1}).json()["label_space_changed"]

    # ---- the same change again reuses the label space and the revisions
    again = make("v2b", merge).json()
    assert hub.get(f"/api/releases/{again['release_id']}").json()["label_space"]["name"] == view["label_space"]["name"]
    assert len(hub.get(f"/api/releases/{again['release_id']}/items/{item['id']}").json()["revisions"]) == 2

    # ---- a class dropped: its boxes go, the images stay
    dropped = hub.get(f"/api/releases/{make('v3', {'2': ''}).json()['release_id']}").json()
    assert [c["name"] for c in dropped["label_space"]["classes"]] == ["person", "car"]
    assert dropped["stats"]["boxes"] == before["stats"]["boxes"] - boxes["dog"] and "2" not in dropped["stats"]["classes"]
    assert dropped["stats"]["items"] == 8
    assert dropped["stats"]["states"]["background"] >= before["stats"]["states"]["background"]

    # ---- nothing changed: the same label space; nothing left: refused
    same = make("v4", {"0": "person"}).json()
    assert hub.get(f"/api/releases/{same['release_id']}").json()["label_space"]["name"] == before["label_space"]["name"]
    assert make("none", {"0": "", "1": " ", "2": ""}).status_code == 400


def test_dropping_a_class_of_a_classification_version(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "분류 클래스", "task": "classify"}).json()
    v1 = _import(hub, project["id"], make_imagefolder(tmp_path / "cls"), "v1")
    ids = {c["name"]: str(c["id"]) for c in hub.get(f"/api/releases/{v1}").json()["label_space"]["classes"]}
    # dog merged into cat, bird dropped: the bird images leave the new version
    body = {"sources": [v1], "version": "v2", "classes": {ids["bird"]: "", ids["dog"]: "cat"}}
    made = hub.post("/api/releases", json=body).json()
    assert (made["items"], made["classes"]) == (12, 1)
    view = hub.get(f"/api/releases/{made['release_id']}").json()
    assert view["stats"]["classes"] == {"0": {"boxes": 12, "items": 12}}


def test_empty_the_trash(hub, tmp_path, settings):
    project = hub.post("/api/projects", json={"name": "비울 것", "task": "detect"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")
    v2 = hub.post("/api/releases", json={"sources": [v1], "version": "v2", "filter": {"split": "val"}}).json()["release_id"]
    thumbs = {i["thumb_url"].split("/")[3].split("?")[0] for i in hub.get(f"/api/releases/{v1}/items").json()["items"]}

    def blob(digest):
        return settings.storage_dir / "blobs" / digest[:2] / digest[2:4] / digest

    assert thumbs and all(blob(t).is_file() for t in thumbs)

    # ---- a version for good: its image records stay while another version holds them
    assert hub.delete(f"/api/releases/{v2}").status_code == 200
    assert hub.get("/api/trash").json()["releases"][0]["in_use"] == []
    done = hub.post("/api/trash/empty", json={"releases": [v2]}).json()
    assert (done["projects"], done["releases"], done["kept"]) == (0, 1, [])
    assert hub.get("/api/trash").json() == {"projects": [], "releases": []}
    tree = hub.get(f"/api/projects/{project['id']}/datasets", params={"trash": True}).json()
    assert [r["version"] for d in tree for r in d["releases"]] == ["v1"]
    assert hub.get(f"/api/releases/{v1}").json()["stats"]["items"] == 8 and all(blob(t).is_file() for t in thumbs)
    assert hub.post("/api/trash/empty", json={"releases": [v2]}).status_code == 404  # not in the trash any more

    # ---- a version a run trained on stays, with the reason
    run = hub.post("/api/runs", json={"release_id": v1, "template": "dummy", "params": {"epochs": 1}, "gpus": 0}).json()
    deadline = time.monotonic() + 120
    while hub.get(f"/api/jobs/{run['job_id']}").json()["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.3)
    assert hub.delete(f"/api/releases/{v1}").status_code == 200
    (binned,) = hub.get("/api/trash").json()["releases"]
    assert "학습 1개" in binned["in_use"]
    kept = hub.post("/api/trash/empty", json={}).json()
    assert kept["releases"] == 0 and kept["kept"][0]["id"] == v1 and "학습" in kept["kept"][0]["reason"]

    # ---- the whole project: datasets, runs, models and the files only they used
    listed = hub.get(f"/api/projects/{project['id']}/models").json()
    model = hub.get(f"/api/models/{listed[0]['id']}").json()
    assert hub.post(f"/api/releases/{v1}/restore").status_code == 200
    assert hub.delete(f"/api/projects/{project['id']}").status_code == 200
    gone = hub.post("/api/trash/empty", json={}).json()
    assert (gone["projects"], gone["kept"]) == (1, []) and gone["files"] >= len(thumbs) + 1 and gone["bytes"] > 0
    assert not any(blob(t).exists() for t in thumbs) and not blob(model["sha256"]).exists()
    assert hub.get("/api/trash").json() == {"projects": [], "releases": []}
    assert hub.get(f"/api/runs/{run['id']}").status_code == 404
    assert hub.post("/api/projects", json={"name": "비울 것"}).status_code == 200  # the name is free again
    actions = [a["action"] for a in hub.get("/api/activity?limit=5").json()]
    assert "trash.empty" in actions

    # ---- retention: emptied by itself after so many days; work folders on the nodes
    policy = hub.put("/api/storage/policy", json={"trash_days": 7}).json()
    assert (policy["trash_days"], policy["workdir_days"], policy["outputs_days"]) == (7, 30, 0)
    assert hub.put("/api/storage/policy", json={"workdir_days": 4000}).status_code == 400


def test_old_work_folders_on_a_node(tmp_path):
    from badamoa.node import workdirs

    root = tmp_path / "runs"
    now = time.time()
    for name, age_days in (("old-failed", 40), ("recent", 2), ("running-old", 40)):
        out = root / name / "out"
        out.mkdir(parents=True)
        (out / "last.pt").write_bytes(b"x" * 1000)
        for path in (out / "last.pt", out, root / name):
            os.utime(path, (now - age_days * 86400, now - age_days * 86400))
    assert workdirs.sweep(tmp_path, 0, set(), now) == (0, 0)  # 0: keep
    assert workdirs.sweep(tmp_path, 30, {"running-old"}, now) == (1, 1000)
    assert sorted(p.name for p in root.iterdir()) == ["recent", "running-old"]
