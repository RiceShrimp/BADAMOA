"""Dataset catalog: in-place import by a real Node Agent, releases, label revisions, comparison."""

from __future__ import annotations

import threading
import time

import httpx
import pytest
from conftest import ADMIN
from sample_data import CLASSES, make_coco, make_yolo

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
    assert states["train_0001.jpg"] == "background" and states["train_0002.jpg"] == "unlabelled"
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
    assert stats["states"] == {"labelled": 6, "background": 1, "unlabelled": 1}
    assert stats["sites"] == ["Hangang"] and stats["bytes"] > 0
    assert [c["name"] for c in release["label_space"]["classes"]] == CLASSES and release["base"] is None
    assert sum(c["boxes"] for c in stats["classes"].values()) == stats["boxes"] > 0

    # ---- filters and paging
    def items(**params):
        return hub.get(f"/api/releases/{v1}/items", params=params).json()

    assert items()["total"] == 8 and items(split="val")["total"] == 2
    assert [i["file_name"] for i in items(state="background")["items"]] == ["train_0001.jpg"]
    assert items(state="unlabelled")["items"][0]["box_count"] == 0
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
