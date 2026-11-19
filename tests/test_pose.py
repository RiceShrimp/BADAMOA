"""Keypoint (pose) datasets: read from YOLO pose and COCO keypoints, kept with each box, shown on the grid."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from conftest import ADMIN
from PIL import Image
from sample_data import make_yolo

from badamoa.node import agent as node_agent
from badamoa.node import importer


def make_yolo_pose(root, images=4):
    """images/train/*.jpg + labels with a box and 3 keypoints (x, y, visibility) + kpt_shape in data.yaml."""
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for i in range(images):
        Image.new("RGB", (200, 100), (40 * i, 90, 120)).save(root / "images" / "train" / f"p{i}.jpg")
        line = "0 0.5 0.5 0.4 0.6 0.40 0.30 2 0.60 0.30 1 0.50 0.80 0"
        (root / "labels" / "train" / f"p{i}.txt").write_text(line + "\n", encoding="utf-8")
    yaml = "path: .\ntrain: images/train\nkpt_shape: [3, 3]\nnames:\n  0: person\n"
    (root / "data.yaml").write_text(yaml, encoding="utf-8")
    return root


def test_yolo_pose_keeps_keypoints(tmp_path):
    found = importer.scan(make_yolo_pose(tmp_path / "pose"))
    assert found.keypoints == {"count": 3, "names": ["1", "2", "3"], "skeleton": []}
    box = found.entries[0].boxes[0]
    assert box[:5] == [0, 0.5, 0.5, 0.4, 0.6] and box[5:] == [0.4, 0.3, 2.0, 0.6, 0.3, 1.0, 0.5, 0.8, 0.0]
    # 17 points without names: COCO's names and limbs
    assert importer.keypoint_info(17)["names"][0] == "nose" and len(importer.keypoint_info(17)["skeleton"]) == 19
    # a plain YOLO dataset is not taken for keypoints
    assert importer.scan(make_yolo(tmp_path / "plain")).keypoints is None


def test_coco_keypoints(tmp_path):
    root = tmp_path / "coco"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(root / "images" / "a.jpg")
    data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["head", "hand", "foot"], "skeleton": [[1, 2], [1, 3]]}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 20, 100, 60],
                "keypoints": [100, 30, 2, 60, 50, 1, 0, 0, 0],
            }
        ],
    }
    (root / "annotations").mkdir()
    (root / "annotations" / "person_keypoints_train.json").write_text(json.dumps(data), encoding="utf-8")
    found = importer.scan(root)
    assert found.keypoints == {"count": 3, "names": ["head", "hand", "foot"], "skeleton": [[0, 1], [0, 2]]}
    assert found.entries[0].boxes[0][5:] == [0.5, 0.3, 2.0, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def hub(live_hub, tmp_path, monkeypatch):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        join = client.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        client.node_id = runner.config.node_id
        yield client
        runner.stop()


def _import(hub, project_id, path, dataset="people", task=""):
    body = {"node_id": hub.node_id, "path": str(path), "dataset": dataset, "version": "v1", "task": task}
    resp = hub.post(f"/api/projects/{project_id}/imports", json=body)
    assert resp.status_code == 200, resp.text
    job = resp.json()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and job["status"] in ("queued", "running"):
        time.sleep(0.3)
        job = hub.get(f"/api/jobs/{job['id']}").json()
    return job


def test_keypoint_datasets_end_to_end(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "자세"}).json()
    job = _import(hub, project["id"], make_yolo_pose(tmp_path / "pose"))  # kpt_shape: Kpt, without being told
    assert job["status"] == "succeeded", job.get("error")
    release = hub.get(f"/api/releases/{job['result']['release_id']}").json()
    assert release["task"] == "pose"
    (cls,) = release["label_space"]["classes"]
    assert (cls["name"], cls["keypoints"], cls["skeleton"]) == ("person", ["1", "2", "3"], [])
    item = hub.get(f"/api/releases/{release['id']}/items").json()["items"][0]
    assert len(item["boxes"][0]) == 5 + 9  # the grid gets the keypoints too
    detail = hub.get(f"/api/releases/{release['id']}/items/{item['id']}").json()
    assert detail["revisions"][0]["boxes"][0][5:8] == [0.4, 0.3, 2.0]
    # keypoints train since 0.60.0: Ultralytics YOLO is offered for Kpt, with its pose models
    offered = hub.get("/api/templates", params={"task": "pose"}).json()
    assert offered[0]["name"] == "yolo" and offered[0]["data_format"] == "yolo-pose"
    assert {p["name"]: p for p in offered[0]["params"]}["model"]["default"] == "yolo11n-pose.pt"
    checked = hub.post("/api/runs/check", json={"release_id": release["id"]}).json()
    assert not [p for p in checked["problems"] if "태스크" in p or "Kpt" in p], checked

    # keypoint data imported as Det keeps the boxes; plain boxes are not imported as Kpt
    job = _import(hub, project["id"], make_yolo_pose(tmp_path / "pose2"), "boxes", "Det")
    item = hub.get(f"/api/releases/{job['result']['release_id']}/items").json()["items"][0]
    assert len(item["boxes"][0]) == 5
    refused = _import(hub, project["id"], make_yolo(tmp_path / "plain"), "plain", "Kpt")
    logs = " ".join(line["text"] for line in hub.get(f"/api/jobs/{refused['id']}/logs").json())
    assert refused["status"] == "failed" and "폴더는 Det 데이터, 가져오려는 태스크는 Kpt" in logs
    cards = hub.get(f"/api/projects/{project['id']}/overview").json()
    assert cards["tasks"] == ["pose", "detect"] and {d["name"]: d["task"] for d in cards["dataset_cards"]} == {
        "people": "pose",
        "boxes": "detect",
    }

    # a task the team names: labels kept whole, trained only by a template that declares it
    job = _import(hub, project["id"], make_yolo_pose(tmp_path / "ocr"), "plates", "OCR")
    release = hub.get(f"/api/releases/{job['result']['release_id']}").json()
    assert release["task"] == "OCR"
    item = hub.get(f"/api/releases/{release['id']}/items").json()["items"][0]
    assert len(item["boxes"][0]) == 5 + 9
    resp = hub.post("/api/runs", json={"release_id": release["id"]})
    assert resp.status_code == 400 and "OCR 태스크를 학습하는 템플릿이 없습니다" in resp.json()["detail"]
    bad = hub.post(
        f"/api/projects/{project['id']}/imports",
        json={"node_id": hub.node_id, "path": str(tmp_path), "dataset": "x", "version": "v1", "task": "문자 인식"},
    )
    assert bad.status_code == 400 and "태스크 이름은" in bad.json()["detail"]
