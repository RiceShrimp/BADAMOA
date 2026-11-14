"""Light real-data check with COCO128 (128 COCO images with YOLO labels, about 7 MB).

Opt-in: runs only when the dataset is on disk. Fetch it once with

    uv run python scripts/fetch_coco128.py

Routine verification uses well-known, small data like this rather than large benchmarks.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import httpx
import pytest
from conftest import ADMIN

from badamoa.node import agent as node_agent
from badamoa.node import importer

COCO128 = Path(os.environ.get("BADAMOA_COCO128", Path(__file__).parent.parent / ".badamoa" / "data" / "coco128"))
pytestmark = pytest.mark.skipif(not (COCO128 / "images").is_dir(), reason="COCO128 not downloaded")


def test_scan_reads_coco128_layout():
    found = importer.scan(COCO128)
    assert found.format == "yolo" and len(found.entries) == 128
    assert len(found.classes) == 80 and found.classes[0] == "person" and found.classes[79] == "toothbrush"
    assert {e.split for e in found.entries} == {"train"}  # from the "train2017" folder
    assert sum(len(e.boxes) for e in found.entries) == 929  # instances in COCO128
    # two images ship without a label file: nobody recorded that they are empty, so they count as unlabelled
    assert sum(e.state == "unlabelled" for e in found.entries) == 2


def test_split_comes_from_coco_style_folder_names_without_yaml(tmp_path):
    for name in ("train2017", "val2017"):
        (tmp_path / "images" / name).mkdir(parents=True)
        (tmp_path / "labels" / name).mkdir(parents=True)
    sample = sorted((COCO128 / "images" / "train2017").glob("*.jpg"))[:2]
    (tmp_path / "images" / "train2017" / sample[0].name).write_bytes(sample[0].read_bytes())
    (tmp_path / "images" / "val2017" / sample[1].name).write_bytes(sample[1].read_bytes())
    found = importer.scan(tmp_path)
    assert sorted(e.split for e in found.entries) == ["train", "val"]


def test_import_coco128_end_to_end(live_hub, tmp_path, monkeypatch):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as hub:
        join = hub.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="data-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        try:
            project = hub.post("/api/projects", json={"name": "COCO"}).json()
            started = time.monotonic()
            job = hub.post(
                f"/api/projects/{project['id']}/imports",
                json={"node_id": runner.config.node_id, "path": str(COCO128), "dataset": "coco128", "version": "v1"},
            ).json()
            while job["status"] in ("queued", "running") and time.monotonic() - started < 180:
                time.sleep(0.3)
                job = hub.get(f"/api/jobs/{job['id']}").json()
            assert job["status"] == "succeeded", job
            print(f"\nCOCO128 import: {time.monotonic() - started:.1f}s for 128 images")

            release = hub.get(f"/api/releases/{job['result']['release_id']}").json()
            stats = release["stats"]
            assert stats["items"] == 128 and stats["boxes"] == 929
            assert stats["states"] == {"labelled": 126, "background": 0, "unlabelled": 2}
            assert len(release["label_space"]["classes"]) == 80
            person = stats["classes"]["0"]
            assert person["boxes"] == 254 and person["items"] == 61  # "person" in COCO128

            people = hub.get(f"/api/releases/{release['id']}/items", params={"class_id": 0, "limit": 200}).json()
            assert people["total"] == 61
            first = hub.get(f"/api/releases/{release['id']}/items/{people['items'][0]['id']}").json()
            image = httpx.get(str(hub.base_url).rstrip("/") + first["image_url"])
            assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
            assert any(box[0] == 0 for box in first["revisions"][0]["boxes"])
        finally:
            runner.stop()


@pytest.mark.skipif(not os.environ.get("BADAMOA_TEST_YOLO"), reason="set BADAMOA_TEST_YOLO=1 to train a real YOLO")
def test_train_yolo_on_coco128(live_hub, tmp_path, monkeypatch):
    """One epoch of the smallest YOLO on CPU: builds the plugin environment, trains, registers the model.

    The first run downloads PyTorch (CPU build, a few hundred MB) into uv's cache; later runs take under a minute.
    """
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    monkeypatch.setenv("BADAMOA_TORCH_BACKEND", "cpu")
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as hub:
        join = hub.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="train-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()

        def wait(job_id: str, timeout: float) -> dict:
            deadline = time.monotonic() + timeout
            job = hub.get(f"/api/jobs/{job_id}").json()
            while job["status"] in ("queued", "running") and time.monotonic() < deadline:
                time.sleep(0.5)
                job = hub.get(f"/api/jobs/{job_id}").json()
            return job

        try:
            project = hub.post("/api/projects", json={"name": "COCO"}).json()
            body = {"node_id": runner.config.node_id, "path": str(COCO128), "dataset": "coco128", "version": "v1"}
            imported = wait(hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"], 180)
            assert imported["status"] == "succeeded", imported

            settings = {"model": "yolo11n.pt", "epochs": 1, "imgsz": 160, "batch": 16, "gpus": 0, "workers": 0}
            run = hub.post("/api/runs", json={"release_id": imported["result"]["release_id"], **settings}).json()
            job = wait(run["job_id"], 1800)
            logs = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs?limit=5000").json())
            assert job["status"] == "succeeded", logs[-3000:]
            assert "\x1b" not in logs  # colours and cursor codes are stripped

            done = hub.get(f"/api/runs/{run['id']}").json()
            assert done["history"]["steps"] == [1]  # the closing validation is not a second epoch
            recorded = set(done["history"]["series"])
            assert {"map50_95", "map50", "precision", "recall", "train_box_loss", "val_box_loss"} <= recorded
            assert done["export"]["items"]["train"] + done["export"]["items"]["val"] == 126  # 2 unlabelled left out
            assert done["export"]["val_source"] == "holdout" and done["info"]["torch"].endswith("+cpu")
            model = hub.get(f"/api/models/{done['model']['id']}").json()
            assert model["arch"] == "yolo11n" and len(model["classes"]) == 80 and model["bytes"] > 1_000_000
            assert 0 <= model["metrics"]["map50_95"] <= 1 and model["class_metrics"]
            weights = httpx.get(str(hub.base_url).rstrip("/") + model["download_url"])
            assert weights.status_code == 200 and len(weights.content) == model["bytes"]
        finally:
            runner.stop()
