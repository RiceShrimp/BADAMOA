"""A new Hub starts with COCO8 for detection, segmentation and keypoints, and a 검수 example with planted mistakes."""

from __future__ import annotations

import threading
import time

import httpx
from conftest import ADMIN

from badamoa.hub import samples, services
from badamoa.hub.db import Database
from badamoa.node import agent as node_agent


def test_the_first_admin_finds_samples_and_a_review_example(live_hub, settings, monkeypatch):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    database = Database(settings.database_url)
    with database.session() as db:  # this PC's own node, as `badamoa hub run` makes it before anyone signs in
        node, token = services.ensure_local_node(db, {"hostname": "hubpc"}, "0.80.0")
        node_id = node.id
    database.engine.dispose()
    home = settings.home / "local-node"
    home.mkdir(parents=True)
    runner = node_agent.NodeAgent(node_agent.NodeConfig(hub_url=live_hub, node_id=node_id, node_token=token), home)
    threading.Thread(target=runner.run_forever, daemon=True).start()
    try:
        made = httpx.post(f"{live_hub}/api/setup", json=ADMIN, timeout=60).json()
        assert made["samples"] == {"project": samples.PROJECT, "imports": 4}
        hub = httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {made['token']}"}, timeout=60)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            jobs = [j for j in hub.get("/api/jobs").json() if j["type"] == "data.import"]
            if jobs and all(j["status"] in ("succeeded", "failed") for j in jobs):
                break
            time.sleep(0.5)
        assert [j["status"] for j in jobs] == ["succeeded"] * 4, jobs
        project = next(p for p in hub.get("/api/projects").json() if p["name"] == samples.PROJECT)
        tree = {d["name"]: d for d in hub.get(f"/api/projects/{project['id']}/datasets").json()}
        assert {name: d["task"] for name, d in tree.items()} == {
            "COCO8 검출": "detect",
            "COCO8 분할": "segment",
            "COCO8 키포인트": "pose",
            "COCO8 검수 예시": "detect",
        }
        assert (settings.home / "samples" / "coco8-review" / "labels" / "train").is_dir()

        # ---- 검수 finds each planted mistake
        example = tree["COCO8 검수 예시"]["releases"][0]["id"]
        counts = {c["id"]: c["images"] for c in hub.get(f"/api/releases/{example}/qa").json()["checks"]}
        assert {k: counts[k] for k in ("duplicate", "tiny", "huge", "class_overlap", "unlabelled")} == {
            "duplicate": 1, "tiny": 1, "huge": 1, "class_overlap": 1, "unlabelled": 1,
        }  # fmt: skip
        plain = tree["COCO8 검출"]["releases"][0]["id"]
        counts = {c["id"]: c["images"] for c in hub.get(f"/api/releases/{plain}/qa").json()["checks"]}
        assert counts["duplicate"] == counts["tiny"] == counts["huge"] == counts["unlabelled"] == 0
    finally:
        runner.stop()


def test_a_hub_without_its_own_node_starts_without_samples(client):
    made = client.post("/api/setup", json=ADMIN).json()
    assert made["samples"] == {"project": "", "imports": 0}
