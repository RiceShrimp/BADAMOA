"""A new Hub starts with COCO8 for detection, segmentation and keypoints, and a 검수 example with planted mistakes."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
from conftest import ADMIN

from badamoa.core.paths import component_home
from badamoa.hub import samples, services
from badamoa.hub.config import HubSettings
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


def test_a_relative_data_folder_hands_the_node_paths_it_can_find(tmp_path, monkeypatch):
    """`--home .badamoa/hub` from the install folder: the samples' folders handed to this PC's node are absolute, for the
    node's imports run in folders of their own (a relative path failed with "폴더를 찾을 수 없습니다")."""
    monkeypatch.chdir(tmp_path)
    assert component_home("hub", ".badamoa/hub") == tmp_path / ".badamoa" / "hub"
    settings = HubSettings(home=Path(".badamoa") / "hub", web_dir=tmp_path / "no-web")
    assert settings.home == tmp_path / ".badamoa" / "hub"
    database = Database(settings.database_url)
    database.create_all()
    with database.session() as db:
        services.ensure_local_node(db, {"hostname": "hubpc"}, "0.80.0")
        user = services.create_user(db, "admin", "admin-pass-1", role="admin")
        paths = [Path(job.params["path"]) for job in samples.seed(db, user, settings)]
    database.engine.dispose()
    assert len(paths) == 4 and all(p.is_absolute() and p.is_dir() for p in paths)


def test_a_hub_without_its_own_node_starts_without_samples(client):
    made = client.post("/api/setup", json=ADMIN).json()
    assert made["samples"] == {"project": "", "imports": 0}


def test_samples_go_in_while_this_pcs_node_beats(client, settings, monkeypatch):
    """The node of the Hub's own PC writes its heartbeat while the samples are copied: the samples still go in
    (they used to fail with "database is locked", a write after a read another connection's write came between)."""
    import sqlite3

    database = Database(settings.database_url)
    with database.session() as db:
        services.ensure_local_node(db, {"hostname": "hubpc"}, "0.80.0")
        user = services.create_user(db, "admin", "admin-pass-1", role="admin")
        user_id = user.id
    path = settings.database_url.removeprefix("sqlite:///")
    copy = samples._copy

    def beating(target):
        copy(target)
        beat = sqlite3.connect(path, timeout=0.3)
        try:
            beat.execute("UPDATE nodes SET last_seen_at = CURRENT_TIMESTAMP")
            beat.commit()
        except sqlite3.OperationalError:
            pass  # it waits for the samples' write lock (a real node tries again on its next beat)
        finally:
            beat.close()

    monkeypatch.setattr(samples, "_copy", beating)
    with database.session() as db:
        made = samples.safely(db, db.get(services.User, user_id), settings)
    database.engine.dispose()
    assert made == {"project": samples.PROJECT, "imports": 4}
