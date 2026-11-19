"""The Hub's own storage (0.61.0): datasets' thumbnails counted per project, the folder counted in the background,
imports refused before they read when they would not fit, and the database's empty pages given back."""

from __future__ import annotations

import threading
import time
from collections import namedtuple

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_yolo
from sqlalchemy import select, text, update

from badamoa.hub import hub_storage
from badamoa.hub.catalog_models import Item
from badamoa.hub.db import Database
from badamoa.hub.models import Job
from badamoa.node import agent as node_agent

Disk = namedtuple("Disk", "total used free")


@pytest.fixture
def hub(live_hub, tmp_path, monkeypatch):
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


def _import(hub, project_id: str, path, version: str) -> dict:
    body = {"node_id": hub.node_id, "path": str(path), "dataset": "street", "version": version}
    job = hub.post(f"/api/projects/{project_id}/imports", json=body).json()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = hub.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("succeeded", "failed", "canceled"):
            break
        time.sleep(0.3)
    job["log"] = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
    return job


def test_datasets_take_their_own_share_of_the_hub(hub, tmp_path, settings):
    project = hub.post("/api/projects", json={"name": "저장", "task": "detect"}).json()
    done = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")
    assert done["status"] == "succeeded", done["log"]
    assert "Hub에 썸네일 · 기록으로 약" in done["log"]  # the node says what the Hub will keep

    usage = hub.get("/api/storage").json()
    assert usage["parts"]["datasets"] > 0 and usage["parts"]["database"] > 0 and usage["per_image"] > 0
    row = next(p for p in usage["projects"] if p["id"] == project["id"])
    assert row["images"] == 8 and row["datasets"] == usage["parts"]["datasets"] and row["total"] >= row["datasets"]
    assert usage["keep_free"] >= hub_storage.KEEP_FREE_MIN

    # The folder is counted in the background; the screens read the last count.
    database = Database(settings.database_url)
    counted = hub_storage.measure(database, settings)
    assert counted["files"] >= 8 and counted["bytes"] >= usage["parts"]["datasets"]
    after = hub.get("/api/storage").json()
    assert after["counted_at"] and after["files"] == counted["files"]

    # Images registered before sizes were kept get theirs filled in by that same pass.
    with database.session() as db:
        db.execute(update(Item).values(thumb_bytes=0))
    hub_storage.mark_changed()  # written behind the Hub's back: the kept count does not know
    assert hub.get("/api/storage").json()["parts"]["datasets"] == 0
    assert hub_storage.fill_thumb_sizes(database, settings) == 8
    assert hub.get("/api/storage").json()["parts"]["datasets"] == usage["parts"]["datasets"]
    database.engine.dispose()


def test_an_import_that_would_not_fit_stops_before_it_reads(hub, tmp_path, monkeypatch):
    project = hub.post("/api/projects", json={"name": "꽉 참", "task": "detect"}).json()
    full = Disk(total=100 * 1024**3, used=99 * 1024**3, free=1 * 1024**3)  # less than what is always kept free
    monkeypatch.setattr(hub_storage.shutil, "disk_usage", lambda _path: full)
    done = _import(hub, project["id"], make_yolo(tmp_path / "a"), "v1")
    assert done["status"] == "failed"
    assert "Hub의 디스크가 모자랍니다" in done["log"] and "이미지 8장" in done["log"]
    assert hub.get(f"/api/projects/{project['id']}/datasets").json() == []  # nothing half made is left


def test_compacting_gives_the_empty_pages_back(hub, settings):
    database = Database(settings.database_url)
    with database.engine.begin() as conn:  # make room in the file, then free it: empty pages
        conn.execute(text("CREATE TABLE filler (blob BLOB)"))
        conn.execute(text("INSERT INTO filler SELECT randomblob(4096) FROM (WITH RECURSIVE n(i) AS "
                          "(SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 2000) SELECT i FROM n)"))  # fmt: skip
        conn.execute(text("DROP TABLE filler"))
    usage = hub.get("/api/storage").json()["database"]
    assert usage["empty"] > 4 * 1024**2 and not usage["compacting"]

    # not while a job runs: imports write as they go
    with database.session() as db:
        job = Job(type="data.import", params={}, status="queued")
        db.add(job)
    refused = hub.post("/api/storage/compact")
    assert refused.status_code == 409 and "작업이 돌고 있어" in refused.json()["detail"]
    with database.session() as db:
        db.get(Job, db.scalar(select(Job.id).where(Job.type == "data.import"))).status = "canceled"

    assert hub.post("/api/storage/compact").status_code == 200
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        usage = hub.get("/api/storage").json()["database"]
        if not usage["compacting"] and usage["last_compact"]:
            break
        time.sleep(0.3)
    assert usage["empty"] == 0 and usage["last_compact"]["after"] < usage["last_compact"]["before"]
    database.engine.dispose()


def test_thumbnails_are_small_webp_and_older_ones_are_shrunk(hub, tmp_path, settings):
    """Nodes make 256 px WebP thumbnails; ones from before (320 px JPEG) the Hub makes so from its own copy."""
    import io

    from PIL import Image

    from badamoa.hub import catalog

    project = hub.post("/api/projects", json={"name": "썸네일", "task": "detect"}).json()
    done = _import(hub, project["id"], make_yolo(tmp_path / "a", size=(640, 480)), "v1")
    assert done["status"] == "succeeded", done["log"]
    database = Database(settings.database_url)
    storage = catalog.storage_for(str(settings.storage_dir))
    with database.session() as db:
        items = db.scalars(select(Item).where(Item.project_id == project["id"])).all()
        assert {i.thumb_v for i in items} == {1}
        with Image.open(io.BytesIO(storage.get_blob(items[0].thumb_key))) as made:
            assert (made.format, max(made.size)) == ("WEBP", 256)
        # one as a node before 0.61 sent it: 320 px JPEG at quality 75
        old = io.BytesIO()
        Image.new("RGB", (320, 240), (200, 120, 40)).save(old, format="JPEG", quality=75)
        old_key = storage.put_blob(old.getvalue())
        items[0].thumb_key, items[0].thumb_bytes, items[0].thumb_v = old_key, len(old.getvalue()), 0
        first = items[0].id
    page = hub.get(f"/api/releases/{done['result']['release_id']}/items").json()
    url = next(i["thumb_url"] for i in page["items"] if i["id"] == first)
    base = str(hub.base_url).rstrip("/")
    assert httpx.get(base + url).headers["content-type"] == "image/jpeg"
    assert hub.get("/api/storage").json()["old_thumbs"] == 1

    assert hub_storage.shrink_thumbs(database, settings) == 1
    with database.session() as db:
        item = db.get(Item, first)
        assert item.thumb_v == 1 and item.thumb_key != old_key and item.thumb_bytes < len(old.getvalue())
    page = hub.get(f"/api/releases/{done['result']['release_id']}/items").json()
    url = next(i["thumb_url"] for i in page["items"] if i["id"] == first)
    assert httpx.get(base + url).headers["content-type"] == "image/webp"
    assert hub.get("/api/storage").json()["old_thumbs"] == 0
    database.engine.dispose()
