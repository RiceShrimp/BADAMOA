"""Copies of versions on the servers that received them: seen, kept or removed, checked against the Hub.

And a version's fingerprint, like a commit hash.
"""

from __future__ import annotations

import re
import time

import pytest
from sample_data import make_yolo
from test_transfer import _logs, _wait
from test_transfer import two_nodes as _two_nodes  # noqa: F401  (a fixture: pytest finds it by this name)

from badamoa.hub import catalog
from badamoa.hub.catalog_models import Release
from badamoa.hub.db import Database
from badamoa.node import cache


@pytest.fixture
def two_nodes(request):
    """A node holding the images and another to train on (``test_transfer``)."""
    return request.getfixturevalue("_two_nodes")


def _import(hub, tmp_path, name="street"):
    data = hub.data.config.node_id
    project = hub.post("/api/projects", json={"name": f"{name} 프로젝트"}).json()
    body = {"node_id": data, "path": str(make_yolo(tmp_path / name)), "dataset": name, "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    return job["result"]["release_id"]


def _run(hub, release, **extra):
    body = {"release_id": release, "template": "dummy", "params": {"epochs": 1}, "gpus": 0, **extra}
    resp = hub.post("/api/runs", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _cached(hub) -> list[str]:
    images = cache.root(hub.train.home) / "images"
    return sorted(p.name for p in images.rglob("*") if p.is_file()) if images.is_dir() else []


def _until(check, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.3)
    return check()


def _copy(hub, release, node):
    copies = hub.get(f"/api/releases/{release}/copies").json()["copies"]
    return next((c for c in copies if c["node_id"] == node), None)


def test_a_version_has_a_fingerprint_and_can_be_named_by_it(two_nodes, tmp_path, settings):
    hub = two_nodes
    release = _import(hub, tmp_path)
    view = hub.get(f"/api/releases/{release}").json()
    assert re.fullmatch(r"[0-9a-f]{64}", view["digest"]) and view["same_as"] == []

    # the same content as another version: the same fingerprint, and each names the other
    again = hub.post("/api/releases", json={"sources": [release], "version": "v2"}).json()["release_id"]
    twin = hub.get(f"/api/releases/{again}").json()
    assert twin["digest"] == view["digest"]
    assert twin["same_as"] == [{"id": release, "version": "v1", "dataset": "street"}]
    # other content (the images dealt into other splits): another fingerprint
    split = {"train": 50, "val": 50}
    other = hub.post("/api/releases", json={"sources": [release], "version": "v3", "split": split}).json()["release_id"]
    third = hub.get(f"/api/releases/{other}").json()["digest"]
    assert third != view["digest"]

    # named by id, by dataset@version or by the start of the fingerprint (7 characters or more)
    assert hub.get("/api/release-ref", params={"ref": "street@v3"}).json()["id"] == other
    assert hub.get("/api/release-ref", params={"ref": third[:9]}).json()["id"] == other
    both = hub.get("/api/release-ref", params={"ref": view["digest"][:12]})
    assert both.status_code == 409 and "2개" in both.json()["detail"]
    assert hub.get("/api/release-ref", params={"ref": "street@v9"}).status_code == 404
    assert hub.get("/api/release-ref", params={"ref": "abc"}).status_code == 404  # too short to be one

    # versions made before there were fingerprints get theirs from the Hub's background pass
    database = Database(settings.database_url)
    with database.session() as db:
        db.get(Release, release).digest = ""
    with database.session() as db:
        catalog.fill_digests(db)  # (the live Hub's own pass may have got there first)
        assert db.get(Release, release).digest == view["digest"]


def test_a_run_can_remove_the_copy_its_server_received(two_nodes, tmp_path):
    hub = two_nodes
    data, train = hub.data.config.node_id, hub.train.config.node_id
    release = _import(hub, tmp_path)

    # the copy goes once the run succeeded, the images on the server that registered them stay
    run = _run(hub, release, nodes=[train], data_after="delete")
    assert run["config"]["data_after"] == "delete"
    assert _wait(hub, run["job_id"])["status"] == "succeeded", _logs(hub, run["job_id"])
    assert _until(lambda: _copy(hub, release, train) is None)
    assert _until(lambda: not _cached(hub)), _cached(hub)
    copies = hub.get(f"/api/releases/{release}/copies").json()
    assert copies["origin"]["node_id"] == data and copies["copies"] == []
    assert {n["id"]: n["images"] for n in hub.get(f"/api/releases/{release}/nodes").json()["nodes"]}[train] == ""

    # a run that keeps it: the next one starts right there
    kept = _run(hub, release, nodes=[train], data_after="keep")
    assert _wait(hub, kept["job_id"])["status"] == "succeeded", _logs(hub, kept["job_id"])
    copy = _copy(hub, release, train)
    assert copy["state"] == "kept" and copy["items"] == 8 and copy["present_items"] == 8
    assert copy["runs"] == 2 and copy["last_used_at"] and copy["removable"]
    assert len(_cached(hub)) == 8
    on_node = hub.get(f"/api/nodes/{train}/copies").json()
    assert [c["release"]["dataset"] for c in on_node["copies"]] == ["street"] and on_node["can_check"]

    # a run that failed with "delete": kept so it can continue, and shown as such
    failed = _run(hub, release, nodes=[train], data_after="delete", params={"epochs": 1, "fail": True})
    assert _wait(hub, failed["job_id"])["status"] == "failed"
    assert _copy(hub, release, train)["state"] == "held" and len(_cached(hub)) == 8


def test_a_copy_is_removed_from_the_hub_but_not_while_a_job_uses_it(two_nodes, tmp_path):
    hub = two_nodes
    train = hub.train.config.node_id
    release = _import(hub, tmp_path)
    first = _run(hub, release, nodes=[train])
    assert _wait(hub, first["job_id"])["status"] == "succeeded", _logs(hub, first["job_id"])
    assert len(_cached(hub)) == 8

    # while a run there uses the version, removing it is refused with the reason
    slow = _run(hub, release, nodes=[train], params={"epochs": 40, "sleep": 0.25})
    busy = hub.delete(f"/api/releases/{release}/copies/{train}")
    assert busy.status_code == 409 and "학습" in busy.json()["detail"]
    assert _copy(hub, release, train)["in_use"] and not _copy(hub, release, train)["removable"]
    assert _wait(hub, slow["job_id"])["status"] == "succeeded"

    # afterwards it goes; an image another kept version shares would have stayed
    gone = hub.delete(f"/api/releases/{release}/copies/{train}")
    assert gone.status_code == 200 and gone.json()["copies"][0]["state"] == "removing"
    assert _until(lambda: _copy(hub, release, train) is None)
    assert _until(lambda: not _cached(hub)), _cached(hub)
    assert hub.delete(f"/api/releases/{release}/copies/{train}").status_code == 404


def test_a_node_checks_its_copies_against_the_hub(two_nodes, tmp_path):
    hub = two_nodes
    train = hub.train.config.node_id
    release = _import(hub, tmp_path)
    run = _run(hub, release, nodes=[train])
    assert _wait(hub, run["job_id"])["status"] == "succeeded", _logs(hub, run["job_id"])
    root = cache.root(hub.train.home)

    # one image gone by hand, one file nobody asked for
    (lost,) = sorted((root / "images").rglob("*.jpg"))[:1]
    lost.unlink()
    stray = root / "images" / "00" / "00" / ("0" * 64 + ".jpg")
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"not a version's image")

    looked = hub.post(f"/api/nodes/{train}/copies/check", json={}).json()
    assert _wait(hub, looked["job_id"])["status"] == "succeeded", _logs(hub, looked["job_id"])
    copy = _copy(hub, release, train)
    assert (copy["items"], copy["present_items"]) == (8, 7) and copy["checked_at"]
    check = hub.get(f"/api/nodes/{train}/copies").json()["check"]
    assert check["unknown"]["files"] == 1 and check["removed"]["files"] == 0 and stray.exists()

    cleaned = hub.post(f"/api/nodes/{train}/copies/check", json={"clean": True}).json()
    assert _wait(hub, cleaned["job_id"])["status"] == "succeeded", _logs(hub, cleaned["job_id"])
    check = hub.get(f"/api/nodes/{train}/copies").json()["check"]
    assert check["removed"]["files"] == 1 and not stray.exists() and len(_cached(hub)) == 7

    # the next run brings back what was missing
    again = _run(hub, release, nodes=[train])
    assert _wait(hub, again["job_id"])["status"] == "succeeded", _logs(hub, again["job_id"])
    assert len(_cached(hub)) == 8


def test_cache_sweep_spares_what_arrives_after_the_list_was_made(tmp_path):
    root = tmp_path / "cache"
    old = cache.path_of(root, "a" * 64, ".jpg")
    keep = cache.path_of(root, "b" * 64, ".jpg")
    for path in (old, keep):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 10)
    needed: set[str] = set()
    assert cache.tally(root, [{"sha256": "b" * 64, "ext": ".jpg", "bytes": 10}], needed) == {"items": 1, "bytes": 10}
    before = time.time() - 60  # the list was made before both files arrived: nothing goes
    assert cache.sweep(root, needed, clean=True, since=before)["removed"]["files"] == 0 and old.exists()
    swept = cache.sweep(root, needed, clean=True, since=time.time() + 1)
    assert swept == {"unknown": {"files": 1, "bytes": 10}, "removed": {"files": 1, "bytes": 10}}
    assert not old.exists() and keep.exists()
