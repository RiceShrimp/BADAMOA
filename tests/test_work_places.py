"""A node's work folder: environments, work folders of jobs and fetched files on the disk with the most room,
what was kept in the home before still used (and moved when idle), and warnings for nearly full disks."""

from __future__ import annotations

import json
import os
import threading
import time

import httpx
from conftest import ADMIN

from badamoa.core import sysinfo
from badamoa.hub import disk_space
from badamoa.node import agent as node_agent
from badamoa.node import envs, places, storage, trainer, workdirs
from badamoa.node.places import choose as real_choose

ENV = {"requirements": ["ultralytics>=8.3.0"], "python": "3.12", "torch": True}


def _disks(monkeypatch, *disks):
    monkeypatch.setattr(sysinfo.MetricsSampler, "_disks", staticmethod(lambda: [dict(d) for d in disks]))


def _disk(mount, free_gb, total_gb=1000.0):
    return {"mount": str(mount), "free_gb": free_gb, "total_gb": total_gb}


def _ready_env(base, name, spec=ENV):
    folder = base / "envs" / name
    folder.mkdir(parents=True)
    (folder / "badamoa-env.json").write_text(json.dumps(spec), encoding="utf-8")
    (folder / "lib.bin").write_bytes(b"x" * 1000)
    return folder


# ---- where the work folder is ------------------------------------------------------------------------------


def test_the_work_folder_goes_to_the_roomiest_disk_and_stays(tmp_path, monkeypatch):
    monkeypatch.setattr(places, "choose", real_choose)
    system, big, other = (tmp_path / name for name in ("system", "big", "other"))
    home = system / "var" / "lib" / "badamoa" / "node"
    home.mkdir(parents=True)
    big.mkdir()
    other.mkdir()
    _disks(monkeypatch, _disk(system, 2), _disk(big, 2125), _disk(other, 400))

    work, error = places.root(home)
    assert work == big / places.FOLDER and work.is_dir() and not error
    assert (home / places.SAVED).read_text(encoding="utf-8") == str(work)

    # remembered: the disks filling up differently later does not move it
    _disks(monkeypatch, _disk(system, 2), _disk(big, 100), _disk(other, 400))
    assert places.root(home)[0] == work
    # an admin's folder wins; one the node cannot make is said, and the node's pick is used meanwhile
    chosen = other / "mine"
    assert places.root(home, str(chosen)) == (chosen, "")
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    fallback, error = places.root(home, str(blocker / "sub"))
    assert fallback == work and "쓸 수 없어" in error
    # every folder it used stays a place to look things up in
    assert places.of(home).roots()[:1] == [work]
    assert set(map(str, places.used_before(home))) == {str(work), str(chosen)}


def test_on_the_homes_own_disk_the_work_folder_is_the_home(tmp_path, monkeypatch):
    monkeypatch.setattr(places, "choose", real_choose)
    system, small = tmp_path / "system", tmp_path / "small"
    home = system / "node"
    home.mkdir(parents=True)
    small.mkdir()
    _disks(monkeypatch, _disk(system, 900), _disk(small, 50))
    assert places.root(home)[0] == home
    assert places.of(home).envs == home / "envs"  # as before


def test_a_disk_the_node_cannot_write_to_is_passed_over(tmp_path, monkeypatch):
    monkeypatch.setattr(places, "choose", real_choose)
    system, locked, next_best = tmp_path / "system", tmp_path / "locked", tmp_path / "next"
    home = system / "node"
    home.mkdir(parents=True)
    next_best.mkdir()
    locked.write_text("not a folder", encoding="utf-8")  # mkdir under it fails, like a disk the account cannot write
    _disks(monkeypatch, _disk(system, 5), _disk(locked, 3000), _disk(next_best, 800))
    assert places.choose(home) == next_best / places.FOLDER


def test_a_pick_whose_disk_is_gone_is_made_again(tmp_path, monkeypatch):
    monkeypatch.setattr(places, "choose", real_choose)
    system, gone, other = tmp_path / "system", tmp_path / "gone", tmp_path / "other"
    home = system / "node"
    home.mkdir(parents=True)
    other.mkdir()
    (home / places.SAVED).write_text(str(gone / places.FOLDER), encoding="utf-8")  # a disk no longer mounted
    _disks(monkeypatch, _disk(system, 5), _disk(other, 800))
    assert places.root(home)[0] == other / places.FOLDER


# ---- what was kept in the home before ----------------------------------------------------------------------


def test_environments_kept_in_the_home_are_used_where_they_are_and_new_ones_go_to_the_work_folder(tmp_path):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    places.use(home, work)
    try:
        old = envs.env_dir(home, ENV["requirements"], ENV["python"], ENV["torch"])
        assert old.parent == work / "envs"  # nothing yet: built in the work folder
        before = _ready_env(home, old.name)
        assert envs.env_dir(home, ENV["requirements"], ENV["python"], ENV["torch"]) == before
        assert envs.ensure(home, ENV["requirements"], ENV["python"], ENV["torch"], log=lambda _: None) == envs.python_of(
            before
        )
        new = envs.env_dir(home, ["timm"], "3.11", False)
        assert new.parent == work / "envs"
        _ready_env(work, new.name, {"requirements": ["timm"], "python": "3.11", "torch": False})
        assert [e["id"] for e in envs.listing(home)] == sorted([before.name, new.name])
        # removing finds it wherever it is
        assert envs.remove(home, ENV, log=lambda _: None)["removed"] is True and not before.exists()
    finally:
        places.use(home, None)


def test_a_job_finds_its_folders_from_the_node(tmp_path, monkeypatch):
    home, work = tmp_path / "home", tmp_path / "work"
    (home / "cache" / "models").mkdir(parents=True)
    (home / "cache" / "models" / "blob.pt").write_bytes(b"w")
    monkeypatch.setenv(places.ENV, str(work))  # what the agent gives every job
    where = places.of(home)
    assert where.work == work and where.runs == work / "runs"
    assert where.find("cache", "models", "blob.pt") == home / "cache" / "models" / "blob.pt"  # still there
    assert where.find("cache", "models", "other.pt") == work / "cache" / "models" / "other.pt"  # new: work folder
    # a container is given the folders its config names, wherever they are
    job = work / "runs" / "j1"
    job.mkdir(parents=True)
    command, _, _ = trainer.launch(
        job_id="j1",
        home=home,
        spec={"kind": "docker", "image": "img:1"},
        python=None,
        entrypoint="python train.py --config {config}",
        config={"cache": str(work / "cache" / "template-data" / "yolo")},
        config_path=job / "config.json",
        code=tmp_path,
        work=job,
        mounts=[(home / "cache" / "models", True)],
    )
    assert (work / "cache" / "ultralytics").is_dir()  # Ultralytics' settings: in the work folder
    mounted = " ".join(command)
    assert str(home / "cache" / "models") in mounted and str(work / "cache" / "ultralytics") in mounted


def _symlinks_work(tmp_path) -> bool:
    try:
        os.symlink(tmp_path, tmp_path / "probe-link", target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    (tmp_path / "probe-link").unlink()
    return True


def test_what_was_kept_in_the_home_moves_into_the_work_folder_once_and_environments_stay(tmp_path, monkeypatch):
    home, work, images = tmp_path / "home", tmp_path / "work", tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"i" * 10)
    env = _ready_env(home, "env-a")
    run = home / "runs" / "job-old"
    (run / "out").mkdir(parents=True)
    (run / "out" / "last.pt").write_bytes(b"c" * 500)
    links = _symlinks_work(tmp_path)
    if links:  # an export links to the images: the link moves, not the images
        (run / "data").mkdir()
        os.symlink(images / "a.jpg", run / "data" / "a.jpg")
    (home / "cache" / "models").mkdir(parents=True)
    (home / "cache" / "models" / "blob1.pt").write_bytes(b"w" * 300)
    (home / "cache" / "models" / "blob2.pt").write_bytes(b"w" * 300)
    (work / "cache" / "models").mkdir(parents=True)
    (work / "cache" / "models" / "blob2.pt").write_bytes(b"w" * 300)  # fetched again already: same content
    (home / "cache" / "templates" / "sha-a").mkdir(parents=True)
    (home / "cache" / "templates" / "sha-a" / "badamoa.toml").write_text('name = "a"', encoding="utf-8")
    (home / "cache" / "template-data" / "yolo").mkdir(parents=True)
    (home / "cache" / "images" / "ab").mkdir(parents=True)  # received images (the home's disk was the roomiest)
    (home / "node.json").write_text("{}", encoding="utf-8")

    # across disks: a copy beside the target, then the rename (here the first rename of each item fails)
    renames = []
    real_rename = os.rename

    def rename(source, target):
        renames.append(target)
        if not os.path.basename(source).startswith("."):  # the item itself: another disk
            raise OSError(18, "Invalid cross-device link")
        return real_rename(source, target)  # the copy beside the target

    monkeypatch.setattr(os, "rename", rename)
    where = places.Places(home, work)
    assert places.pending(where)
    done = places.move_old(where, log=lambda _: None)
    monkeypatch.setattr(os, "rename", real_rename)
    assert done == {"moved": 4, "removed": 1, "bytes": done["bytes"], "failed": 0}
    assert (work / "runs" / "job-old" / "out" / "last.pt").read_bytes() == b"c" * 500 and not run.exists()
    if links:
        moved = work / "runs" / "job-old" / "data" / "a.jpg"
        assert moved.is_symlink() and moved.read_bytes() == b"i" * 10
    assert sorted(p.name for p in (work / "cache" / "models").iterdir()) == ["blob1.pt", "blob2.pt"]
    assert not (home / "cache" / "models").exists()
    assert (work / "cache" / "templates" / "sha-a" / "badamoa.toml").is_file()
    assert (work / "cache" / "template-data" / "yolo").is_dir()
    # environments, received images and the node's own files stay
    assert env.is_dir() and (home / "cache" / "images" / "ab").is_dir() and (home / "node.json").is_file()
    assert not places.pending(where) and not list(work.rglob("*.moving"))


def test_the_storage_report_and_cleaning_cover_both_places(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "uv_cache_dir", lambda *_: None)
    monkeypatch.setattr(storage.containers, "docker", lambda: None)
    home, work = tmp_path / "home", tmp_path / "work"
    _ready_env(home, "env-old")
    _ready_env(work, "env-new")
    for base, job in ((home, "job-old"), (work, "job-new")):
        (base / "runs" / job).mkdir(parents=True)
        (base / "runs" / job / "x").write_bytes(b"c" * 10)
    places.use(home, work)
    try:
        report = storage.measure(home, home / "cache")
        assert report["work"] == str(work)
        items = {i["id"]: i for i in report["kinds"]["envs"]["items"]}
        assert report["kinds"]["envs"]["path"] == str(work / "envs")
        assert "path" not in items["env-new"] and items["env-old"]["path"] == str(home / "envs" / "env-old")
        assert sorted(i["id"] for i in report["kinds"]["runs"]["items"]) == ["job-new", "job-old"]
        assert str(work) in report["disks"] and str(home / "envs" / "env-old") in report["disks"]

        done = storage.clean(home, "runs", [], everything=True, log=lambda _: None, running={"job-new"})
        assert done["removed"] == 1 and not (home / "runs" / "job-old").exists() and (work / "runs" / "job-new").exists()
        assert storage.clean(home, "envs", ["env-old"], log=lambda _: None)["removed"] == 1
        assert not (home / "envs" / "env-old").exists()
        # the old work folders' sweep reaches both places
        (home / "runs" / "job-stale").mkdir(parents=True)
        os.utime(home / "runs" / "job-stale", (0, 0))
        assert workdirs.sweep(home, 1, set())[0] == 1
    finally:
        places.use(home, None)


def test_uvs_cache_is_kept_on_the_work_folders_disk(tmp_path, monkeypatch):
    home, work = tmp_path / "home", tmp_path / "work"
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    envs._uv_own_cache.cache_clear()
    monkeypatch.setattr(envs, "_uv_own_cache", lambda: tmp_path / "uv-own")
    assert envs.uv_env(places.Places(home, home)) == {}  # no work folder of its own: uv's own cache
    monkeypatch.setattr(places, "same_disk", lambda a, b: True)
    assert envs.uv_env(places.Places(home, work)) == {}  # uv's own is on that disk already
    monkeypatch.setattr(places, "same_disk", lambda a, b: False)
    assert envs.uv_env(places.Places(home, work)) == {"UV_CACHE_DIR": str(work / "uv-cache")}
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "admins"))
    assert envs.uv_env(places.Places(home, work)) == {}  # the server's own setting stays


# ---- nearly full disks -------------------------------------------------------------------------------------


def test_nearly_full_disks_are_named_with_what_is_on_them():
    assert disk_space.level(2.0, 468) == "full"
    assert disk_space.level(10.8, 468) == "low"  # under 5 % of the disk
    assert disk_space.level(30, 468) == ""
    assert disk_space.level(9, 100) == "low" and disk_space.level(12, 100) == ""
    assert disk_space.level(362, 7392) == ""  # 4.9 % of a big disk is plenty
    assert disk_space.level(40, 7392) == "low" and disk_space.level(None, 100) == ""

    metrics = {
        "places": [
            {"kind": "home", "path": "/var/lib/badamoa/node", "mount": "/", "free_gb": 10.8, "total_gb": 467.9},
            {"kind": "work", "path": "/home4/badamoa-work", "mount": "/home4", "free_gb": 2130.8, "total_gb": 7392.2},
            {"kind": "images", "path": "/home4/badamoa-cache", "mount": "/home4", "free_gb": 2130.8, "total_gb": 7392.2},
            {"kind": "docker", "path": "/var/lib/docker", "mount": "/", "free_gb": 10.8, "total_gb": 467.9},
        ],
        "disks": [_disk("/", 10.8, 467.9), _disk("/home4", 2130.8, 7392.2), _disk("/boot/efi", 0.4, 0.5)],
    }
    (alert,) = disk_space.alerts(metrics)
    assert alert["mount"] == "/" and alert["level"] == "low" and alert["uses"] == ["home", "docker"]
    assert alert["docker"] == "/var/lib/docker" and alert["label"] == "BADAMOA 홈 · Docker 이미지"
    assert disk_space.docker_target(metrics, "/var/lib/docker") == "/home4/docker"
    # Docker Desktop keeps images in its own virtual disk: nothing to say about this machine's disks
    desktop = {"places": [{"kind": "docker", "path": "/var/lib/docker", "desktop": True}]}
    assert disk_space.alerts(desktop) == []
    # a node before 0.74.0: its home (from the storage report) and the images' folder
    older = {"disks": [_disk("C:\\", 1.5, 930), _disk("D:\\", 500, 930)], "cache": {"dir": "D:\\badamoa-cache"}}
    (alert,) = disk_space.alerts(older, {"home": "C:\\ProgramData\\badamoa\\node"})
    assert alert["mount"] == "C:\\" and alert["level"] == "full" and alert["uses"] == ["home"]


def _join(client, admin, version="0.74.0"):
    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    reg = client.post("/api/agent/register", json={"join_token": token, "agent_version": version, "info": {}}).json()
    return reg["node_id"], {"Authorization": f"Bearer {reg['node_token']}"}


def test_the_hub_keeps_an_admins_work_folder_and_shows_low_disks(client, admin):
    node_id, node = _join(client, admin)
    patch = client.patch(f"/api/nodes/{node_id}", json={"work_dir": "data/work"}, headers=admin)
    assert patch.status_code == 400 and "절대 경로" in patch.json()["detail"]
    assert client.patch(f"/api/nodes/{node_id}", json={"work_dir": "/home4/work"}, headers=admin).status_code == 200
    metrics = {
        "disks": [_disk("/", 2.0, 467.9), _disk("/home4", 2125, 7392.2)],
        "work": {"dir": "/home4/work", "auto": False, "free_gb": 2125, "error": ""},
        "places": [
            {"kind": "home", "path": "/var/lib/badamoa/node", "mount": "/", "free_gb": 2.0, "total_gb": 467.9},
            {"kind": "work", "path": "/home4/work", "mount": "/home4", "free_gb": 2125, "total_gb": 7392.2},
            {"kind": "docker", "path": "/var/lib/docker", "mount": "/", "free_gb": 2.0, "total_gb": 467.9},
        ],
    }
    body = {"agent_version": "0.74.0", "metrics": metrics, "running_job_ids": []}
    reply = client.post("/api/agent/heartbeat", json=body, headers=node).json()
    assert reply["work_dir"] == "/home4/work"
    (row,) = client.get("/api/gpu-board", headers=admin).json()["nodes"]
    assert row["work_dir"] == "/home4/work" and row["work"]["dir"] == "/home4/work"
    assert [(a["mount"], a["level"], a["uses"]) for a in row["disk_alerts"]] == [("/", "full", ["home", "docker"])]
    detail = client.get(f"/api/nodes/{node_id}", headers=admin).json()
    assert detail["work_dir"] == "/home4/work" and detail["disk_alerts"][0]["docker"] == "/var/lib/docker"
    assert detail["docker_target"] == "/home4/docker"


def test_a_real_node_moves_into_the_work_folder_an_admin_set(live_hub, tmp_path, monkeypatch):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 0.5)
    monkeypatch.setattr(storage, "uv_cache_dir", lambda *_: None)
    monkeypatch.setattr(storage.containers, "docker", lambda: None)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as hub:
        join = hub.post("/api/nodes/join-tokens", json={}).json()
        home, work = tmp_path / "node", tmp_path / "big-disk" / "badamoa-work"
        home.mkdir()
        (home / "runs" / "job-old" / "out").mkdir(parents=True)
        (home / "runs" / "job-old" / "out" / "last.pt").write_bytes(b"c" * 50)
        env = _ready_env(home, "env-a")
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="gpu-1"), home)
        node_id = runner.config.node_id
        assert hub.patch(f"/api/nodes/{node_id}", json={"work_dir": str(work)}).status_code == 200
        threading.Thread(target=runner.run_forever, daemon=True).start()
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                row = next(n for n in hub.get("/api/gpu-board").json()["nodes"] if n["id"] == node_id)
                moved = (work / "runs" / "job-old" / "out" / "last.pt").is_file()
                if moved and (row.get("work") or {}).get("dir") == str(work) and not row["work"]["moving"]:
                    break
                time.sleep(0.3)
            assert (work / "runs" / "job-old" / "out" / "last.pt").read_bytes() == b"c" * 50
            assert not (home / "runs" / "job-old").exists() and env.is_dir()  # the environment is used where it is
            assert row["work"]["dir"] == str(work) and row["work"]["auto"] is False and not row["work"]["error"]
            assert places.of(home).work == work and [e["id"] for e in envs.listing(home)] == ["env-a"]
        finally:
            runner.stop()
            places.use(home, None)



def test_removing_the_node_takes_the_work_folders_it_made(tmp_path, monkeypatch):
    from badamoa.node import remover

    root, home = tmp_path / "opt" / "badamoa", tmp_path / "var" / "node"
    made, named = tmp_path / "home4" / places.FOLDER, tmp_path / "data" / "mine"
    for folder in (root, home, made, named):
        folder.mkdir(parents=True)
    (home / places.USED).write_text(json.dumps([str(made), str(named)]), encoding="utf-8")
    monkeypatch.setattr(remover.updater, "layout", lambda: {"root": root})
    seen: dict[str, list] = {}

    def capture(_root, _home, work):
        seen["work"] = work
        return ""

    monkeypatch.setattr(remover, "_linux", capture)
    monkeypatch.setattr(remover, "_windows", capture)
    places.use(home, named)  # an admin's folder of another name: what else is in it is not BADAMOA's
    try:
        remover.remove(home)
    finally:
        places.use(home, None)
    assert seen["work"] == [made]
