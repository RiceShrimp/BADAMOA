"""Node resources: metric history, per-GPU scheduling on shared servers, disk forecast."""

from __future__ import annotations

import sqlite3
import struct
import time
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from conftest import ADMIN

from badamoa.core import sysinfo
from badamoa.hub import resources
from badamoa.hub.db import Database, utcnow
from badamoa.hub.models import NodeMetric

TWO_GPUS = {
    "hostname": "ai-200",
    "os": "linux",
    "gpus": [
        {"index": 0, "name": "RTX 4090", "memory_total_mb": 24000},
        {"index": 1, "name": "RTX 4090", "memory_total_mb": 24000},
    ],
}


def _gpu(index, used_mb, util=0, users=()):
    return {
        "index": index,
        "memory_used_mb": used_mb,
        "memory_total_mb": 24000,
        "utilization_pct": util,
        "temperature_c": 50,
        "power_w": 100,
        "users": list(users),
    }


def _join(client, admin):
    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    reg = client.post(
        "/api/agent/register", json={"join_token": token, "agent_version": "0.3.0", "info": TWO_GPUS}
    ).json()
    return reg["node_id"], {"Authorization": f"Bearer {reg['node_token']}"}


def _beat(client, node, gpus, running=()):
    body = {"agent_version": "0.3.0", "metrics": {"cpu_pct": 10, "gpus": gpus}, "running_job_ids": list(running)}
    assert client.post("/api/agent/heartbeat", json=body, headers=node).status_code == 200


def _claim(client, node):
    return client.post("/api/agent/claim", json={"wait_seconds": 0}, headers=node).json()["job"]


def test_gpu_jobs_avoid_gpus_someone_else_is_using(client, admin):
    node_id, node = _join(client, admin)
    # GPU 0 is in use by a person who logged in; GPU 1 is idle.
    _beat(client, node, [_gpu(0, 18000, util=100, users=["kgu"]), _gpu(1, 300)])

    states = client.get("/api/nodes", headers=admin).json()[0]["gpu_states"]
    assert [(g["index"], g["state"], g["users"]) for g in states] == [(0, "busy", ["kgu"]), (1, "free", [])]

    first = client.post("/api/jobs", json={"type": "system.gpu"}, headers=admin).json()
    second = client.post("/api/jobs", json={"type": "system.gpu"}, headers=admin).json()
    assert first["gpus_requested"] == 1

    claimed = _claim(client, node)
    assert claimed["id"] == first["id"] and claimed["gpu_indices"] == [1]
    assert _claim(client, node) is None  # GPU 1 is now reserved, GPU 0 still belongs to kgu

    waiting = client.get(f"/api/jobs/{second['id']}", headers=admin).json()
    assert waiting["status"] == "queued"
    assert "빈 GPU가 부족합니다" in waiting["wait_reason"] and "kgu" in waiting["wait_reason"]
    states = client.get(f"/api/nodes/{node_id}", headers=admin).json()["gpu_states"]
    assert (states[1]["state"], states[1]["job_id"]) == ("job", first["id"])

    # The first job ends: its GPU is released and the waiting job gets it.
    client.post(f"/api/agent/jobs/{first['id']}/complete", json={"status": "succeeded"}, headers=node)
    assert _claim(client, node)["gpu_indices"] == [1]


def test_job_needing_more_gpus_than_idle_waits_and_cpu_jobs_still_run(client, admin):
    _, node = _join(client, admin)
    _beat(client, node, [_gpu(0, 200), _gpu(1, 20000)])
    big = client.post("/api/jobs", json={"type": "system.gpu", "gpus": 2}, headers=admin).json()
    cpu = client.post("/api/jobs", json={"type": "system.info"}, headers=admin).json()
    assert _claim(client, node)["id"] == cpu["id"]  # the GPU job is skipped, not blocking the queue
    assert "필요 2개" in client.get(f"/api/jobs/{big['id']}", headers=admin).json()["wait_reason"]
    assert client.post("/api/jobs", json={"type": "system.gpu", "gpus": 99}, headers=admin).status_code == 400

    _beat(client, node, [_gpu(0, 200), _gpu(1, 200)], running=[cpu["id"]])
    assert _claim(client, node)["gpu_indices"] == [0, 1]


def test_min_vram_filters_small_gpus(client, admin):
    _, node = _join(client, admin)
    _beat(client, node, [_gpu(0, 200), _gpu(1, 200)])
    job = client.post("/api/jobs", json={"type": "system.gpu", "min_vram_mb": 40000}, headers=admin).json()
    assert _claim(client, node) is None
    assert "빈 GPU 0/2개" in client.get(f"/api/jobs/{job['id']}", headers=admin).json()["wait_reason"]


def test_heartbeats_build_history_that_the_api_serves(client, admin, settings):
    node_id, node = _join(client, admin)
    _beat(client, node, [_gpu(0, 12000, util=40), _gpu(1, 0, util=0)])
    _beat(client, node, [_gpu(0, 12000, util=60), _gpu(1, 0, util=0)])

    data = client.get(f"/api/nodes/{node_id}/metrics?range=1h", headers=admin).json()
    assert data["resolution"] == "1m" and len(data["timestamps"]) >= 1
    series = data["series"]
    assert series["gpu0_mem"]["avg"][-1] == 50.0
    # both heartbeats usually land in one minute bucket; either way the peak is 60
    assert max(v for v in series["gpu0_util"]["max"] if v is not None) == 60.0
    assert sum(v for v in series["gpu_util"]["avg"] if v is not None) > 0
    assert series["gpu_power"]["avg"][-1] == 200.0
    assert client.get(f"/api/nodes/{node_id}/metrics?range=1y", headers=admin).status_code == 400
    assert client.get(f"/api/nodes/{node_id}/metrics", headers={}).status_code == 401


def test_disk_forecast_uses_growth_over_recorded_history(settings):
    db = Database(settings.database_url)
    db.create_all()
    from badamoa.hub.models import Node

    now = utcnow()
    with db.session() as s:
        node = Node(name="n", token_hash="x", info={}, metrics={})
        node.metrics = {"disks": [{"mount": "/home", "used_gb": 1600.0, "free_gb": 200.0}, {"mount": "/", "used_gb": 50.0}]}
        s.add(node)
        s.flush()
        # two days ago /home held 1580 GB → +10 GB/day; / did not grow
        s.add(
            NodeMetric(
                node_id=node.id,
                res="1h",
                ts=now - timedelta(days=2),
                data={"disk_used:/home": [1580.0, 1, 1580.0], "disk_used:/": [100.0, 2, 50.0]},
            )
        )
        s.flush()
        forecast = {f["mount"]: f for f in resources.disk_forecast(s, node, now)}
    assert forecast["/home"]["delta_7d_gb"] == 70.0 and forecast["/home"]["days_until_full"] == 20
    assert forecast["/"]["delta_7d_gb"] == 0.0 and forecast["/"]["days_until_full"] is None


def test_old_history_is_pruned_per_resolution(settings):
    db = Database(settings.database_url)
    db.create_all()
    from badamoa.hub.models import Node

    now = utcnow()
    with db.session() as s:
        node = Node(name="n", token_hash="y", info={}, metrics={})
        s.add(node)
        s.flush()
        for res, age in [("1m", timedelta(hours=30)), ("1m", timedelta(minutes=30)), ("1h", timedelta(days=20))]:
            s.add(NodeMetric(node_id=node.id, res=res, ts=now - age, data={"cpu": [1.0, 1, 1.0]}))
        s.flush()
        assert resources.prune(s, now) == 1
        assert resources.summarize(s, node.id, "1h")["series"]["cpu"] == {"avg": 1.0, "max": 1.0, "last": 1.0}


def test_database_from_an_older_version_gains_new_columns(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE jobs (id VARCHAR(32) PRIMARY KEY, type VARCHAR(64), status VARCHAR(16))")
    con.execute("INSERT INTO jobs VALUES ('j1', 'system.info', 'succeeded')")
    con.commit()
    con.close()

    db = Database(f"sqlite:///{path.as_posix()}")
    db.create_all()
    con = sqlite3.connect(path)
    columns = {row[1] for row in con.execute("PRAGMA table_info(jobs)")}
    assert "gpu_indices" in columns
    assert con.execute("SELECT gpu_indices FROM jobs WHERE id='j1'").fetchone()[0] == "[]"
    con.close()


def test_sampler_reports_this_machine():
    sampler = sysinfo.MetricsSampler()
    sampler.sample()
    time.sleep(0.2)
    snap = sampler.sample()
    assert 0 <= snap["cpu_pct"] <= 100 and snap["memory_total_mb"] > 0 and snap["uptime_s"] > 0
    assert snap["disks"] and all(d["total_gb"] > 0 for d in snap["disks"])
    assert snap["users"] and all({"user", "cpu_cores", "memory_mb", "active"} <= set(u) for u in snap["users"])
    flat = resources.flatten(snap)
    assert "cpu" in flat and "mem" in flat and any(k.startswith("disk_pct:") for k in flat)


def test_windows_disk_busy_is_task_manager_active_time(monkeypatch):
    """Windows disks have no busy_time: 디스크 I/O 사용률 is 100 − % idle time between two idle-counter readings."""
    readings = iter(
        [
            (1_000, 10_000),
            (4_000, 20_000),  # idle 3 000 of 10 000 → 70 % active
            (4_000, 30_000),  # never idle → 100
            (16_000, 40_000),  # more idle than time passed → clamped to 0
            (100, 50_000),  # counter went back (restarted) → nothing
            None,  # the disk did not answer → nothing
            (200, 60_000),  # first reading after a miss → nothing yet
            (1_200, 70_000),
        ]
    )
    asked = []
    monkeypatch.setattr(sysinfo, "_windows_disk_times", lambda name: asked.append(name) or next(readings))
    windows = {"PhysicalDrive0": SimpleNamespace(read_bytes=0, write_bytes=0)}
    monkeypatch.setattr(sysinfo.psutil, "disk_io_counters", lambda perdisk: windows)
    sampler = sysinfo.MetricsSampler()
    busy = [sampler._physical_disks(1.0)[0]["busy_pct"] for _ in range(8)]
    assert busy == [None, 70.0, 100.0, 0.0, None, None, None, 90.0]

    # Linux keeps using busy_time and never asks the Windows counter.
    asked.clear()
    linux = {"sda": SimpleNamespace(read_bytes=0, write_bytes=0, busy_time=0)}
    monkeypatch.setattr(sysinfo.psutil, "disk_io_counters", lambda perdisk: linux)
    sampler = sysinfo.MetricsSampler()
    sampler._physical_disks(None)
    linux["sda"] = SimpleNamespace(read_bytes=0, write_bytes=0, busy_time=250)
    assert sampler._physical_disks(1.0)[0]["busy_pct"] == 25.0 and asked == []


def test_disk_performance_buffer_gives_idle_and_query_time():
    pack = struct.Struct("<5q4Iq").pack
    assert sysinfo._parse_disk_performance(pack(1, 2, 3, 4, 500, 1, 1, 0, 0, 900) + bytes(24)) == (500, 900)
    assert sysinfo._parse_disk_performance(pack(1, 2, 3, 4, 0, 1, 1, 0, 0, 900)) is None  # keeps no idle time
    assert sysinfo._parse_disk_performance(b"\0" * 40) is None  # short answer
    assert sysinfo._windows_disk_times("sda") is None and sysinfo._busy_from_idle(None, (1, 2)) is None


def test_agent_runs_gpu_job_on_reserved_gpu(live_hub, settings, tmp_path, monkeypatch):
    """Real agent, real GPU: the job sees exactly the GPU the Hub reserved."""
    if not sysinfo.gpus():
        pytest.skip("no NVIDIA GPU on this machine")
    import threading

    from badamoa.node import agent as node_agent

    # This machine's GPU may be busy with desktop apps; treat it as idle for the test.
    settings.gpu_busy_memory_pct = settings.gpu_busy_util_pct = 1000.0
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 0.5)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=30) as hub:
        join = hub.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="gpu-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        try:
            job = hub.post("/api/jobs", json={"type": "system.gpu"}).json()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                job = hub.get(f"/api/jobs/{job['id']}").json()
                if job["status"] in {"succeeded", "failed", "canceled"}:
                    break
                time.sleep(0.3)
            assert job["status"] == "succeeded", job
            assert job["gpu_indices"] == [0] and job["result"]["assigned"] == [0]
            logs = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
            assert "CUDA_VISIBLE_DEVICES=0" in logs
            node = hub.get("/api/nodes").json()[0]
            while not node["metrics"].get("gpus") and time.monotonic() < deadline:  # the first heartbeat may be on its way
                time.sleep(0.3)
                node = hub.get("/api/nodes").json()[0]
            assert node["metrics"]["gpus"][0]["power_w"] is not None and node["gpu_states"][0]["state"] == "free"
        finally:
            runner.stop()


def test_a_job_waits_for_the_gpus_it_names(client, admin, settings):
    """A run that picked its GPUs gets those and no others: it waits while one of them is in use."""
    from sqlalchemy import select

    from badamoa.hub import services, transfers
    from badamoa.hub.models import User
    from badamoa.hub.services import ServiceError

    node_id, node = _join(client, admin)
    _beat(client, node, [_gpu(0, 200), _gpu(1, 20000, util=100, users=["kgu"])])
    client.patch(f"/api/nodes/{node_id}", json={"reserved_gpus": [0]}, headers=admin)
    database = Database(settings.database_url)
    with database.session() as db:
        # what the new-run dialog shows of the node: each GPU and its state
        (choice,) = transfers.node_choices(db, settings, "no-such-version")["nodes"]
        assert [(g["index"], g["state"], g["reserved"], g["users"]) for g in choice["gpu_list"]] == [
            (0, "reserved", True, []),
            (1, "busy", False, ["kgu"]),
        ]
        client.patch(f"/api/nodes/{node_id}", json={"reserved_gpus": []}, headers=admin)
        user = db.scalar(select(User))
        job = services.create_job(db, user, "system.gpu", {}, node_id, gpu_indices=[1, 1])
        job_id = job.id
        assert job.requirements["gpus"] == 1 and job.requirements["gpu_indices"] == [1]
        with pytest.raises(ServiceError, match="하나로"):
            services.create_job(db, user, "system.gpu", {}, None, gpu_indices=[0])
    database.engine.dispose()

    assert _claim(client, node) is None  # GPU 0 is free, but the job named GPU 1, which kgu is using
    reason = client.get(f"/api/jobs/{job_id}", headers=admin).json()["wait_reason"]
    assert "고른 GPU" in reason and "GPU 1 사용 중 (kgu)" in reason
    _beat(client, node, [_gpu(0, 200), _gpu(1, 200)])
    assert _claim(client, node)["gpu_indices"] == [1]


def test_a_run_names_gpus_on_one_server_and_never_the_ones_kept_for_people():
    from badamoa.hub import training
    from badamoa.hub.services import ServiceError

    node = SimpleNamespace(name="ai-200", info=TWO_GPUS, reserved_gpus=[1])
    other = SimpleNamespace(name="ai-201", info=TWO_GPUS, reserved_gpus=[])
    assert training.pick_gpus(node, [node], [0, 0]) == [0]
    assert training.pick_gpus(node, [node, other], None) == []  # none named: any free ones, on any of them
    with pytest.raises(ServiceError, match="사람 전용"):
        training.pick_gpus(node, [node], [1])
    with pytest.raises(ServiceError, match="없습니다"):
        training.pick_gpus(node, [node], [5])
    with pytest.raises(ServiceError, match="하나여야"):
        training.pick_gpus(node, [node, other], [0])
