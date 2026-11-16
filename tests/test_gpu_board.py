"""The node screens: every GPU on one board, GPUs kept for people, and who held each GPU over time."""

from __future__ import annotations

from datetime import timedelta

from test_resources import _beat, _claim, _gpu, _join

from badamoa.hub import resources
from badamoa.hub.db import Database, utcnow
from badamoa.hub.models import GpuSpan


def _person(index, used_mb, user, program, util=90):
    gpu = _gpu(index, used_mb, util=util, users=[user])
    gpu["processes"] = [{"pid": 100 + index, "user": user, "name": program, "memory_mb": used_mb}]
    return gpu


def test_board_shows_who_holds_each_gpu_and_why_jobs_wait(client, admin):
    node_id, node = _join(client, admin)
    _beat(client, node, [_person(0, 18000, "kgu", "python"), _gpu(1, 300)])
    job = client.post("/api/jobs", json={"type": "system.gpu"}, headers=admin).json()
    assert _claim(client, node)["gpu_indices"] == [1]
    waiting = client.post("/api/jobs", json={"type": "system.gpu"}, headers=admin).json()

    board = client.get("/api/gpu-board", headers=admin).json()
    (server,) = board["nodes"]
    assert server["id"] == node_id and server["status"] == "online" and server["gpu_name"] == "RTX 4090"
    person, ours = server["gpus"]
    assert person["state"] == "busy" and person["occupant"] == {
        "kind": "person",
        "who": "kgu",
        "programs": "python",
        "light": False,
    }
    assert ours["state"] == "job" and ours["occupant"]["job_id"] == job["id"]
    assert ours["occupant"]["label"] == "GPU 점검" and ours["occupant"]["href"] == "#/jobs"
    assert board["counts"] == {"free": 0, "job": 1, "busy": 1, "reserved": 0, "offline": 0, "waiting": 1}
    assert board["waiting"][0]["job_id"] == waiting["id"] and "kgu" in board["waiting"][0]["reason"]


def test_gpus_kept_for_people_are_never_given_to_jobs(client, admin):
    node_id, node = _join(client, admin)
    _beat(client, node, [_gpu(0, 200), _gpu(1, 200)])
    kept = client.patch(f"/api/nodes/{node_id}", json={"reserved_gpus": [1, 7]}, headers=admin)
    assert kept.status_code == 200 and kept.json()["reserved_gpus"] == [1]  # an index the server lacks is dropped

    states = client.get(f"/api/nodes/{node_id}", headers=admin).json()["gpu_states"]
    assert [(s["state"], s["reserved"]) for s in states] == [("free", False), ("reserved", True)]
    first = client.post("/api/jobs", json={"type": "system.gpu"}, headers=admin).json()
    second = client.post("/api/jobs", json={"type": "system.gpu"}, headers=admin).json()
    assert _claim(client, node)["gpu_indices"] == [0] and _claim(client, node) is None
    reason = client.get(f"/api/jobs/{second['id']}", headers=admin).json()["wait_reason"]
    assert "사람 전용 1개" in reason
    assert client.get("/api/gpu-board", headers=admin).json()["counts"]["reserved"] == 1

    # someone uses the kept GPU: it shows as theirs; released, it is kept again
    _beat(client, node, [_gpu(0, 200), _person(1, 20000, "lee", "jupyter")], running=[first["id"]])
    gpu = client.get("/api/gpu-board", headers=admin).json()["nodes"][0]["gpus"][1]
    assert gpu["state"] == "busy" and gpu["reserved"] and gpu["occupant"]["who"] == "lee"
    assert client.patch(f"/api/nodes/{node_id}", json={"reserved_gpus": []}, headers=admin).json()["reserved_gpus"] == []


def test_timeline_has_jobs_and_people_on_each_gpu(client, admin, settings):
    node_id, node = _join(client, admin)
    _beat(client, node, [_person(0, 9000, "kgu", "python"), _gpu(1, 300)])
    _beat(client, node, [_person(0, 12000, "kgu", "train.py"), _gpu(1, 300)])  # the same person: one span
    job = client.post("/api/jobs", json={"type": "system.gpu"}, headers=admin).json()
    assert _claim(client, node)["gpu_indices"] == [1]

    database = Database(settings.database_url)
    with database.session() as db:
        spans = db.query(GpuSpan).all()
        assert [(s.gpu_index, s.who, s.programs, s.memory_mb) for s in spans] == [(0, "kgu", "python, train.py", 12000)]
        # a span from long ago stretches to now no more; a later use by someone else is a new span
        spans[0].ended_at = spans[0].started_at = utcnow() - timedelta(hours=2)
    _beat(client, node, [_person(0, 5000, "lee", "jupyter"), _gpu(1, 300)], running=[job["id"]])

    line = client.get(f"/api/nodes/{node_id}/timeline", params={"range": "24h"}, headers=admin).json()
    assert line["range"] == "24h" and len(line["gpus"]) == 2 and len(line["timestamps"]) >= 1
    first, second = line["gpus"]
    assert [(s["kind"], s.get("who")) for s in first["spans"]] == [("person", "kgu"), ("person", "lee")]
    assert [(s["kind"], s["job_id"], s["label"]) for s in second["spans"]] == [("job", job["id"], "GPU 점검")]
    assert first["util"] and first["util"][-1] is not None
    assert client.get(f"/api/nodes/{node_id}/timeline", params={"range": "1y"}, headers=admin).status_code == 400

    # kept as long as the coarsest history
    with database.session() as db:
        db.query(GpuSpan).update({GpuSpan.ended_at: utcnow() - timedelta(days=40)})
    with database.session() as db:
        resources.prune(db)
        assert db.query(GpuSpan).count() == 0
    database.engine.dispose()
