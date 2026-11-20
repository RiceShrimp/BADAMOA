"""Models of different projects compared on the same data; an evaluation deleted with what it kept."""

from __future__ import annotations

from sample_data import make_yolo
from test_labeling import _login
from test_training import _logs, _train, _wait, hub, release  # noqa: F401  (the fixtures)


def test_models_of_two_projects_compared_and_an_evaluation_deleted(hub, release, tmp_path):  # noqa: F811
    run, _ = _train(hub, release, {"epochs": 1})
    ours = hub.get(f"/api/runs/{run['id']}").json()["model"]
    other = hub.post("/api/projects", json={"name": "다른 현장"}).json()
    body = {"node_id": hub.node_id, "path": str(make_yolo(tmp_path / "other")), "dataset": "road", "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{other['id']}/imports", json=body).json()["id"])
    theirs_release = job["result"]["release_id"]
    their_run, _ = _train(hub, {"id": theirs_release, "project_id": other["id"]}, {"epochs": 1})
    theirs = hub.get(f"/api/runs/{their_run['id']}").json()["model"]

    # ---- a model evaluated on another project's version (same task and classes)
    on_theirs = {"release_id": theirs_release, "split": "val", "gpus": 0}
    op = hub.post(f"/api/models/{ours['id']}/eval", json=on_theirs).json()
    assert _wait(hub, op["job_id"])["status"] == "succeeded", _logs(hub, op["job_id"])
    mine = hub.post(f"/api/models/{theirs['id']}/eval", json=on_theirs).json()
    assert _wait(hub, mine["job_id"])["status"] == "succeeded"

    # ---- found in every project, compared side by side
    found = hub.get("/api/models-find", params={"q": "다른 현장"}).json()
    assert [m["id"] for m in found] == [theirs["id"]] and found[0]["project"]["name"] == "다른 현장"
    both = hub.get("/api/models-compare", params={"ids": f"{ours['id']},{theirs['id']}"}).json()
    assert [m["project"]["name"] for m in both["models"]] == ["주차장", "다른 현장"]
    [shared] = both["evaluations"]
    assert shared["release"]["project"]["name"] == "다른 현장" and shared["split"] == "val"
    assert set(shared["rows"]) == {ours["id"], theirs["id"]}
    board = hub.get(f"/api/releases/{theirs_release}/evaluations").json()["rows"]
    assert {r["model"]["id"] for r in board} == {ours["id"], theirs["id"]}

    # ---- deleted: by whoever ran it or an admin, with its predictions
    hub.post("/api/users", json={"username": "kim", "password": "kim-compare-pw"})
    kim = _login(hub, "kim", "kim-compare-pw")
    assert kim.delete(f"/api/model-ops/{op['id']}").status_code == 403
    assert hub.delete(f"/api/model-ops/{op['id']}").json() == {"ok": True}
    assert hub.get(f"/api/model-ops/{op['id']}").status_code == 404
    again = hub.get("/api/models-compare", params={"ids": f"{ours['id']},{theirs['id']}"}).json()
    assert list(again["evaluations"][0]["rows"]) == [theirs["id"]]
    assert all(p["op_id"] != op["id"] for p in hub.get(f"/api/releases/{theirs_release}/prediction-runs").json())
