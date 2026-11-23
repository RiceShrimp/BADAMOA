"""The first screens of 라벨링, 검수, 학습 and 모델: every project at a glance."""

from __future__ import annotations

import sqlite3
import time

from sample_data import make_imagefolder, make_yolo
from test_labeling import _login
from test_training import _logs, _train, _wait, hub, release  # noqa: F401  (the fixtures)


def _import(hub, project_id, root, dataset="street", **extra):  # noqa: F811
    body = {"node_id": hub.node_id, "path": str(root), "dataset": dataset, "version": "v1", **extra}
    job = _wait(hub, hub.post(f"/api/projects/{project_id}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    return job["result"]["release_id"]


def test_labeling_and_review_boards(hub, tmp_path):  # noqa: F811
    project = hub.post("/api/projects", json={"name": "보드", "folder": "현장"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "ds"), unlabelled="unlabelled")
    other = hub.post("/api/projects", json={"name": "묶음 없음"}).json()
    _import(hub, other["id"], make_yolo(tmp_path / "other", seed=3))
    kim_user = hub.post("/api/users", json={"username": "kim", "password": "kim-boards-pw"}).json()
    kim = _login(hub, "kim", "kim-boards-pw")

    # ---- a set of every image, four to a part; kim labels part 0 and admin reviews it
    made = hub.post("/api/label-sets", json={"release_id": v1, "name": "전체 보기", "part_size": 4}).json()
    me = hub.get("/api/auth/me").json()
    given = {"assignee": kim_user["id"], "reviewer": me["id"]}
    assert hub.patch(f"/api/label-sets/{made['id']}/parts/0", json=given).status_code == 200

    board = kim.get("/api/boards/labeling").json()
    assert board["me"]["name"] == "kim" and len(board["mine"]) == 1
    mine = board["mine"][0]
    assert mine["set"]["id"] == made["id"] and mine["project"]["name"] == "보드" and mine["roles"] == ["label"]
    assert (mine["part"], mine["size"], mine["to_label"], mine["to_review"], mine["next_label"]) == (0, 4, 4, 0, 0)
    [group] = board["projects"]
    assert group["project"]["name"] == "보드" and group["project"]["folder"] == "현장"
    [card] = group["sets"]
    assert card["items"] == 8 and card["parts"] == 2 and card["columns"]["labeling"] == 1 and card["mine"] == 1
    assert len(card["collage"]) == 4 and card["open_release_id"] == v1
    assert [p["name"] for p in board["without"]] == ["묶음 없음"]
    assert board["totals"] == {"sets": 1, "items": 8, "todo": 8, "review": 0, "done": 0}
    waiting = hub.get("/api/boards/labeling").json()["mine"]
    assert [m["roles"] for m in waiting] == [["review"]] and waiting[0]["to_review"] == 0  # nothing labelled yet

    # ---- kim labels one image of the part (two boxes on one object: a check flags it)
    page = kim.get(f"/api/releases/{v1}/items", params={"label_set": made["id"], "part": 0}).json()["items"]
    item = page[1]
    twice = [[0, 0.5, 0.5, 0.3, 0.3], [0, 0.5, 0.5, 0.3, 0.31]]
    saved = kim.put(f"/api/releases/{v1}/items/{item['id']}/labels", json={"boxes": twice, "label_set": made["id"]})
    assert saved.status_code == 200, saved.text
    draft = saved.json()["release_id"]
    assert kim.get("/api/boards/labeling").json()["mine"][0]["to_label"] == 3
    review = hub.get("/api/boards/labeling").json()["mine"][0]
    assert review["to_review"] == 1 and review["next_review"] == 1

    # ---- 검수: each dataset's newest labels (the work in progress here), the most problems first
    checked = hub.get("/api/boards/review").json()
    assert [d["project"]["name"] for d in checked["datasets"]] == ["보드", "묶음 없음"]
    first = checked["datasets"][0]
    assert first["release"] == {"id": draft, "version": "작업 중", "draft": True, "base": "v1", "edited": 1}
    assert first["checked"] and first["items"] == 8 and first["evaluation"] is None
    assert first["groups"]["missing"] == 1  # the image with no label file
    assert first["groups"]["boxes"] >= 1 and first["groups"]["any"] >= 2
    assert {"unlabelled", "duplicate"} <= {c["id"] for c in first["checks"]}
    assert first["flagged_shots"] and 1 <= len(first["collage"]) <= 4
    assert checked["totals"]["datasets"] == 2 and checked["totals"]["checked"] == 2
    assert checked["datasets"][1]["release"]["draft"] is False and checked["datasets"][1]["groups"]["missing"] == 0


def test_training_and_model_boards(hub, release):  # noqa: F811
    empty = hub.post("/api/projects", json={"name": "빈 프로젝트"}).json()
    done, job = _train(hub, release, {"epochs": 2})
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    failed, job = _train(hub, release, {"fail": True})
    assert job["status"] == "failed"
    model = hub.get(f"/api/runs/{done['id']}").json()["model"]
    assert hub.post(f"/api/models/{model['id']}/alias", json={"alias": "SOTA"}).status_code == 200

    # ---- a run going on: how far, about how long is left, where
    body = {"release_id": release["id"], "template": "dummy", "params": {"epochs": 8, "sleep": 0.4}, "gpus": 0}
    slow = hub.post("/api/runs", json=body).json()
    seen = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        going = hub.get("/api/boards/training").json()["going"]
        now = next((r for r in going if r["id"] == slow["id"]), None)
        if now and now["status"] == "running" and now["epoch"] >= 3:
            seen = now
            break
        time.sleep(0.2)
    assert seen is not None, "the run was never seen going"
    assert seen["project"]["id"] == release["project_id"] and seen["node_name"] == "train-node" and seen["epochs"] == 8
    assert seen["metric"] == "map50_95" and isinstance(seen["eta_s"], int) and seen["eta_s"] >= 0
    assert _wait(hub, slow["job_id"])["status"] == "succeeded"

    board = hub.get("/api/boards/training").json()
    assert board["going"] == []
    assert [c["project"]["name"] for c in board["projects"]] == ["주차장", "빈 프로젝트"]  # no runs last
    card = board["projects"][0]
    assert (card["runs"], card["succeeded"], card["failed"], card["running"]) == (3, 2, 1, 0)
    assert card["metric"] == "map50_95" and len(card["trend"]) == 2 and card["best"]["value"] == max(card["trend"])
    assert [r["id"] for r in card["recent"]] == [slow["id"], failed["id"], done["id"]]
    assert card["recent"][1]["status"] == "failed" and card["recent"][1]["value"] is None
    assert board["projects"][1]["runs"] == 0 and board["projects"][1]["last_at"] is None
    assert board["totals"]["runs"] == 3 and board["totals"]["failed"] == 1

    models = hub.get("/api/boards/models").json()
    card = models["projects"][0]
    assert card["project"]["name"] == "주차장" and card["models"] == 2 and card["families"] == 1
    assert [(m["id"], m["aliases"]) for m in card["aliased"]] == [(model["id"], ["SOTA"])]
    assert card["aliased"][0]["value"] is not None and card["best"]["value"] >= card["aliased"][0]["value"]
    assert card["latest"]["version"] == 2
    assert models["projects"][1]["project"]["id"] == empty["id"] and models["projects"][1]["latest"] is None
    assert models["totals"] == {"models": 2, "families": 1, "aliased": 1, "production": 0, "projects": 1}


def test_review_checks_with_models_that_fit(hub, release, tmp_path):  # noqa: F811
    done, job = _train(hub, release, {"epochs": 2})
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    model = hub.get(f"/api/runs/{done['id']}").json()["model"]
    pets = _import(hub, release["project_id"], make_imagefolder(tmp_path / "pets"), dataset="pets")

    # ---- which of the project's models can check a version: the dataset's task, the version's classes
    choices = hub.get(f"/api/releases/{release['id']}/eval-models").json()
    assert choices["task"] == "detect" and choices["evaluate"]["id"] == release["id"]
    assert choices["splits"] == {"train": 6, "val": 2}
    [fit] = choices["models"]
    assert (fit["id"], fit["fits"], fit["why"], fit["task"], fit["trained_on"]) == (model["id"], True, "", "detect", True)
    other = hub.get(f"/api/releases/{pets}/eval-models").json()
    assert other["task"] == "classify" and other["models"][0]["fits"] is False
    assert other["models"][0]["why"] == "Det 모델이라 Cls 데이터셋에는 쓸 수 없습니다"

    # ---- a work in progress is checked as its base version
    item = hub.get(f"/api/releases/{release['id']}/items", params={"limit": 1}).json()["items"][0]
    draft = hub.put(f"/api/releases/{release['id']}/items/{item['id']}/labels", json={"boxes": []}).json()["release_id"]
    assert hub.get(f"/api/releases/{draft}/eval-models").json()["evaluate"]["id"] == release["id"]

    # ---- checked with the model: the 검수 screen names it, with its task and aliases
    hub.post(f"/api/models/{model['id']}/alias", json={"alias": "SOTA"})
    op = hub.post(f"/api/models/{model['id']}/eval", json={"release_id": release["id"], "split": "val", "gpus": 0}).json()
    pending = hub.get(f"/api/releases/{draft}/qa").json()["pending"]
    assert pending == [] or (pending[0]["id"] == op["id"] and pending[0]["model"]["id"] == model["id"])
    assert _wait(hub, op["job_id"])["status"] == "succeeded", _logs(hub, op["job_id"])
    summary = hub.get(f"/api/releases/{draft}/qa").json()
    assert summary["pending"] == []
    [used] = summary["evaluations"]
    assert used["id"] == op["id"] and used["model"]["task"] == "detect" and used["model"]["aliases"] == ["SOTA"]

    # ---- an evaluation by a model of another task is no check of these labels (one made before evaluations were refused)
    database = sqlite3.connect(hub.hub_home / "hub.db")
    database.execute("UPDATE model_versions SET task = 'segment' WHERE id = ?", (model["id"],))
    database.commit()
    database.close()
    assert hub.get(f"/api/releases/{release['id']}/checks").json() == []
    assert hub.get(f"/api/releases/{draft}/qa").json()["evaluations"] == []
    assert hub.get(f"/api/releases/{release['id']}/eval-models").json()["models"][0]["fits"] is False
