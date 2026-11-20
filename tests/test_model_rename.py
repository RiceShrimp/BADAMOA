"""Renaming a model (every version of it) and renumbering a version: the copies of the name follow."""

from __future__ import annotations

from sqlalchemy import select, update
from test_labeling import _login
from test_training import _logs, _train, _wait, hub, release  # noqa: F401  (the fixtures)

from badamoa.hub.catalog_models import LabelRevision
from badamoa.hub.db import Database


def test_a_model_is_renamed_and_its_copies_follow(hub, release):  # noqa: F811
    first, _ = _train(hub, release, {"epochs": 1})
    second, _ = _train(hub, release, {"epochs": 1})
    v1 = hub.get(f"/api/runs/{first['id']}").json()["model"]
    v2 = hub.get(f"/api/runs/{second['id']}").json()["model"]
    assert v1["name"] == v2["name"] and (v1["version"], v2["version"]) == (1, 2)
    old = v1["name"]
    # a run continued from v1 keeps a copy of its name; so does a label 오토라벨 wrote with it
    body = {"release_id": release["id"], "template": "dummy", "params": {"epochs": 1}, "gpus": 0, "base_model_id": v1["id"]}
    continued = hub.post("/api/runs", json=body).json()
    assert _wait(hub, continued["job_id"])["status"] == "succeeded", _logs(hub, continued["job_id"])
    database = Database(f"sqlite:///{(hub.hub_home / 'hub.db').as_posix()}")
    with database.session() as db:
        some = db.scalars(select(LabelRevision.id).limit(1)).first()
        db.execute(update(LabelRevision).where(LabelRevision.id == some).values(source=f"model:{old} v1"))

    # ---- the family takes the new name, every version of it
    renamed = hub.patch(f"/api/models/{v1['id']}", json={"name": "  도로 검출!  "}).json()
    assert (renamed["name"], renamed["version"]) == ("도로-검출", 1)
    models = hub.get(f"/api/projects/{release['project_id']}/models").json()
    assert sorted((m["name"], m["version"]) for m in models) == [("도로-검출", 1), ("도로-검출", 2), ("도로-검출", 3)]
    assert hub.get(f"/api/runs/{continued['id']}").json()["base_model"]["name"] == "도로-검출"
    with database.session() as db:
        assert db.get(LabelRevision, some).source == "model:도로-검출 v1"

    # ---- a version takes another number; one in use is refused
    assert hub.patch(f"/api/models/{v2['id']}", json={"version": 10}).json()["version"] == 10
    taken = hub.patch(f"/api/models/{v1['id']}", json={"version": 10})
    assert taken.status_code == 409 and "이미 있습니다" in taken.json()["detail"]
    assert hub.patch(f"/api/models/{v1['id']}", json={"version": 0}).status_code == 400
    with database.session() as db:
        db.execute(update(LabelRevision).where(LabelRevision.id == some).values(source="model:도로-검출 v10"))
    hub.patch(f"/api/models/{v2['id']}", json={"version": 2})
    with database.session() as db:
        assert db.get(LabelRevision, some).source == "model:도로-검출 v2"

    # ---- a name another model has is refused; a rerun registers under the new name
    other = {"release_id": release["id"], "template": "dummy", "params": {"epochs": 1}, "gpus": 0, "model_name": "다른"}
    third = hub.post("/api/runs", json=other).json()
    assert _wait(hub, third["job_id"])["status"] == "succeeded"
    assert hub.patch(f"/api/models/{v1['id']}", json={"name": "다른"}).status_code == 409
    again = hub.post(f"/api/runs/{first['id']}/rerun", json={})
    if again.status_code == 200:  # a rerun of a run of the family: its next version under the new name
        assert _wait(hub, again.json()["job_id"])["status"] == "succeeded"
        assert hub.get(f"/api/runs/{again.json()['id']}").json()["model"]["name"] == "도로-검출"

    # ---- only the project's maker or an admin
    hub.post("/api/users", json={"username": "kim", "password": "kim-model-pw"})
    kim = _login(hub, "kim", "kim-model-pw")
    assert kim.patch(f"/api/models/{v1['id']}", json={"name": "김"}).status_code == 403
    assert kim.patch(f"/api/models/{v1['id']}", json={"note": "메모는 누구나"}).status_code == 200
