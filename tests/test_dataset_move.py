"""Moving a dataset to another project: its versions, work in progress, labels and labeling sets go with it; images
another dataset shares, or the other project has already, are copied or taken as they are there."""

from __future__ import annotations

import sqlite3

from test_labeling import _login
from test_training import _train, hub, release  # noqa: F401  (the fixtures)


def _items(client, release_id):
    """file name → (image id, split, state, boxes) of a version."""
    page = client.get(f"/api/releases/{release_id}/items", params={"limit": 200}).json()
    return {i["file_name"]: (i["id"], i["split"], i["state"], i["boxes"]) for i in page["items"]}


def _project_items(client, project_id):
    return next(p["items"] for p in client.get("/api/projects").json() if p["id"] == project_id)


def _tree(client, project_id):
    tree = client.get(f"/api/projects/{project_id}/datasets").json()
    return [(d["name"], [r["version"] for r in d["releases"]]) for d in tree]


def test_a_dataset_moves_with_its_versions_labels_and_sets(hub, release):  # noqa: F811
    source = release["project_id"]
    before = hub.get(f"/api/releases/{release['id']}").json()
    dataset_id = before["dataset"]["id"]
    images = _items(hub, release["id"])
    first = images["train_0000.jpg"][0]

    # a fixed label (the work in progress), a labeling set and a training run on the version
    fixed = hub.put(f"/api/releases/{release['id']}/items/{first}/labels", json={"boxes": [[1, 0.4, 0.4, 0.2, 0.2]]})
    assert fixed.status_code == 200, fixed.text
    draft = hub.get(f"/api/releases/{release['id']}").json()["draft"]["id"]
    made = hub.post("/api/label-sets", json={"release_id": release["id"], "name": "확인", "part_size": 4})
    assert made.status_code == 200, made.text
    run, job = _train(hub, release, {"ceiling": 0.5})
    assert job["status"] == "succeeded"
    model = hub.get(f"/api/runs/{run['id']}").json()["model"]

    target = hub.post("/api/projects", json={"name": "창고"}).json()
    plan = hub.get(f"/api/datasets/{dataset_id}/move", params={"project_id": target["id"]}).json()
    assert (plan["versions"], plan["draft"], plan["items"], plan["shared"], plan["label_sets"]) == (1, True, 8, 0, 1)
    assert plan["stays"]["runs"] == 1 and plan["stays"]["models"] == 1 and plan["busy"] == []
    assert plan["target"] == {"id": target["id"], "name": "창고", "name_taken": False, "known": 0}

    # not to where it is, not by someone who made neither project
    assert hub.post(f"/api/datasets/{dataset_id}/move", json={"project_id": source}).status_code == 400
    hub.post("/api/users", json={"username": "kim", "password": "kim-move-pw"})
    kim = _login(hub, "kim", "kim-move-pw")
    assert kim.post(f"/api/datasets/{dataset_id}/move", json={"project_id": target["id"]}).status_code == 403

    moved = hub.post(f"/api/datasets/{dataset_id}/move", json={"project_id": target["id"]})
    assert moved.status_code == 200, moved.text
    assert moved.json()["project"]["id"] == target["id"] and moved.json()["copied"] == 0
    assert _tree(hub, source) == [] and _tree(hub, target["id"]) == [("street", ["작업 중", "v1"])]
    after = hub.get(f"/api/releases/{release['id']}").json()
    assert after["project_id"] == target["id"] and after["label_space"] == before["label_space"]
    assert after["digest"] == before["digest"]  # the same content, the same fingerprint
    # only this dataset held its images: they moved as they were, ids and labels
    assert _items(hub, release["id"]) == images
    assert _items(hub, draft)["train_0000.jpg"][3] == [[1, 0.4, 0.4, 0.2, 0.2]]
    assert (_project_items(hub, source), _project_items(hub, target["id"])) == (0, 8)
    [labeling] = hub.get("/api/label-sets", params={"project_id": target["id"]}).json()
    assert labeling["name"] == "확인" and hub.get("/api/label-sets", params={"project_id": source}).json() == []

    # the run and its model stay where they were made and still point at the version
    assert hub.get(f"/api/runs/{run['id']}").json()["release"]["id"] == release["id"]
    assert hub.get(f"/api/models/{model['id']}").json()["project_id"] == source
    log = sqlite3.connect(hub.hub_home / "hub.db")
    [(target_id,)] = log.execute("SELECT target FROM audit_logs WHERE action = 'dataset.move'").fetchall()
    log.close()
    assert target_id == dataset_id

    # a new training of the moved version is the new project's
    run2, job2 = _train(hub, release, {"ceiling": 0.5})
    assert job2["status"] == "succeeded" and hub.get(f"/api/runs/{run2['id']}").json()["project_id"] == target["id"]


def test_shared_images_are_copied_and_known_ones_taken_as_they_are(hub, release):  # noqa: F811
    source = release["project_id"]
    images = _items(hub, release["id"])
    # a second dataset of the same images and labels: the project holds each image once, for both
    other = hub.post("/api/releases", json={"sources": [release["id"]], "version": "v1", "dataset": "다른"})
    assert other.status_code == 200, other.text
    other_id = other.json()["release_id"]
    digest = hub.get(f"/api/releases/{other_id}").json()["digest"]
    other_dataset = hub.get(f"/api/releases/{other_id}").json()["dataset"]["id"]
    street = hub.get(f"/api/releases/{release['id']}").json()["dataset"]["id"]
    target = hub.post("/api/projects", json={"name": "창고"}).json()

    plan = hub.get(f"/api/datasets/{other_dataset}/move", params={"project_id": target["id"]}).json()
    assert (plan["items"], plan["shared"], plan["target"]["known"]) == (8, 8, 0)
    moved = hub.post(f"/api/datasets/{other_dataset}/move", json={"project_id": target["id"]})
    assert moved.status_code == 200, moved.text
    assert moved.json()["copied"] == 8
    # the moved one holds copies with the same labels; the one left keeps its images as they were
    copied = _items(hub, other_id)
    assert {k: v[1:] for k, v in copied.items()} == {k: v[1:] for k, v in images.items()}
    assert not {v[0] for v in copied.values()} & {v[0] for v in images.values()}
    assert _items(hub, release["id"]) == images
    assert (_project_items(hub, source), _project_items(hub, target["id"])) == (8, 8)
    detail = hub.get(f"/api/releases/{other_id}/items/{copied['train_0000.jpg'][0]}").json()
    assert [r["dataset"] for r in detail["releases"]] == ["다른"]  # its history there is its own
    assert [r["rev"] for r in detail["revisions"]] == [1]
    assert hub.get(f"/api/releases/{other_id}").json()["digest"] == digest  # a version keeps its fingerprint
    assert hub.get(f"/api/releases/{other_id}").json()["label_space"]["name"]

    # the other project has the images and a "다른" now: the second goes onto those images, not by a name taken
    plan = hub.get(f"/api/datasets/{street}/move", params={"project_id": target["id"]}).json()
    assert plan["target"]["name_taken"] is False and plan["target"]["known"] == 8 and plan["shared"] == 0
    taken = hub.post(f"/api/datasets/{street}/move", json={"project_id": target["id"], "name": "다른"})
    assert taken.status_code == 409 and "이미 있습니다" in taken.json()["detail"]
    moved = hub.post(f"/api/datasets/{street}/move", json={"project_id": target["id"], "name": "길"})
    assert moved.status_code == 200, moved.text
    there = _items(hub, release["id"])
    assert {k: v[0] for k, v in there.items()} == {k: v[0] for k, v in copied.items()}  # the images it had
    assert {k: v[1:] for k, v in there.items()} == {k: v[1:] for k, v in images.items()}
    assert (_project_items(hub, source), _project_items(hub, target["id"])) == (0, 8)  # nothing left behind
    assert _tree(hub, target["id"]) == [("길", ["v1"]), ("다른", ["v1"])] and _tree(hub, source) == []
    # one label space for both there (the same name and classes), none left behind; the same content is seen as such
    street_view, other_view = (hub.get(f"/api/releases/{r}").json() for r in (release["id"], other_id))
    assert street_view["label_space"] == other_view["label_space"]
    assert [r["id"] for r in street_view["same_as"]] == [other_id]
    database = sqlite3.connect(hub.hub_home / "hub.db")
    left = database.execute("SELECT count(*) FROM label_spaces WHERE project_id = ?", (source,)).fetchone()[0]
    database.close()
    assert left == 0
    assert hub.get(f"/api/releases/{release['id']}/qa").status_code == 200
    # both versions name their label space; training on either still works
    run, job = _train(hub, {"id": other_id}, {"ceiling": 0.5})
    assert job["status"] == "succeeded", job


def test_nothing_moves_while_something_works_on_it(hub, release):  # noqa: F811
    dataset_id = hub.get(f"/api/releases/{release['id']}").json()["dataset"]["id"]
    target = hub.post("/api/projects", json={"name": "창고"}).json()
    database = sqlite3.connect(hub.hub_home / "hub.db")
    database.execute(
        "INSERT INTO sweeps (id, project_id, name, release_id, method, metric, goal, space, settings, max_runs, parallel,"
        " status, error, created_at) VALUES ('s1', ?, 'search', ?, 'grid', 'map50', 'maximize', '{}', '{}', 2, 1,"
        " 'running', '', '2026-11-24 00:00:00')",
        (release["project_id"], release["id"]),
    )
    database.commit()
    plan = hub.get(f"/api/datasets/{dataset_id}/move").json()
    assert plan["busy"] == ["탐색 1개"]
    refused = hub.post(f"/api/datasets/{dataset_id}/move", json={"project_id": target["id"]})
    assert refused.status_code == 409 and "탐색 1개" in refused.json()["detail"]
    database.execute("UPDATE sweeps SET status = 'stopped' WHERE id = 's1'")
    database.commit()
    database.close()
    assert hub.post(f"/api/datasets/{dataset_id}/move", json={"project_id": target["id"]}).status_code == 200
