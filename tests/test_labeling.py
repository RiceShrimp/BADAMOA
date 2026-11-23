"""Labeling sets: images handed to people part by part, labelled into the work in progress, reviewed."""

from __future__ import annotations

import shutil

import httpx
from sample_data import make_yolo
from test_catalog import _import, hub  # noqa: F401  (the fixture)
from test_drafts import _detail


def _login(client, username, password):
    login = {"username": username, "password": password}
    token = httpx.post(f"{client.base_url}/api/auth/login", json=login).json()["token"]
    return httpx.Client(base_url=client.base_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)


def test_a_set_split_into_parts_labelled_and_reviewed(hub, tmp_path):  # noqa: F811
    root = make_yolo(tmp_path / "raw")
    shutil.rmtree(root / "labels")
    (root / "data.yaml").unlink()
    project = hub.post("/api/projects", json={"name": "라벨링"}).json()
    v1 = _import(hub, project["id"], root, "v1", classes=["car", "person"])
    member = hub.post("/api/users", json={"username": "kim", "password": "kim-labels-pw"}).json()
    kim = _login(hub, "kim", "kim-labels-pw")

    # ---- a set of the images nobody labelled yet, three to a part
    body = {"release_id": v1, "name": "첫 라벨링", "instructions": "차와 사람을 박스로", "part_size": 3}
    body["filter"] = {"state": "unlabelled"}
    made = hub.post("/api/label-sets", json=body)
    assert made.status_code == 200, made.text
    one = made.json()
    assert one["items"] == 8 and one["parts"] == 3 and [p["size"] for p in one["part_list"]] == [3, 3, 2]
    assert one["columns"] == {"unassigned": 3, "labeling": 0, "review": 0, "done": 0}
    assert one["open_release_id"] == v1 and {p["name"] for p in one["people"]} >= {"admin", "kim"}
    assert hub.post("/api/label-sets", json={**body, "filter": {"state": "labelled"}}).status_code == 400  # none
    # every image to one person: a part as big as the set, however big a number is asked for (no 10,000 bound)
    whole = hub.post("/api/label-sets", json={**body, "name": "한 사람이 모두", "part_size": 99_999}).json()
    assert (whole["items"], whole["parts"], whole["part_size"]) == (8, 1, 8)
    assert hub.delete(f"/api/label-sets/{whole['id']}").status_code == 200
    assert hub.post("/api/label-sets", json={**body, "part_size": 0}).status_code == 400

    # ---- part 0 to kim (admin reviews); the board moves it to 라벨링
    given = {"assignee": member["id"], "reviewer": one["me"]}
    assert hub.patch(f"/api/label-sets/{one['id']}/parts/0", json=given).status_code == 200
    seen = kim.get(f"/api/label-sets/{one['id']}").json()
    assert seen["mine"] == 1 and seen["part_list"][0]["column"] == "labeling"
    assert seen["part_list"][0]["assignee"]["name"] == "kim" and seen["part_list"][0]["next_label"] == 0

    # ---- the part's images, in the set's order, each with its state
    page = kim.get(f"/api/releases/{v1}/items", params={"label_set": one["id"], "part": 0}).json()
    assert page["total"] == 3 and [i["set"]["state"] for i in page["items"]] == ["todo"] * 3
    first, second, third = (i["id"] for i in page["items"])

    # ---- kim labels two (into the work in progress) and leaves the third as it is
    labelled = kim.put(
        f"/api/releases/{v1}/items/{second}/labels", json={"boxes": [[1, 0.3, 0.3, 0.1, 0.2]], "label_set": one["id"]}
    ).json()
    assert labelled["set"] == {"item_id": second, "state": "labelled"}
    kim.put(f"/api/releases/{v1}/items/{first}/labels", json={"boxes": [[0, 0.5, 0.5, 0.2, 0.2]], "label_set": one["id"]})
    assert kim.post(f"/api/label-sets/{one['id']}/items/{third}/done").json()["state"] == "labelled"
    board = hub.get(f"/api/label-sets/{one['id']}").json()
    part = board["part_list"][0]
    assert part["column"] == "review" and part["states"]["labelled"] == 3 and part["next_review"] == 0
    assert board["states"] == {"todo": 5, "labelled": 3, "approved": 0, "rejected": 0}

    # ---- review: one sent back with a note, two approved; fixed and approved, the part is done
    assert hub.post(f"/api/label-sets/{one['id']}/items/{first}/review", json={"approve": False}).status_code == 400
    sent_back = {"approve": False, "note": "박스가 작음"}
    back = hub.post(f"/api/label-sets/{one['id']}/items/{first}/review", json=sent_back).json()
    assert back == {"item_id": first, "state": "rejected", "note": "박스가 작음"}
    for item in (second, third):
        hub.post(f"/api/label-sets/{one['id']}/items/{item}/review", json={"approve": True})
    assert hub.get(f"/api/label-sets/{one['id']}").json()["part_list"][0]["column"] == "labeling"
    again = kim.get(f"/api/releases/{v1}/items", params={"label_set": one["id"], "part": 0}).json()["items"]
    assert again[0]["set"] == {"state": "rejected", "note": "박스가 작음", "part": 0}
    draft = kim.put(
        f"/api/releases/{v1}/items/{first}/labels", json={"boxes": [[0, 0.5, 0.5, 0.4, 0.4]], "label_set": one["id"]}
    ).json()["release_id"]
    hub.post(f"/api/label-sets/{one['id']}/items/{first}/review", json={"approve": True})
    board = hub.get(f"/api/label-sets/{one['id']}").json()
    assert board["part_list"][0]["column"] == "done" and board["columns"]["done"] == 1
    assert board["open_release_id"] == draft  # labelled into the dataset's work in progress
    assert _detail(hub, draft, first)["revisions"][-1]["boxes"] == [[0, 0.5, 0.5, 0.4, 0.4]]

    # ---- listed, and removed only by its maker or an admin (the labels stay)
    assert [s["name"] for s in hub.get("/api/label-sets", params={"project_id": project["id"]}).json()] == ["첫 라벨링"]
    assert kim.delete(f"/api/label-sets/{one['id']}").status_code == 403
    assert hub.delete(f"/api/label-sets/{one['id']}").status_code == 200
    assert hub.get(f"/api/label-sets/{one['id']}").status_code == 404
    assert _detail(hub, draft, first)["revisions"][-1]["boxes"] == [[0, 0.5, 0.5, 0.4, 0.4]]
    kim.close()


def test_a_set_without_review_is_done_once_labelled(hub, tmp_path):  # noqa: F811
    project = hub.post("/api/projects", json={"name": "검수 없이"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "ds"), "v1")
    one = hub.post(
        "/api/label-sets", json={"release_id": v1, "name": "빠른 확인", "filter": {"split": "val"}, "review": False}
    ).json()
    assert one["items"] == 2 and one["parts"] == 1
    items = hub.get(f"/api/releases/{v1}/items", params={"label_set": one["id"]}).json()["items"]
    for item in items:
        hub.post(f"/api/label-sets/{one['id']}/items/{item['id']}/done")
    board = hub.get(f"/api/label-sets/{one['id']}").json()
    assert board["part_list"][0]["column"] == "done" and board["states"]["approved"] == 2
    assert hub.post(f"/api/label-sets/{one['id']}/items/{items[0]['id']}/review", json={"approve": True}).status_code == 400
    # an older version is not where sets are made
    _import(hub, project["id"], make_yolo(tmp_path / "ds"), "v2")
    old = hub.post("/api/label-sets", json={"release_id": v1, "name": "옛 버전"})
    assert old.status_code == 409 and "최신 버전" in old.json()["detail"]
