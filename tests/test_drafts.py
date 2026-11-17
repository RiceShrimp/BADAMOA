"""Fixing labels in place: the dataset's work in progress ("작업 중"), confirming it as a version,
and images that nobody has labelled yet."""

from __future__ import annotations

import shutil

from conftest import ADMIN
from sample_data import CLASSES, make_yolo
from test_catalog import _import, hub  # noqa: F401  (the fixture)

from badamoa.node import importer


def _items(client, release_id, query=""):
    return client.get(f"/api/releases/{release_id}/items?limit=100&{query}").json()


def _detail(client, release_id, item_id):
    return client.get(f"/api/releases/{release_id}/items/{item_id}").json()


def _save(client, release_id, item_id, boxes, parent=None, force=False):
    body = {"boxes": boxes, "parent_revision_id": parent, "force": force}
    return client.put(f"/api/releases/{release_id}/items/{item_id}/labels", json=body)


def _saved_revision(client, item_id, saved):
    revisions = _detail(client, saved["release_id"], item_id)["revisions"]
    return next(r for r in revisions if r["id"] == saved["revision_id"])


def test_fix_labels_into_the_work_in_progress_and_confirm_it(hub, tmp_path):  # noqa: F811
    project = hub.post("/api/projects", json={"name": "고치기"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "ds"), "v1")
    items = _items(hub, v1)["items"]
    labelled = next(i for i in items if i["state"] == "labelled")
    empty = next(i for i in items if i["state"] == "background")

    # ---- the newest version can be fixed; nothing is in progress yet
    detail = _detail(hub, v1, labelled["id"])
    assert detail["edit"]["allowed"] and detail["edit"]["draft_id"] is None
    start = detail["edit"]["revision_id"]
    assert start == detail["revision_id"]
    boxes = next(r for r in detail["revisions"] if r["id"] == start)["boxes"]

    # ---- the first fix makes the work in progress; the version itself stays as it was
    fixed = [[1 - int(boxes[0][0]) if len(CLASSES) > 1 else 0, 0.5, 0.5, 0.2, 0.2], *boxes[1:]]
    resp = _save(hub, v1, labelled["id"], fixed, parent=start)
    assert resp.status_code == 200, resp.text
    draft = resp.json()["release_id"]
    assert draft != v1 and resp.json()["edited"] == 1 and resp.json()["state"] == "labelled"
    assert _detail(hub, v1, labelled["id"])["revision_id"] == start  # v1 untouched
    revision = next(r for r in _detail(hub, draft, labelled["id"])["revisions"] if r["id"] == resp.json()["revision_id"])
    assert revision["source"] == "human:admin" and revision["boxes"][0][1:5] == [0.5, 0.5, 0.2, 0.2]
    assert len(revision["labelled_classes"]) == len(CLASSES)  # a person who saves has seen the whole image

    tree = hub.get(f"/api/projects/{project['id']}/datasets").json()
    assert [(r["version"], r["status"], r.get("edited")) for r in tree[0]["releases"]] == [
        ("작업 중", "draft", 1),
        ("v1", "ready", None),
    ]
    view = hub.get(f"/api/releases/{v1}").json()
    assert view["draft"]["id"] == draft and view["draft"]["base"]["version"] == "v1"
    assert view["draft"]["edited"] == 1 and view["draft"]["editors"] == [ADMIN["username"]]
    card = hub.get(f"/api/projects/{project['id']}/overview").json()["dataset_cards"][0]
    assert card["draft"] == {"id": draft, "edited": 1} and card["releases"] == 1 and card["release"]["id"] == v1

    # ---- a background image gets a box; a stale save is turned away unless forced
    assert _save(hub, v1, empty["id"], [[0, 0.3, 0.3, 0.1, 0.1]]).json()["edited"] == 2
    stale = _save(hub, draft, labelled["id"], boxes, parent=start)
    assert stale.status_code == 412, stale.text
    assert _save(hub, draft, labelled["id"], fixed, parent=start, force=True).status_code == 200

    # ---- "fixed only" in the work in progress and in its base; the base still shows its own labels
    edited = _items(hub, draft, "edited=true")
    assert {i["id"] for i in edited["items"]} == {labelled["id"], empty["id"]} and all(i["fixed"] for i in edited["items"])
    in_base = _items(hub, v1, "edited=true")["items"]
    assert {i["id"]: i["state"] for i in in_base}[empty["id"]] == "background"
    draft_view = hub.get(f"/api/releases/{draft}").json()
    assert draft_view["status"] == "draft"
    assert draft_view["stats"]["states"]["labelled"] == view["stats"]["states"]["labelled"] + 1
    assert not draft_view["stats"].get("stale")

    # ---- one image back to the base's labels
    assert hub.post(f"/api/releases/{draft}/items/{empty['id']}/revert").json()["edited"] == 1

    # ---- not a version yet: no training, no download, no trash
    assert hub.post("/api/runs", json={"release_id": draft, "template": "dummy"}).status_code == 409
    assert hub.post(f"/api/releases/{draft}/exports", json={"format": "yolo"}).status_code in (404, 409)
    assert hub.delete(f"/api/releases/{draft}").status_code == 400

    # ---- confirm: an ordinary version based on v1, one image relabelled
    assert hub.post(f"/api/releases/{draft}/confirm", json={"version": "작업 중"}).status_code == 400
    assert hub.post(f"/api/releases/{draft}/confirm", json={"version": "v1"}).status_code == 409
    done = hub.post(f"/api/releases/{draft}/confirm", json={"version": "v2"})
    assert done.status_code == 200, done.text
    v2 = hub.get(f"/api/releases/{draft}").json()
    assert v2["version"] == "v2" and v2["status"] == "ready" and v2["base"]["version"] == "v1"
    assert v2["note"] == "이미지 1장의 라벨을 고침" and v2["draft"] is None
    assert hub.get(f"/api/releases/{draft}/compare?other={v1}").json()["relabelled"] == 1

    # ---- an older version is not fixed any more; the newest makes a new work in progress
    old = _save(hub, v1, labelled["id"], boxes)
    assert old.status_code == 409 and "최신 버전(v2)" in old.json()["detail"]
    assert not _detail(hub, v1, labelled["id"])["edit"]["allowed"]
    again = _save(hub, draft, empty["id"], [], parent=_detail(hub, draft, empty["id"])["edit"]["revision_id"])
    assert again.status_code == 200 and again.json()["state"] == "background"
    second = _save(hub, draft, labelled["id"], [])
    second_draft = second.json()["release_id"]
    assert second_draft != draft
    gone = hub.post(f"/api/releases/{second_draft}/discard").json()
    assert gone["discarded"] == 1 and gone["latest_id"] == draft
    assert hub.get(f"/api/releases/{draft}").json()["draft"] is None
    # the discarded fix stays in the image's label history
    history = _detail(hub, draft, labelled["id"])["revisions"]
    assert any(r["source"] == "human:admin" and r["state"] == "background" for r in history)


def test_labels_are_checked_and_kept_inside_the_image(hub, tmp_path):  # noqa: F811
    project = hub.post("/api/projects", json={"name": "검사"}).json()
    v1 = _import(hub, project["id"], make_yolo(tmp_path / "ds"), "v1")
    item = _items(hub, v1)["items"][0]
    assert _save(hub, v1, item["id"], [[99, 0.5, 0.5, 0.1, 0.1]]).status_code == 400  # no such class
    assert _save(hub, v1, item["id"], [[0, 0.5]]).status_code == 400  # not a box
    saved = _save(hub, v1, item["id"], [[0, 0.95, 0.5, 0.2, 0.2], [0, 1.5, 0.5, 0.1, 0.1]]).json()
    rev = next(r for r in _detail(hub, saved["release_id"], item["id"])["revisions"] if r["id"] == saved["revision_id"])
    assert rev["boxes"] == [[0, 0.925, 0.5, 0.15, 0.2]]  # cut at the edge; the one outside is gone


def test_the_work_in_progress_moves_onto_a_newer_version(hub, tmp_path):  # noqa: F811
    project = hub.post("/api/projects", json={"name": "옮기기"}).json()
    root = make_yolo(tmp_path / "ds")
    v1 = _import(hub, project["id"], root, "v1")
    item = _items(hub, v1)["items"][0]
    draft = _save(hub, v1, item["id"], [[0, 0.4, 0.4, 0.2, 0.2]]).json()["release_id"]
    v2 = _import(hub, project["id"], root, "v2")  # same folder again: a newer version

    blocked = _save(hub, v2, item["id"], [[0, 0.4, 0.4, 0.3, 0.3]])
    assert blocked.status_code == 409 and "바탕" in blocked.json()["detail"]
    assert hub.get(f"/api/releases/{v2}").json()["draft"]["behind"]["version"] == "v2"
    moved = hub.post(f"/api/releases/{draft}/rebase").json()
    assert moved == {"base": "v2", "kept": 1, "overlapping": [], "dropped": 0}
    view = hub.get(f"/api/releases/{draft}").json()
    assert view["base"]["version"] == "v2" and view["draft"]["edited"] == 1 and view["draft"]["behind"] is None
    assert _save(hub, v2, item["id"], [[0, 0.4, 0.4, 0.3, 0.3]]).json()["release_id"] == draft


def test_a_folder_of_images_only_is_not_labelled_yet(hub, tmp_path):  # noqa: F811
    root = make_yolo(tmp_path / "ds")
    shutil.rmtree(root / "labels")
    (root / "data.yaml").unlink()
    project = hub.post("/api/projects", json={"name": "라벨할 것"}).json()
    v1 = _import(hub, project["id"], root, "v1", classes=["car", "person", " car "])
    view = hub.get(f"/api/releases/{v1}").json()
    assert [c["name"] for c in view["label_space"]["classes"]] == ["car", "person"]
    assert view["stats"]["states"] == {"labelled": 0, "background": 0, "unlabelled": 8}
    item = _items(hub, v1, "state=unlabelled")["items"][0]
    assert _detail(hub, v1, item["id"])["revisions"][0]["labelled_classes"] == []

    # labelling one: it counts as labelled once confirmed
    draft = _save(hub, v1, item["id"], [[1, 0.5, 0.5, 0.3, 0.6]]).json()["release_id"]
    hub.post(f"/api/releases/{draft}/confirm", json={"version": "v2"}).raise_for_status()
    assert hub.get(f"/api/releases/{draft}").json()["stats"]["states"] == {"labelled": 1, "background": 0, "unlabelled": 7}

    # the same folder with "background": nothing to find in any of them
    v3 = _import(hub, project["id"], root, "v3", unlabelled="background")
    assert hub.get(f"/api/releases/{v3}").json()["stats"]["states"] == {"labelled": 0, "background": 8, "unlabelled": 0}
    assert hub.post(f"/api/projects/{project['id']}/imports", json={
        "node_id": hub.node_id, "path": str(root), "dataset": "street", "version": "작업 중",
    }).status_code == 400


def test_images_without_a_label_file_next_to_labelled_ones_are_background(tmp_path):
    root = make_yolo(tmp_path / "ds")  # one train image has no label file
    states = {e.path.name: e.state for e in importer.scan(root).entries}
    assert states["train_0002.jpg"] == "background"
    states = {e.path.name: e.state for e in importer.scan(root, unlabelled="unlabelled").entries}
    assert states["train_0002.jpg"] == "unlabelled" and states["train_0001.jpg"] == "background"  # empty file


def test_fix_outlines_keypoints_and_classes(hub, tmp_path):  # noqa: F811
    from sample_data import make_imagefolder
    from test_classify import _make_seg
    from test_pose import make_yolo_pose

    project = hub.post("/api/projects", json={"name": "모양"}).json()

    # ---- segmentation: the box follows the outline; an outline needs three corners
    seg = _import(hub, project["id"], _make_seg(tmp_path / "seg"), "v1", task="segment")
    item = _items(hub, seg, "q=train_0000")["items"][0]
    outline = [2, 0, 0, 0, 0, 0.1, 0.2, 0.6, 0.2, 0.6, 0.7, 0.1, 0.7]
    saved = _save(hub, seg, item["id"], [outline])
    assert saved.status_code == 200, saved.text
    rev = _saved_revision(hub, item["id"], saved.json())
    assert rev["boxes"] == [[2, 0.35, 0.45, 0.5, 0.5, 0.1, 0.2, 0.6, 0.2, 0.6, 0.7, 0.1, 0.7]]
    assert _save(hub, seg, item["id"], [[2, 0.5, 0.5, 0.2, 0.2, 0.1, 0.2, 0.6, 0.2]]).status_code == 400

    # ---- pose: points as (x, y, visibility); a missing point is (0, 0, 0); the count follows the class
    pose = _import(hub, project["id"], make_yolo_pose(tmp_path / "pose"), "v1", dataset="people")
    item = _items(hub, pose)["items"][0]
    moved = [0, 0.5, 0.5, 0.4, 0.6, 0.45, 0.35, 2, 0.7, 0.3, 1, 0.9, 0.9, 0]
    saved = _save(hub, pose, item["id"], [moved])
    assert saved.status_code == 200, saved.text
    rev = _saved_revision(hub, item["id"], saved.json())
    assert rev["boxes"][0][5:] == [0.45, 0.35, 2.0, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0]
    assert _save(hub, pose, item["id"], [[0, 0.5, 0.5, 0.4, 0.6, 0.4, 0.3, 2]]).status_code == 400  # one point of three
    assert _save(hub, pose, item["id"], [[0, 0.5, 0.5, 0.4, 0.6, 0.4, 0.3, 5, 0, 0, 0, 0, 0, 0]]).status_code == 400
    new = _save(hub, pose, item["id"], [moved, [0, 0.2, 0.2, 0.1, 0.1]]).json()  # a new one, points not placed yet
    rev = _saved_revision(hub, item["id"], new)
    assert rev["boxes"][1] == [0, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---- classification: one class per image
    cls = _import(hub, project["id"], make_imagefolder(tmp_path / "cls"), "v1", dataset="kinds")
    item = _items(hub, cls)["items"][0]
    other = 1 - int(_detail(hub, cls, item["id"])["revisions"][0]["boxes"][0][0])
    saved = _save(hub, cls, item["id"], [[other]])
    assert saved.status_code == 200 and saved.json()["state"] == "labelled"
    assert _save(hub, cls, item["id"], [[0], [1]]).status_code == 400
