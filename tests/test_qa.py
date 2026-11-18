"""Label checks (검수): rules that find labels which are probably missing or wrong."""

from __future__ import annotations

from sample_data import make_yolo
from test_catalog import _import, hub  # noqa: F401  (the fixture)

LABELS = {
    "train_0000": "0 0.5 0.5 0.3 0.3\n0 0.505 0.5 0.3 0.3\n",  # the same object boxed twice
    "train_0001": "",  # background
    "train_0003": "0 0.3 0.3 0.2 0.2\n1 0.3 0.3 0.2 0.21\n",  # one object, two classes
    "train_0004": "1 0.5 0.5 0.005 0.2\n",  # 1.6 pixels wide
    "train_0005": "1 0.5 0.5 1.0 1.0\n",  # the whole image
    "val_0000": "2 0.5 0.5 0.2 0.2\n",  # the only "dog" in the dataset
    "val_0001": "1 0.4 0.4 0.2 0.2\n",
}


def _counts(client, release_id):
    return {c["id"]: c["images"] for c in client.get(f"/api/releases/{release_id}/qa").json()["checks"]}


def test_rules_flag_duplicates_class_overlaps_tiny_huge_and_rare(hub, tmp_path):  # noqa: F811
    root = make_yolo(tmp_path / "ds")  # 320×240; train_0002 has no label file (background here)
    for name, text in LABELS.items():
        (root / "labels" / name.split("_")[0] / f"{name}.txt").write_text(text, encoding="utf-8")
    project = hub.post("/api/projects", json={"name": "검수"}).json()
    v1 = _import(hub, project["id"], root, "v1")

    summary = hub.get(f"/api/releases/{v1}/qa").json()
    assert {c["id"]: c["group"] for c in summary["checks"]}["rare_class"] == "wrong"
    assert _counts(hub, v1) == {
        "unlabelled": 0,
        "duplicate": 1,
        "class_overlap": 1,
        "rare_class": 1,
        "tiny": 1,
        "huge": 1,
    }
    flagged = hub.get(f"/api/releases/{v1}/items", params={"check": "duplicate"}).json()
    assert flagged["total"] == 1 and flagged["items"][0]["file_name"] == "train_0000.jpg"
    assert flagged["items"][0]["qa"] == {"check": "duplicate", "boxes": [0, 1], "note": "person 2개 겹침"}
    rare = hub.get(f"/api/releases/{v1}/items", params={"check": "rare_class"}).json()["items"]
    assert [(i["file_name"], i["qa"]["note"]) for i in rare] == [("val_0000.jpg", "dog")]
    # each note names the classes involved, so the list says which labels to look at
    notes = {
        check: hub.get(f"/api/releases/{v1}/items", params={"check": check}).json()["items"][0]["qa"]["note"]
        for check in ("class_overlap", "tiny", "huge")
    }
    assert notes == {"class_overlap": "car ↔ person 겹침", "tiny": "car 1개", "huge": "car 1개"}
    assert hub.get(f"/api/releases/{v1}/items", params={"check": "nope"}).status_code == 400

    # ---- fixing in the work in progress: its checks follow the fix
    item = flagged["items"][0]["id"]
    fixed = hub.put(f"/api/releases/{v1}/items/{item}/labels", json={"boxes": [[0, 0.5, 0.5, 0.3, 0.3]]})
    draft = fixed.json()["release_id"]
    assert _counts(hub, draft)["duplicate"] == 0 and _counts(hub, v1)["duplicate"] == 1  # the version stays as it was

    # ---- a labeling set of what a check found
    made = hub.post("/api/label-sets", json={"release_id": draft, "name": "작은 박스", "filter": {"check": "tiny"}}).json()
    assert made["items"] == 1


def test_let_be_follows_the_labels_and_versions_keep_their_review(hub, tmp_path):  # noqa: F811
    root = make_yolo(tmp_path / "ds")
    for name, text in LABELS.items():
        (root / "labels" / name.split("_")[0] / f"{name}.txt").write_text(text, encoding="utf-8")
    project = hub.post("/api/projects", json={"name": "검수 기록"}).json()
    v1 = _import(hub, project["id"], root, "v1")

    def check(release_id, check_id):
        return next(c for c in hub.get(f"/api/releases/{release_id}/qa").json()["checks"] if c["id"] == check_id)

    # ---- "문제 없음": the two boxes really are two objects; the check stops counting the image
    twice = hub.get(f"/api/releases/{v1}/items", params={"check": "duplicate"}).json()["items"][0]["id"]
    body = {"check": "duplicate", "note": "실제로 겹쳐 선 두 사람"}
    judged = hub.put(f"/api/releases/{v1}/items/{twice}/let-be", json=body)
    assert judged.status_code == 200, judged.text
    assert judged.json()["let_be"]["note"] == "실제로 겹쳐 선 두 사람" and judged.json()["let_be"]["by"] == "admin"
    assert (check(v1, "duplicate")["images"], check(v1, "duplicate")["let_be"]) == (0, 1)
    assert hub.get(f"/api/releases/{v1}/items", params={"check": "duplicate"}).json()["total"] == 0
    kept = hub.get(f"/api/releases/{v1}/items", params={"check": "duplicate", "let_be": True}).json()["items"]
    assert [i["id"] for i in kept] == [twice] and kept[0]["let_be"]["note"] == "실제로 겹쳐 선 두 사람"
    assert hub.put(f"/api/releases/{v1}/items/{twice}/let-be", json={"check": "nope"}).status_code == 400

    # ---- a fix in the work in progress: the judgement follows the unchanged labels; the preview says what was done
    tiny = hub.get(f"/api/releases/{v1}/items", params={"check": "tiny"}).json()["items"][0]["id"]
    draft = hub.put(f"/api/releases/{v1}/items/{tiny}/labels", json={"boxes": []}).json()["release_id"]
    assert (check(draft, "duplicate")["images"], check(draft, "tiny")["images"]) == (0, 0)
    preview = hub.get(f"/api/releases/{draft}/review").json()
    assert preview["preview"] and not preview["recorded"] and preview["base"]["version"] == "v1"
    assert preview["before"]["groups"]["boxes"] == 2 and preview["after"]["groups"]["boxes"] == 1  # tiny fixed, huge left
    assert preview["before"]["checks"]["tiny"] == 1 and preview["after"]["checks"]["tiny"] == 0
    assert (preview["fixed"], preview["fixed_flagged"], preview["let_be"], preview["people"]) == (1, 1, 1, ["admin"])

    # ---- confirmed: the version keeps the record; the list of versions says which ones have one
    hub.post(f"/api/releases/{draft}/confirm", json={"version": "v2"})
    record = hub.get(f"/api/releases/{draft}/review").json()
    assert record["recorded"] and record["after"]["groups"]["boxes"] == 1 and record["let_be"] == 1
    imported = hub.get(f"/api/releases/{v1}/review").json()
    assert not imported["recorded"] and imported["now"]["groups"]["boxes"] == 2
    tree = hub.get(f"/api/projects/{project['id']}/datasets").json()
    listed = {r["version"]: r.get("review") for d in tree for r in d["releases"]}
    assert listed["v2"] == {"recorded": True, "open": 3} and listed["v1"]["recorded"] is False

    # ---- labels changed later: the judgement no longer holds, the check counts the image again
    moved = [[0, 0.5, 0.5, 0.3, 0.3], [0, 0.505, 0.505, 0.3, 0.3]]
    again = hub.put(f"/api/releases/{draft}/items/{twice}/labels", json={"boxes": moved}).json()["release_id"]
    assert check(again, "duplicate")["images"] == 1 and check(draft, "duplicate")["images"] == 0

    # ---- taken back: counted again where it was let be
    assert hub.delete(f"/api/releases/{v1}/items/{twice}/let-be", params={"check": "duplicate"}).status_code == 200
    assert (check(v1, "duplicate")["images"], check(v1, "duplicate")["let_be"]) == (1, 0)
