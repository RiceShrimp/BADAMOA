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
    # 검수 narrows what a check found like the dataset screen: by split, by name, by label state
    by = lambda **f: hub.get(f"/api/releases/{v1}/items", params={"check": "duplicate", **f}).json()["total"]  # noqa: E731
    assert (by(split="train"), by(split="val"), by(q="train_0000"), by(q="val_"), by(state="background")) == (1, 0, 1, 0, 0)
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
    moved = [[0, 0.5, 0.5, 0.3, 0.3], [0, 0.502, 0.502, 0.3, 0.3]]  # IoU 0.97: the same object (0.95 and up)
    again = hub.put(f"/api/releases/{draft}/items/{twice}/labels", json={"boxes": moved}).json()["release_id"]
    assert check(again, "duplicate")["images"] == 1 and check(draft, "duplicate")["images"] == 0

    # ---- taken back: counted again where it was let be
    assert hub.delete(f"/api/releases/{v1}/items/{twice}/let-be", params={"check": "duplicate"}).status_code == 200
    assert (check(v1, "duplicate")["images"], check(v1, "duplicate")["let_be"]) == (1, 0)


def test_the_class_filter_narrows_findings_by_their_own_class(hub, tmp_path):  # noqa: F811
    root = make_yolo(tmp_path / "ds")
    labels = {
        "train_0000": "0 0.5 0.5 0.005 0.2\n2 0.3 0.3 0.2 0.2\n",  # a tiny person beside a dog of a normal size
        "train_0001": "1 0.5 0.5 0.005 0.2\n",  # a tiny car
        "train_0003": "2 0.6 0.6 0.2 0.2\n",
    }
    for name, text in labels.items():
        (root / "labels" / name.split("_")[0] / f"{name}.txt").write_text(text, encoding="utf-8")
    project = hub.post("/api/projects", json={"name": "검수 클래스"}).json()
    v1 = _import(hub, project["id"], root, "v1")

    def tiny(classes: str):
        found = hub.get(f"/api/releases/{v1}/items", params={"check": "tiny", "classes": classes}).json()
        return sorted((i["file_name"], tuple(i["qa"]["boxes"])) for i in found["items"])

    assert tiny("") == [("train_0000.jpg", (0,)), ("train_0001.jpg", (0,))]
    assert tiny("2") == []  # train_0000 holds a dog, but its tiny box is a person
    assert tiny("0") == [("train_0000.jpg", (0,))]
    assert tiny("0,1") == [("train_0000.jpg", (0,)), ("train_0001.jpg", (0,))]

    # outside a check: the images holding any of them, as the dataset screen
    def names(**params):
        found = hub.get(f"/api/releases/{v1}/items", params={**params, "limit": 200}).json()["items"]
        return {i["file_name"] for i in found}

    assert names(classes="1,2") == names(class_id=1) | names(class_id=2)
    assert {"train_0000.jpg", "train_0001.jpg", "train_0003.jpg"} <= names(classes="1,2")


def test_a_model_check_counts_as_fiftyone_does():
    """FiftyOne's compute_mistakenness: boxes paired whatever their classes (IoU 0.5, the surest first); an unpaired
    prediction counts only when the model is sure (over 0.95), a pair of other classes only when it is sure of its
    own, and a label only when no prediction even touches it and it is big enough to see."""
    from badamoa.hub.checks import judge

    rule = {"conf": 0.95, "min_area": 0.001}
    person, bike = [0, 0.5, 0.5, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]
    truth = [person, bike]

    # the same box, the same class: found; the other label nothing touches: missed
    assert judge([person], [0.9], truth, rule) == {"found": [0], "extra": [], "missed": [1], "reclass": []}
    # an object no label has: only a sure one is a missing label (0.97), an unsure one (0.6) is the model's own error
    assert judge([person, bike, [2, 0.8, 0.8, 0.1, 0.1]], [0.9, 0.9, 0.97], truth, rule)["extra"] == [2]
    assert judge([person, bike, [2, 0.8, 0.8, 0.1, 0.1]], [0.9, 0.9, 0.6], truth, rule)["extra"] == []
    # the right box, another class: the label's class may be wrong only when the model is sure; never missed twice
    other = [1, 0.5, 0.5, 0.2, 0.2]
    assert judge([other], [0.98], truth[:1], rule) == {"found": [], "extra": [], "missed": [], "reclass": [0]}
    assert judge([other], [0.7], truth[:1], rule) == {"found": [], "extra": [], "missed": [], "reclass": []}
    # a label a prediction touches (an imprecise box) or too small to see (under 0.1% of the image) is not missed
    assert judge([[0, 0.55, 0.55, 0.2, 0.2]], [0.5], [person], rule)["missed"] == []
    assert judge([], [], [[0, 0.5, 0.5, 0.02, 0.02]], rule)["missed"] == []
    assert judge([], [], [[0, 0.5, 0.5, 0.05, 0.05]], rule)["missed"] == [0]
    # a class the model never learned is not its to judge
    assert judge([], [], truth, rule, known={0})["missed"] == [0]
    # classification: the class it gives against the image's own, when it is sure
    assert judge([[3]], [0.99], [[1]], rule)["reclass"] == [0] and judge([[3]], [0.8], [[1]], rule)["reclass"] == []
    assert judge([[1]], [0.6], [[1]], rule)["found"] == [0]


def test_model_checks_stay_on_the_review_screen_after_they_end(hub, settings, tmp_path):  # noqa: F811
    """검수 keeps its model checks like a training's record: each one after it ends, the failed one with why, on
    which server, how long, who started it, and the images it had sent to that server first; the log is the job's."""
    from datetime import timedelta

    from badamoa.hub.db import Database, utcnow
    from badamoa.hub.models import Job
    from badamoa.hub.train_models import ModelJob, ModelVersion

    root = make_yolo(tmp_path / "ds")
    project = hub.post("/api/projects", json={"name": "검수 기록"}).json()
    v1 = _import(hub, project["id"], root, "v1")
    me = hub.get("/api/auth/me").json()
    database = Database(settings.database_url)
    with database.session() as db:
        now = utcnow()
        model = ModelVersion(project_id=project["id"], name="det", version=1, blob="0" * 64, trainer="yolo", task="detect")
        db.add(model)
        db.flush()
        sent = Job(type="data.sync", params={"release_id": v1}, status="succeeded", target_node_id=hub.node_id,
                   node_id=hub.node_id, started_at=now - timedelta(minutes=5),
                   finished_at=now - timedelta(minutes=3))  # fmt: skip
        db.add(sent)
        db.flush()
        failed = Job(type="model.run", params={}, status="failed", error="CUDA out of memory", node_id=hub.node_id,
                     requirements={"after": sent.id}, started_at=now - timedelta(minutes=3),
                     finished_at=now - timedelta(minutes=2))  # fmt: skip
        done = Job(type="model.run", params={}, status="succeeded", node_id=hub.node_id,
                   started_at=now - timedelta(hours=1, minutes=2), finished_at=now - timedelta(hours=1))  # fmt: skip
        db.add_all([failed, done])
        db.flush()
        db.add_all(
            [
                ModelJob(
                    model_id=model.id,
                    kind="eval",
                    release_id=v1,
                    params={"split": "val"},
                    job_id=done.id,
                    created_by=me["id"],
                    created_at=now - timedelta(hours=1, minutes=3),
                ),  # fmt: skip
                ModelJob(
                    model_id=model.id,
                    kind="eval",
                    release_id=v1,
                    params={"split": "val", "splits": ["val", "test"]},
                    job_id=failed.id,
                    created_by=me["id"],
                    created_at=now - timedelta(minutes=6),
                ),  # fmt: skip
            ]
        )
    database.engine.dispose()

    rows = hub.get(f"/api/releases/{v1}/qa/history").json()
    assert [r["status"] for r in rows] == ["failed", "succeeded"]  # the newest first, kept after they ended
    newest, older = rows
    assert (newest["error"], newest["node_name"], newest["by"]) == ("CUDA out of memory", "data-node", "admin")
    assert newest["splits"] == ["val", "test"] and newest["model"]["name"] == "det"
    assert newest["duration_s"] == 60 and older["duration_s"] == 120
    assert newest["sent"]["status"] == "succeeded" and newest["sent"]["node_name"] == "data-node"
    assert newest["sent"]["duration_s"] == 120 and older["sent"] is None
    assert older["basis"] is False  # it kept no per-image predictions: no model basis
    assert hub.get(f"/api/jobs/{newest['job_id']}/logs").status_code == 200


def test_a_model_checks_counts_move_with_the_thresholds(hub, settings, tmp_path):  # noqa: F811
    """Counts an older Hub stored as the template gave them are counted again at the start. FiftyOne's thresholds
    count by default; a screen asks for others and sees the counts at each level beside: an object no label has at
    0.9 is not one at FiftyOne's 0.95, it is at 0.8."""
    from sqlalchemy import select

    from badamoa.hub import checks
    from badamoa.hub.catalog_models import ItemPrediction, LabelRevision, ReleaseItem
    from badamoa.hub.db import Database
    from badamoa.hub.models import HubSetting
    from badamoa.hub.train_models import ModelJob, ModelVersion

    root = make_yolo(tmp_path / "ds")
    project = hub.post("/api/projects", json={"name": "셈"}).json()
    v1 = _import(hub, project["id"], root, "v1")
    database = Database(settings.database_url)
    with database.session() as db:
        rows = db.execute(
            select(ReleaseItem.item_id, LabelRevision.boxes)
            .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id)
            .where(ReleaseItem.release_id == v1)
        ).all()
        item_id, boxes = next((i, b) for i, b in rows if b)
        model = ModelVersion(project_id=project["id"], name="m", version=1, blob="0" * 64, trainer="yolo", task="detect")
        db.add(model)
        db.flush()
        op = ModelJob(model_id=model.id, kind="eval", release_id=v1, params={"split": "val"})
        db.add(op)
        db.flush()
        away = [0, 0.01, 0.01, 0.01, 0.01]  # a corner no label is at
        touching = [[b[0], *b[1:5]] for b in boxes]  # every label touched (paired, same class)
        db.add(ItemPrediction(op_id=op.id, item_id=item_id, labels=[*touching, away], scores=[0.99] * len(boxes) + [0.9],
                              found=0, missed=7, extra=7, wrong=14))  # fmt: skip
        op_id = op.id
    with database.session() as db:
        marked = db.get(HubSetting, checks.RULE_KEY)  # this Hub counted at its start: as a Hub before 0.89.0, it had not
        if marked is not None:
            db.delete(marked)
        db.flush()
        assert checks.recount_if_needed(db) == 1 and checks.recount_if_needed(db) == 0  # once, then it is counted
        row = db.get(ItemPrediction, (op_id, item_id))
        assert (row.found, row.missed, row.extra, row.reclass, row.wrong) == (len(boxes), 0, 0, 0, 0)
        assert (row.extra_top, row.reclass_top, row.missed_area) == (0.9, 0.0, 0.0)  # the values any threshold counts by
    database.engine.dispose()

    fiftyone = hub.get(f"/api/releases/{v1}/qa").json()
    assert fiftyone["rule"] == {"conf": 0.95, "min_area": 0.001, "defaults": {"conf": 0.95, "min_area": 0.001}}
    first = fiftyone["evaluations"][0]
    assert (first["extra"], first["reclass"], first["missed"]) == (0, 0, 0)
    # beside the chosen threshold, the count at each level: the corner box (0.9) counts below 0.9, not from it
    assert dict(map(tuple, first["levels"]["extra"])) == {0.5: 1, 0.7: 1, 0.8: 1, 0.9: 0, 0.95: 0, 0.99: 0}
    lower = hub.get(f"/api/releases/{v1}/qa", params={"conf": 0.8}).json()
    assert lower["rule"]["conf"] == 0.8 and lower["evaluations"][0]["extra"] == 1
    found = hub.get(f"/api/releases/{v1}/items", params={"suspect": op_id, "mistake": "extra", "conf": 0.8}).json()
    assert [i["id"] for i in found["items"]] == [item_id] and found["items"][0]["check"]["extra"] == 1
    assert hub.get(f"/api/releases/{v1}/items", params={"suspect": op_id, "mistake": "extra"}).json()["total"] == 0
    assert hub.get(f"/api/releases/{v1}/qa", params={"conf": 0.01}).json()["rule"]["conf"] == 0.25  # kept from 0.25


def test_box_checks_count_at_the_thresholds_asked_for(hub, tmp_path):  # noqa: F811
    """The box checks count at the thresholds 검수 asks for (IoU 0.95 · 3 px · 98% by default), with the counts at
    each level beside; the images and the labels framed follow."""
    root = make_yolo(tmp_path / "ds")
    for name, text in LABELS.items():
        (root / "labels" / name.split("_")[0] / f"{name}.txt").write_text(text, encoding="utf-8")
    project = hub.post("/api/projects", json={"name": "임계값"}).json()
    v1 = _import(hub, project["id"], root, "v1")

    def checks_at(rules=""):
        return {c["id"]: c for c in hub.get(f"/api/releases/{v1}/qa", params={"rules": rules}).json()["checks"]}

    plain = checks_at()  # train_0000: two boxes of a class at IoU 0.97; train_0004: 1.6 px wide; train_0005: all of it
    assert (plain["duplicate"]["limit"], plain["duplicate"]["images"]) == (0.95, 1) and "0.95" in plain["duplicate"]["hint"]
    assert (plain["tiny"]["images"], plain["huge"]["images"]) == (1, 1)
    assert dict(map(tuple, plain["duplicate"]["levels"]))[0.99] == 0 and dict(map(tuple, plain["tiny"]["levels"]))[1] == 0
    strict = checks_at("duplicate:0.99,tiny:1,huge:0.999")
    assert (strict["duplicate"]["images"], strict["tiny"]["images"], strict["huge"]["images"]) == (0, 0, 1)
    assert "IoU 0.99" in strict["duplicate"]["hint"] and checks_at("duplicate:7")["duplicate"]["limit"] == 0.99  # bounded

    def listed(rules):
        return hub.get(f"/api/releases/{v1}/items", params={"check": "duplicate", "rules": rules}).json()

    assert listed("duplicate:0.99")["total"] == 0
    loose = listed("duplicate:0.9")
    assert loose["total"] == 1 and loose["items"][0]["qa"]["boxes"] == [0, 1] and "겹침" in loose["items"][0]["qa"]["note"]
