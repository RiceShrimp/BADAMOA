"""Finding labels that are likely wrong, from what models predicted when they were evaluated.

An evaluation records, for every image it measured, the model's predictions and how they compared with
the labels: found, missed (labels the model did not find) and extra (predictions no label backs). Many
mistakes on one image often mean its labels are wrong, so the dataset screen can list the images by them
and the item view can show the predictions over the labels and take one as a label.

Images are matched by their sha256: the evaluation exports them under that name. Predictions keep the
evaluated release's label space; an evaluation is offered on the versions of the same dataset that share it.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names

from .catalog_models import Dataset, Item, ItemPrediction, LabelRevision, QaDismissal, Release, ReleaseItem
from .db import iso
from .models import HubSetting, Job
from .services import ServiceError
from .train_models import ModelJob, ModelVersion

CHUNK = 2000


def ingest(db: Session, op: ModelJob, raw: bytes) -> int:
    """Store an evaluation's per-image predictions (``bm.log_predictions``). Returns how many images matched."""
    release = db.get(Release, op.release_id) if op.release_id else None
    if release is None:
        return 0
    try:
        images = json.loads(raw.decode("utf-8")).get("images") or []
    except (ValueError, UnicodeDecodeError, AttributeError):
        raise ServiceError(400, "예측 파일을 읽을 수 없습니다.") from None
    dataset = db.get(Dataset, release.dataset_id)
    job = db.get(Job, op.job_id) if op.job_id else None
    # The model numbers its classes 0..n-1; the release's label space may number them otherwise.
    class_ids = list((job.params or {}).get("class_ids") or []) if job else []

    def label_space_id(n: int) -> int | None:
        if not class_ids:
            return n
        return class_ids[n] if 0 <= n < len(class_ids) else None

    by_sha: dict[str, dict[str, Any]] = {}
    for entry in images:
        if isinstance(entry, dict) and isinstance(entry.get("image"), str):
            by_sha[entry["image"].lower()] = entry
    # Each upload replaces the predictions of the images it names: an evaluation of several splits uploads once per
    # split, and its images add up to one basis.
    shas = list(by_sha)
    matched = 0
    known = set(class_ids)
    for start in range(0, len(shas), CHUNK):
        chunk = shas[start : start + CHUNK]
        items = db.execute(
            select(Item.id, Item.sha256).where(Item.project_id == dataset.project_id, Item.sha256.in_(chunk))
        ).all()
        truths = _truths(db, release.id, [i for i, _ in items])
        db.execute(
            delete(ItemPrediction).where(ItemPrediction.op_id == op.id, ItemPrediction.item_id.in_([i for i, _ in items]))
        )
        rows = []
        for item_id, sha in items:
            entry = by_sha[sha]
            labels, scores = [], []
            for label, score in zip(entry.get("labels") or [], entry.get("scores") or [], strict=False):
                try:
                    cls = label_space_id(int(label[0]))
                    values = [float(v) for v in label[1:]]
                except (TypeError, ValueError, IndexError):
                    continue
                if cls is None:
                    continue
                labels.append([cls, *values])
                scores.append(round(float(score), 4))
            # counted here (FiftyOne's way, ``_pairing``), not as the template counted them
            counts = _stored(_pairing(labels, scores, truths.get(item_id, []), known))
            rows.append({"op_id": op.id, "item_id": item_id, **counts, "labels": labels, "scores": scores})
        if rows:
            db.execute(insert(ItemPrediction), rows)
            matched += len(rows)
    return matched


def _offered(db: Session, release: Release, kinds: tuple[str, ...] = ("eval",)) -> list[ModelJob]:
    """Evaluations on this release or another version of its dataset with the same classes, newest first,
    by a model of the dataset's task: a Seg model's view of a Det dataset is no check of its labels.
    ``kinds``: also 오토라벨 runs (``prediction_runs``)."""
    siblings = select(Release.id).where(
        Release.dataset_id == release.dataset_id, Release.label_space_id == release.label_space_id
    )
    has_rows = select(ItemPrediction.op_id).where(ItemPrediction.op_id == ModelJob.id).exists()
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind.in_(kinds), ModelJob.release_id.in_(siblings), has_rows)
        .order_by(ModelJob.created_at.desc())
    ).all()
    dataset = db.get(Dataset, release.dataset_id)
    task = (dataset.task if dataset else "") or "detect"
    models = {m.id: m for m in db.scalars(select(ModelVersion).where(ModelVersion.id.in_({op.model_id for op in ops})))}
    return [op for op in ops if op.model_id not in models or task_names.same(models[op.model_id].task or "detect", task)]


MISTAKES = ("extra", "missed", "reclass")  # what an evaluation finds (``judge``), judged like a rule check's finding


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# ---- counting a model's view of the labels: FiftyOne's way (compute_mistakenness) ----------------------------

RULE_KEY = "qa.model"  # what the stored counts were made with (``recount_if_needed``)
# 1: FiftyOne's (0.89.0); 2: with the values that count at any threshold; 3: no size cut for a label the model
# did not find (0.95.0); 4: how near the nearest prediction came to each such label, any IoU (0.100.1); 5: IoU 0.01
# by default (0.103.0). A new one counts again
RULE_VERSION = 5
# FiftyOne's 0.95; every label no prediction touches, however small (FiftyOne has no size cut either). ``min_area``
# is still taken from an older screen's query
# ``conf``: how sure the model must be of an object no label has (extra); ``reclass_conf``: of another class on a
# label (reclass), set apart on 검수 since 0.100.1
# ``missed_iou``: a label paired with no prediction counts as not found when none overlaps it by this much (0.01:
# the model put nothing near it; 0.95 would take about every unpaired label, as FiftyOne flags them)
DEFAULT_RULE = {"conf": 0.95, "reclass_conf": 0.95, "missed_iou": 0.01, "min_area": 0.0}
MATCH_IOU = 0.5  # a prediction and a label are one object (FiftyOne's _DETECTION_IOU)
MISSED_IOU_LEVELS = (0.01, 0.1, 0.3, 0.5, 0.7, 0.95)  # 물체가 없을 수 있는 라벨: the counts beside the one chosen
# What 검수 shows the counts at besides the one chosen: how they change with the threshold
CONF_LEVELS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def rule(
    conf: float | None = None,
    min_area: float | None = None,
    reclass_conf: float | None = None,
    missed_iou: float | None = None,
) -> dict[str, float]:
    """How sure the model must be of an object no label has (``conf``) and of another class on a label
    (``reclass_conf``), and how big a label must be (``min_area``, a share of the image): FiftyOne's by default; a
    screen may ask for others (predictions are kept from 0.25 up)."""

    def sure(value: float | None, default: float) -> float:
        return default if value is None else min(0.99, max(0.25, float(value)))

    picked = {
        "conf": sure(conf, DEFAULT_RULE["conf"]),
        "reclass_conf": sure(reclass_conf, DEFAULT_RULE["reclass_conf"]),
        "missed_iou": DEFAULT_RULE["missed_iou"] if missed_iou is None else min(0.99, max(0.01, float(missed_iou))),
        "min_area": DEFAULT_RULE["min_area"] if min_area is None else min(0.1, max(0.0, float(min_area))),
    }
    return picked


def _pairing(
    predicted: list[list[float]], scores: list[float], truth: list[list[float]], known: set[int] | None = None
) -> dict[str, list[Any]]:
    """An image's predictions against its labels before any threshold, FiftyOne's way: each prediction is paired
    with the label its box overlaps most (IoU 0.5, whatever the classes, the surest prediction first).
    ``found``: labels paired with their class; ``extra``: (prediction, score) paired with none; ``reclass``:
    (label, score) paired with another class; ``missed``: (label, its share of the image) no prediction even
    touches (IoU under 0.1). ``known``: the classes the model learned, a label of another is not its to judge.
    Classification (no box): the class it gives against the image's own."""
    out: dict[str, list[Any]] = {"found": [], "extra": [], "reclass": [], "missed": []}
    mine = [j for j, b in enumerate(truth) if b and (not known or int(b[0]) in known)]
    score = [scores[i] if i < len(scores) else 0.0 for i in range(len(predicted))]
    if any(len(b) < 5 for b in [*predicted, *(truth[j] for j in mine)]):
        if predicted and mine:
            top = max(range(len(predicted)), key=lambda i: score[i])
            held = [j for j in mine if int(truth[j][0]) == int(predicted[top][0])]
            if held:
                out["found"] = held[:1]
            else:
                out["reclass"] = [(mine[0], score[top])]
        return out
    left = list(mine)
    for i in sorted(range(len(predicted)), key=lambda i: -score[i]):
        best, at = MATCH_IOU, None
        for j in left:
            if (overlap := _iou(predicted[i][1:5], truth[j][1:5])) >= best:
                best, at = overlap, j
        if at is None:
            out["extra"].append((i, score[i]))
            continue
        left.remove(at)
        if int(predicted[i][0]) == int(truth[at][0]):
            out["found"].append(at)
        else:
            out["reclass"].append((at, score[i]))
    for j in left:  # labels paired with none: how near the nearest prediction came (IoU, any class)
        box = truth[j]
        near = max((_iou(p[1:5], box[1:5]) for p in predicted if len(p) >= 5), default=0.0)
        out["missed"].append((j, box[3] * box[4], near))
    return out


def judge(
    predicted: list[list[float]],
    scores: list[float],
    truth: list[list[float]],
    counting: dict[str, float] | None = None,
    known: set[int] | None = None,
) -> dict[str, list[int]]:
    """An image's mistakes as FiftyOne counts them (``_pairing``), at a threshold:
    ``extra``    a prediction paired with no label, the model sure of it (over ``conf``): a label may be missing;
    ``reclass``  a pair whose classes differ, the model sure of its own (over ``reclass_conf``): the label's class
                 may be wrong;
    ``missed``   a label no prediction even touches, big enough to see (``min_area`` of the image): it may be
                 spurious (FiftyOne flags every unpaired label; most are objects too small for the model);
    ``found``    pairs of the same class.
    Indices: ``extra`` into the predictions, the rest into the labels."""
    counting = counting or DEFAULT_RULE
    conf, min_area = counting["conf"], counting["min_area"]
    reclass_conf = counting.get("reclass_conf", conf)
    missed_iou = counting.get("missed_iou", DEFAULT_RULE["missed_iou"])
    paired = _pairing(predicted, scores, truth, known)
    return {
        "found": paired["found"],
        "extra": [i for i, s in paired["extra"] if s > conf],
        "reclass": [j for j, s in paired["reclass"] if s > reclass_conf],
        "missed": [j for j, area, near in paired["missed"] if area >= min_area and near < missed_iou],
    }


def _stored(paired: dict[str, list[Any]]) -> dict[str, Any]:
    """What a row keeps: the counts at FiftyOne's settings (sorting, boards) and the values any threshold counts by."""
    conf, min_area = DEFAULT_RULE["conf"], DEFAULT_RULE["min_area"]
    counts = {
        "found": len(paired["found"]),
        "extra": sum(1 for _, s in paired["extra"] if s > conf),
        "reclass": sum(1 for _, s in paired["reclass"] if s > DEFAULT_RULE["reclass_conf"]),
        "missed": sum(1 for _, area, near in paired["missed"] if area >= min_area and near < DEFAULT_RULE["missed_iou"]),
    }
    return {
        **counts,
        "wrong": counts["extra"] + counts["missed"] + counts["reclass"],
        "extra_top": max((s for _, s in paired["extra"]), default=0.0),
        "reclass_top": max((s for _, s in paired["reclass"]), default=0.0),
        "missed_area": max(
            (area for _, area, near in paired["missed"] if near < DEFAULT_RULE["missed_iou"]), default=0.0
        ),
        # the least any unpaired label was overlapped (1: none unpaired): under a threshold, the image has one
        "missed_near": min((near for *_, near in paired["missed"]), default=1.0),
    }


def over(kind: str, counting: dict[str, float]) -> Any:
    """SQL: an image an evaluation found ``kind`` on, at these thresholds."""
    if kind == "extra":
        return ItemPrediction.extra_top > counting["conf"]
    if kind == "reclass":
        return ItemPrediction.reclass_top > counting.get("reclass_conf", counting["conf"])
    near = ItemPrediction.missed_near < counting.get("missed_iou", DEFAULT_RULE["missed_iou"])
    if counting.get("min_area"):  # a size cut an older screen asks for
        return and_(near, ItemPrediction.missed_area >= counting["min_area"])
    return near


def order_of(kind: str) -> Any:
    """SQL: the surest (or, for labels nothing touches, the biggest) first, as FiftyOne sorts by mistakenness."""
    return {"extra": ItemPrediction.extra_top, "reclass": ItemPrediction.reclass_top}.get(kind, ItemPrediction.missed_area)


def levels(kind: str) -> tuple[float, ...]:
    return MISSED_IOU_LEVELS if kind == "missed" else CONF_LEVELS


def at_level(kind: str, counting: dict[str, float], value: float) -> dict[str, float]:
    """``counting`` with ``kind``'s own threshold at ``value`` (the counts at each level beside the chosen one)."""
    key = {"extra": "conf", "reclass": "reclass_conf", "missed": "missed_iou"}[kind]
    return {**counting, key: value, "min_area": 0.0}


def _truths(db: Session, release_id: str, item_ids: list[int]) -> dict[int, list[list[float]]]:
    """Each image's labels in a version."""
    out: dict[int, list[list[float]]] = {}
    for start in range(0, len(item_ids), CHUNK):
        rows = db.execute(
            select(ReleaseItem.item_id, LabelRevision.boxes)
            .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id)
            .where(ReleaseItem.release_id == release_id, ReleaseItem.item_id.in_(item_ids[start : start + CHUNK]))
        )
        out.update({item_id: boxes or [] for item_id, boxes in rows})
    return out


def _known(db: Session, op: ModelJob) -> set[int]:
    job = db.get(Job, op.job_id) if op.job_id else None
    return {int(c) for c in ((job.params or {}).get("class_ids") or [])} if job is not None else set()


def recount(db: Session, op: ModelJob) -> int:
    """An evaluation's stored values, made again from its predictions and the evaluated version's labels."""
    known = _known(db, op)
    rows = db.execute(
        select(ItemPrediction.item_id, ItemPrediction.labels, ItemPrediction.scores).where(ItemPrediction.op_id == op.id)
    ).all()
    truths = _truths(db, op.release_id or "", [item_id for item_id, *_ in rows])
    changed = []
    for item_id, labels, scores in rows:
        paired = _pairing(labels or [], scores or [], truths.get(item_id, []), known)
        changed.append({"op_id": op.id, "item_id": item_id, **_stored(paired)})
    for start in range(0, len(changed), CHUNK):
        db.execute(update(ItemPrediction), changed[start : start + CHUNK])
    return len(changed)


def recount_all(db: Session) -> int:
    """Every evaluation counted again; marked done with the way it was counted."""
    total = 0
    for op in db.scalars(select(ModelJob).where(ModelJob.kind == "eval", ModelJob.release_id.is_not(None))):
        if db.scalar(select(ItemPrediction.item_id).where(ItemPrediction.op_id == op.id).limit(1)) is not None:
            total += recount(db, op)
    row = db.get(HubSetting, RULE_KEY)
    if row is None:
        row = HubSetting(key=RULE_KEY, value={})
        db.add(row)
    row.value = {"counted": {"version": RULE_VERSION}}
    db.flush()
    return total


def recount_if_needed(db: Session) -> int:
    """At the Hub's start: stored counts made another way (before 0.89.0, or without the threshold values), again."""
    row = db.get(HubSetting, RULE_KEY)
    counted = ((row.value or {}) if row is not None else {}).get("counted") or {}
    if counted == {"version": RULE_VERSION}:
        return 0
    import logging

    total = recount_all(db)
    logging.getLogger("badamoa.hub").info("검수: 모델 검사 %d장을 FiftyOne 방식으로 다시 셌습니다", total)
    return total


def page_counts(
    db: Session, release: Release, op_id: str, item_ids: list[int], counting: dict[str, float]
) -> dict[int, dict[str, int]]:
    """For a page of images: an evaluation's findings at these thresholds against this version's labels."""
    op = db.get(ModelJob, op_id)
    if op is None or not item_ids:
        return {}
    truths, known = _truths(db, release.id, item_ids), _known(db, op)
    out = {}
    for row in db.scalars(select(ItemPrediction).where(ItemPrediction.op_id == op.id, ItemPrediction.item_id.in_(item_ids))):
        found = judge(row.labels or [], row.scores or [], truths.get(row.item_id, []), counting, known)
        out[row.item_id] = {kind: len(found[kind]) for kind in ("found", "missed", "extra", "reclass")}
    return out


def flagged(
    db: Session, release: Release, op_id: str, mistake: str, item_ids: list[int], counting: dict[str, float] | None = None
) -> dict[int, dict[str, Any]]:
    """For a page of images: the labels an evaluation's ``mistake`` (missed or reclass) is on, against this
    version's labels, and what the model said of them — 검수 frames them as it frames a rule check's. For
    ``extra`` (objects no label has) the model's own boxes instead: ``predicted`` with their ``scores``, the surest
    first, which 검수 frames and numbers the same way (and can take as labels)."""
    if mistake not in MISTAKES or not item_ids:
        return {}
    from .catalog_models import LabelSpace

    op = db.get(ModelJob, op_id)
    if op is None:
        return {}
    space = db.get(LabelSpace, release.label_space_id)
    names = {int(c["id"]): str(c["name"]) for c in (space.classes if space else [])}
    truths = _truths(db, release.id, item_ids)
    counting, known = counting or rule(), _known(db, op)
    out: dict[int, dict[str, Any]] = {}
    for row in db.scalars(select(ItemPrediction).where(ItemPrediction.op_id == op.id, ItemPrediction.item_id.in_(item_ids))):
        truth = truths.get(row.item_id, [])
        found = judge(row.labels or [], row.scores or [], truth, counting, known)
        boxes = found[mistake]
        if not boxes:
            continue
        if mistake == "extra":  # the predictions: no label is there, and the model is sure of each
            scores = row.scores or []
            picks = sorted(boxes, key=lambda i: -(scores[i] if i < len(scores) else 0.0))
            predicted = [list(row.labels[i]) for i in picks]
            sure = [round(float(scores[i]), 4) if i < len(scores) else 0.0 for i in picks]
            note = " · ".join(f"{names.get(int(b[0]), b[0])} {v:.2f}" for b, v in zip(predicted, sure, strict=True))
            out[row.item_id] = {"boxes": [], "predicted": predicted, "scores": sure, "note": note}
            continue
        if mistake == "reclass":  # what the model is sure each one is
            said = []
            for j in boxes:
                theirs = max(
                    (
                        (s, int(p[0]))
                        for p, s in zip(row.labels or [], row.scores or [], strict=False)
                        if len(p) < 5 or _iou(p[1:5], truth[j][1:5]) >= MATCH_IOU
                    ),
                    default=None,
                )
                if theirs is not None:
                    label, model = names.get(int(truth[j][0]), truth[j][0]), names.get(theirs[1], theirs[1])
                    said.append(f"{label} → 모델 {model} {theirs[0]:.2f}")
            note = " · ".join(said)
        else:
            note = "모델이 찾지 못한 라벨"
        out[row.item_id] = {"boxes": boxes, "note": note}
    return out


def mistakes_of_classes(
    db: Session, release: Release, op_id: str, mistake: str, classes: set[int], counting: dict[str, float] | None = None
) -> set[int]:
    """The images an evaluation found ``mistake`` (extra, missed or reclass) on for one of ``classes``: its
    predictions against each image's labels in this release (``judge``)."""
    from .catalog_models import LabelRevision, ReleaseItem

    counting = counting or rule()
    rows = db.execute(
        select(ItemPrediction.item_id, ItemPrediction.labels, ItemPrediction.scores, ReleaseItem.revision_id)
        .join(ReleaseItem, and_(ReleaseItem.release_id == release.id, ReleaseItem.item_id == ItemPrediction.item_id))
        .where(ItemPrediction.op_id == op_id, over(mistake, counting))
    ).all()
    revisions: dict[str, list[list[float]]] = {}
    ids = list({rid for *_, rid in rows if rid})
    for start in range(0, len(ids), 500):
        found = db.execute(
            select(LabelRevision.id, LabelRevision.boxes).where(LabelRevision.id.in_(ids[start : start + 500]))
        )
        revisions.update({rid: boxes or [] for rid, boxes in found})
    op = db.get(ModelJob, op_id)
    known = _known(db, op) if op is not None else set()
    picked = set()
    for item_id, labels, scores, revision_id in rows:
        truth = revisions.get(revision_id, [])
        found = judge(labels or [], scores or [], truth, counting, known)  # labels of classes it never learned: not its
        if mistake == "extra":
            of = {int((labels or [])[i][0]) for i in found["extra"]}
        else:  # the label's class (and, when it is another, the one the model gives it)
            of = {int(truth[j][0]) for j in found[mistake]}
            if mistake == "reclass":
                of |= {
                    int(p[0])
                    for p in labels or []
                    for j in found["reclass"]
                    if len(p) < 5 or _iou(p[1:5], truth[j][1:5]) >= MATCH_IOU
                }
        if classes & of:
            picked.add(item_id)
    return picked


def let_be(check: Any, item_id: Any, revision_id: Any) -> Any:
    """SQL: someone found this image's labels (this revision of them) right for this check ("문제 없음")."""
    return (
        select(QaDismissal.item_id)
        .where(QaDismissal.item_id == item_id, QaDismissal.check == check, QaDismissal.revision_id == revision_id)
        .exists()
    )


# ---- predictions over the dataset: every model run that predicted its images ------------------------------

PREDICTING = ("eval", "autolabel")


def prediction_runs(db: Session, release: Release) -> list[ModelJob]:
    """Evaluations and 오토라벨 runs whose predictions belong to this version's images (same dataset, same classes,
    a model of its task), newest first."""
    return _offered(db, release, PREDICTING)


def run_ref(db: Session, op: ModelJob, sets: dict[str, str] | None = None) -> dict[str, Any]:
    """A model run as the screens tell it apart: which model, evaluation or 오토라벨, of what, when."""
    params = op.params or {}
    ref: dict[str, Any] = {
        "op_id": op.id,
        "kind": op.kind,  # eval | autolabel
        "model": model_ref(db.get(ModelVersion, op.model_id)),
        "created_at": iso(op.created_at),
    }
    if op.kind == "eval":
        ref["split"] = params.get("split", "")
    else:
        set_id = str(params.get("set_id") or "")
        ref["set"] = {"id": set_id, "name": (sets or {}).get(set_id, "")} if set_id else None
        ref["keep"] = params.get("keep")
    return ref


def _set_names(db: Session, ops: list[ModelJob]) -> dict[str, str]:
    from .catalog_models import LabelSet

    ids = {str((op.params or {}).get("set_id") or "") for op in ops if op.kind == "autolabel"} - {""}
    return dict(db.execute(select(LabelSet.id, LabelSet.name).where(LabelSet.id.in_(ids))).all()) if ids else {}


def release_prediction_runs(db: Session, release: Release) -> list[dict[str, Any]]:
    """The runs whose predictions can be laid over this version's images, with how many images each predicted."""
    ops = prediction_runs(db, release)
    sets = _set_names(db, ops)
    counts = dict(
        db.execute(
            select(ItemPrediction.op_id, func.count())
            .where(ItemPrediction.op_id.in_([op.id for op in ops]))
            .group_by(ItemPrediction.op_id)
        ).all()
    )
    return [{**run_ref(db, op, sets), "images": counts.get(op.id, 0)} for op in ops]


def item_prediction_runs(db: Session, release: Release, item_id: int) -> list[dict[str, Any]]:
    """Every run that predicted this image, newest first, each with what it predicted (in the version's label form)."""
    ops = prediction_runs(db, release)
    rows = {
        row.op_id: row
        for row in db.scalars(
            select(ItemPrediction).where(ItemPrediction.item_id == item_id, ItemPrediction.op_id.in_([op.id for op in ops]))
        )
    }
    sets = _set_names(db, [op for op in ops if op.id in rows])
    out = []
    for op in ops:
        row = rows.get(op.id)
        if row is None:
            continue
        run = {**run_ref(db, op, sets), "labels": row.labels or [], "scores": row.scores or []}
        if op.kind == "eval":
            run.update(found=row.found, missed=row.missed, extra=row.extra, reclass=row.reclass)
        out.append(run)
    return out


def page_predictions(db: Session, release: Release, op_id: str, item_ids: list[int], limit: int) -> dict[int, dict]:
    """One run's predictions for a page of images (the grid lays them over the thumbnails)."""
    if op_id not in {op.id for op in prediction_runs(db, release)}:
        raise ServiceError(404, "이 버전의 이미지에 겹쳐 볼 수 있는 예측이 아닙니다.")
    rows = db.scalars(select(ItemPrediction).where(ItemPrediction.op_id == op_id, ItemPrediction.item_id.in_(item_ids)))
    return {r.item_id: {"labels": (r.labels or [])[:limit], "scores": (r.scores or [])[:limit]} for r in rows}


def model_ref(model: ModelVersion | None) -> dict[str, Any] | None:
    """The model an evaluation is by, as the 검수 screen names it."""
    if model is None:
        return None
    aliases = list(model.aliases or ([model.alias] if model.alias else []))
    return {"id": model.id, "name": model.name, "version": model.version, "task": model.task or "detect", "aliases": aliases}


def release_checks(db: Session, release: Release) -> list[dict[str, Any]]:
    """The evaluations that can sort this release's images by likely wrong labels."""
    out = []
    for op in _offered(db, release):
        mine = (ItemPrediction.op_id == op.id, ReleaseItem.release_id == release.id)
        base = (
            select(func.count()).select_from(ItemPrediction).join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
        )
        images = db.scalar(base.where(*mine)) or 0
        if not images:
            continue
        wrong = db.scalar(base.where(*mine, ItemPrediction.wrong > 0)) or 0
        model = db.get(ModelVersion, op.model_id)
        evaluated = db.get(Release, op.release_id)
        out.append(
            {
                "id": op.id,
                "model": model_ref(model),
                "release": {"id": evaluated.id, "version": evaluated.version} if evaluated else None,
                "split": (op.params or {}).get("split", ""),
                "created_at": iso(op.created_at),
                "images": images,
                "wrong": wrong,
            }
        )
    return out


def newest(db: Session, release: Release) -> dict[str, Any] | None:
    """The newest evaluation that measured this release's images: how many it found objects no label
    has on (``extra``) and labels it did not find on (``missed``)."""
    for op in _offered(db, release):
        mine = (ItemPrediction.op_id == op.id, ReleaseItem.release_id == release.id)
        base = (
            select(func.count()).select_from(ItemPrediction).join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
        )
        images = db.scalar(base.where(*mine)) or 0
        if not images:
            continue
        model = db.get(ModelVersion, op.model_id)
        return {
            "id": op.id,
            "model": model_ref(model),
            "created_at": iso(op.created_at),
            "images": images,
            **{
                kind: db.scalar(
                    base.where(
                        *mine,
                        getattr(ItemPrediction, kind) > 0,
                        ~let_be(kind, ItemPrediction.item_id, ReleaseItem.revision_id),
                    )
                )
                or 0
                for kind in MISTAKES
            },
        }
    return None


def pending(db: Session, release: Release) -> list[dict[str, Any]]:
    """Evaluations of this dataset's versions still waiting or running: their view comes when they end."""
    siblings = select(Release.id).where(
        Release.dataset_id == release.dataset_id, Release.label_space_id == release.label_space_id
    )
    rows = db.execute(
        select(ModelJob, Job.status)
        .join(Job, Job.id == ModelJob.job_id)
        .where(ModelJob.kind == "eval", ModelJob.release_id.in_(siblings), Job.status.in_(("queued", "running")))
        .order_by(ModelJob.created_at.desc())
    ).all()
    return [
        {
            "id": op.id,
            "job_id": op.job_id,
            "status": status,
            "split": (op.params or {}).get("split", ""),
            "model": model_ref(db.get(ModelVersion, op.model_id)),
            "created_at": iso(op.created_at),
        }
        for op, status in rows
    ]


HISTORY = 30  # the model checks 검수 lists for a dataset


def history(db: Session, settings: Any, release: Release, limit: int = HISTORY) -> list[dict[str, Any]]:
    """검수's record of the model checks on this dataset's versions (those with the same classes), newest first.
    Like a training's, each stays after it ends: how it went and where, how long, who started it, the images it had
    sent to that server first, and why it failed — the Jobs screen shows it only briefly. The log is the job's."""
    from . import modelops, services
    from .models import User

    siblings = select(Release.id).where(
        Release.dataset_id == release.dataset_id, Release.label_space_id == release.label_space_id
    )
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.release_id.in_(siblings))
        .order_by(ModelJob.created_at.desc())
        .limit(limit)
    ).all()
    names = services.node_name_map(db)
    users = {
        u.id: u.username for u in db.scalars(select(User).where(User.id.in_({op.created_by for op in ops if op.created_by})))
    }
    basis = {op.id for op in _offered(db, release)}  # usable as 검수's model basis (it kept per-image predictions)
    out = []
    for op in ops:
        view = modelops.op_view(db, settings, op, names)
        job = db.get(Job, op.job_id) if op.job_id else None
        sent = None
        after = (job.requirements or {}).get("after") if job else None
        before = db.get(Job, after) if after else None
        if before is not None and before.type == "data.sync":  # the images went to that server first
            shown = services.job_view(db, settings, before, names)
            sent = {
                "job_id": before.id,
                "status": shown["status"],
                "error": shown["error"],
                "node_name": shown["node_name"] or names.get(before.target_node_id or "", ""),
                "duration_s": shown["duration_s"],
            }
        out.append(
            {
                "id": op.id,
                "job_id": op.job_id,
                "status": view["status"],
                "error": view["error"],
                "wait_reason": view["wait_reason"],
                "node_name": view["node_name"] or (names.get(job.target_node_id or "", "") if job else ""),
                "duration_s": view["duration_s"],
                "created_at": view["created_at"],
                "finished_at": iso(job.finished_at) if job else None,
                "log_count": job.log_count if job else 0,
                "by": users.get(op.created_by or "", ""),
                "model": model_ref(db.get(ModelVersion, op.model_id)),
                "release": view["release"],
                "splits": [str(s) for s in (op.params or {}).get("splits") or [(op.params or {}).get("split") or "val"]],
                "code": view["code"],
                "images": view["predicted"],
                "basis": op.id in basis,
                "sent": sent,
            }
        )
    return out


def check_of(db: Session, release: Release, op_id: str) -> ModelJob:
    op = db.get(ModelJob, op_id)
    if op is None or op.kind != "eval" or op not in _offered(db, release):
        raise ServiceError(404, "이 버전에 쓸 수 있는 평가가 아닙니다.")
    return op


def item_predictions(db: Session, release: Release, item_id: int, op_id: str) -> dict[str, Any]:
    op = check_of(db, release, op_id)
    row = db.get(ItemPrediction, (op.id, item_id))
    model = db.get(ModelVersion, op.model_id)
    return {
        "op_id": op.id,
        "model": model_ref(model),
        "split": (op.params or {}).get("split", ""),
        "evaluated": row is not None,
        "labels": row.labels if row else [],
        "scores": row.scores if row else [],
        "found": row.found if row else 0,
        "missed": row.missed if row else 0,
        "extra": row.extra if row else 0,
        "reclass": row.reclass if row else 0,
    }
