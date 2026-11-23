"""Label checks (검수): images whose labels are probably missing (미라벨) or wrong (오라벨).

Two kinds of signal:

- Rules on the labels alone, no model needed: an image nobody labelled, the same object boxed twice,
  one object boxed with two classes, a box too small to be anything, a box covering the whole image, a
  class that appears only once or twice (often a slip of the class list).
- What a trained model predicted when it was evaluated (``checks``): objects it found that no label
  has (likely missing), and labels it did not find (likely wrong).

Rule results are kept per release with a fingerprint of its labels, and worked out again when the
labels change (the work in progress, as people fix things).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from sqlalchemy import and_, case, delete, func, insert, select, text
from sqlalchemy.orm import Session

from . import catalog, checks
from .catalog_models import (
    Dataset,
    ItemPrediction,
    LabelSpace,
    QaDismissal,
    QaResult,
    QaRun,
    QaScore,
    Release,
    ReleaseItem,
    RevisionClass,
)
from .db import iso, utcnow
from .models import User
from .services import ServiceError, audit

# What each check finds, grouped as the screen shows them.
CHECKS: dict[str, dict[str, str]] = {
    "unlabelled": {"group": "missing", "title": "라벨 안 한 이미지", "hint": "아무도 라벨하지 않은 이미지(아직 안 함)"},
    "duplicate": {"group": "boxes", "title": "같은 물체에 박스 둘", "hint": "같은 클래스 박스가 거의 겹침(IoU 0.95 이상)"},
    "class_overlap": {
        "group": "wrong",
        "title": "한 물체에 클래스 둘",
        "hint": "다른 클래스 박스가 거의 겹침: 둘 중 하나가 틀렸을 수 있음",
    },
    "rare_class": {
        "group": "wrong",
        "title": "드문 클래스",
        "hint": "데이터셋 전체에 두 번 이하로 나오는 클래스: 잘못 고른 클래스일 수 있음",
    },
    "tiny": {"group": "boxes", "title": "아주 작은 박스", "hint": "가로나 세로가 3픽셀 미만"},
    "huge": {"group": "boxes", "title": "이미지를 거의 덮는 박스", "hint": "이미지 넓이의 98% 이상"},
}
RARE = 2
CHUNK = 2000
RULES = "3"  # bumped when what the checks find or how they word it changes: results are worked out again

# The box checks a threshold decides. ``QaResult`` keeps what they find at ``DEFAULTS`` (boards, labeling sets,
# the review record); ``QaScore`` keeps each image's value from the ``LOOSE`` bound up, so 검수 counts at any
# threshold it asks for (within ``BOUNDS``) and shows the counts at ``LEVELS`` beside.
DEFAULTS: dict[str, float] = {"duplicate": 0.95, "class_overlap": 0.85, "tiny": 3, "huge": 0.98}
LOOSE: dict[str, float] = {"duplicate": 0.5, "class_overlap": 0.5, "tiny": 16, "huge": 0.8}
BOUNDS: dict[str, tuple[float, float]] = {
    "duplicate": (0.5, 0.99),
    "class_overlap": (0.5, 0.99),
    "tiny": (1, 16),
    "huge": (0.8, 0.999),
}
LEVELS: dict[str, tuple[float, ...]] = {
    "duplicate": (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    "class_overlap": (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    "tiny": (1, 2, 3, 5, 8, 12, 16),
    "huge": (0.8, 0.85, 0.9, 0.95, 0.98, 0.99),
}


# a box check's image and its labels in the release
_SCORED_ITEM = and_(ReleaseItem.release_id == QaScore.release_id, ReleaseItem.item_id == QaScore.item_id)


def limits(text: str = "") -> dict[str, float]:
    """The box checks' thresholds a screen asks for (``duplicate:0.9,tiny:5``), the defaults for the rest."""
    out = dict(DEFAULTS)
    for part in (text or "").split(","):
        check, _, value = part.partition(":")
        if check.strip() in DEFAULTS:
            try:
                low, high = BOUNDS[check.strip()]
                out[check.strip()] = min(high, max(low, float(value)))
            except ValueError:
                continue
    return out


def passes(check: str, value: float, limit: float) -> bool:
    """A label's value counts at this threshold: shorter than it (tiny), at or over it (the others)."""
    return value < limit if check == "tiny" else value >= limit


def _within(check: str, limit: float) -> Any:
    return QaScore.score < limit if check == "tiny" else QaScore.score >= limit


def hint_of(check: str, limit: float) -> str:
    """What a check finds, worded with its threshold."""
    if check == "duplicate":
        return f"같은 클래스 박스가 거의 겹침(IoU {limit:g} 이상)"
    if check == "class_overlap":
        return f"다른 클래스 박스가 거의 겹침(IoU {limit:g} 이상): 둘 중 하나가 틀렸을 수 있음"
    if check == "tiny":
        return f"가로나 세로가 {limit:g}픽셀 미만"
    if check == "huge":
        return f"이미지 넓이의 {limit * 100:g}% 이상"
    return CHECKS[check]["hint"]


def _iou(a: list[float], b: list[float]) -> float:
    _, ax, ay, aw, ah = a[:5]
    _, bx, by, bw, bh = b[:5]
    w = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    h = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / (aw * ah + bw * bh - w * h)


def _overlapping(geometric: list[tuple[int, list[float]]], at: float = 0.5) -> list[tuple[int, int, float]]:
    """Pairs of boxes (their label indices, in label order, and their IoU) that overlap by ``at`` (IoU) or more.

    The boxes are swept along x: once a box starts right of another's right edge, no later one can overlap it. And
    an IoU of ``at`` needs the smaller area to be at least ``at`` of the larger (the IoU is at most their ratio), so
    most pairs are passed over before any IoU. Comparing every pair took 34 s for 684,049 images."""
    if len(geometric) < 2:
        return []
    spans = sorted(
        (b[1] - b[3] / 2, b[1] + b[3] / 2, b[2] - b[4] / 2, b[2] + b[4] / 2, b[3] * b[4], i) for i, b in geometric
    )
    found = []
    for n, (x0, x1, y0, y1, area, i) in enumerate(spans):
        for u0, u1, v0, v1, other, j in spans[n + 1 :]:
            if u0 >= x1:
                break
            small, big = (area, other) if area <= other else (other, area)
            if big <= 0 or small < at * big:
                continue
            w = min(x1, u1) - max(x0, u0)
            h = min(y1, v1) - max(y0, v0)
            if w <= 0 or h <= 0:
                continue
            inter = w * h
            if (iou := inter / (area + other - inter)) >= at:
                found.append((*((i, j) if i < j else (j, i)), round(iou, 4)))
    return sorted(found)


def _measure(boxes: list[list[float]], width: int, height: int, task: str) -> dict[str, dict[int, tuple[float, int]]]:
    """Each box check's value per label from its loose bound up: {check: {label index: (value, partner or -1)}} — the
    shortest side in pixels (tiny), the share of the image (huge), the most it overlaps a box of its class
    (duplicate) or of another (class_overlap), and with which."""
    out: dict[str, dict[int, tuple[float, int]]] = {check: {} for check in DEFAULTS}
    if task == "classify":
        return out
    geometric = [(i, b) for i, b in enumerate(boxes) if len(b) >= 5]
    for i, b in geometric:
        side = min(b[3] * width, b[4] * height)
        if side < LOOSE["tiny"]:
            out["tiny"][i] = (round(side, 2), -1)
        if (share := b[3] * b[4]) >= LOOSE["huge"]:
            out["huge"][i] = (round(share, 4), -1)
    for i, j, iou in _overlapping(geometric, min(LOOSE["duplicate"], LOOSE["class_overlap"])):
        kind = "duplicate" if int(boxes[i][0]) == int(boxes[j][0]) else "class_overlap"
        for one, two in ((i, j), (j, i)):
            if one not in out[kind] or iou > out[kind][one][0]:
                out[kind][one] = (iou, two)
    return out


def _fingerprint(db: Session, release: Release) -> str:
    count, total = db.execute(
        select(func.count(), func.coalesce(func.sum(ReleaseItem.revision_id), 0)).where(ReleaseItem.release_id == release.id)
    ).one()
    return hashlib.sha1(f"{count}:{total}:{release.label_space_id}:{RULES}".encode()).hexdigest()


def _flags(
    boxes: list[list[float]],
    width: int,
    height: int,
    rare: set[int],
    task: str,
    names: dict[int, str] | None = None,
    measured: dict[str, dict[int, tuple[float, int]]] | None = None,
    at: dict[str, float] | None = None,
) -> dict[str, tuple[list[int], str]]:
    """The checks one image fails: {check: (label indices, note)}, the box checks at ``at`` (the defaults). The
    note names the classes involved ("car 2개", "person ↔ bicycle 겹침"), so a list of images says which labels
    to look at."""
    names = names or {}
    at = at or DEFAULTS
    measured = measured if measured is not None else _measure(boxes, width, height, task)

    def name(i: int) -> str:
        return str(names.get(int(boxes[i][0]), int(boxes[i][0])))

    def named(indices: list[int]) -> str:
        counts: dict[str, int] = {}
        for i in indices:
            counts[name(i)] = counts.get(name(i), 0) + 1
        return ", ".join(f"{n} {c}개" for n, c in counts.items())

    out: dict[str, tuple[list[int], str]] = {}
    for check in DEFAULTS:
        hit = sorted(i for i, (value, _) in measured[check].items() if passes(check, value, at[check]))
        if not hit:
            continue
        if check == "class_overlap":
            pairs = dict.fromkeys(" ↔ ".join(sorted((name(i), name(measured[check][i][1])))) for i in hit)
            out[check] = (hit, f"{' · '.join(pairs)} 겹침")
        else:
            out[check] = (hit, f"{named(hit)} 겹침" if check == "duplicate" else named(hit))
    odd = [i for i, b in enumerate(boxes) if b and int(b[0]) in rare]
    if odd:
        out["rare_class"] = (odd, ", ".join(dict.fromkeys(name(i) for i in odd)))
    return out


def run(db: Session, release: Release) -> QaRun:
    """Run the rule checks on the release's labels (again only when they changed)."""
    fingerprint = _fingerprint(db, release)
    done = db.get(QaRun, release.id)
    if done is not None and done.fingerprint == fingerprint:
        return done
    task = catalog.release_task(db, release)
    space = db.get(LabelSpace, release.label_space_id)
    names = {int(c["id"]): c["name"] for c in (space.classes if space else [])}
    # how often each class occurs: the per-class counts every label record keeps (no label read for it)
    per_class = dict(
        db.execute(
            select(RevisionClass.class_id, func.sum(RevisionClass.count))
            .join(ReleaseItem, ReleaseItem.revision_id == RevisionClass.revision_id)
            .where(ReleaseItem.release_id == release.id)
            .group_by(RevisionClass.class_id)
        ).all()
    )
    rare = {int(cid) for cid, n in per_class.items() if (n or 0) <= RARE}
    if len(per_class) < 3:  # a dataset of one or two classes has no "rare" one
        rare = set()

    # Everything is read and judged first, and only then written: the database's write lock is held for the
    # writing alone (deleting the old results first held it for the whole check, 55 s on 684,049 images, and
    # every other write of the Hub waited and failed: "database is locked").
    counts = Counter()
    # images nobody labelled: no labels to read
    waiting = db.scalars(
        select(ReleaseItem.item_id).where(ReleaseItem.release_id == release.id, ReleaseItem.state == "unlabelled")
    ).all()
    counts["unlabelled"] = len(waiting)
    flagged: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []  # the box checks' values, for 검수's thresholds
    # the images with labels, in order of their number, each label record read once as text and parsed here
    # (faster than SQLAlchemy turning each into a Python value); images without labels fail no check
    last = 0
    while True:
        rows = db.execute(
            text(
                "SELECT ri.item_id, lr.boxes, i.width, i.height FROM release_items ri "
                "JOIN label_revisions lr ON lr.id = ri.revision_id JOIN items i ON i.id = ri.item_id "
                "WHERE ri.release_id = :release AND ri.box_count > 0 AND ri.state != 'unlabelled' AND ri.item_id > :last "
                "ORDER BY ri.item_id LIMIT :chunk"
            ),
            {"release": release.id, "last": last, "chunk": CHUNK * 5},
        ).all()
        if not rows:
            break
        for item_id, raw, width, height in rows:
            boxes = json.loads(raw) if isinstance(raw, str) else raw
            measured = _measure(boxes or [], width or 1, height or 1, task)
            for check, (indices, note) in _flags(boxes or [], width or 1, height or 1, rare, task, names, measured).items():
                row = {"release_id": release.id, "check": check, "item_id": item_id, "boxes": indices, "note": note[:255]}
                flagged.append(row)
                counts[check] += 1
            for check, per_label in measured.items():
                if per_label:
                    values = [value for value, _ in per_label.values()]
                    scored.append(
                        {
                            "release_id": release.id,
                            "check": check,
                            "item_id": item_id,
                            "score": min(values) if check == "tiny" else max(values),
                            "labels": [[i, value, partner] for i, (value, partner) in sorted(per_label.items())],
                        }
                    )
        last = rows[-1][0]
    db.execute(delete(QaResult).where(QaResult.release_id == release.id))
    db.execute(delete(QaScore).where(QaScore.release_id == release.id))
    for start in range(0, len(scored), CHUNK):
        db.execute(insert(QaScore), scored[start : start + CHUNK])
    for start in range(0, len(waiting), CHUNK):
        part = waiting[start : start + CHUNK]
        db.execute(
            insert(QaResult),
            [{"release_id": release.id, "check": "unlabelled", "item_id": i, "boxes": [], "note": ""} for i in part],
        )
    for start in range(0, len(flagged), CHUNK):
        db.execute(insert(QaResult), flagged[start : start + CHUNK])
    if done is None:
        done = QaRun(release_id=release.id)
        db.add(done)
    done.fingerprint = fingerprint
    done.counts = {check: counts[check] for check in CHECKS}
    done.computed_at = utcnow()
    db.flush()
    return done


def _in_release(release_id: str) -> Any:
    """QaResult rows with the release's own labels of the image (whose revision a judgement is kept for)."""
    return (
        select(QaResult)
        .join(ReleaseItem, and_(ReleaseItem.release_id == QaResult.release_id, ReleaseItem.item_id == QaResult.item_id))
        .where(QaResult.release_id == release_id)
    )


def check_counts(db: Session, release_id: str) -> dict[str, tuple[int, int]]:
    """Per check: (images still flagged, images someone let be)."""
    judged = checks.let_be(QaResult.check, QaResult.item_id, ReleaseItem.revision_id)
    rows = db.execute(
        _in_release(release_id)
        .with_only_columns(QaResult.check, func.count(), func.sum(case((judged, 1), else_=0)))
        .group_by(QaResult.check)
    ).all()
    return {check: (total - (gone or 0), gone or 0) for check, total, gone in rows}


def _scored_counts(db: Session, release_id: str, at: dict[str, float]) -> dict[str, dict[str, Any]]:
    """The box checks at these thresholds: images still flagged, images let be, and the count at each level."""
    out = {}
    for check, limit in at.items():
        judged = checks.let_be(check, QaScore.item_id, ReleaseItem.revision_id)
        base = (
            select(QaScore.item_id)
            .join(ReleaseItem, _SCORED_ITEM)
            .where(QaScore.release_id == release_id, QaScore.check == check)
        )
        flagged = base.where(_within(check, limit))
        steps = LEVELS[check]
        row = db.execute(
            select(*[func.coalesce(func.sum(case((_within(check, v), 1), else_=0)), 0) for v in steps])
            .select_from(QaScore)
            .join(ReleaseItem, _SCORED_ITEM)
            .where(QaScore.release_id == release_id, QaScore.check == check, ~judged)
        ).one()
        out[check] = {
            "images": db.scalar(select(func.count()).select_from(flagged.where(~judged).subquery())) or 0,
            "let_be": db.scalar(select(func.count()).select_from(flagged.where(judged).subquery())) or 0,
            "limit": limit,
            "default": DEFAULTS[check],
            "hint": hint_of(check, limit),
            "levels": [[v, int(n or 0)] for v, n in zip(steps, row, strict=True)],
        }
    return out


def summary(
    db: Session, release: Release, counting: dict[str, float] | None = None, at: dict[str, float] | None = None
) -> dict[str, Any]:
    """The checks of a release: how many images each found (and how many were let be), and the evaluations that
    can add the model's view: counted at ``counting`` (FiftyOne's thresholds unless asked for others), and at each of
    ``checks.levels`` beside, so the screen shows how the counts move with the threshold."""
    counting = counting or checks.rule()
    at = at or dict(DEFAULTS)
    done = run(db, release)
    counts = check_counts(db, release.id)
    scored = _scored_counts(db, release.id, at)  # the box checks at the thresholds asked for
    evaluations = []
    for check in checks.release_checks(db, release):
        mine = (ItemPrediction.op_id == check["id"], ReleaseItem.release_id == release.id)
        base = (
            select(func.count()).select_from(ItemPrediction).join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
        )
        found: dict[str, Any] = {"levels": {}}
        for kind in checks.MISTAKES:  # extra: objects no label has; reclass: another class; missed: labels untouched
            judged = checks.let_be(kind, ItemPrediction.item_id, ReleaseItem.revision_id)
            flagged = base.where(*mine, checks.over(kind, counting))
            found[kind] = db.scalar(flagged.where(~judged)) or 0
            found[f"{kind}_let_be"] = db.scalar(flagged.where(judged)) or 0
            steps = checks.levels(kind)
            at = {**counting, "conf": 0.0, "min_area": 0.0}
            row = db.execute(
                select(*[func.sum(case((checks.over(kind, {**at, "conf": v, "min_area": v}), 1), else_=0)) for v in steps])
                .select_from(ItemPrediction)
                .join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
                .where(*mine, ~judged)
            ).one()
            found["levels"][kind] = [[v, int(n or 0)] for v, n in zip(steps, row, strict=True)]
        evaluations.append({**check, **found})
    return {
        "release_id": release.id,
        "computed_at": iso(done.computed_at),
        "checks": [
            {
                "id": cid,
                **meta,
                "images": counts.get(cid, (0, 0))[0],
                "let_be": counts.get(cid, (0, 0))[1],
                **scored.get(cid, {}),  # a box check: at the threshold asked for, with its levels
            }
            for cid, meta in CHECKS.items()
        ],
        "evaluations": evaluations,
        "pending": checks.pending(db, release),  # evaluations still going: the model's view comes when they end
        "rule": {**counting, "defaults": checks.DEFAULT_RULE},  # the thresholds counted at, and FiftyOne's
    }


def _revision_boxes(db: Session, revision_ids: set[str]) -> dict[str, list[list[float]]]:
    from .catalog_models import LabelRevision

    found: dict[str, list[list[float]]] = {}
    ids = list(revision_ids)
    for start in range(0, len(ids), 500):
        rows = db.execute(
            select(LabelRevision.id, LabelRevision.boxes).where(LabelRevision.id.in_(ids[start : start + 500]))
        )
        found.update({rid: boxes or [] for rid, boxes in rows})
    return found


def scored_items(release_id: str, check: str, limit: float) -> Any:
    """SQL: the images a box check finds at this threshold."""
    return select(QaScore.item_id).where(QaScore.release_id == release_id, QaScore.check == check, _within(check, limit))


def _at_limit(labels: list[list[float]], check: str, limit: float) -> list[int]:
    return sorted(int(i) for i, value, _ in labels if passes(check, value, limit))


def of_classes(
    db: Session, release_id: str, check: str, classes: set[int], at: dict[str, float] | None = None
) -> dict[int, list[int]]:
    """The images a rule check flagged for a label of one of ``classes``, each with those labels (indices). An
    image flagged with no label (nobody labelled it) has no class, so it is never one of them. A box check at the
    threshold in ``at``."""
    if check in DEFAULTS:
        limit = (at or DEFAULTS)[check]
        rows = [
            (item_id, _at_limit(labels or [], check, limit), revision_id)
            for item_id, labels, revision_id in db.execute(
                select(QaScore.item_id, QaScore.labels, ReleaseItem.revision_id)
                .join(ReleaseItem, _SCORED_ITEM)
                .where(QaScore.release_id == release_id, QaScore.check == check, _within(check, limit))
            )
        ]
    else:
        rows = db.execute(
            select(QaResult.item_id, QaResult.boxes, ReleaseItem.revision_id)
            .join(ReleaseItem, and_(ReleaseItem.release_id == QaResult.release_id, ReleaseItem.item_id == QaResult.item_id))
            .where(QaResult.release_id == release_id, QaResult.check == check)
        ).all()
    revisions = _revision_boxes(db, {rid for _, _, rid in rows if rid})
    found: dict[int, list[int]] = {}
    for item_id, flagged_boxes, revision_id in rows:
        boxes = revisions.get(revision_id, [])
        mine = [i for i in flagged_boxes or [] if 0 <= i < len(boxes) and boxes[i] and int(boxes[i][0]) in classes]
        if mine:
            found[item_id] = mine
    return found


def flagged(
    db: Session, release_id: str, check: str, item_ids: list[int], at: dict[str, float] | None = None
) -> dict[int, dict[str, Any]]:
    """For a page of images: which labels the check flagged and its note (a box check at the threshold in ``at``)."""
    if not item_ids:
        return {}
    if check in DEFAULTS and at is not None and at[check] != DEFAULTS[check]:
        release = db.get(Release, release_id)
        space = db.get(LabelSpace, release.label_space_id) if release else None
        names = {int(c["id"]): c["name"] for c in (space.classes if space else [])}
        rows = db.execute(
            select(QaScore.item_id, QaScore.labels, ReleaseItem.revision_id)
            .join(ReleaseItem, _SCORED_ITEM)
            .where(QaScore.release_id == release_id, QaScore.check == check, QaScore.item_id.in_(item_ids))
        ).all()
        revisions = _revision_boxes(db, {rid for *_, rid in rows if rid})
        out = {}
        for item_id, labels, revision_id in rows:
            boxes = revisions.get(revision_id, [])
            measured = {c: {} for c in DEFAULTS}
            measured[check] = {int(i): (value, int(partner)) for i, value, partner in labels or []}
            found = _flags(boxes, 1, 1, set(), "detect", names, measured, at).get(check)
            if found:
                out[item_id] = {"check": check, "boxes": found[0], "note": found[1][:255]}
        return out
    rows = db.scalars(
        select(QaResult).where(QaResult.release_id == release_id, QaResult.check == check, QaResult.item_id.in_(item_ids))
    )
    return {r.item_id: {"check": check, "boxes": r.boxes, "note": r.note} for r in rows}


def is_fresh(db: Session, release: Release) -> bool:
    """The rule checks have run on the release's labels as they are now."""
    done = db.get(QaRun, release.id)
    return done is not None and done.fingerprint == _fingerprint(db, release)


def groups(db: Session, release_id: str) -> dict[str, int]:
    """Images the rule checks flag, by group (an image flagged twice counts once), and in all; what someone
    let be ("문제 없음") is left out."""
    distinct = (
        _in_release(release_id)
        .with_only_columns(func.count(func.distinct(QaResult.item_id)))
        .where(~checks.let_be(QaResult.check, QaResult.item_id, ReleaseItem.revision_id))
    )
    out = {}
    for group in ("missing", "wrong", "boxes"):
        ids = [cid for cid, meta in CHECKS.items() if meta["group"] == group]
        out[group] = db.scalar(distinct.where(QaResult.check.in_(ids))) or 0
    out["any"] = db.scalar(distinct) or 0
    return out


def judgements(db: Session, release_id: str, check: str, item_ids: list[int]) -> dict[int, dict[str, Any]]:
    """For a page of images: the "문제 없음" kept for this check on the release's labels of each."""
    if not item_ids:
        return {}
    rows = db.execute(
        select(QaDismissal, User.username)
        .join(
            ReleaseItem,
            and_(ReleaseItem.item_id == QaDismissal.item_id, ReleaseItem.revision_id == QaDismissal.revision_id),
        )
        .outerjoin(User, User.id == QaDismissal.created_by)
        .where(ReleaseItem.release_id == release_id, QaDismissal.check == check, QaDismissal.item_id.in_(item_ids))
    ).all()
    return {d.item_id: {"note": d.note, "by": name, "at": iso(d.created_at)} for d, name in rows}


def let_be(db: Session, user: User, release: Release, item_id: int, check: str, note: str = "", undo: bool = False) -> dict:
    """Judge what a check flagged on this image right ("문제 없음"), or take that back. Kept for the image's labels
    as they are in this release: the next version keeps it while the labels stay the same."""
    if check not in CHECKS and check not in checks.MISTAKES:
        raise ServiceError(400, "그런 검사가 없습니다.")
    member = db.get(ReleaseItem, (release.id, item_id))
    if member is None:
        raise ServiceError(404, "이 버전에 없는 이미지입니다.")
    found = db.get(QaDismissal, (item_id, check, member.revision_id))
    if undo:
        if found is not None:
            db.delete(found)
            audit(db, user.id, "ui", "qa.undo", release.id, item=item_id, check=check)
        db.flush()
        return {"item_id": item_id, "check": check, "let_be": None}
    dataset = db.get(Dataset, release.dataset_id)
    if found is None:
        found = QaDismissal(item_id=item_id, check=check, revision_id=member.revision_id, dataset_id=dataset.id)
        db.add(found)
    found.note = note.strip()[:255]
    found.created_by, found.created_at = user.id, utcnow()
    audit(db, user.id, "ui", "qa.let_be", release.id, item=item_id, check=check)
    db.flush()
    judged = {"note": found.note, "by": user.username, "at": iso(found.created_at)}
    return {"item_id": item_id, "check": check, "let_be": judged}


def fill_runs(db: Session, limit: int = 1) -> int:
    """Label checks of finished versions run ahead (the Hub's background loop, one at a time): the 검수 tab of even
    a large version then opens at once. A finished version never changes; one checked under older rules is checked
    again when it is opened."""
    from .catalog_models import Release as Version

    waiting = db.scalars(
        select(Version)
        .where(Version.status == "ready", Version.deleted_at.is_(None))
        .where(~select(QaRun.release_id).where(QaRun.release_id == Version.id).exists())
        .order_by(Version.created_at.desc())
        .limit(limit)
    ).all()
    for release in waiting:
        run(db, release)
        db.commit()
    return len(waiting)
