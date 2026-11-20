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
    "duplicate": {"group": "boxes", "title": "같은 물체에 박스 둘", "hint": "같은 클래스 박스가 거의 겹침(IoU 0.85 이상)"},
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
OVERLAP = 0.85
TINY_PX = 3
HUGE = 0.98
RARE = 2
CHUNK = 2000
RULES = "2"  # bumped when what the checks find or how they word it changes: results are worked out again


def _iou(a: list[float], b: list[float]) -> float:
    _, ax, ay, aw, ah = a[:5]
    _, bx, by, bw, bh = b[:5]
    w = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    h = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / (aw * ah + bw * bh - w * h)


def _overlapping(geometric: list[tuple[int, list[float]]]) -> list[tuple[int, int]]:
    """Pairs of boxes (their label indices, in label order) that overlap by ``OVERLAP`` (IoU) or more.

    The boxes are swept along x: once a box starts right of another's right edge, no later one can overlap it. And
    an IoU of 0.85 needs areas within 15% of each other (the IoU is at most the smaller area over the larger), so
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
            if big <= 0 or small < OVERLAP * big:
                continue
            w = min(x1, u1) - max(x0, u0)
            h = min(y1, v1) - max(y0, v0)
            if w <= 0 or h <= 0:
                continue
            inter = w * h
            if inter / (area + other - inter) >= OVERLAP:
                found.append((i, j) if i < j else (j, i))
    return sorted(found)


def _fingerprint(db: Session, release: Release) -> str:
    count, total = db.execute(
        select(func.count(), func.coalesce(func.sum(ReleaseItem.revision_id), 0)).where(ReleaseItem.release_id == release.id)
    ).one()
    return hashlib.sha1(f"{count}:{total}:{release.label_space_id}:{RULES}".encode()).hexdigest()


def _flags(
    boxes: list[list[float]], width: int, height: int, rare: set[int], task: str, names: dict[int, str] | None = None
) -> dict[str, tuple[list[int], str]]:
    """The checks one image fails: {check: (label indices, note)}. The note names the classes involved
    ("car 2개", "person ↔ bicycle 겹침"), so a list of images says which labels to look at."""
    names = names or {}

    def name(i: int) -> str:
        return str(names.get(int(boxes[i][0]), int(boxes[i][0])))

    def named(indices: list[int]) -> str:
        counts: dict[str, int] = {}
        for i in indices:
            counts[name(i)] = counts.get(name(i), 0) + 1
        return ", ".join(f"{n} {c}개" for n, c in counts.items())

    out: dict[str, tuple[list[int], str]] = {}
    if task != "classify":
        geometric = [(i, b) for i, b in enumerate(boxes) if len(b) >= 5]
        tiny = [i for i, b in geometric if b[3] * width < TINY_PX or b[4] * height < TINY_PX]
        if tiny:
            out["tiny"] = (tiny, named(tiny))
        huge = [i for i, b in geometric if b[3] * b[4] >= HUGE]
        if huge:
            out["huge"] = (huge, named(huge))
        same, other, pairs = set(), set(), {}
        for i, j in _overlapping(geometric):
            if int(boxes[i][0]) == int(boxes[j][0]):
                same.update((i, j))
            else:
                other.update((i, j))
                pairs[" ↔ ".join(sorted((name(i), name(j))))] = True
        if same:
            out["duplicate"] = (sorted(same), f"{named(sorted(same))} 겹침")
        if other:
            out["class_overlap"] = (sorted(other), f"{' · '.join(pairs)} 겹침")
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
            for check, (indices, note) in _flags(boxes or [], width or 1, height or 1, rare, task, names).items():
                row = {"release_id": release.id, "check": check, "item_id": item_id, "boxes": indices, "note": note[:255]}
                flagged.append(row)
                counts[check] += 1
        last = rows[-1][0]
    db.execute(delete(QaResult).where(QaResult.release_id == release.id))
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


def summary(db: Session, release: Release) -> dict[str, Any]:
    """The checks of a release: how many images each found (and how many were let be), and the evaluations that
    can add the model's view."""
    done = run(db, release)
    counts = check_counts(db, release.id)
    evaluations = []
    for check in checks.release_checks(db, release):
        mine = (ItemPrediction.op_id == check["id"], ReleaseItem.release_id == release.id)
        base = (
            select(func.count()).select_from(ItemPrediction).join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
        )
        found = {}
        for kind in checks.MISTAKES:  # extra: objects no label has; missed: labels it did not find
            judged = checks.let_be(kind, ItemPrediction.item_id, ReleaseItem.revision_id)
            flagged = base.where(*mine, getattr(ItemPrediction, kind) > 0)
            found[kind] = db.scalar(flagged.where(~judged)) or 0
            found[f"{kind}_let_be"] = db.scalar(flagged.where(judged)) or 0
        evaluations.append({**check, **found})
    return {
        "release_id": release.id,
        "computed_at": iso(done.computed_at),
        "checks": [
            {"id": cid, **meta, "images": counts.get(cid, (0, 0))[0], "let_be": counts.get(cid, (0, 0))[1]}
            for cid, meta in CHECKS.items()
        ],
        "evaluations": evaluations,
        "pending": checks.pending(db, release),  # evaluations still going: the model's view comes when they end
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


def of_classes(db: Session, release_id: str, check: str, classes: set[int]) -> dict[int, list[int]]:
    """The images a rule check flagged for a label of one of ``classes``, each with those labels (indices). An
    image flagged with no label (nobody labelled it) has no class, so it is never one of them."""
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


def flagged(db: Session, release_id: str, check: str, item_ids: list[int]) -> dict[int, dict[str, Any]]:
    """For a page of images: which labels the check flagged and its note."""
    if not item_ids:
        return {}
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
