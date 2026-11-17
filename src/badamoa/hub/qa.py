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
from collections import Counter
from typing import Any

from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from . import catalog, checks
from .catalog_models import Item, ItemPrediction, LabelRevision, LabelSpace, QaResult, QaRun, Release, ReleaseItem
from .db import iso, utcnow

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


def _iou(a: list[float], b: list[float]) -> float:
    _, ax, ay, aw, ah = a[:5]
    _, bx, by, bw, bh = b[:5]
    w = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    h = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / (aw * ah + bw * bh - w * h)


def _fingerprint(db: Session, release: Release) -> str:
    count, total = db.execute(
        select(func.count(), func.coalesce(func.sum(ReleaseItem.revision_id), 0)).where(
            ReleaseItem.release_id == release.id
        )
    ).one()
    return hashlib.sha1(f"{count}:{total}:{release.label_space_id}".encode()).hexdigest()


def _flags(boxes: list[list[float]], width: int, height: int, rare: set[int], task: str) -> dict[str, tuple[list[int], str]]:
    """The checks one image fails: {check: (label indices, note)}."""
    out: dict[str, tuple[list[int], str]] = {}
    if task != "classify":
        geometric = [(i, b) for i, b in enumerate(boxes) if len(b) >= 5]
        tiny = [i for i, b in geometric if b[3] * width < TINY_PX or b[4] * height < TINY_PX]
        if tiny:
            out["tiny"] = (tiny, f"{len(tiny)}개")
        huge = [i for i, b in geometric if b[3] * b[4] >= HUGE]
        if huge:
            out["huge"] = (huge, f"{len(huge)}개")
        same, other = set(), set()
        for n, (i, a) in enumerate(geometric):
            for j, b in geometric[n + 1 :]:
                if _iou(a, b) >= OVERLAP:
                    (same if int(a[0]) == int(b[0]) else other).update((i, j))
        if same:
            out["duplicate"] = (sorted(same), f"{len(same)}개가 겹침")
        if other:
            out["class_overlap"] = (sorted(other), f"{len(other)}개가 겹침")
    odd = [i for i, b in enumerate(boxes) if b and int(b[0]) in rare]
    if odd:
        out["rare_class"] = (odd, "")
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
    per_class = Counter()
    for (boxes,) in db.execute(
        select(LabelRevision.boxes)
        .join(ReleaseItem, ReleaseItem.revision_id == LabelRevision.id)
        .where(ReleaseItem.release_id == release.id, ReleaseItem.box_count > 0)
    ):
        per_class.update(int(b[0]) for b in boxes or [] if b)
    rare = {cid for cid, n in per_class.items() if n <= RARE}
    if len(per_class) < 3:  # a dataset of one or two classes has no "rare" one
        rare = set()

    db.execute(delete(QaResult).where(QaResult.release_id == release.id))
    counts = Counter()
    last = 0
    while True:
        rows = db.execute(
            select(ReleaseItem.item_id, ReleaseItem.state, LabelRevision.boxes, Item.width, Item.height)
            .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id)
            .join(Item, Item.id == ReleaseItem.item_id)
            .where(ReleaseItem.release_id == release.id, ReleaseItem.item_id > last)
            .order_by(ReleaseItem.item_id)
            .limit(CHUNK)
        ).all()
        if not rows:
            break
        found = []
        for item_id, state, boxes, width, height in rows:
            if state == "unlabelled":
                found.append({"release_id": release.id, "check": "unlabelled", "item_id": item_id, "boxes": [], "note": ""})
                continue
            for check, (indices, note) in _flags(boxes or [], width or 1, height or 1, rare, task).items():
                if check == "rare_class":
                    note = ", ".join(sorted({names.get(int(boxes[i][0]), str(boxes[i][0])) for i in indices}))
                row = {"release_id": release.id, "check": check, "item_id": item_id, "boxes": indices, "note": note[:255]}
                found.append(row)
        if found:
            db.execute(insert(QaResult), found)
            counts.update(f["check"] for f in found)
        last = rows[-1][0]
    if done is None:
        done = QaRun(release_id=release.id)
        db.add(done)
    done.fingerprint = fingerprint
    done.counts = {check: counts[check] for check in CHECKS}
    done.computed_at = utcnow()
    db.flush()
    return done


def summary(db: Session, release: Release) -> dict[str, Any]:
    """The checks of a release: how many images each found, and the evaluations that can add the model's view."""
    done = run(db, release)
    evaluations = []
    for check in checks.release_checks(db, release):
        mine = (ItemPrediction.op_id == check["id"], ReleaseItem.release_id == release.id)
        base = (
            select(func.count())
            .select_from(ItemPrediction)
            .join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
        )
        evaluations.append(
            {
                **check,
                "extra": db.scalar(base.where(*mine, ItemPrediction.extra > 0)) or 0,  # objects no label has
                "missed": db.scalar(base.where(*mine, ItemPrediction.missed > 0)) or 0,  # labels it did not find
            }
        )
    return {
        "release_id": release.id,
        "computed_at": iso(done.computed_at),
        "checks": [{"id": cid, **meta, "images": done.counts.get(cid, 0)} for cid, meta in CHECKS.items()],
        "evaluations": evaluations,
    }


def flagged(db: Session, release_id: str, check: str, item_ids: list[int]) -> dict[int, dict[str, Any]]:
    """For a page of images: which labels the check flagged and its note."""
    if not item_ids:
        return {}
    rows = db.scalars(
        select(QaResult).where(QaResult.release_id == release_id, QaResult.check == check, QaResult.item_id.in_(item_ids))
    )
    return {r.item_id: {"check": check, "boxes": r.boxes, "note": r.note} for r in rows}
