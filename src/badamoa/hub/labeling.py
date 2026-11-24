"""Labeling sets: images of a dataset handed to people to label (and review), part by part.

A set is made from a version's images, usually the filtered ones ("아직 안 함"), split into parts of a
fixed size. Each part has someone who labels it and, when the set asks for review, someone who checks
it. Labels are saved like any fix (``drafts.save_labels``): new revisions in the dataset's work in
progress, which becomes a version when confirmed. The set only tracks who does what and how far each
image got: todo → labelled → approved, or rejected (back to the labeller with a note).

The board puts each part in a column: 미배정 (nobody yet), 라벨링, 검수 (all labelled, waiting for the
reviewer), 완료 (all approved; without review, all labelled).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import and_, delete, false, func, insert, select
from sqlalchemy.orm import Session

from . import catalog, checks, drafts, qa, services
from .catalog_models import (
    Dataset,
    ItemPrediction,
    LabelSet,
    LabelSetItem,
    LabelSetPart,
    QaResult,
    Release,
    ReleaseItem,
)
from .db import iso, utcnow
from .models import User
from .services import ServiceError

STATES = ("todo", "labelled", "approved", "rejected")
MAX_ITEMS = 200_000
COLUMNS = ("unassigned", "labeling", "review", "done")


def get_set(db: Session, set_id: str) -> LabelSet:
    found = db.get(LabelSet, set_id)
    if found is None:
        raise ServiceError(404, "라벨링 묶음을 찾을 수 없습니다.")
    return found


def create_set(
    db: Session,
    user: User,
    *,
    release_id: str,
    name: str,
    instructions: str = "",
    match: dict[str, Any] | None = None,
    part_size: int = 50,
    review: bool = True,
    via: str = "ui",
    parts: int = 0,
) -> LabelSet:
    """A set of the release's images that match the version screen's filters (all of them without). ``parts``: split
    into that many parts instead (as even as can be; 1: all of it to one person), whatever the images come to.

    ``match`` also takes ``edited`` (only the images the work in progress changed) and ``suspect`` (an
    evaluation: the images it got wrong, the most mistakes first — a set to check labels that are likely wrong).
    """
    release = catalog.get_release(db, release_id)
    _, why = drafts.edit_place(db, release)
    if why:
        raise ServiceError(409, why.replace("라벨은", "라벨링 묶음은").replace("고칩니다", "만듭니다"))
    name = name.strip()
    if not name:
        raise ServiceError(400, "이름을 적으세요.")
    if part_size < 1:
        raise ServiceError(400, "일감 하나의 이미지 수는 1장 이상입니다.")
    wanted = match or {}
    query = catalog._item_filter(
        release.id,
        wanted.get("split"),
        wanted.get("state") or "",
        catalog.class_ids(wanted.get("class_id")),  # one class or several: the images holding any
        wanted.get("site") or "",
        wanted.get("camera") or "",
        str(wanted.get("q") or "").strip(),
    ).with_only_columns(ReleaseItem.item_id)
    order = [ReleaseItem.item_id]
    if wanted.get("edited"):  # only the images the work in progress changed
        draft = drafts.fixing(db, release)
        query = query.where(ReleaseItem.item_id.in_(drafts.edited_items(draft))) if draft else query.where(false())
    if wanted.get("suspect") and not wanted.get("picks"):  # an evaluation's images it got wrong, the most first
        op = checks.check_of(db, release, str(wanted["suspect"]))
        query = query.join(
            ItemPrediction, and_(ItemPrediction.op_id == op.id, ItemPrediction.item_id == ReleaseItem.item_id)
        ).where(ItemPrediction.wrong > 0)
        order = [ItemPrediction.wrong.desc(), ItemPrediction.missed.desc(), ReleaseItem.item_id]
        if wanted.get("mistake") in checks.MISTAKES:
            column = getattr(ItemPrediction, wanted["mistake"])
            query = query.where(column > 0)
            order = [column.desc(), ReleaseItem.item_id]
    chosen = catalog.parse_picks([str(p) for p in wanted.get("picks") or []])
    if chosen:  # several of 검수's checks at once: the images any of them found
        found_by, hits = catalog.picks_condition(db, release, chosen, str(wanted.get("suspect") or ""), None, None, set())
        query = query.where(found_by)
        order = [hits.desc(), ReleaseItem.item_id]
    elif wanted.get("check"):  # a label check's images (검수)
        if wanted["check"] not in qa.CHECKS:
            raise ServiceError(400, "그런 검사가 없습니다.")
        qa.run(db, release)
        flagged = and_(
            QaResult.release_id == release.id, QaResult.check == wanted["check"], QaResult.item_id == ReleaseItem.item_id
        )
        query = query.join(QaResult, flagged)
    item_ids = list(db.scalars(query.order_by(*order).limit(MAX_ITEMS + 1)))
    if not item_ids:
        raise ServiceError(400, "고른 조건에 맞는 이미지가 없습니다.")
    if len(item_ids) > MAX_ITEMS:
        raise ServiceError(400, f"라벨링 묶음 하나에는 이미지 {MAX_ITEMS:,}장까지 담을 수 있습니다. 조건을 좁혀 주세요.")
    if parts > 0:
        part_size = -(-len(item_ids) // parts)  # ceil: no more parts than asked for
    part_size = min(part_size, len(item_ids))  # no bound but the set's own size: all of it to one person is one part
    dataset = db.get(Dataset, release.dataset_id)
    found = LabelSet(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        release_id=release.id,
        name=name[:128],
        instructions=instructions.strip(),
        part_size=part_size,
        review=review,
        created_by=user.id,
    )
    db.add(found)
    db.flush()
    rows = [
        {"set_id": found.id, "item_id": item_id, "part": n // part_size, "position": n, "state": "todo", "note": ""}
        for n, item_id in enumerate(item_ids)
    ]
    for start in range(0, len(rows), 5000):
        db.execute(insert(LabelSetItem), rows[start : start + 5000])
    parts = (len(item_ids) + part_size - 1) // part_size
    db.execute(insert(LabelSetPart), [{"set_id": found.id, "part": n} for n in range(parts)])
    services.audit(db, user.id, via, "label_set.create", found.id, name=found.name, items=len(item_ids), parts=parts)
    db.flush()
    return found


def _counts(db: Session, set_id: str) -> dict[int, Counter]:
    rows = db.execute(
        select(LabelSetItem.part, LabelSetItem.state, func.count())
        .where(LabelSetItem.set_id == set_id)
        .group_by(LabelSetItem.part, LabelSetItem.state)
    ).all()
    out: dict[int, Counter] = {}
    for part, state, n in rows:
        out.setdefault(part, Counter())[state] += n
    return out


def column_of(counts: Counter, review: bool, assignee: str | None) -> str:
    size = sum(counts.values())
    todo, rejected, labelled, approved = (counts[s] for s in ("todo", "rejected", "labelled", "approved"))
    if review and approved == size:
        return "done"
    if not review and todo == 0 and rejected == 0:
        return "done"
    if review and todo == 0 and rejected == 0 and labelled > 0:
        return "review"
    if assignee or labelled + approved + rejected > 0:
        return "labeling"
    return "unassigned"


def _names(db: Session, ids: set[str]) -> dict[str, str]:
    ids.discard("")
    if not ids:
        return {}
    return dict(db.execute(select(User.id, User.username).where(User.id.in_(ids))).all())


def _open_release(db: Session, found: LabelSet) -> Release | None:
    """Where the set's images are labelled: the dataset's work in progress, else its newest version."""
    return drafts.draft_of(db, found.dataset_id) or drafts.latest_ready(db, found.dataset_id)


def _first(db: Session, set_id: str, part: int, states: tuple[str, ...]) -> int | None:
    """Index (within the part) of its first image in one of ``states``."""
    position = db.scalar(
        select(func.min(LabelSetItem.position)).where(
            LabelSetItem.set_id == set_id, LabelSetItem.part == part, LabelSetItem.state.in_(states)
        )
    )
    first = db.scalar(
        select(func.min(LabelSetItem.position)).where(LabelSetItem.set_id == set_id, LabelSetItem.part == part)
    )
    return None if position is None or first is None else position - first


def set_summary(db: Session, found: LabelSet, user: User, counts: dict[int, Counter] | None = None) -> dict[str, Any]:
    counts = _counts(db, found.id) if counts is None else counts
    total = Counter()
    for c in counts.values():
        total.update(c)
    parts = db.scalars(select(LabelSetPart).where(LabelSetPart.set_id == found.id)).all()
    columns = Counter(column_of(counts.get(p.part, Counter()), found.review, p.assignee) for p in parts)
    dataset = db.get(Dataset, found.dataset_id)
    release = db.get(Release, found.release_id)
    return {
        "id": found.id,
        "name": found.name,
        "project_id": found.project_id,
        "dataset": {"id": dataset.id, "name": dataset.name} if dataset else None,
        "release": {"id": release.id, "version": release.version} if release else None,
        "review": found.review,
        "part_size": found.part_size,
        "items": sum(total.values()),
        "states": {s: total[s] for s in STATES},
        "parts": len(parts),
        "columns": {c: columns[c] for c in COLUMNS},
        "mine": sum(1 for p in parts if user.id in (p.assignee, p.reviewer)),
        "created_by": _names(db, {found.created_by or ""}).get(found.created_by or ""),
        "created_at": iso(found.created_at),
        "autolabel": _autolabel_status(db, found),
    }


def _autolabel_status(db: Session, found: LabelSet) -> dict[str, Any] | None:
    from . import autolabel

    return autolabel.status(db, found)


def list_sets(db: Session, user: User, project_id: str = "") -> list[dict[str, Any]]:
    query = select(LabelSet).order_by(LabelSet.created_at.desc())
    if project_id:
        query = query.where(LabelSet.project_id == project_id)
    return [set_summary(db, s, user) for s in db.scalars(query)]


def set_view(db: Session, user: User, set_id: str) -> dict[str, Any]:
    """A set with its parts on the board and the people who can be given one."""
    found = get_set(db, set_id)
    counts = _counts(db, found.id)
    parts = db.scalars(select(LabelSetPart).where(LabelSetPart.set_id == found.id).order_by(LabelSetPart.part)).all()
    names = _names(db, {p.assignee or "" for p in parts} | {p.reviewer or "" for p in parts})
    people = db.execute(
        select(User.id, User.username).where(User.disabled_at.is_(None)).order_by(User.username)
    ).all()
    opened = _open_release(db, found)
    out = []
    for p in parts:
        c = counts.get(p.part, Counter())
        out.append(
            {
                "part": p.part,
                "size": sum(c.values()),
                "states": {s: c[s] for s in STATES},
                "assignee": {"id": p.assignee, "name": names.get(p.assignee, "")} if p.assignee else None,
                "reviewer": {"id": p.reviewer, "name": names.get(p.reviewer, "")} if p.reviewer else None,
                "column": column_of(c, found.review, p.assignee),
                # where to open it: the first image to label (or sent back), the first to review
                "next_label": _first(db, found.id, p.part, ("todo", "rejected")),
                "next_review": _first(db, found.id, p.part, ("labelled",)) if found.review else None,
            }
        )
    return {
        **set_summary(db, found, user, counts),
        "instructions": found.instructions,
        "open_release_id": opened.id if opened else None,
        "people": [{"id": i, "name": n} for i, n in people],
        "part_list": out,
        "me": user.id,
    }


def assign(
    db: Session, user: User, set_id: str, part: int, *, assignee: str | None = None, reviewer: str | None = None
) -> dict[str, Any]:
    """Give a part to someone to label and/or review (``""`` takes it back)."""
    found = get_set(db, set_id)
    row = db.get(LabelSetPart, (found.id, part))
    if row is None:
        raise ServiceError(404, "그런 일감이 없습니다.")
    for value in (assignee, reviewer):
        if value and db.get(User, value) is None:
            raise ServiceError(400, "그런 사용자가 없습니다.")
    if assignee is not None:
        row.assignee = assignee or None
    if reviewer is not None:
        row.reviewer = reviewer or None
    services.audit(db, user.id, "ui", "label_set.assign", found.id, part=part, assignee=row.assignee, reviewer=row.reviewer)
    db.flush()
    return {"part": part, "assignee": row.assignee, "reviewer": row.reviewer}


def _member(db: Session, set_id: str, item_id: int) -> LabelSetItem:
    row = db.get(LabelSetItem, (set_id, item_id))
    if row is None:
        raise ServiceError(404, "이 라벨링 묶음에 없는 이미지입니다.")
    return row


def mark_labelled(db: Session, user: User, set_id: str, item_id: int) -> dict[str, Any]:
    """The image is labelled (saved, or looked at and left as it is). A reviewer's approval is not undone by
    their own save; anyone else's change sends it back to review."""
    found = get_set(db, set_id)
    row = _member(db, found.id, item_id)
    if not (row.state == "approved" and row.reviewed_by == user.id):
        row.state = "labelled" if found.review else "approved"  # a note it was sent back with stays for the reviewer
    row.labelled_by, row.labelled_at = user.id, utcnow()
    db.flush()
    return {"item_id": item_id, "state": row.state}


def review_item(db: Session, user: User, set_id: str, item_id: int, approve: bool, note: str = "") -> dict[str, Any]:
    found = get_set(db, set_id)
    if not found.review:
        raise ServiceError(400, "이 라벨링 묶음은 검수를 하지 않습니다.")
    row = _member(db, found.id, item_id)
    if row.state == "todo":
        raise ServiceError(409, "아직 라벨링하지 않은 이미지입니다.")
    if not approve and not note.strip():
        raise ServiceError(400, "돌려보내는 이유를 적어 주세요.")
    row.state = "approved" if approve else "rejected"
    row.note = "" if approve else note.strip()[:1000]
    row.reviewed_by, row.reviewed_at = user.id, utcnow()
    services.audit(db, user.id, "ui", "label_set.review", found.id, item=item_id, approve=approve)
    db.flush()
    return {"item_id": item_id, "state": row.state, "note": row.note}


def item_states(db: Session, set_id: str, item_ids: list[int]) -> dict[int, dict[str, Any]]:
    """For the image grid and view: each image's state in the set and the note it was sent back with."""
    if not item_ids:
        return {}
    rows = db.scalars(select(LabelSetItem).where(LabelSetItem.set_id == set_id, LabelSetItem.item_id.in_(item_ids)))
    return {r.item_id: {"state": r.state, "note": r.note, "part": r.part} for r in rows}


def remove_set(db: Session, user: User, set_id: str) -> dict[str, Any]:
    """Drop the set. The labels saved through it stay (in the work in progress or the versions made since)."""
    found = get_set(db, set_id)
    if user.role != "admin" and user.id != found.created_by:
        raise ServiceError(403, "만든 사람이나 관리자만 지울 수 있습니다.")
    db.execute(delete(LabelSetItem).where(LabelSetItem.set_id == found.id))
    db.execute(delete(LabelSetPart).where(LabelSetPart.set_id == found.id))
    db.delete(found)
    services.audit(db, user.id, "ui", "label_set.delete", set_id, name=found.name)
    db.flush()
    return {"ok": True}
