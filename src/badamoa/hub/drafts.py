"""Fixing labels in place: the work in progress ("작업 중") of a dataset, and making it a version.

A version is never changed. A label fixed while looking at the newest version is saved as a new
label revision and lands in the dataset's work in progress: a release in the ``draft`` state that
starts as a copy of that version's item list and whose rows for fixed images point at the new
revisions. Everything that reads a release (grid, filters, statistics, compare) reads the draft as
it is; training, export and evaluation take only ``ready`` releases. Confirming the draft makes it
an ordinary version; discarding it leaves the new revisions in each image's label history.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import and_, delete, func, insert, literal, select
from sqlalchemy.orm import Session, aliased

from . import catalog, label_diff, labeling, services
from .catalog_models import Dataset, LabelRevision, LabelSpace, Release, ReleaseItem, RevisionClass
from .db import iso, utcnow
from .models import User
from .services import ServiceError

DRAFT = "작업 중"  # the draft's version name; no version may be called this
MAX_BOXES = 5000
MIN_SIDE = 0.0005  # a box thinner than this (as a share of the image) is a slip of the mouse

# HTTP status of "someone else saved this image after you opened it": the screen offers to look or overwrite.
CHANGED_MEANWHILE = 412


def latest_ready(db: Session, dataset_id: str) -> Release | None:
    return db.scalar(
        select(Release)
        .where(Release.dataset_id == dataset_id, Release.status == "ready", Release.deleted_at.is_(None))
        .order_by(Release.created_at.desc())
        .limit(1)
    )


def draft_of(db: Session, dataset_id: str) -> Release | None:
    return db.scalar(select(Release).where(Release.dataset_id == dataset_id, Release.status == "draft"))


def _changed_rows(draft: Release):
    """(draft row, base row) pairs of the images whose labels the draft changed."""
    mine, base = aliased(ReleaseItem), aliased(ReleaseItem)
    query = (
        select(mine, base)
        .join(base, and_(base.item_id == mine.item_id, base.release_id == draft.base_release_id))
        .where(mine.release_id == draft.id, mine.revision_id != base.revision_id)
    )
    return query, mine, base


def edited_count(db: Session, draft: Release) -> int:
    query, mine, _ = _changed_rows(draft)
    return db.scalar(select(func.count()).select_from(query.with_only_columns(mine.item_id).subquery())) or 0


def edited_ids(db: Session, draft: Release, item_ids: list[int] | None = None) -> set[int]:
    query, mine, _ = _changed_rows(draft)
    query = query.with_only_columns(mine.item_id)
    if item_ids is not None:
        query = query.where(mine.item_id.in_(item_ids))
    return set(db.scalars(query))


def fixing(db: Session, release: Release) -> Release | None:
    """The work in progress that changes this release's images: the release itself, or the draft based on it."""
    if release.status == "draft":
        return release
    draft = draft_of(db, release.dataset_id)
    return draft if draft is not None and draft.base_release_id == release.id else None


def edited_items(draft: Release):
    """A subquery of the item ids the work in progress changed."""
    return select(ReleaseItem.item_id).where(ReleaseItem.release_id == draft.id, edited_condition(draft))


def edited_condition(draft: Release):
    """A WHERE condition on ``ReleaseItem`` (the draft's rows): only the images the draft changed."""
    base = aliased(ReleaseItem)
    changed = (
        select(base.item_id)
        .where(
            base.release_id == draft.base_release_id,
            base.item_id == ReleaseItem.item_id,
            base.revision_id != ReleaseItem.revision_id,
        )
        .exists()
    )
    return changed


def _editors(db: Session, draft: Release) -> list[str]:
    query, mine, _ = _changed_rows(draft)
    ids = query.with_only_columns(mine.revision_id).subquery()
    users = db.scalars(
        select(User.username)
        .join(LabelRevision, LabelRevision.created_by == User.id)
        .where(LabelRevision.id.in_(select(ids.c.revision_id)))
        .distinct()
    ).all()
    return sorted(users)


def draft_info(db: Session, dataset_id: str) -> dict[str, Any] | None:
    """The dataset's work in progress, for the version screens: what it is based on and how much it changed."""
    draft = draft_of(db, dataset_id)
    if draft is None:
        return None
    base = db.get(Release, draft.base_release_id) if draft.base_release_id else None
    latest = latest_ready(db, dataset_id)
    return {
        "id": draft.id,
        "base": {"id": base.id, "version": base.version} if base else None,
        "edited": edited_count(db, draft),
        "editors": _editors(db, draft),
        "created_at": iso(draft.created_at),
        # a newer version came in since: the draft can be moved onto it
        "behind": {"id": latest.id, "version": latest.version} if latest and base and latest.id != base.id else None,
    }


def edit_place(db: Session, release: Release) -> tuple[Release | None, str]:
    """Where a fix made while looking at ``release`` goes: the draft (None: one is made on the first save),
    or why fixing is not possible here."""
    if release.status == "draft":
        return release, ""
    if release.status != "ready" or release.deleted_at is not None:
        return None, "등록이 끝난 버전에서만 라벨을 고칠 수 있습니다."
    latest = latest_ready(db, release.dataset_id)
    if latest is not None and latest.id != release.id:
        return None, f"라벨은 최신 버전({latest.version})에서 고칩니다. 이 버전은 그대로 남습니다."
    draft = draft_of(db, release.dataset_id)
    if draft is not None and draft.base_release_id != release.id:
        base = db.get(Release, draft.base_release_id)
        name = base.version if base else "이전 버전"
        return None, f"작업 중의 바탕이 {name}입니다. 작업 중을 열어 바탕을 {release.version}(으)로 옮긴 뒤 고치세요."
    return draft, ""


def _new_draft(db: Session, user: User, base: Release) -> Release:
    draft = Release(
        dataset_id=base.dataset_id,
        version=DRAFT,
        label_space_id=base.label_space_id,
        base_release_id=base.id,
        status="draft",
        stats={**(base.stats or {})},
        created_by=user.id,
    )
    db.add(draft)
    db.flush()
    _copy_rows(db, base.id, draft.id)
    return draft


def _copy_rows(db: Session, source_id: str, target_id: str) -> None:
    columns = ["release_id", "item_id", "split", "revision_id", "state", "box_count"]
    rows = select(
        literal(target_id),
        ReleaseItem.item_id,
        ReleaseItem.split,
        ReleaseItem.revision_id,
        ReleaseItem.state,
        ReleaseItem.box_count,
    ).where(ReleaseItem.release_id == source_id)
    db.execute(insert(ReleaseItem).from_select(columns, rows))


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ServiceError(400, "라벨 값은 숫자여야 합니다.") from e
    if number != number or number in (float("inf"), float("-inf")):
        raise ServiceError(400, "라벨 값은 숫자여야 합니다.")
    return number


def _unit(v: float) -> float:
    return round(min(1.0, max(0.0, v)), 5)


def _after_box(rest: list[float], task: str, keypoints: int) -> list[float]:
    """What follows a box: an outline (segmentation), keypoints (pose), nothing (detection), or a named
    task's own values (kept as they are)."""
    if task == "segment":
        if rest and (len(rest) < 6 or len(rest) % 2):
            raise ServiceError(400, "윤곽은 점 3개 이상의 [x1, y1, x2, y2, …]입니다.")
        return [_unit(v) for v in rest]
    if task == "pose":
        if not rest:
            return [0.0, 0.0, 0.0] * keypoints
        if len(rest) % 3 or (keypoints and len(rest) != keypoints * 3):
            raise ServiceError(400, f"키포인트는 점 {keypoints}개의 [x, y, 보임(0 · 1 · 2)]입니다.")
        out: list[float] = []
        for i in range(0, len(rest), 3):
            x, y, seen = rest[i : i + 3]
            if seen not in (0, 1, 2):
                raise ServiceError(400, "키포인트의 보임 값은 0(없음) · 1(가려짐) · 2(보임)입니다.")
            out += [_unit(x), _unit(y), float(seen)] if seen else [0.0, 0.0, 0.0]
        return out
    if task == "detect":
        return []
    return rest


def clean_boxes(boxes: list[list[Any]], space: LabelSpace, task: str) -> list[list[float]]:
    """The labels as stored: known classes, boxes inside the image, coordinates rounded to 5 places."""
    if len(boxes) > MAX_BOXES:
        raise ServiceError(400, f"한 이미지의 라벨은 {MAX_BOXES:,}개까지입니다.")
    known = {int(c["id"]) for c in space.classes}
    points = {int(c["id"]): len(c.get("keypoints") or []) for c in space.classes}  # pose: keypoints per class
    out: list[list[float]] = []
    for box in boxes:
        if not box:
            raise ServiceError(400, "빈 라벨이 있습니다.")
        cls = int(_number(box[0]))
        if cls not in known:
            raise ServiceError(400, f"클래스 목록에 없는 클래스({cls})입니다.")
        if task == "classify":
            out.append([cls])
            continue
        if len(box) < 5:
            raise ServiceError(400, "박스는 [클래스, cx, cy, w, h] 형식입니다.")
        cx, cy, w, h = (_number(v) for v in box[1:5])
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        rest = _after_box([_number(v) for v in box[5:]], task, points.get(cls, 0))
        if task == "segment" and rest:  # the box is the outline's bounds
            x1, y1, x2, y2 = min(rest[0::2]), min(rest[1::2]), max(rest[0::2]), max(rest[1::2])
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            continue  # nothing left inside the image
        out.append([cls, round((x1 + x2) / 2, 5), round((y1 + y2) / 2, 5), round(x2 - x1, 5), round(y2 - y1, 5), *rest])
    if task == "classify" and len(out) > 1:
        raise ServiceError(400, "분류 이미지에는 클래스 하나만 고를 수 있습니다.")
    return out


def save_labels(
    db: Session,
    user: User,
    release_id: str,
    item_id: int,
    boxes: list[list[Any]],
    *,
    parent_id: int | None = None,
    force: bool = False,
    label_set: str = "",
    source: str = "",
    seen: list[int] | None = None,
    via: str = "ui",
) -> dict[str, Any]:
    """Save a person's labels of one image into the dataset's work in progress (made on the first fix).

    ``source`` and ``seen`` are for labels a model drew (오토라벨): where they came from, and the classes it
    looked for (a person looks for all of them).

    ``parent_id`` is the revision the person started from: if someone else saved this image in the
    meantime the save is refused (unless ``force``) so nobody overwrites a fix without seeing it.
    No boxes means background: the person looked and there is nothing to find.
    """
    release = catalog.get_release(db, release_id)
    draft, why = edit_place(db, release)
    if why:
        raise ServiceError(409, why)
    if db.get(ReleaseItem, (release.id, item_id)) is None:
        raise ServiceError(404, "이 버전에 없는 이미지입니다.")
    if draft is None:
        draft = _new_draft(db, user, release)
    member = db.get(ReleaseItem, (draft.id, item_id))
    if member is None:
        raise ServiceError(404, "작업 중에 없는 이미지입니다.")
    if parent_id is not None and member.revision_id != parent_id and not force:
        raise ServiceError(CHANGED_MEANWHILE, "다른 사람이 이 이미지의 라벨을 먼저 고쳤습니다.")

    space = db.get(LabelSpace, draft.label_space_id)
    labels = clean_boxes(boxes, space, catalog.release_task(db, draft))
    state = "labelled" if labels else "background"
    digest = catalog._digest(space.id, state, labels)
    history = db.scalars(select(LabelRevision).where(LabelRevision.item_id == item_id)).all()
    revision = next((r for r in history if r.digest == digest), None)
    if revision is None:
        revision = LabelRevision(
            item_id=item_id,
            rev=max((r.rev for r in history), default=0) + 1,
            label_space_id=space.id,
            source=source or f"human:{user.username}",
            state=state,
            box_count=len(labels),
            boxes=labels,
            # A person who saves an image has seen all of it: every class was looked for.
            labelled_class_ids=list(seen) if seen is not None else [int(c["id"]) for c in space.classes],
            digest=digest,
            parent_id=member.revision_id,
            created_by=user.id,
        )
        db.add(revision)
        db.flush()
        counts = Counter(int(b[0]) for b in labels)
        db.add_all(RevisionClass(revision_id=revision.id, class_id=cid, count=n) for cid, n in counts.items())
    member.revision_id, member.state, member.box_count = revision.id, revision.state, revision.box_count
    draft.stats = {**(draft.stats or {}), "stale": True}
    services.audit(db, user.id, via, "label.edit", draft.id, item=item_id, revision=revision.id, boxes=len(labels))
    db.flush()
    in_set = labeling.mark_labelled(db, user, label_set, item_id) if label_set else None
    return {
        **({"set": in_set} if in_set else {}),
        "release_id": draft.id,
        "revision_id": revision.id,
        "state": revision.state,
        "box_count": revision.box_count,
        "edited": edited_count(db, draft),
    }


def _get_draft(db: Session, draft_id: str) -> Release:
    draft = catalog.get_release(db, draft_id)
    if draft.status != "draft":
        raise ServiceError(400, "작업 중이 아닙니다.")
    return draft


def revert_item(db: Session, user: User, draft_id: str, item_id: int, via: str = "ui") -> dict[str, Any]:
    """Put one image of the work in progress back to its labels in the base version."""
    draft = _get_draft(db, draft_id)
    member = db.get(ReleaseItem, (draft.id, item_id))
    base = db.get(ReleaseItem, (draft.base_release_id, item_id)) if draft.base_release_id else None
    if member is None or base is None:
        raise ServiceError(404, "되돌릴 이미지를 찾을 수 없습니다.")
    member.revision_id, member.state, member.box_count = base.revision_id, base.state, base.box_count
    draft.stats = {**(draft.stats or {}), "stale": True}
    services.audit(db, user.id, via, "label.revert", draft.id, item=item_id)
    db.flush()
    return {"release_id": draft.id, "revision_id": base.revision_id, "edited": edited_count(db, draft)}


def discard(db: Session, user: User, draft_id: str, via: str = "ui") -> dict[str, Any]:
    """Throw the work in progress away. The fixed labels stay in each image's label history."""
    draft = _get_draft(db, draft_id)
    edited = edited_count(db, draft)
    dataset_id = draft.dataset_id
    db.execute(delete(ReleaseItem).where(ReleaseItem.release_id == draft.id))
    db.delete(draft)
    services.audit(db, user.id, via, "release.discard", draft_id, edited=edited)
    db.flush()
    latest = latest_ready(db, dataset_id)
    return {"discarded": edited, "latest_id": latest.id if latest else None}


def confirm(db: Session, user: User, draft_id: str, version: str, note: str = "", via: str = "ui") -> dict[str, Any]:
    """Make the work in progress a version: it stops changing and can be trained on."""
    draft = _get_draft(db, draft_id)
    edited = edited_count(db, draft)
    if edited == 0:
        raise ServiceError(400, "고친 이미지가 없어 새 버전을 만들 수 없습니다.")
    version = version.strip()
    if not version:
        raise ServiceError(400, "버전 이름을 적으세요.")
    if version == DRAFT:
        raise ServiceError(400, f"'{DRAFT}'은 버전 이름으로 쓸 수 없습니다.")
    taken = select(Release).where(Release.dataset_id == draft.dataset_id, Release.version == version)
    if db.scalar(taken) is not None:
        dataset = db.get(Dataset, draft.dataset_id)
        raise ServiceError(409, f"{dataset.name} {version} 버전이 이미 있습니다. 다른 이름을 쓰세요.")
    from . import review  # noqa: PLC0415  (review reads drafts)

    draft.review = review.record(db, draft)  # 검수 kept with the version: problems before and after, who
    draft.version = version
    draft.note = note.strip() or f"이미지 {edited:,}장의 라벨을 고침"
    draft.status = "ready"
    draft.created_at = utcnow()
    draft.created_by = user.id
    db.flush()
    draft.stats = catalog.compute_stats(db, draft)
    catalog.stamp_digest(db, draft)
    services.audit(db, user.id, via, "release.create", draft.id, version=version, edited=edited, base=draft.base_release_id)
    return {"release_id": draft.id, "version": version, "edited": edited}


def rebase(db: Session, user: User, draft_id: str, via: str = "ui") -> dict[str, Any]:
    """Move the work in progress onto the dataset's newest version, carrying the fixes along.

    An image whose labels the newer version changed too keeps the fix and is counted as overlapping;
    an image the newer version no longer has is left out.
    """
    draft = _get_draft(db, draft_id)
    latest = latest_ready(db, draft.dataset_id)
    if latest is None or latest.id == draft.base_release_id:
        return {"base": latest.version if latest else None, "kept": 0, "overlapping": [], "dropped": 0}
    if latest.label_space_id != draft.label_space_id:
        raise ServiceError(
            409, f"{latest.version}은(는) 클래스 목록이 달라 옮길 수 없습니다. 작업 중을 먼저 버전으로 확정하세요."
        )
    query, mine, base = _changed_rows(draft)
    fixes = {m.item_id: (m.revision_id, m.state, m.box_count, b.revision_id) for m, b in db.execute(query).all()}
    db.execute(delete(ReleaseItem).where(ReleaseItem.release_id == draft.id))
    _copy_rows(db, latest.id, draft.id)
    db.flush()
    kept, overlapping, dropped = 0, [], 0
    for item_id, (revision_id, state, box_count, was) in fixes.items():
        member = db.get(ReleaseItem, (draft.id, item_id))
        if member is None:
            dropped += 1
            continue
        if member.revision_id != was:
            overlapping.append(item_id)
        member.revision_id, member.state, member.box_count = revision_id, state, box_count
        kept += 1
    draft.base_release_id = latest.id
    draft.stats = {**(latest.stats or {}), "stale": True}
    services.audit(db, user.id, via, "release.rebase", draft.id, base=latest.id, kept=kept, dropped=dropped)
    db.flush()
    return {"base": latest.version, "kept": kept, "overlapping": overlapping, "dropped": dropped}


def fresh_stats(db: Session, release: Release) -> None:
    """A draft's statistics are worked out again when someone looks, not on every save."""
    if release.status == "draft" and (release.stats or {}).get("stale"):
        release.stats = catalog.compute_stats(db, release)


def item_edit_view(db: Session, release: Release, item_id: int) -> dict[str, Any]:
    """For the item view: may this image be fixed here, and which labels a fix starts from."""
    draft, why = edit_place(db, release)
    latest = latest_ready(db, release.dataset_id)
    here = db.get(ReleaseItem, (release.id, item_id))
    member = db.get(ReleaseItem, (draft.id, item_id)) if draft is not None else None
    start = member.revision_id if member else (here.revision_id if here else None)
    base = db.get(ReleaseItem, (draft.base_release_id, item_id)) if draft is not None and draft.base_release_id else None
    return {
        "allowed": not why,
        "reason": why,
        "latest": {"id": latest.id, "version": latest.version} if latest else None,
        "draft_id": draft.id if draft is not None else None,
        "revision_id": start,  # the labels a fix starts from: the work in progress's, else this version's
        # the work in progress changed this image (its labels differ from the base version's)
        "fixed": bool(member and base and member.revision_id != base.revision_id),
        "base_revision_id": base.revision_id if base else None,  # the base version's labels, what a fix is compared with
    }


# ---- what the work in progress changed ------------------------------------------------

KINDS = ("added", "removed", "moved", "reclassed")


def _changed_labels(db: Session, draft: Release, item_ids: list[int] | None = None):
    """(item id, base revision, draft revision) of the images the work in progress changed."""
    query, mine, base = _changed_rows(draft)
    if item_ids is not None:
        query = query.where(mine.item_id.in_(item_ids))
    rows = [(m.item_id, b.revision_id, m.revision_id) for m, b in db.execute(query).all()]
    wanted = list({r for _, b, m in rows for r in (b, m)})
    revisions: dict[int, LabelRevision] = {}
    for start in range(0, len(wanted), 2000):
        chunk = wanted[start : start + 2000]
        revisions.update({r.id: r for r in db.scalars(select(LabelRevision).where(LabelRevision.id.in_(chunk)))})
    return [(item, revisions[b], revisions[m]) for item, b, m in rows if b in revisions and m in revisions]


def change_counts(db: Session, draft: Release, item_ids: list[int]) -> dict[int, dict[str, int]]:
    """For a page of images: how many labels each one got added, removed, moved and reclassed."""
    if not item_ids:
        return {}
    return {
        item: label_diff.counts(label_diff.diff(before.boxes or [], after.boxes or []))
        for item, before, after in _changed_labels(db, draft, item_ids)
    }


def items_changed(db: Session, draft: Release, kind: str) -> list[int]:
    """The images the work in progress changed in this way (added, removed, moved or reclassed labels)."""
    return [
        item
        for item, before, after in _changed_labels(db, draft)
        if label_diff.diff(before.boxes or [], after.boxes or [])[kind]
    ]


def changes(db: Session, draft: Release) -> dict[str, Any]:
    """What the work in progress changed against its base version, as a commit's summary would."""
    space = db.get(LabelSpace, draft.label_space_id)
    names = {int(c["id"]): c["name"] for c in (space.classes if space else [])}
    boxes: Counter = Counter()
    images: Counter = Counter()
    reclassed: Counter = Counter()
    states: Counter = Counter()
    editors: Counter = Counter()
    rows = _changed_labels(db, draft)
    for _, before, after in rows:
        found = label_diff.diff(before.boxes or [], after.boxes or [])
        for kind in KINDS:
            boxes[kind] += len(found[kind])
            images[kind] += 1 if found[kind] else 0
        for i, j in found["reclassed"]:
            old, new = int(before.boxes[i][0]), int(after.boxes[j][0])
            reclassed[(names.get(old, str(old)), names.get(new, str(new)))] += 1
        if before.state != after.state:
            states[(before.state, after.state)] += 1
        if after.created_by:
            editors[after.created_by] += 1
    people = dict(db.execute(select(User.id, User.username).where(User.id.in_(list(editors)))).all()) if editors else {}
    base = db.get(Release, draft.base_release_id) if draft.base_release_id else None
    return {
        "release_id": draft.id,
        "base": {"id": base.id, "version": base.version} if base else None,
        "images": len(rows),
        "boxes": {kind: boxes[kind] for kind in KINDS},
        "images_by_kind": {kind: images[kind] for kind in KINDS},
        "class_changes": [{"from": a, "to": b, "count": n} for (a, b), n in reclassed.most_common(20)],
        "states": [{"from": a, "to": b, "count": n} for (a, b), n in states.most_common()],
        "editors": [{"name": people.get(uid, "?"), "images": n} for uid, n in editors.most_common()],
    }
