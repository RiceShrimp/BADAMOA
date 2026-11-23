"""검수 kept with a version: what a work in progress did about the problems its base version had.

When a work in progress is confirmed, the version it becomes keeps a record, like a review kept with a
commit: the images the label checks flagged before (in the base version) and after (in the new one), by
group and by check; how many of the flagged images were fixed and how many someone judged right ("문제
없음"); the newest model evaluation's flagged images and what became of them; who did it.

Before it is confirmed, the same record is worked out live as a preview (the confirm dialog shows it).
A version that came from an import has no record; its current problems still show on the 검수 screen.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from . import checks, drafts, qa
from .catalog_models import ItemPrediction, LabelRevision, QaDismissal, QaResult, Release, ReleaseItem
from .db import iso, utcnow
from .models import User

QA_NOW = 20_000  # images: a bigger version without a fresh check shows its problems on the 검수 screen


def _problems(db: Session, release: Release) -> dict[str, Any]:
    """The release's flagged images now: by group and by check (what was let be left out), and how many were let be."""
    qa.run(db, release)
    counts = qa.check_counts(db, release.id)
    return {
        "groups": qa.groups(db, release.id),
        "checks": {cid: counts.get(cid, (0, 0))[0] for cid in qa.CHECKS},
        "let_be": sum(gone for _, gone in counts.values()),
    }


def _flagged_items(release_id: str) -> Any:
    """The images a rule check flags in the release and nobody let be (SQL)."""
    return (
        select(QaResult.item_id)
        .join(ReleaseItem, and_(ReleaseItem.release_id == QaResult.release_id, ReleaseItem.item_id == QaResult.item_id))
        .where(
            QaResult.release_id == release_id,
            ~checks.let_be(QaResult.check, QaResult.item_id, ReleaseItem.revision_id),
        )
        .distinct()
    )


def record(db: Session, draft: Release) -> dict[str, Any]:
    """The 검수 record of a work in progress against its base version (see the module's docstring)."""
    base = db.get(Release, draft.base_release_id) if draft.base_release_id else None
    after = _problems(db, draft)
    before = _problems(db, base) if base else None
    edited = drafts.edited_items(draft)  # SQL: the images the work in progress changed
    fixed_flagged = 0
    if base is not None:
        fixed_flagged = (
            db.scalar(
                select(func.count()).select_from(
                    _flagged_items(base.id).where(QaResult.item_id.in_(edited)).subquery()
                )
            )
            or 0
        )
    # judged right in the work in progress: kept for the draft's labels of the image
    let_be = (
        db.scalar(
            select(func.count(func.distinct(QaDismissal.item_id)))
            .join(
                ReleaseItem,
                and_(ReleaseItem.item_id == QaDismissal.item_id, ReleaseItem.revision_id == QaDismissal.revision_id),
            )
            .where(ReleaseItem.release_id == draft.id)
        )
        or 0
    )
    model = None
    newest = checks.newest(db, base) if base is not None else None
    if newest is not None:
        flagged = select(ItemPrediction.item_id).where(
            ItemPrediction.op_id == newest["id"],
            or_(ItemPrediction.extra > 0, ItemPrediction.missed > 0, ItemPrediction.reclass > 0),
        )
        in_draft = select(ReleaseItem.item_id, ReleaseItem.revision_id).where(
            ReleaseItem.release_id == draft.id, ReleaseItem.item_id.in_(flagged)
        )
        rows = db.execute(in_draft).all()
        fixed_ids = set(
            db.scalars(
                select(ReleaseItem.item_id).where(ReleaseItem.release_id == draft.id, ReleaseItem.item_id.in_(edited))
            )
        )
        judged = {
            (d.item_id, d.revision_id)
            for d in db.scalars(
                select(QaDismissal).where(
                    QaDismissal.check.in_(checks.MISTAKES), QaDismissal.item_id.in_([item for item, _ in rows])
                )
            )
        }
        model = {
            "evaluation": newest["id"],
            "model": newest["model"],
            "flagged": len(rows),
            "fixed": sum(1 for item, _ in rows if item in fixed_ids),
            "let_be": sum(1 for item, revision in rows if item not in fixed_ids and (item, revision) in judged),
        }
    # the people: who fixed labels in it and who judged flagged labels right
    people = set(
        db.scalars(
            select(func.distinct(LabelRevision.created_by))
            .join(ReleaseItem, ReleaseItem.revision_id == LabelRevision.id)
            .where(
                ReleaseItem.release_id == draft.id,
                ReleaseItem.item_id.in_(edited),
                LabelRevision.created_by.is_not(None),
            )
        )
    )
    people |= set(
        db.scalars(
            select(func.distinct(QaDismissal.created_by))
            .join(
                ReleaseItem,
                and_(ReleaseItem.item_id == QaDismissal.item_id, ReleaseItem.revision_id == QaDismissal.revision_id),
            )
            .where(ReleaseItem.release_id == draft.id, QaDismissal.created_by.is_not(None))
        )
    )
    names = sorted(db.scalars(select(User.username).where(User.id.in_(people)))) if people else []
    return {
        "base": {"id": base.id, "version": base.version} if base else None,
        "before": before,
        "after": after,
        "fixed": drafts.edited_count(db, draft),  # images whose labels changed
        "fixed_flagged": fixed_flagged,  # of the images flagged in the base version
        "let_be": let_be,  # images whose flagged labels someone judged right
        "model": model,
        "people": names,
        "at": iso(utcnow()),
    }


def view(db: Session, release: Release) -> dict[str, Any]:
    """A version's 검수 record; for a work in progress, the record it would keep if confirmed now."""
    if release.status == "draft":
        return {"recorded": False, "preview": True, **record(db, release)}
    if release.review:
        return {"recorded": True, "preview": False, **release.review}
    small = ((release.stats or {}).get("items") or 0) <= QA_NOW
    now = _problems(db, release) if small or qa.is_fresh(db, release) else None  # a big one: on the 검수 screen
    return {"recorded": False, "preview": False, "now": now}


def badge(db: Session, release: Release) -> dict[str, Any] | None:
    """For a list of versions: whether it keeps a 검수 record, and how many flagged images are left (when the
    checks have run on it)."""
    if release.status != "ready":
        return None
    done = qa.is_fresh(db, release)
    return {
        "recorded": bool(release.review),
        "open": qa.groups(db, release.id)["any"] if done else None,
    }
