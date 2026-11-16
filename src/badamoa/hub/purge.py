"""Emptying the trash: deleted projects and versions removed for good, with the stored files only they used.

What goes with a project: its datasets, versions, image records and labels, runs (metrics, files,
checkpoints), models and their evaluations, exports and tests, searches and tags. The images
themselves stay where they were registered; BADAMOA never wrote them and does not remove them.

A version that runs, models or evaluations point at stays in the trash: removing it would leave
those records saying "trained on something that no longer exists". Remove the project instead,
or keep the version.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import catalog, cleanup, services
from .catalog_models import Dataset, Item, Project, Release, ReleaseItem
from .config import HubSettings
from .db import utcnow
from .models import User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run, RunFile

CHUNK = 900  # SQLite's limit on bound parameters is 999


def _chunks(values: list[Any]) -> list[list[Any]]:
    return [values[i : i + CHUNK] for i in range(0, len(values), CHUNK)]


# ---- the files that may be left over ---------------------------------------------------------


def _project_blobs(db: Session, project_id: str) -> set[str]:
    blobs = {k for k in db.scalars(select(Item.thumb_key).where(Item.project_id == project_id)) if k}
    runs = list(db.scalars(select(Run.id).where(Run.project_id == project_id)))
    for chunk in _chunks(runs):
        blobs.update(db.scalars(select(RunFile.blob).where(RunFile.run_id.in_(chunk))))
    for checkpoint in db.scalars(select(Run.checkpoint).where(Run.project_id == project_id)):
        if (checkpoint or {}).get("blob"):
            blobs.add(checkpoint["blob"])
    models = list(db.scalars(select(ModelVersion.id).where(ModelVersion.project_id == project_id)))
    blobs.update(b for b in db.scalars(select(ModelVersion.blob).where(ModelVersion.project_id == project_id)) if b)
    for chunk in _chunks(models):
        for op in db.scalars(select(ModelJob).where(ModelJob.model_id.in_(chunk))):
            blobs.update(f["blob"] for f in op.files or [] if f.get("blob"))
            blobs.update(i["blob"] for i in (op.params or {}).get("images") or [] if i.get("blob"))
    return blobs


def _remove_unused(settings: HubSettings, db: Session, candidates: set[str]) -> tuple[int, int]:
    """Delete the stored files among ``candidates`` that nothing uses any more. Returns (files, bytes)."""
    if not candidates:
        return 0, 0
    live = cleanup.live_blobs(db)
    removed = freed = 0
    for digest in candidates - live:
        path = settings.storage_dir / "blobs" / digest[:2] / digest[2:4] / digest
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        removed, freed = removed + 1, freed + size
    return removed, freed


# ---- removing ---------------------------------------------------------------------------------


def _drop_project(db: Session, project: Project) -> None:
    models = list(db.scalars(select(ModelVersion.id).where(ModelVersion.project_id == project.id)))
    for chunk in _chunks(models):
        db.execute(delete(ModelJob).where(ModelJob.model_id.in_(chunk)))
    # Everything else hangs off the project and goes with it (ON DELETE CASCADE).
    db.execute(delete(Project).where(Project.id == project.id))


def _drop_release(db: Session, release: Release) -> set[str]:
    """Remove a version, the dataset if it was its last one, and the image records no version holds now."""
    dataset = db.get(Dataset, release.dataset_id)
    project_id = dataset.project_id
    held = list(db.scalars(select(ReleaseItem.item_id).where(ReleaseItem.release_id == release.id)))
    db.execute(delete(ReleaseItem).where(ReleaseItem.release_id == release.id))
    db.execute(delete(Release).where(Release.id == release.id))
    if not db.scalar(select(func.count(Release.id)).where(Release.dataset_id == dataset.id)):
        db.execute(delete(Dataset).where(Dataset.id == dataset.id))
    thumbs: set[str] = set()
    for chunk in _chunks(held):
        still = set(db.scalars(select(ReleaseItem.item_id).where(ReleaseItem.item_id.in_(chunk))))
        gone = [i for i in chunk if i not in still]
        if gone:
            thumbs.update(k for k in db.scalars(select(Item.thumb_key).where(Item.id.in_(gone))) if k)
            db.execute(delete(Item).where(Item.id.in_(gone), Item.project_id == project_id))
    return thumbs


def empty_trash(
    db: Session,
    settings: HubSettings,
    user: User | None,
    *,
    projects: list[str] | None = None,
    releases: list[str] | None = None,
    older_than_days: int = 0,
    via: str = "ui",
) -> dict[str, Any]:
    """Remove deleted projects and versions for good. With no ids, everything in the trash.

    ``older_than_days`` keeps what went to the trash more recently (the Hub's own pass under the
    retention policy). A version something still points at, and a project with work under way, stay.
    """
    if user is not None:
        services.require_admin(user)
    everything = projects is None and releases is None
    query_p = select(Project).where(Project.deleted_at.is_not(None))
    query_r = (
        select(Release)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .join(Project, Project.id == Dataset.project_id)
        .where(Release.deleted_at.is_not(None), Project.deleted_at.is_(None))
    )
    if older_than_days > 0:
        cutoff = utcnow() - timedelta(days=older_than_days)
        query_p = query_p.where(Project.deleted_at < cutoff)
        query_r = query_r.where(Release.deleted_at < cutoff)
    picked_p = [p for p in db.scalars(query_p) if everything or p.id in (projects or [])]
    picked_r = [r for r in db.scalars(query_r) if everything or r.id in (releases or [])]
    if not everything and len(picked_p) + len(picked_r) < len(projects or []) + len(releases or []):
        raise ServiceError(404, "휴지통에 없는 항목이 있습니다. 화면을 새로 고친 뒤 다시 시도하세요.")

    kept: list[dict[str, str]] = []
    candidates: set[str] = set()
    done_p = done_r = 0
    for project in picked_p:
        busy = catalog._busy(db, project.id)
        if busy:
            kept.append({"id": project.id, "name": project.name, "reason": f"{', '.join(busy)} 진행 중"})
            continue
        candidates |= _project_blobs(db, project.id)
        _drop_project(db, project)
        done_p += 1
    for release in picked_r:
        dataset = db.get(Dataset, release.dataset_id)
        users = catalog.release_users(db, release.id)
        if users:
            reason = f"{', '.join(users)}가 이 버전을 씀"
            kept.append({"id": release.id, "name": f"{dataset.name} {release.version}", "reason": reason})
            continue
        candidates |= _drop_release(db, release)
        done_r += 1
    db.flush()
    files, freed = _remove_unused(settings, db, candidates)
    if done_p or done_r:
        services.audit(
            db, user.id if user else None, via, "trash.empty", "", projects=done_p, releases=done_r, files=files, bytes=freed
        )
    return {"projects": done_p, "releases": done_r, "files": files, "bytes": freed, "kept": kept}
