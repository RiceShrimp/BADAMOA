"""Emptying the trash: what is in it removed for good, with the stored files only it used.

What goes with a project: its datasets, versions, image records and labels, runs (metrics, files,
checkpoints), models and their evaluations, exports and tests, searches and tags. The images
themselves stay where they were registered; BADAMOA never wrote them and does not remove them.
A run goes with its metrics, files, checkpoint and tags; a model with its weights, evaluations,
exports and tests; a template with every version of its code (runs made with it keep their record
but can no longer be started again).

Lineage is kept whole: a version that runs, models or evaluations point at, and a run a model was
trained by, stay in the trash (with the reason) until what points at them goes first. Models go
before runs and runs before versions, so emptying everything at once removes the whole chain.

Copies of a removed version that other servers received are removed from those servers too
(``node_data.forget``).
"""

from __future__ import annotations

import logging
import threading
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import catalog, cleanup, node_data, services, trash
from .catalog_models import Dataset, Item, ItemPrediction, Project, Release, ReleaseItem
from .config import HubSettings
from .db import utcnow
from .models import Job, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run, RunFile, Tag, Template

log = logging.getLogger(__name__)

CHUNK = 900  # SQLite's limit on bound parameters is 999
KINDS = ("projects", "releases", "runs", "models", "templates")


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


FILES_NOW = 2000  # left-over files removed before going on; more go on a thread of their own


def _remove_unused(settings: HubSettings, db: Session, candidates: set[str]) -> tuple[int, int]:
    """Delete the stored files among ``candidates`` that nothing uses any more. Returns (files, bytes).

    What the caller removed is committed first, so the database is not held while the files go (an import of
    150,000 images stopped leaves as many thumbnails: minutes of deleting). Each batch is looked up again just
    before it goes: an import may have taken a thumbnail back meanwhile."""
    if not candidates:
        return 0, 0
    db.commit()
    unused = sorted(candidates - cleanup.live_blobs(db))
    db.commit()  # the reading ends: writers do not wait on it
    removed = freed = 0
    for chunk in _chunks(unused):
        taken = set(db.scalars(select(Item.thumb_key).where(Item.thumb_key.in_(chunk))))
        db.commit()
        for digest in chunk:
            if digest in taken:
                continue
            path = settings.storage_dir / "blobs" / digest[:2] / digest[2:4] / digest
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            removed, freed = removed + 1, freed + size
    return removed, freed


def remove_unused_soon(settings: HubSettings, db: Session, candidates: set[str]) -> None:
    """``_remove_unused``, on a thread of its own when there are many: the request or the Hub's sweep that let them
    go does not wait minutes for the files. Left over if the Hub stops first: its cleanup takes them a day on."""
    if len(candidates) <= FILES_NOW:
        _remove_unused(settings, db, candidates)
        return
    db.commit()
    engine = db.get_bind()

    def remove() -> None:
        try:
            with Session(engine) as own:
                files, size = _remove_unused(settings, own, set(candidates))
            log.info("left-over files removed: %d (%.1f MB)", files, size / 1024**2)
        except Exception:  # noqa: BLE001  (a thread of its own: say so and leave the rest to the cleanup)
            log.exception("removing left-over files failed")

    threading.Thread(target=remove, name="remove-unused", daemon=True).start()


# ---- removing ---------------------------------------------------------------------------------


def _drop_project(db: Session, project: Project) -> None:
    models = list(db.scalars(select(ModelVersion.id).where(ModelVersion.project_id == project.id)))
    for chunk in _chunks(models):
        ops = select(ModelJob.id).where(ModelJob.model_id.in_(chunk))
        db.execute(delete(ItemPrediction).where(ItemPrediction.op_id.in_(ops)))
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


def _drop_model(db: Session, model: ModelVersion) -> set[str]:
    blobs = {model.blob}
    for op in db.scalars(select(ModelJob).where(ModelJob.model_id == model.id)):
        blobs.update(f["blob"] for f in op.files or [] if f.get("blob"))
        blobs.update(i["blob"] for i in (op.params or {}).get("images") or [] if i.get("blob"))
        db.execute(delete(ItemPrediction).where(ItemPrediction.op_id == op.id))
    db.execute(delete(ModelJob).where(ModelJob.model_id == model.id))
    db.execute(delete(Tag).where(Tag.kind == "model", Tag.object_id == model.id))
    db.execute(delete(ModelVersion).where(ModelVersion.id == model.id))
    return blobs


def _drop_run(db: Session, run: Run) -> set[str]:
    blobs = set(db.scalars(select(RunFile.blob).where(RunFile.run_id == run.id)))
    if (run.checkpoint or {}).get("blob"):
        blobs.add(run.checkpoint["blob"])
    db.execute(delete(Tag).where(Tag.kind == "run", Tag.object_id == run.id))
    db.execute(delete(Run).where(Run.id == run.id))  # its metrics and file records go with it (ON DELETE CASCADE)
    return blobs


def empty_trash(
    db: Session,
    settings: HubSettings,
    user: User | None,
    *,
    projects: list[str] | None = None,
    releases: list[str] | None = None,
    runs: list[str] | None = None,
    models: list[str] | None = None,
    templates: list[str] | None = None,
    older_than_days: int = 0,
    via: str = "ui",
) -> dict[str, Any]:
    """Remove what is in the trash for good. With no ids at all, everything in it.

    ``older_than_days`` keeps what went to the trash more recently (the Hub's own pass under the
    retention policy). What something still points at, and what has work under way, stays.
    A template is named by the id of any of its versions.
    """
    from . import hub_storage

    hub_storage.mark_changed()  # the storage screen counts again
    if user is not None:
        services.require_admin(user)
    wanted = {"projects": projects, "releases": releases, "runs": runs, "models": models, "templates": templates}
    everything = all(ids is None for ids in wanted.values())
    query_p = select(Project).where(Project.deleted_at.is_not(None))
    query_r = (
        select(Release)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .join(Project, Project.id == Dataset.project_id)
        .where(Release.deleted_at.is_not(None), Project.deleted_at.is_(None))
    )
    query_run, query_m = trash.runs_query(), trash.models_query()
    groups = trash.template_groups(db)
    if older_than_days > 0:
        cutoff = utcnow() - timedelta(days=older_than_days)
        query_p = query_p.where(Project.deleted_at < cutoff)
        query_r = query_r.where(Release.deleted_at < cutoff)
        query_run = query_run.where(Run.deleted_at < cutoff)
        query_m = query_m.where(ModelVersion.deleted_at < cutoff)
        groups = [g for g in groups if max(t.deleted_at for t in g) < cutoff]
    binned: dict[str, list[Any]] = {
        "projects": list(db.scalars(query_p)),
        "releases": list(db.scalars(query_r)),
        "runs": list(db.scalars(query_run)),
        "models": list(db.scalars(query_m)),
        "templates": groups,
    }
    picked: dict[str, list[Any]] = {}
    for kind, rows in binned.items():
        ids_of = (lambda g: {t.id for t in g}) if kind == "templates" else (lambda r: {r.id})
        asked = set(wanted[kind] or [])
        if not asked <= set().union(*(ids_of(r) for r in rows)):
            raise ServiceError(404, "휴지통에 없는 항목이 있습니다. 화면을 새로 고친 뒤 다시 시도하세요.")
        picked[kind] = [r for r in rows if everything or ids_of(r) & asked]

    kept: list[dict[str, str]] = []
    candidates: set[str] = set()
    done = dict.fromkeys(KINDS, 0)
    for model in picked["models"]:
        running = [
            op
            for op in db.scalars(select(ModelJob).where(ModelJob.model_id == model.id))
            if not cleanup._finished(db, op.job_id)
        ]
        if running:
            kept.append({"id": model.id, "name": f"{model.name} v{model.version}", "reason": "평가 · 내보내기 진행 중"})
            continue
        candidates |= _drop_model(db, model)
        done["models"] += 1
    db.flush()
    for run in picked["runs"]:
        made = db.scalar(select(func.count(ModelVersion.id)).where(ModelVersion.run_id == run.id)) or 0
        if made:
            kept.append({"id": run.id, "name": run.name, "reason": f"모델 {made}개가 이 학습에서 나옴"})
            continue
        candidates |= _drop_run(db, run)
        done["runs"] += 1
    db.flush()
    for project in picked["projects"]:
        busy = catalog._busy(db, project.id)
        if busy:
            kept.append({"id": project.id, "name": project.name, "reason": f"{', '.join(busy)} 진행 중"})
            continue
        candidates |= _project_blobs(db, project.id)
        versions = select(Release.id).join(Dataset, Dataset.id == Release.dataset_id)
        # copies other servers received of its versions go too
        node_data.forget(db, list(db.scalars(versions.where(Dataset.project_id == project.id))), user and user.id, via)
        _drop_project(db, project)
        done["projects"] += 1
    for release in picked["releases"]:
        dataset = db.get(Dataset, release.dataset_id)
        users = catalog.release_users(db, release.id)
        if users:
            reason = f"{', '.join(users)}가 이 버전을 씀"
            kept.append({"id": release.id, "name": f"{dataset.name} {release.version}", "reason": reason})
            continue
        node_data.forget(db, [release.id], user and user.id, via)
        candidates |= _drop_release(db, release)
        done["releases"] += 1
    for group in picked["templates"]:
        name = group[-1].name
        running = db.scalar(
            select(func.count(Run.id))
            .join(Job, Job.id == Run.job_id)
            .where(Run.trainer == name, Job.status.in_(catalog.ACTIVE))
        )
        if running:
            kept.append({"id": group[-1].id, "name": group[-1].title or name, "reason": f"학습 {running}개가 쓰는 중"})
            continue
        candidates |= {t.blob for t in group if t.blob}
        db.execute(delete(Template).where(Template.id.in_([t.id for t in group])))
        done["templates"] += 1
    db.flush()
    files, freed = _remove_unused(settings, db, candidates)
    if any(done.values()):
        services.audit(db, user.id if user else None, via, "trash.empty", "", **done, files=files, bytes=freed)
    return {**done, "files": files, "bytes": freed, "kept": kept}
