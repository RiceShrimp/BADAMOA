"""What is in the trash: deleted projects, dataset versions, runs, models and templates.

Deleting moves a thing here and keeps everything it had (records and stored files) until it is
restored or removed for good (``purge.py``). Things inside a deleted project are not listed one by
one: they come back, or go, with their project. Templates are listed by name, all versions together;
the built-in ones that leave the list when the Hub moves to a newer version are not in the trash.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core.storage import Storage

from . import catalog
from .catalog_models import Dataset, Project, Release
from .db import iso
from .train_models import ModelVersion, Run, Template


def _in_live_project(column: Any) -> Any:
    return column.in_(select(Project.id).where(Project.deleted_at.is_(None)))


def runs_query() -> Any:
    return select(Run).where(Run.deleted_at.is_not(None), _in_live_project(Run.project_id))


def models_query() -> Any:
    return select(ModelVersion).where(ModelVersion.deleted_at.is_not(None), _in_live_project(ModelVersion.project_id))


def template_groups(db: Session) -> list[list[Template]]:
    """Deleted templates by name, oldest version first; a name with a version still listed is not in the trash."""
    listed = set(db.scalars(select(Template.name).where(Template.deleted_at.is_(None))))
    groups: dict[str, list[Template]] = {}
    query = select(Template).where(Template.deleted_at.is_not(None), Template.builtin.is_(False))
    for row in db.scalars(query.order_by(Template.created_at, Template.version)):
        if row.name not in listed:
            groups.setdefault(row.name, []).append(row)
    return list(groups.values())


def count(db: Session) -> dict[str, int]:
    """How many things are in the trash, by kind, for the sidebar."""
    found = catalog.trash_count(db)
    found["runs"] = db.scalar(select(func.count()).select_from(runs_query().subquery())) or 0
    found["models"] = db.scalar(select(func.count()).select_from(models_query().subquery())) or 0
    found["templates"] = len(template_groups(db))
    found["total"] = sum(v for k, v in found.items() if k != "total")
    return found


def _release_label(db: Session, release_id: str | None) -> str:
    release = db.get(Release, release_id) if release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    return f"{dataset.name} {release.version}" if release and dataset else ""


def listing(db: Session, storage: Storage) -> dict[str, Any]:
    found = catalog.trash(db)  # projects and dataset versions
    runs = db.scalars(runs_query().order_by(Run.deleted_at.desc())).all()
    models = db.scalars(models_query().order_by(ModelVersion.deleted_at.desc())).all()
    ids = {r.project_id for r in runs} | {m.project_id for m in models}
    names = {p.id: p.name for p in db.scalars(select(Project).where(Project.id.in_(ids)))} if ids else {}
    made: dict[str, int] = (
        dict(
            db.execute(
                select(ModelVersion.run_id, func.count(ModelVersion.id))
                .where(ModelVersion.run_id.in_([r.id for r in runs]))
                .group_by(ModelVersion.run_id)
            ).all()
        )
        if runs
        else {}
    )
    found["runs"] = [
        {
            "id": r.id,
            "name": r.name,
            "project": {"id": r.project_id, "name": names.get(r.project_id, "")},
            "release": _release_label(db, r.release_id),
            "template": (r.template or {}).get("title") or r.trainer,
            "epoch": r.epoch,
            "created_by": r.created_by,
            "deleted_at": iso(r.deleted_at),
            # Models trained by it (listed or in the trash) keep it from going for good.
            "in_use": [f"모델 {made[r.id]}개"] if made.get(r.id) else [],
        }
        for r in runs
    ]
    runs_by_id = {r.id: r for r in db.scalars(select(Run).where(Run.id.in_([m.run_id for m in models if m.run_id])))}
    found["models"] = [
        {
            "id": m.id,
            "name": m.name,
            "version": m.version,
            "task": m.task,
            "bytes": m.bytes,
            "project": {"id": m.project_id, "name": names.get(m.project_id, "")},
            "run": runs_by_id[m.run_id].name if m.run_id in runs_by_id else "",
            "deleted_at": iso(m.deleted_at),
            "restorable": storage.has_blob(m.blob),  # deleted before the trash kept files: its weights may be gone
        }
        for m in models
    ]
    used: dict[str, int] = dict(db.execute(select(Run.trainer, func.count(Run.id)).group_by(Run.trainer)).all())
    found["templates"] = sorted(
        (
            {
                "id": group[-1].id,
                "name": group[-1].name,
                "title": group[-1].title,
                "task": group[-1].task,
                "versions": [t.version for t in group],
                "deleted_at": iso(max(t.deleted_at for t in group)),
                "runs": used.get(group[-1].name, 0),
            }
            for group in template_groups(db)
        ),
        key=lambda t: t["deleted_at"] or "",
        reverse=True,
    )
    return found
