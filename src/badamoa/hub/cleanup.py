"""Keeping the Hub's storage in check: what is kept for how long, and removing what nothing uses.

Three kinds of things can go, each only when it is safe to:

    unreferenced  files of runs, models and test images that were deleted, and files nothing
                  records (an upload never used): always, a day after they were written
    outputs       the files runs and model jobs produced, older than ``outputs_days``
                  (0 = keep for ever). Metrics, logs and the models themselves stay; runs whose
                  model has an alias, and runs tagged ``keep`` or ``보관``, are left alone.
    checkpoints   the checkpoints of interrupted runs older than ``checkpoint_days``
                  (such a run can then no longer be continued)

The Hub runs this every few hours with the saved policy; an admin can look at what it would
remove and run it now from the storage card in Settings.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from badamoa.core.protocol import TERMINAL_STATUSES

from . import services
from .catalog_models import Item
from .config import HubSettings
from .db import utcnow
from .models import HubSetting, Job, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run, RunFile, Tag, Template

POLICY_KEY = "storage.policy"
DEFAULT_POLICY = {"outputs_days": 0, "checkpoint_days": 0}
GRACE_S = 24 * 3600  # a younger file may be on its way to being recorded (an upload, an artifact in flight)
KEEP_TAGS = ("keep", "보관")
EVERY_S = 6 * 3600  # the Hub's own pass under the saved policy

lock = threading.Lock()  # one cleanup at a time, the Hub's own or an admin's
log = logging.getLogger("badamoa.hub.cleanup")


def scheduled(database: Any, settings: HubSettings) -> dict[str, Any] | None:
    """The Hub's regular pass (background loop): under the saved policy, plus what nothing uses."""
    if not lock.acquire(blocking=False):
        return None
    try:
        with database.session() as db:
            report = cleanup(db, settings, apply=True)
        if report["removed"]["files"]:
            log.info("storage cleanup: %d files, %.1f MB", report["removed"]["files"], report["removed"]["bytes"] / 1024**2)
        return report
    finally:
        lock.release()


# ---- policy -----------------------------------------------------------------------------------


def get_policy(db: Session) -> dict[str, int]:
    row = db.get(HubSetting, POLICY_KEY)
    return {**DEFAULT_POLICY, **((row.value if row else None) or {})}


def set_policy(db: Session, user: User, outputs_days: int, checkpoint_days: int) -> dict[str, int]:
    services.require_admin(user)
    if not (0 <= outputs_days <= 3650 and 0 <= checkpoint_days <= 3650):
        raise ServiceError(400, "보관 기간은 0(계속 보관)에서 3650일 사이여야 합니다.")
    value = {"outputs_days": int(outputs_days), "checkpoint_days": int(checkpoint_days)}
    row = db.get(HubSetting, POLICY_KEY)
    if row is None:
        db.add(HubSetting(key=POLICY_KEY, value=value))
    else:
        row.value = value
    services.audit(db, user.id, "ui", "storage.policy", "", **value)
    return value


# ---- what goes ----------------------------------------------------------------------------------


def _protected_runs(db: Session) -> set[str]:
    """Runs whose outputs stay whatever their age: tagged keep, or their model carries an alias."""
    kept = set(db.scalars(select(Tag.object_id).where(Tag.kind == "run", Tag.name.in_(KEEP_TAGS))))
    for model in db.scalars(select(ModelVersion).where(ModelVersion.deleted_at.is_(None))):
        if (model.aliases or model.alias) and model.run_id:
            kept.add(model.run_id)
    return kept


def _aliased_models(db: Session) -> set[str]:
    return {
        m.id for m in db.scalars(select(ModelVersion).where(ModelVersion.deleted_at.is_(None))) if m.aliases or m.alias
    }


def _finished(db: Session, job_id: str | None) -> bool:
    job = db.get(Job, job_id) if job_id else None
    return job is None or job.status in TERMINAL_STATUSES


def _old_outputs(db: Session, days: int) -> tuple[list[RunFile], list[ModelJob]]:
    if days <= 0:
        return [], []
    cutoff = utcnow() - timedelta(days=days)
    protected = _protected_runs(db)
    runs = {
        r.id
        for r in db.scalars(select(Run).where(Run.deleted_at.is_(None), Run.created_at < cutoff))
        if r.id not in protected and _finished(db, r.job_id)
    }
    files = [f for f in db.scalars(select(RunFile).where(RunFile.run_id.in_(runs)))] if runs else []
    aliased = _aliased_models(db)
    ops = [
        op
        for op in db.scalars(select(ModelJob).where(ModelJob.created_at < cutoff))
        if op.files and op.model_id not in aliased and _finished(db, op.job_id)
    ]
    return files, ops


def _old_checkpoints(db: Session, days: int) -> list[Run]:
    if days <= 0:
        return []
    cutoff = utcnow() - timedelta(days=days)
    return [
        r
        for r in db.scalars(select(Run).where(Run.created_at < cutoff))
        if (r.checkpoint or {}).get("blob") and _finished(db, r.job_id)
    ]


def _blobs_in(value: Any) -> Iterator[str]:
    """Every blob a job's parameters name (weights it is about to fetch, a checkpoint to resume from …)."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("blob", "sha256") and isinstance(item, str):
                yield item
            else:
                yield from _blobs_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _blobs_in(item)


def live_blobs(
    db: Session,
    without_files: set[str] = frozenset(),
    without_ops: set[str] = frozenset(),
    without_checkpoints: set[str] = frozenset(),
) -> set[str]:
    """Every stored file something still uses; anything else in storage is left over.

    The ``without_*`` sets leave out records a cleanup is about to let go (run files by id,
    the files of model jobs by id, checkpoints by run id), so a preview counts what a cleanup would free.
    """
    live: set[str] = set()
    live.update(k for k in db.scalars(select(Item.thumb_key)) if k)
    live.update(b for b in db.scalars(select(Template.blob)) if b)  # every version: old runs may run again
    live.update(db.scalars(select(ModelVersion.blob).where(ModelVersion.deleted_at.is_(None))))
    deleted_runs = set(db.scalars(select(Run.id).where(Run.deleted_at.is_not(None))))
    for file_id, run_id, blob in db.execute(select(RunFile.id, RunFile.run_id, RunFile.blob)):
        if run_id not in deleted_runs and file_id not in without_files:
            live.add(blob)
    for run_id, checkpoint in db.execute(select(Run.id, Run.checkpoint).where(Run.deleted_at.is_(None))):
        if (checkpoint or {}).get("blob") and run_id not in without_checkpoints:
            live.add(checkpoint["blob"])
    deleted_models = set(db.scalars(select(ModelVersion.id).where(ModelVersion.deleted_at.is_not(None))))
    for op in db.scalars(select(ModelJob)):
        if op.model_id in deleted_models:
            continue
        if op.id not in without_ops:
            live.update(f["blob"] for f in op.files or [] if f.get("blob"))
        live.update(i["blob"] for i in (op.params or {}).get("images") or [] if i.get("blob"))
    for params in db.scalars(select(Job.params).where(Job.status.not_in(TERMINAL_STATUSES))):
        live.update(_blobs_in(params))
    return live


def _stored(settings: HubSettings) -> Iterator[tuple[Path, str, int, float]]:
    root = settings.storage_dir / "blobs"
    for folder, _, names in os.walk(root):
        for name in names:
            path = Path(folder) / name
            try:
                info = path.stat()
            except OSError:
                continue
            yield path, name, info.st_size, info.st_mtime


def cleanup(db: Session, settings: HubSettings, apply: bool, user: User | None = None) -> dict[str, Any]:
    """What a cleanup removes under the saved policy; with ``apply`` it removes it."""
    policy = get_policy(db)
    files, ops = _old_outputs(db, policy["outputs_days"])
    checkpoints = _old_checkpoints(db, policy["checkpoint_days"])
    report = {
        "policy": policy,
        "outputs": {
            "files": len(files) + sum(len(op.files or []) for op in ops),
            "bytes": sum(f.bytes for f in files) + sum(int(f.get("bytes") or 0) for op in ops for f in op.files or []),
        },
        "checkpoints": {"runs": len(checkpoints), "bytes": sum(int(r.checkpoint.get("bytes") or 0) for r in checkpoints)},
    }
    # What nothing will use once those are let go: deleted things, the files above, uploads never used.
    live = live_blobs(db, {f.id for f in files}, {op.id for op in ops}, {r.id for r in checkpoints})
    if apply:
        for f in files:
            db.delete(f)
        for op in ops:
            op.files = []
        for run in checkpoints:
            run.checkpoint = {}
        db.flush()
    now = time.time()
    freed = removed = 0
    for path, digest, size, mtime in _stored(settings):
        if digest in live or now - mtime < GRACE_S:
            continue
        removed, freed = removed + 1, freed + size
        if apply:
            try:
                path.unlink()
            except OSError:
                removed, freed = removed - 1, freed - size
    leftovers = 0
    incoming = settings.storage_dir / "incoming"
    if incoming.is_dir():
        for part in incoming.iterdir():  # uploads cut off half-way
            try:
                if now - part.stat().st_mtime >= GRACE_S:
                    leftovers += 1
                    if apply:
                        part.unlink()
            except OSError:
                pass
    report["removed"] = {"files": removed, "bytes": freed, "partial_uploads": leftovers}
    if apply and user is not None:
        services.audit(db, user.id, "ui", "storage.cleanup", "", files=removed, bytes=freed)
    return report
