"""Finding labels that are likely wrong, from what models predicted when they were evaluated.

An evaluation records, for every image it measured, the model's predictions and how they compared with
the labels: found, missed (labels the model did not find) and extra (predictions no label backs). Many
mistakes on one image often mean its labels are wrong, so the dataset screen can list the images by them
and the item view can show the predictions over the labels and take one as a label.

Images are matched by their sha256: the evaluation exports them under that name. Predictions keep the
evaluated release's label space; an evaluation is offered on the versions of the same dataset that share it.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names

from .catalog_models import Dataset, Item, ItemPrediction, QaDismissal, Release, ReleaseItem
from .db import iso
from .models import Job
from .services import ServiceError
from .train_models import ModelJob, ModelVersion

CHUNK = 2000


def ingest(db: Session, op: ModelJob, raw: bytes) -> int:
    """Store an evaluation's per-image predictions (``bm.log_predictions``). Returns how many images matched."""
    release = db.get(Release, op.release_id) if op.release_id else None
    if release is None:
        return 0
    try:
        images = json.loads(raw.decode("utf-8")).get("images") or []
    except (ValueError, UnicodeDecodeError, AttributeError):
        raise ServiceError(400, "예측 파일을 읽을 수 없습니다.") from None
    dataset = db.get(Dataset, release.dataset_id)
    job = db.get(Job, op.job_id) if op.job_id else None
    # The model numbers its classes 0..n-1; the release's label space may number them otherwise.
    class_ids = list((job.params or {}).get("class_ids") or []) if job else []

    def label_space_id(n: int) -> int | None:
        if not class_ids:
            return n
        return class_ids[n] if 0 <= n < len(class_ids) else None

    by_sha: dict[str, dict[str, Any]] = {}
    for entry in images:
        if isinstance(entry, dict) and isinstance(entry.get("image"), str):
            by_sha[entry["image"].lower()] = entry
    db.execute(delete(ItemPrediction).where(ItemPrediction.op_id == op.id))
    shas = list(by_sha)
    matched = 0
    for start in range(0, len(shas), CHUNK):
        chunk = shas[start : start + CHUNK]
        items = db.execute(
            select(Item.id, Item.sha256).where(Item.project_id == dataset.project_id, Item.sha256.in_(chunk))
        ).all()
        rows = []
        for item_id, sha in items:
            entry = by_sha[sha]
            labels, scores = [], []
            for label, score in zip(entry.get("labels") or [], entry.get("scores") or [], strict=False):
                try:
                    cls = label_space_id(int(label[0]))
                    values = [float(v) for v in label[1:]]
                except (TypeError, ValueError, IndexError):
                    continue
                if cls is None:
                    continue
                labels.append([cls, *values])
                scores.append(round(float(score), 4))
            counts = {k: max(0, int(entry.get(k) or 0)) for k in ("found", "missed", "extra")}
            wrong = counts["missed"] + counts["extra"]
            rows.append({"op_id": op.id, "item_id": item_id, **counts, "wrong": wrong, "labels": labels, "scores": scores})
        if rows:
            db.execute(insert(ItemPrediction), rows)
            matched += len(rows)
    return matched


def _offered(db: Session, release: Release) -> list[ModelJob]:
    """Evaluations on this release or another version of its dataset with the same classes, newest first,
    by a model of the dataset's task: a Seg model's view of a Det dataset is no check of its labels."""
    siblings = select(Release.id).where(
        Release.dataset_id == release.dataset_id, Release.label_space_id == release.label_space_id
    )
    has_rows = select(ItemPrediction.op_id).where(ItemPrediction.op_id == ModelJob.id).exists()
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.release_id.in_(siblings), has_rows)
        .order_by(ModelJob.created_at.desc())
    ).all()
    dataset = db.get(Dataset, release.dataset_id)
    task = (dataset.task if dataset else "") or "detect"
    models = {m.id: m for m in db.scalars(select(ModelVersion).where(ModelVersion.id.in_({op.model_id for op in ops})))}
    return [op for op in ops if op.model_id not in models or task_names.same(models[op.model_id].task or "detect", task)]


MISTAKES = ("extra", "missed")  # what an evaluation finds, judged like a rule check's finding


def let_be(check: Any, item_id: Any, revision_id: Any) -> Any:
    """SQL: someone found this image's labels (this revision of them) right for this check ("문제 없음")."""
    return (
        select(QaDismissal.item_id)
        .where(QaDismissal.item_id == item_id, QaDismissal.check == check, QaDismissal.revision_id == revision_id)
        .exists()
    )


def model_ref(model: ModelVersion | None) -> dict[str, Any] | None:
    """The model an evaluation is by, as the 검수 screen names it."""
    if model is None:
        return None
    aliases = list(model.aliases or ([model.alias] if model.alias else []))
    return {"id": model.id, "name": model.name, "version": model.version, "task": model.task or "detect", "aliases": aliases}


def release_checks(db: Session, release: Release) -> list[dict[str, Any]]:
    """The evaluations that can sort this release's images by likely wrong labels."""
    out = []
    for op in _offered(db, release):
        mine = (ItemPrediction.op_id == op.id, ReleaseItem.release_id == release.id)
        base = (
            select(func.count())
            .select_from(ItemPrediction)
            .join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
        )
        images = db.scalar(base.where(*mine)) or 0
        if not images:
            continue
        wrong = db.scalar(base.where(*mine, ItemPrediction.wrong > 0)) or 0
        model = db.get(ModelVersion, op.model_id)
        evaluated = db.get(Release, op.release_id)
        out.append(
            {
                "id": op.id,
                "model": model_ref(model),
                "release": {"id": evaluated.id, "version": evaluated.version} if evaluated else None,
                "split": (op.params or {}).get("split", ""),
                "created_at": iso(op.created_at),
                "images": images,
                "wrong": wrong,
            }
        )
    return out


def newest(db: Session, release: Release) -> dict[str, Any] | None:
    """The newest evaluation that measured this release's images: how many it found objects no label
    has on (``extra``) and labels it did not find on (``missed``)."""
    for op in _offered(db, release):
        mine = (ItemPrediction.op_id == op.id, ReleaseItem.release_id == release.id)
        base = (
            select(func.count())
            .select_from(ItemPrediction)
            .join(ReleaseItem, ReleaseItem.item_id == ItemPrediction.item_id)
        )
        images = db.scalar(base.where(*mine)) or 0
        if not images:
            continue
        model = db.get(ModelVersion, op.model_id)
        return {
            "id": op.id,
            "model": model_ref(model),
            "created_at": iso(op.created_at),
            "images": images,
            **{
                kind: db.scalar(
                    base.where(
                        *mine,
                        getattr(ItemPrediction, kind) > 0,
                        ~let_be(kind, ItemPrediction.item_id, ReleaseItem.revision_id),
                    )
                )
                or 0
                for kind in MISTAKES
            },
        }
    return None


def pending(db: Session, release: Release) -> list[dict[str, Any]]:
    """Evaluations of this dataset's versions still waiting or running: their view comes when they end."""
    siblings = select(Release.id).where(
        Release.dataset_id == release.dataset_id, Release.label_space_id == release.label_space_id
    )
    rows = db.execute(
        select(ModelJob, Job.status)
        .join(Job, Job.id == ModelJob.job_id)
        .where(ModelJob.kind == "eval", ModelJob.release_id.in_(siblings), Job.status.in_(("queued", "running")))
        .order_by(ModelJob.created_at.desc())
    ).all()
    return [
        {
            "id": op.id,
            "job_id": op.job_id,
            "status": status,
            "split": (op.params or {}).get("split", ""),
            "model": model_ref(db.get(ModelVersion, op.model_id)),
            "created_at": iso(op.created_at),
        }
        for op, status in rows
    ]


def check_of(db: Session, release: Release, op_id: str) -> ModelJob:
    op = db.get(ModelJob, op_id)
    if op is None or op.kind != "eval" or op not in _offered(db, release):
        raise ServiceError(404, "이 버전에 쓸 수 있는 평가가 아닙니다.")
    return op


def item_predictions(db: Session, release: Release, item_id: int, op_id: str) -> dict[str, Any]:
    op = check_of(db, release, op_id)
    row = db.get(ItemPrediction, (op.id, item_id))
    model = db.get(ModelVersion, op.model_id)
    return {
        "op_id": op.id,
        "model": model_ref(model),
        "split": (op.params or {}).get("split", ""),
        "evaluated": row is not None,
        "labels": row.labels if row else [],
        "scores": row.scores if row else [],
        "found": row.found if row else 0,
        "missed": row.missed if row else 0,
        "extra": row.extra if row else 0,
    }
