"""오토라벨: a registered model labels a labeling set's images first, and people fix what it drew.

The model's template runs its ``predict`` code in autolabel mode on the server that holds the images,
over the set's images nobody has labelled yet, and records each image's predictions as an evaluation
does (``bm.log_predictions``: the dataset's own label form, so outlines and keypoints too). They are
kept as predictions, not labels:

- below the lowest confidence (``keep``) a prediction is dropped;
- whoever opens the image in the set starts from the rest and saves what they fixed;
- with ``accept``, an image whose predictions all reach it is saved into the work in progress at once
  (source ``model:<name> v<version>``) and goes to review, or is done when the set has no review.

Labels people made are never overwritten: only images still "not labelled yet" in the set, and still
without labels in the work in progress, are accepted.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names

from . import asks, catalog, drafts, labeling, modelops, services, training
from .catalog_models import ItemPrediction, LabelSet, LabelSetItem, LabelSpace, Release, ReleaseItem
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion

SINCE = (0, 52, 0)  # node versions from here on run a template's predict in autolabel mode
KEEP = 0.25  # predictions below this are dropped
ALIASES = ("production", "champion")


def _names(db: Session, release: Release) -> dict[str, int]:
    space = db.get(LabelSpace, release.label_space_id)
    return {str(c["name"]): int(c["id"]) for c in (space.classes if space else [])}


def _problem(db: Session, model: ModelVersion, task: str, names: dict[str, int]) -> str:
    """Why this model cannot label the set's images, or "" when it can."""
    if model.task and not task_names.same(model.task, task):
        return f"{task_names.short(model.task)}용 모델입니다 (이 데이터셋은 {task_names.short(task)})"
    if "predict" not in modelops.actions(db, model):
        return "이 모델 종류에는 예측 코드가 없습니다"
    if not any(name in names for name in model.classes or []):
        return "모델의 클래스가 이 데이터셋의 클래스와 하나도 맞지 않습니다"
    return ""


def _open(db: Session, found: LabelSet) -> Release:
    release = labeling._open_release(db, found)
    if release is None:
        raise ServiceError(409, "이 데이터셋에 등록이 끝난 버전이 없어 오토라벨을 돌릴 곳이 없습니다.")
    return release


def _todo(db: Session, found: LabelSet) -> int:
    return db.scalar(select(func.count()).where(LabelSetItem.set_id == found.id, LabelSetItem.state == "todo")) or 0


def _job_of(db: Session, found: LabelSet) -> Job | None:
    op = db.get(ModelJob, (found.autolabel or {}).get("op_id") or "")
    return db.get(Job, op.job_id) if op is not None and op.job_id else None


def choices(db: Session, set_id: str) -> dict[str, Any]:
    """The project's models that can label the set, the ones with an alias and the best first."""
    found = labeling.get_set(db, set_id)
    release = _open(db, found)
    task = catalog.release_task(db, release)
    names = _names(db, release)
    models = []
    for model in db.scalars(
        select(ModelVersion)
        .where(ModelVersion.project_id == found.project_id, ModelVersion.deleted_at.is_(None))
        .order_by(ModelVersion.created_at.desc())
    ):
        metric = training.PRIMARY_METRIC.get(model.task or "detect", "map50_95")
        value = (model.metrics or {}).get(metric)
        classes = list(model.classes or [])
        models.append(
            {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "aliases": training._aliases(model),
                "task": model.task or "detect",
                "metric": metric,
                "value": value if isinstance(value, int | float) else None,
                "classes": len(classes),
                "matched": sum(1 for name in classes if name in names),
                "missing": [name for name in classes if name not in names][:8],
                "problem": _problem(db, model, task, names),
                "created_at": iso(model.created_at),
            }
        )
    models.sort(
        key=lambda m: (
            bool(m["problem"]),
            not any(a in ALIASES for a in m["aliases"]),
            -(m["value"] if m["value"] is not None else -1),
        )
    )
    return {"todo": _todo(db, found), "keep": KEEP, "models": models}


def start(
    db: Session,
    user: User,
    set_id: str,
    model_id: str,
    keep: float = KEEP,
    accept: float | None = None,
    gpus: int | None = None,
    via: str = "ui",
) -> ModelJob:
    """Run the model over the set's images not labelled yet, on the server that has them (on a GPU when it has
    one, unless ``gpus=0``)."""
    found = labeling.get_set(db, set_id)
    release = _open(db, found)
    job = _job_of(db, found)
    if job is not None and job.status in ("queued", "running"):
        raise ServiceError(409, "이 묶음의 오토라벨이 이미 돌고 있습니다.")
    model = training.get_model(db, model_id)
    if model.project_id != found.project_id:
        raise ServiceError(400, "다른 프로젝트의 모델입니다.")
    names = _names(db, release)
    why = _problem(db, model, catalog.release_task(db, release), names)
    if why:
        raise ServiceError(400, f"이 모델로는 오토라벨을 돌릴 수 없습니다: {why}.")
    if not 0 < keep <= 1:
        raise ServiceError(400, "최소 확신도는 0보다 크고 1 이하여야 합니다.")
    if accept is not None and not keep <= accept <= 1:
        raise ServiceError(400, "자동 승인 확신도는 최소 확신도와 1 사이여야 합니다.")
    todo = _todo(db, found)
    if not todo:
        raise ServiceError(400, "이 묶음에는 아직 라벨링하지 않은 이미지가 없습니다.")
    node: Node = training.data_node(db, release.id)
    if not asks.answers_questions(node.agent_version or "", SINCE):
        need = ".".join(str(n) for n in SINCE)
        raise ServiceError(409, f"{node.name}의 BADAMOA를 {need} 이상으로 업데이트해야 오토라벨을 돌릴 수 있습니다.")
    # the model numbers its classes 0..n-1: each one's id in this dataset, by name (None: not in it, dropped)
    class_ids = [names.get(name) for name in model.classes or []]
    gpus = (1 if (node.info or {}).get("gpus") else 0) if gpus is None else max(0, min(int(gpus), 1))
    params = {"conf": keep, "keep": keep, "accept": accept, "set_id": found.id}
    extra = {"class_ids": class_ids}
    op = modelops._start(db, user, model, "autolabel", node, params, release_id=release.id, extra=extra, gpus=gpus, via=via)
    found.autolabel = {
        "op_id": op.id,
        "model": {"id": model.id, "name": model.name, "version": model.version},
        "keep": keep,
        "accept": accept,
        "items": todo,
        "started_by": user.id,
        "started_at": iso(utcnow()),
    }
    services.audit(db, user.id, via, "label_set.autolabel", found.id, model=model.id, items=todo, keep=keep, accept=accept)
    db.flush()
    return op


def manifest(db: Session, node: Node, op: ModelJob, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    """The set's images still to label, laid out as a version's manifest for the node."""
    release = db.get(Release, op.release_id) if op.release_id else None
    if release is None:
        raise ServiceError(404, "오토라벨할 버전이 없어졌습니다.")
    todo = select(LabelSetItem.item_id).where(
        LabelSetItem.set_id == (op.params or {}).get("set_id"), LabelSetItem.state == "todo"
    )
    return training.release_manifest(db, node, release, after, limit, only=todo)


def _kept(row: ItemPrediction | None, keep: float) -> list[tuple[list[float], float]]:
    if row is None:
        return []
    return [(label, score) for label, score in zip(row.labels or [], row.scores or [], strict=False) if score >= keep]


def apply(db: Session, op: ModelJob) -> dict[str, int]:
    """Once the model has run: count what it found and accept the images it is sure of (when asked)."""
    params = op.params or {}
    found = db.get(LabelSet, params.get("set_id") or "")
    release = db.get(Release, op.release_id) if op.release_id else None
    if found is None or release is None:
        return {}
    model = db.get(ModelVersion, op.model_id)
    starter = db.get(User, op.created_by) if op.created_by else None
    keep, accept = float(params.get("keep") or KEEP), params.get("accept")
    names = _names(db, release)
    seen = sorted({names[n] for n in (model.classes if model else []) or [] if n in names})
    draft, _ = drafts.edit_place(db, release)
    where = draft or release  # where an image's labels are now
    counts: Counter = Counter()
    rows = db.execute(
        select(LabelSetItem, ItemPrediction)
        .join(ItemPrediction, and_(ItemPrediction.op_id == op.id, ItemPrediction.item_id == LabelSetItem.item_id))
        .where(LabelSetItem.set_id == found.id)
    ).all()
    for member, prediction in rows:
        kept = _kept(prediction, keep)
        counts["images"] += 1
        counts["labels"] += len(kept)
        if not kept:
            counts["empty"] += 1
            continue
        if accept is None or starter is None or model is None or member.state != "todo":
            continue
        if any(score < float(accept) for _, score in kept):
            continue
        now = db.get(ReleaseItem, (where.id, member.item_id))
        if now is None or now.state != "unlabelled":  # labels are there already: a person's, not overwritten
            continue
        try:
            saved = drafts.save_labels(
                db,
                starter,
                release.id,
                member.item_id,
                [label for label, _ in kept],
                source=f"model:{model.name} v{model.version}",
                seen=seen,
                via="model",
            )
        except ServiceError:
            counts["skipped"] += 1
            continue
        where = db.get(Release, saved["release_id"]) or where  # the first save makes the work in progress
        member.state = "labelled" if found.review else "approved"
        member.labelled_by, member.labelled_at = op.created_by, utcnow()
        counts["accepted"] += 1
    result = {k: counts[k] for k in ("images", "labels", "accepted", "empty", "skipped")}
    found.autolabel = {**(found.autolabel or {}), "result": result, "finished_at": iso(utcnow())}
    db.flush()
    return result


def predictions_for(db: Session, set_id: str, item_id: int) -> dict[str, Any]:
    """What the set's autolabel predicted for one image (above the lowest confidence): the editor starts from it."""
    found = labeling.get_set(db, set_id)
    info = found.autolabel or {}
    if not info.get("op_id"):
        raise ServiceError(404, "이 묶음은 오토라벨을 돌리지 않았습니다.")
    keep = float(info.get("keep") or KEEP)
    kept = _kept(db.get(ItemPrediction, (info["op_id"], item_id)), keep)
    return {
        "model": info.get("model"),
        "keep": keep,
        "labels": [label for label, _ in kept],
        "scores": [score for _, score in kept],
    }


def status(db: Session, found: LabelSet) -> dict[str, Any] | None:
    """The set's last autolabel: which model, how sure, where it is, and what came of it."""
    info = found.autolabel or {}
    if not info.get("op_id"):
        return None
    job = _job_of(db, found)
    state = job.status if job is not None else "failed"
    return {
        **{k: info.get(k) for k in ("model", "keep", "accept", "items", "started_at", "finished_at", "result")},
        "job_id": job.id if job is not None else None,
        "status": state,
        "error": (job.error or "")[:500] if job is not None and state == "failed" else None,
    }
