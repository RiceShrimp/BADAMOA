"""Training logic: starting runs, the node-side protocol, metric history, the model registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core.jobs import BUILTIN_JOB_TYPES
from badamoa.core.plugins import PLUGINS
from badamoa.core.protocol import TERMINAL_STATUSES, TrainFinish, TrainReport

from . import catalog, services
from .catalog_models import Dataset, Item, LabelRevision, LabelSpace, Release, ReleaseItem, StorageLocation
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelVersion, Run, RunMetric

TRAINERS = {"yolo": "train.yolo", "dummy": "train.dummy"}
ALIASES = ("champion", "challenger")
MANIFEST_MAX = 5000
# Shown first in tables and tiles; anything else a trainer reports is kept and charted too.
KEY_METRICS = ("map50_95", "map50", "precision", "recall")
_NAME_RE = re.compile(r"[^0-9A-Za-z가-힣._-]+")


def _slug(text: str) -> str:
    return _NAME_RE.sub("-", text.strip()).strip("-")


def _unique(base: str, taken: set[str]) -> str:
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


# ---- starting runs -----------------------------------------------------------------


def data_node(db: Session, release_id: str) -> Node:
    """The node that holds the release's image files; training has to run there."""
    node_ids = db.scalars(
        select(StorageLocation.node_id)
        .join(Item, Item.location_id == StorageLocation.id)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release_id)
        .distinct()
    ).all()
    if not node_ids:
        raise ServiceError(400, "이 릴리스의 이미지가 어느 노드에 있는지 알 수 없습니다.")
    if len(node_ids) > 1:
        raise ServiceError(400, "이 릴리스의 이미지가 여러 노드에 흩어져 있습니다. 한 노드에 모인 릴리스만 학습할 수 있습니다.")  # noqa: E501
    node = db.get(Node, node_ids[0])
    if node is None:
        raise ServiceError(400, "이 릴리스의 이미지가 있던 노드가 삭제되었습니다.")
    return node


def start_run(
    db: Session,
    user: User,
    *,
    release_id: str,
    trainer: str = "yolo",
    name: str = "",
    model_name: str = "",
    model: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    pretrained: bool = True,
    classes: list[int] | None = None,
    unlabelled: str = "exclude",
    val_fraction: float = 0.2,
    gpus: int | None = None,
    min_vram_mb: int = 0,
    extra: dict[str, Any] | None = None,
    source_run_id: str | None = None,
    via: str = "api",
) -> Run:
    if trainer not in TRAINERS:
        raise ServiceError(400, f"trainer는 {', '.join(TRAINERS)} 중 하나여야 합니다.")
    release = catalog.get_release(db, release_id)
    if release.status != "ready":
        raise ServiceError(409, "아직 등록 중인 릴리스입니다.")
    dataset = db.get(Dataset, release.dataset_id)
    project = catalog.get_project(db, dataset.project_id)
    if project.task != "detect":
        raise ServiceError(400, "지금은 객체 검출(detect) 프로젝트만 학습할 수 있습니다.")
    if unlabelled not in ("exclude", "background"):
        raise ServiceError(400, "unlabelled는 exclude 또는 background여야 합니다.")
    space = db.get(LabelSpace, release.label_space_id)
    known = {int(c["id"]) for c in space.classes}
    chosen = sorted({int(c) for c in classes or []})
    if not set(chosen) <= known:
        raise ServiceError(400, f"라벨 공간 {space.name}에 없는 클래스 id가 있습니다: {sorted(set(chosen) - known)}")
    model = model.strip() or "yolo11n.pt"
    node = data_node(db, release.id)
    gpu_count = (1 if node.info.get("gpus") else 0) if gpus is None else gpus

    config = {
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "pretrained": pretrained,
        "classes": chosen,
        "unlabelled": unlabelled,
        "val_fraction": val_fraction,
        "extra": extra or {},
    }
    arch = Path(model).stem
    taken = set(db.scalars(select(Run.name).where(Run.project_id == project.id)))
    run = Run(
        project_id=project.id,
        name=_unique(_slug(name) or _slug(f"{dataset.name}-{release.version}-{arch}"), taken),
        release_id=release.id,
        trainer=trainer,
        config={**config, "gpus": gpu_count, "min_vram_mb": min_vram_mb, "model_name": _slug(model_name)},
        epochs=epochs,
        source_run_id=source_run_id,
        created_by=user.id,
    )
    db.add(run)
    db.flush()
    job = services.create_job(
        db,
        user,
        TRAINERS[trainer],
        {"run_id": run.id, "release_id": release.id, **config},
        node.id,
        via=via,
        gpus=gpu_count,
        min_vram_mb=min_vram_mb,
    )
    run.job_id = job.id
    services.audit(db, user.id, via, "run.create", run.id, name=run.name, release=f"{dataset.name} {release.version}")
    return run


def rerun(db: Session, user: User, run_id: str, via: str = "api") -> Run:
    """Start a new run with the same release and settings."""
    source = get_run(db, run_id)
    c = source.config
    return start_run(
        db,
        user,
        release_id=source.release_id,
        trainer=source.trainer,
        name=source.name,
        model_name=c.get("model_name", ""),
        model=c["model"],
        epochs=c["epochs"],
        imgsz=c["imgsz"],
        batch=c["batch"],
        workers=c.get("workers", 4),
        pretrained=c.get("pretrained", True),
        classes=c.get("classes"),
        unlabelled=c.get("unlabelled", "exclude"),
        val_fraction=c.get("val_fraction", 0.2),
        gpus=c.get("gpus"),
        min_vram_mb=c.get("min_vram_mb", 0),
        extra=c.get("extra"),
        source_run_id=source.id,
        via=via,
    )


# ---- node protocol (the node running the job calls these) ---------------------------


def _running(db: Session, node: Node, job_id: str) -> tuple[Job, Run]:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or not job.type.startswith("train.") or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 학습 작업이 아닙니다.")
    run = db.scalar(select(Run).where(Run.job_id == job.id))
    if run is None:
        raise ServiceError(404, "이 작업에 연결된 학습 실행이 없습니다.")
    return job, run


def manifest(db: Session, node: Node, job_id: str, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    """What to train on: every item of the release with its split and the labels the release pins."""
    _, run = _running(db, node, job_id)
    release = db.get(Release, run.release_id)
    dataset = db.get(Dataset, release.dataset_id)
    space = db.get(LabelSpace, release.label_space_id)
    limit = max(1, min(limit, MANIFEST_MAX))
    rows = db.execute(
        select(ReleaseItem, Item, LabelRevision)
        .join(Item, Item.id == ReleaseItem.item_id)
        .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id)
        .where(ReleaseItem.release_id == release.id, Item.id > after)
        .order_by(Item.id)
        .limit(limit)
    ).all()
    locations = {
        loc.id: loc.root for loc in db.scalars(select(StorageLocation).where(StorageLocation.node_id == node.id))
    }
    return {
        "dataset": dataset.name,
        "version": release.version,
        "classes": space.classes,
        "locations": locations,
        "items": [
            {
                "id": item.id,
                "sha256": item.sha256,
                "rel_path": item.rel_path,
                "bytes": item.bytes,
                "location_id": item.location_id,
                "split": member.split,
                "state": revision.state,
                "boxes": revision.boxes,
                "labelled_class_ids": revision.labelled_class_ids,
            }
            for member, item, revision in rows
        ],
        "next": rows[-1][1].id if len(rows) == limit else None,
    }


def report(db: Session, node: Node, job_id: str, body: TrainReport) -> dict[str, Any]:
    _, run = _running(db, node, job_id)
    if body.event == "export":
        run.export = body.data
    elif body.event == "info":
        run.info = {**(run.info or {}), **body.data}
    elif body.event == "epoch":
        step = int(body.data.get("epoch", 0))
        metrics = {k: v for k, v in (body.data.get("metrics") or {}).items() if isinstance(v, int | float)}
        row = db.get(RunMetric, (run.id, step))
        if row is None:
            db.add(RunMetric(run_id=run.id, step=step, data=metrics))
        else:
            row.data = metrics
        run.epoch = max(run.epoch, step)
        if body.data.get("epochs"):
            run.epochs = int(body.data["epochs"])
        # Until the run finishes, its summary is the best epoch so far.
        if metrics.get("map50_95", 0.0) >= (run.summary or {}).get("map50_95", -1.0):
            run.summary = metrics
    return {"ok": True}


def finish(db: Session, node: Node, job_id: str, body: TrainFinish) -> dict[str, Any]:
    """Record the final metrics and register the uploaded weights as a new model version."""
    job, run = _running(db, node, job_id)
    blob = str(body.weights.get("blob", ""))
    if not blob:
        raise ServiceError(400, "가중치가 업로드되지 않았습니다.")
    existing = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id))
    if existing is not None:  # a retried call
        return {"run_id": run.id, "model": {"id": existing.id, "name": existing.name, "version": existing.version}}
    release = db.get(Release, run.release_id)
    dataset = db.get(Dataset, release.dataset_id)
    arch = Path(run.config.get("model", "")).stem
    name = run.config.get("model_name") or _slug(f"{dataset.name}-{arch}")
    latest = db.scalar(
        select(func.max(ModelVersion.version)).where(ModelVersion.project_id == run.project_id, ModelVersion.name == name)
    )
    run.summary = body.metrics
    run.class_metrics = body.classes
    model = ModelVersion(
        project_id=run.project_id,
        name=name,
        version=(latest or 0) + 1,
        run_id=run.id,
        release_id=run.release_id,
        arch=arch,
        trainer=run.trainer,
        blob=blob,
        bytes=int(body.weights.get("bytes", 0)),
        file_name=str(body.weights.get("file_name", "best.pt"))[:128],
        imgsz=int(run.config.get("imgsz", 0)),
        classes=list((run.export or {}).get("classes", [])),
        metrics=body.metrics,
        class_metrics=body.classes,
        created_by=job.created_by,
    )
    db.add(model)
    db.flush()
    services.audit(db, job.created_by, "agent", "model.create", model.id, name=name, version=model.version)
    return {"run_id": run.id, "model": {"id": model.id, "name": model.name, "version": model.version}}


# ---- runs: views ---------------------------------------------------------------------


def get_run(db: Session, run_id: str) -> Run:
    run = db.get(Run, run_id)
    if run is None or run.deleted_at is not None:
        raise ServiceError(404, "학습 실행을 찾을 수 없습니다.")
    return run


def run_view(
    db: Session, settings: HubSettings, run: Run, detail: bool = False, node_names: dict[str, str] | None = None
) -> dict[str, Any]:
    job = db.get(Job, run.job_id) if run.job_id else None
    release = db.get(Release, run.release_id)
    dataset = db.get(Dataset, release.dataset_id) if release else None
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id, ModelVersion.deleted_at.is_(None)))
    job_data = services.job_view(db, settings, job, node_names) if job else None
    spec = BUILTIN_JOB_TYPES.get(TRAINERS.get(run.trainer, ""))
    plugin = PLUGINS.get(spec.plugin) if spec and spec.plugin else None
    view: dict[str, Any] = {
        "id": run.id,
        "name": run.name,
        "project_id": run.project_id,
        "trainer": run.trainer,
        "trainer_title": plugin.title if plugin else (spec.title if spec else run.trainer),
        "status": job_data["status"] if job_data else "failed",
        "job_id": run.job_id,
        "node_name": job_data["node_name"] if job_data else None,
        "wait_reason": job_data["wait_reason"] if job_data else None,
        "error": job_data["error"] if job_data else None,
        "duration_s": job_data["duration_s"] if job_data else None,
        "gpu_indices": job_data["gpu_indices"] if job_data else [],
        "created_via": job_data["created_via"] if job_data else "",
        "created_at": iso(run.created_at),
        "epochs": run.epochs,
        "epoch": run.epoch,
        "config": run.config,
        "release": {"id": release.id, "dataset": dataset.name, "version": release.version} if release and dataset else None,
        "metrics": run.summary or {},  # final metrics once finished, the best epoch so far before that
        "model": {"id": model.id, "name": model.name, "version": model.version, "alias": model.alias} if model else None,
        "source_run_id": run.source_run_id,
    }
    if detail:
        rows = db.scalars(select(RunMetric).where(RunMetric.run_id == run.id).order_by(RunMetric.step)).all()
        keys = sorted({k for r in rows for k in r.data})
        view["history"] = {"steps": [r.step for r in rows], "series": {k: [r.data.get(k) for r in rows] for k in keys}}
        view["export"] = run.export or {}
        view["info"] = run.info or {}
        view["class_metrics"] = run.class_metrics or []
    return view


def list_runs(db: Session, settings: HubSettings, project_id: str, limit: int = 200) -> list[dict[str, Any]]:
    catalog.get_project(db, project_id)
    runs = db.scalars(
        select(Run)
        .where(Run.project_id == project_id, Run.deleted_at.is_(None))
        .order_by(Run.created_at.desc())
        .limit(max(1, min(limit, 500)))
    ).all()
    names = services.node_name_map(db)
    return [run_view(db, settings, run, node_names=names) for run in runs]


def delete_run(db: Session, user: User, run_id: str) -> None:
    run = get_run(db, run_id)
    job = db.get(Job, run.job_id) if run.job_id else None
    if job is not None and job.status not in TERMINAL_STATUSES:
        raise ServiceError(409, "실행 중인 학습은 먼저 중단하세요.")
    if user.role != "admin" and run.created_by != user.id:
        raise ServiceError(403, "자신이 시작한 학습만 지울 수 있습니다.")
    run.deleted_at = utcnow()
    services.audit(db, user.id, "api", "run.delete", run.id, name=run.name)


# ---- models ---------------------------------------------------------------------------


def get_model(db: Session, model_id: str) -> ModelVersion:
    model = db.get(ModelVersion, model_id)
    if model is None or model.deleted_at is not None:
        raise ServiceError(404, "모델을 찾을 수 없습니다.")
    return model


def model_view(db: Session, settings: HubSettings, model: ModelVersion, detail: bool = False) -> dict[str, Any]:
    release = db.get(Release, model.release_id) if model.release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    run = db.get(Run, model.run_id) if model.run_id else None
    source = None
    if release and dataset:
        source = {"id": release.id, "dataset": dataset.name, "version": release.version, "deleted": bool(release.deleted_at)}
    view: dict[str, Any] = {
        "id": model.id,
        "project_id": model.project_id,
        "name": model.name,
        "version": model.version,
        "alias": model.alias,
        "task": model.task,
        "arch": model.arch,
        "trainer": model.trainer,
        "bytes": model.bytes,
        "imgsz": model.imgsz,
        "class_count": len(model.classes or []),
        "metrics": model.metrics or {},
        "created_at": iso(model.created_at),
        "run": {"id": run.id, "name": run.name} if run and run.deleted_at is None else None,
        "release": source,
    }
    if detail:
        view["classes"] = model.classes or []
        view["class_metrics"] = model.class_metrics or []
        view["file_name"] = model.file_name
        view["sha256"] = model.blob
        view["note"] = model.note
        view["config"] = run.config if run else {}
        view["info"] = run.info if run else {}
        view["export"] = run.export if run else {}
        view["download_url"] = catalog.media_url(settings, "model", model.id)
    return view


def list_models(db: Session, settings: HubSettings, project_id: str) -> list[dict[str, Any]]:
    catalog.get_project(db, project_id)
    models = db.scalars(
        select(ModelVersion)
        .where(ModelVersion.project_id == project_id, ModelVersion.deleted_at.is_(None))
        .order_by(ModelVersion.name, ModelVersion.version.desc())
    ).all()
    return [model_view(db, settings, m) for m in models]


def set_alias(db: Session, user: User, model_id: str, alias: str, via: str = "api") -> ModelVersion:
    """Give a version an alias. Within a model family an alias sits on one version at a time."""
    alias = alias.strip()
    if alias and alias not in ALIASES:
        raise ServiceError(400, f"alias는 {', '.join(ALIASES)} 중 하나이거나 비워 두어야 합니다.")
    model = get_model(db, model_id)
    if alias:
        holders = db.scalars(
            select(ModelVersion).where(
                ModelVersion.project_id == model.project_id, ModelVersion.name == model.name, ModelVersion.alias == alias
            )
        )
        for other in holders:
            other.alias = ""
    model.alias = alias
    services.audit(db, user.id, via, "model.alias", model.id, name=model.name, version=model.version, alias=alias)
    return model


def delete_model(db: Session, user: User, model_id: str) -> None:
    services.require_admin(user)
    model = get_model(db, model_id)
    model.deleted_at = utcnow()
    model.alias = ""
    services.audit(db, user.id, "api", "model.delete", model.id, name=model.name, version=model.version)
