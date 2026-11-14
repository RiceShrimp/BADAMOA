"""Training logic: starting runs, the node-side protocol, metric history, the model registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core.protocol import TERMINAL_STATUSES, TrainFinish, TrainReport
from badamoa.core.templates import RESERVED, TemplateError, resolve_params

from . import catalog, registry, resources, services
from .catalog_models import Dataset, Item, LabelRevision, LabelSpace, Project, Release, ReleaseItem, StorageLocation
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelVersion, Run, RunFile, RunMetric, Template

ALIASES = ("champion", "challenger")
STAGES = ("data", "env", "train", "upload")
MANIFEST_MAX = 5000
# The number a run is judged by: decides which step counts as "best so far".
PRIMARY_METRIC = {"detect": "map50_95", "segment": "map50_95", "classify": "top1"}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
MAX_FILES_PER_RUN = 500
HISTORY_POINTS = 600  # more steps than this are thinned out when read
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
    template: str = "",
    environment: str = "",
    params: dict[str, Any] | None = None,
    name: str = "",
    model_name: str = "",
    classes: list[int] | None = None,
    unlabelled: str = "exclude",
    val_fraction: float = 0.2,
    gpus: int | None = None,
    min_vram_mb: int = 0,
    source_run_id: str | None = None,
    base_model_id: str = "",
    via: str = "api",
) -> Run:
    """Queue a training run: this release, with this template's code, in this environment.

    ``template`` and ``environment`` may be left empty: the project's default
    template and the template's default environment are used. ``base_model_id``
    starts from the weights of a registered model instead of pretrained ones
    (training on with new data).
    """
    release = catalog.get_release(db, release_id)
    if release.status != "ready":
        raise ServiceError(409, "아직 등록 중인 릴리스입니다.")
    dataset = db.get(Dataset, release.dataset_id)
    project = catalog.get_project(db, dataset.project_id)
    fallback = registry.DEFAULT_BY_TASK.get(project.task, registry.DEFAULT_TEMPLATE)
    row = registry.get_template(db, template or (project.settings or {}).get("template") or fallback)
    manifest = registry.manifest_of(row)
    if manifest.task != project.task:
        raise ServiceError(400, f"{row.title}은 {manifest.task} 템플릿이고 이 프로젝트는 {project.task}입니다.")
    try:
        settings = resolve_params(manifest, params or {})
    except TemplateError as e:
        raise ServiceError(400, str(e)) from e
    if unlabelled not in ("exclude", "background"):
        raise ServiceError(400, "unlabelled는 exclude 또는 background여야 합니다.")
    if not 0 <= val_fraction <= 0.9:
        raise ServiceError(400, "val_fraction은 0~0.9여야 합니다.")
    space = db.get(LabelSpace, release.label_space_id)
    known = {int(c["id"]) for c in space.classes}
    chosen = sorted({int(c) for c in classes or []})
    if not set(chosen) <= known:
        raise ServiceError(400, f"라벨 공간 {space.name}에 없는 클래스 id가 있습니다: {sorted(set(chosen) - known)}")
    node = data_node(db, release.id)
    env = registry.environment_for_run(db, manifest, environment, node)
    base = None
    if base_model_id:
        base = get_model(db, base_model_id)
        if base.project_id != project.id or base.trainer != row.name:
            raise ServiceError(400, f"{row.title}로 만든 이 프로젝트의 모델에서만 이어서 학습할 수 있습니다.")
    gpu_count = (1 if node.info.get("gpus") else 0) if gpus is None else gpus

    # What the model is called: the chosen variant (yolo11n.pt → yolo11n), else the template itself.
    variant = Path(str(settings.get(manifest.variant, ""))).stem if manifest.variant else ""
    variant = _slug(variant) or row.name
    taken = set(db.scalars(select(Run.name).where(Run.project_id == project.id)))
    epochs = settings.get("epochs")
    run = Run(
        project_id=project.id,
        name=_unique(_slug(name) or _slug(f"{dataset.name}-{release.version}-{variant}"), taken),
        release_id=release.id,
        trainer=row.name,
        template={
            "id": row.id,
            "name": row.name,
            "title": row.title,
            "version": row.version,
            "sha256": row.blob or "",
            "builtin": row.builtin,
            "variant": variant,
        },
        environment=env,
        config={
            **settings,
            "classes": chosen,
            "unlabelled": unlabelled,
            "val_fraction": val_fraction,
            "gpus": gpu_count,
            "min_vram_mb": min_vram_mb,
            "model_name": _slug(model_name),
            **({"base_model": {"id": base.id, "name": base.name, "version": base.version}} if base else {}),
        },
        epochs=int(epochs) if isinstance(epochs, int | float) else 0,
        source_run_id=source_run_id,
        created_by=user.id,
    )
    db.add(run)
    db.flush()
    job = services.create_job(
        db,
        user,
        "train.run",
        {
            "run_id": run.id,
            "release_id": release.id,
            "template": {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "builtin": row.builtin,
                "sha256": row.blob or "",
                "entrypoint": manifest.entrypoints["train"],
                "data_format": manifest.data_format,
            },
            "environment": env,
            "params": settings,
            "classes": chosen,
            "unlabelled": unlabelled,
            "val_fraction": val_fraction,
            **({"init_weights": {"blob": base.blob, "file_name": base.file_name or "best.pt"}} if base else {}),
        },
        node.id,
        via=via,
        gpus=gpu_count,
        min_vram_mb=min_vram_mb,
    )
    run.job_id = job.id
    services.audit(
        db, user.id, via, "run.create", run.id, name=run.name, release=f"{dataset.name} {release.version}", template=row.name
    )
    return run


def rerun(db: Session, user: User, run_id: str, via: str = "api") -> Run:
    """Start a new run with the same release, the same template version and the same settings."""
    source = get_run(db, run_id)
    c = source.config
    row = db.get(Template, (source.template or {}).get("id", "")) or registry.get_template(db, source.trainer)
    if row.deleted_at is not None and row.builtin:
        row = registry.get_template(db, row.name)  # built-in templates move on with the installed version
    if row.deleted_at is not None:
        raise ServiceError(409, f"이 실행이 쓴 템플릿({row.title} {row.version})이 삭제되었습니다.")
    declared = registry.manifest_of(row).params
    env = source.environment or {}
    return start_run(
        db,
        user,
        release_id=source.release_id,
        template=row.id,
        environment=env.get("id") or ("auto" if env.get("kind") == "auto" else ""),
        params={k: c[k] for k in declared if k in c},
        name=source.name,
        model_name=c.get("model_name", ""),
        classes=c.get("classes"),
        unlabelled=c.get("unlabelled", "exclude"),
        val_fraction=c.get("val_fraction", 0.2),
        gpus=c.get("gpus"),
        min_vram_mb=c.get("min_vram_mb", 0),
        base_model_id=(c.get("base_model") or {}).get("id", ""),
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
    return release_manifest(db, node, db.get(Release, run.release_id), after, limit)


def release_manifest(db: Session, node: Node, release: Release, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    """A page of a release's items as the node needs them to lay the data out for a template."""
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


def init_weights(db: Session, node: Node, job_id: str) -> str:
    """The storage key of the weights this run starts from, when it continues a registered model."""
    job, _ = _running(db, node, job_id)
    blob = ((job.params or {}).get("init_weights") or {}).get("blob")
    if not blob:
        raise ServiceError(404, "이 실행은 등록된 모델에서 시작하지 않습니다.")
    return blob


def system(db: Session, run: Run) -> dict[str, Any]:
    """CPU, memory and GPU load of the server while the run was on it."""
    job = db.get(Job, run.job_id) if run.job_id else None
    if job is None or job.node_id is None or job.started_at is None:
        return {"timestamps": [], "series": {}, "gpu_indices": [], "step_s": 60, "from": None, "to": None}
    history = resources.window(db, job.node_id, job.started_at, job.finished_at or utcnow())
    return {**history, "gpu_indices": job.gpu_indices or []}


def template_bundle(db: Session, node: Node, job_id: str) -> str:
    """The storage key (sha256) of the template code this run was started with."""
    _, run = _running(db, node, job_id)
    blob = (run.template or {}).get("sha256")
    if not blob:
        raise ServiceError(404, "이 실행의 템플릿은 내려받을 묶음이 없습니다 (내장 템플릿).")
    return blob


def report(db: Session, node: Node, job_id: str, body: TrainReport) -> dict[str, Any]:
    _, run = _running(db, node, job_id)
    data = body.data
    if body.event == "stage":
        if data.get("stage") in STAGES:
            run.stage = data["stage"]
    elif body.event == "export":
        run.export = data
    elif body.event == "info":
        run.info = {**(run.info or {}), **data}
    elif body.event == "epoch":
        primary = PRIMARY_METRIC.get(db.get(Project, run.project_id).task, "map50_95")
        rows: dict[int, RunMetric] = {}
        # One entry, or several the node gathered while the code logged faster than once a second.
        for entry in data["batch"] if isinstance(data.get("batch"), list) else [data]:
            # ``step`` is where the values go on the x axis; ``epoch`` / ``epochs`` only move the progress bar.
            metrics = {k: v for k, v in (entry.get("metrics") or {}).items() if isinstance(v, int | float)}
            if entry.get("epoch") is not None:
                run.epoch = max(run.epoch, int(entry["epoch"]))
            if entry.get("epochs"):
                run.epochs = int(entry["epochs"])
            if not metrics:
                continue
            step = int(entry.get("step", entry.get("epoch", 0)) or 0)
            row = rows.get(step) or db.get(RunMetric, (run.id, step))
            if row is None:
                row = RunMetric(run_id=run.id, step=step, data=metrics)
                db.add(row)
            else:
                row.data = {**row.data, **metrics}  # several reports for one step add up
            rows[step] = row
            # Until the run finishes, its summary is the best step so far.
            if metrics.get(primary, 0.0) >= (run.summary or {}).get(primary, -1.0):
                run.summary = {**(run.summary or {}), **metrics}
    return {"ok": True}


def add_file(db: Session, node: Node, job_id: str, name: str, blob: str, size: int) -> dict[str, Any]:
    """Attach a file the code recorded to its run. Recording the same name again replaces it."""
    _, run = _running(db, node, job_id)
    name = Path(name).name[:255] or "file"
    existing = db.scalars(select(RunFile).where(RunFile.run_id == run.id)).all()
    row = next((f for f in existing if f.name == name), None)
    if row is None:
        if len(existing) >= MAX_FILES_PER_RUN:
            raise ServiceError(400, f"한 실행에 기록할 수 있는 파일은 {MAX_FILES_PER_RUN}개까지입니다.")
        row = RunFile(run_id=run.id, name=name)
        db.add(row)
    row.blob, row.bytes = blob, size
    row.kind = "image" if name.lower().endswith(IMAGE_SUFFIXES) else "file"
    db.flush()
    return {"id": row.id, "name": row.name}


def get_file(db: Session, file_id: str) -> RunFile:
    row = db.get(RunFile, file_id)
    if row is None:
        raise ServiceError(404, "파일을 찾을 수 없습니다.")
    return row


def finish(db: Session, node: Node, job_id: str, body: TrainFinish) -> dict[str, Any]:
    """Record the final metrics and, when weights were uploaded, register them as a new model version."""
    job, run = _running(db, node, job_id)
    existing = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id))
    if existing is not None:  # a retried call
        return {"run_id": run.id, "model": {"id": existing.id, "name": existing.name, "version": existing.version}}
    if body.metrics:
        run.summary = body.metrics
    run.class_metrics = body.classes
    blob = str(body.weights.get("blob", ""))
    if not blob:  # the code reported results but no weights: metrics only
        return {"run_id": run.id, "model": None}
    release = db.get(Release, run.release_id)
    dataset = db.get(Dataset, release.dataset_id)
    project = db.get(Project, run.project_id)
    variant = (run.template or {}).get("variant") or Path(str(run.config.get("model", ""))).stem or run.trainer
    name = run.config.get("model_name") or _slug(f"{dataset.name}-{variant}")
    latest = db.scalar(
        select(func.max(ModelVersion.version)).where(ModelVersion.project_id == run.project_id, ModelVersion.name == name)
    )
    imgsz = run.config.get("imgsz")
    model = ModelVersion(
        project_id=run.project_id,
        name=name,
        version=(latest or 0) + 1,
        run_id=run.id,
        release_id=run.release_id,
        task=project.task,
        arch=variant,
        trainer=run.trainer,
        template=run.template or {},
        blob=blob,
        bytes=int(body.weights.get("bytes", 0)),
        file_name=str(body.weights.get("file_name", "best.pt"))[:128],
        imgsz=int(imgsz) if isinstance(imgsz, int | float) else 0,
        classes=list((run.export or {}).get("classes", [])),
        metrics=run.summary or {},
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


def _template_of(run: Run) -> dict[str, Any]:
    """The run's template; runs from before templates existed were always the built-in YOLO."""
    if run.template:
        return run.template
    title = "Ultralytics YOLO" if run.trainer == "yolo" else run.trainer
    return {"id": "", "name": run.trainer, "title": title, "version": "", "builtin": True}


def _settings_of(db: Session, run: Run) -> list[dict[str, Any]]:
    """The template settings of a run with their labels, in the template's order."""
    row = db.get(Template, (run.template or {}).get("id", ""))
    declared = registry.manifest_of(row).params if row else {}
    names = list(declared) + [k for k in run.config if k not in declared and k not in RESERVED]
    return [
        {
            "name": k,
            "label": (declared[k].label if k in declared else "") or k,
            "value": run.config[k],
            "group": declared[k].group if k in declared else "",
            "advanced": declared[k].advanced if k in declared else False,
            # Lets the screen show what was changed first and fold the untouched defaults away.
            "changed": k not in declared or run.config[k] != declared[k].default,
        }
        for k in names
        if k in run.config and not isinstance(run.config[k], dict)
    ]


def run_view(
    db: Session, settings: HubSettings, run: Run, detail: bool = False, node_names: dict[str, str] | None = None
) -> dict[str, Any]:
    job = db.get(Job, run.job_id) if run.job_id else None
    release = db.get(Release, run.release_id)
    dataset = db.get(Dataset, release.dataset_id) if release else None
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id, ModelVersion.deleted_at.is_(None)))
    job_data = services.job_view(db, settings, job, node_names) if job else None
    status = job_data["status"] if job_data else "failed"
    template = _template_of(run)
    project = db.get(Project, run.project_id)
    view: dict[str, Any] = {
        "id": run.id,
        "name": run.name,
        "project_id": run.project_id,
        "task": project.task if project else "detect",
        "template": {k: template.get(k, "") for k in ("id", "name", "title", "version", "variant")},
        "trainer_title": template.get("title") or run.trainer,
        "environment": {k: v for k, v in (run.environment or {"kind": "auto", "name": "자동 설치"}).items() if k != "path"},
        "status": status,
        # What a running job is doing right now: data → env → train → upload.
        "stage": (run.stage or "data") if status == "running" else "",
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
        "metrics": run.summary or {},  # final metrics once finished, the best step so far before that
        "model": {"id": model.id, "name": model.name, "version": model.version, "alias": model.alias} if model else None,
        "base_model": run.config.get("base_model"),  # the registered model this run started from, if any
        "source_run_id": run.source_run_id,
    }
    if detail:
        rows = db.scalars(select(RunMetric).where(RunMetric.run_id == run.id).order_by(RunMetric.step)).all()
        if len(rows) > 2 * HISTORY_POINTS:  # logged every iteration: a chart needs far fewer points
            stride = -(-len(rows) // HISTORY_POINTS)
            rows = [r for i, r in enumerate(rows) if i % stride == 0 or i == len(rows) - 1]
        keys = sorted({k for r in rows for k in r.data})
        view["history"] = {"steps": [r.step for r in rows], "series": {k: [r.data.get(k) for r in rows] for k in keys}}
        view["export"] = run.export or {}
        view["info"] = run.info or {}
        view["files"] = [
            {"id": f.id, "name": f.name, "kind": f.kind, "bytes": f.bytes, "url": catalog.media_url(settings, "file", f.id)}
            for f in db.scalars(select(RunFile).where(RunFile.run_id == run.id).order_by(RunFile.created_at, RunFile.name))
        ]
        view["class_metrics"] = run.class_metrics or []
        view["settings"] = _settings_of(db, run)
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
    template = model.template or (_template_of(run) if run else {"name": model.trainer, "title": model.trainer})
    view: dict[str, Any] = {
        "id": model.id,
        "project_id": model.project_id,
        "name": model.name,
        "version": model.version,
        "alias": model.alias,
        "task": model.task,
        "arch": model.arch,
        "trainer": model.trainer,
        "template": {k: template.get(k, "") for k in ("name", "title", "version")},
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
        view["environment"] = {k: v for k, v in ((run.environment if run else None) or {}).items() if k != "path"}
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
