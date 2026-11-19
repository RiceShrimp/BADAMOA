"""What can be done with a registered model: evaluate it on a release, export it, try it on sample images.

Each is a job that runs an entry point (``eval`` / ``export`` / ``predict``) of the
template version that trained the model, so the code that reads the weights is
the code that wrote them. A template that declares no such entry point simply
does not offer the action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
from badamoa.core.storage import Storage

from . import catalog, checks, drafts, registry, services, training
from .catalog_models import Dataset, LabelSpace, Release
from .config import HubSettings
from .db import iso
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run, Template

ACTIONS = {"eval": "평가", "export": "내보내기", "predict": "테스트"}
# 오토라벨 runs the template's predict code in autolabel mode (see ``autolabel``)
ENTRY = {"autolabel": "predict"}
EXPORT_FORMATS = ("onnx", "torchscript")
MAX_TEST_IMAGES = 20
MAX_RESULT_ITEMS = 200
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


def _template_of(db: Session, model: ModelVersion) -> Template:
    row = db.get(Template, (model.template or {}).get("id", "")) or registry.get_template(db, model.trainer)
    if row.deleted_at is not None and row.builtin:
        row = registry.get_template(db, row.name)
    if row.deleted_at is not None:
        raise ServiceError(409, f"이 모델을 만든 템플릿({row.title} {row.version})이 삭제되었습니다.")
    return row


def actions(db: Session, model: ModelVersion) -> list[str]:
    """Which of evaluate / export / test this model's template has code for."""
    try:
        entrypoints = registry.manifest_of(_template_of(db, model)).entrypoints
    except ServiceError:
        return []
    return [action for action in ACTIONS if action in entrypoints]


def _training_node(db: Session, model: ModelVersion) -> Node | None:
    """The server that trained the model: its environment is already there."""
    run = db.get(Run, model.run_id) if model.run_id else None
    job = db.get(Job, run.job_id) if run and run.job_id else None
    node = db.get(Node, job.node_id) if job and job.node_id else None
    return node if node is not None and node.enabled else None


def _start(
    db: Session,
    user: User,
    model: ModelVersion,
    kind: str,
    node: Node | None,
    params: dict[str, Any],
    *,
    release_id: str | None = None,
    extra: dict[str, Any] | None = None,
    gpus: int = 0,
    via: str = "api",
) -> ModelJob:
    row = _template_of(db, model)
    manifest = registry.manifest_of(row)
    entry = ENTRY.get(kind, kind)
    if entry not in manifest.entrypoints:
        raise ServiceError(400, f"{row.title}에는 {ACTIONS[entry]} 코드가 없습니다.")
    if node is None:
        raise ServiceError(409, "이 모델을 학습한 서버가 연결되어 있지 않아 실행할 곳이 없습니다.")
    run = db.get(Run, model.run_id) if model.run_id else None
    wanted = (run.environment or {}) if run else {}
    # the environment it was trained in: a registered one, the one the template installs, or its declared image
    same = wanted.get("id") or {"auto": "auto", "docker": registry.DECLARED_IMAGE}.get(wanted.get("kind", ""), "")
    if same == registry.DECLARED_IMAGE and not manifest.environment.image:
        same = ""
    env = registry.environment_for_run(db, manifest, same, node)
    _, unfit = training.fitting([node], env, gpus)
    if unfit:
        raise ServiceError(400, " ".join(unfit))
    declared = manifest.params
    op = ModelJob(model_id=model.id, kind=kind, release_id=release_id, params=params, created_by=user.id)
    db.add(op)
    db.flush()
    job = services.create_job(
        db,
        user,
        "model.run",
        {
            "op_id": op.id,
            "model_id": model.id,
            "action": kind,
            "template": {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "builtin": row.builtin,
                "sha256": row.blob or "",
                "entrypoint": manifest.entrypoints[entry],
                "data_format": manifest.data_format,
            },
            "environment": env,
            "params": params,
            # How the model was trained (architecture, image size …): the code needs it to rebuild the network.
            "train_params": {k: v for k, v in (run.config if run else {}).items() if k in declared},
            "classes": model.classes or [],
            "weights": {"blob": model.blob, "file_name": model.file_name or "best.pt"},
            **(extra or {}),
        },
        node.id,
        via=via,
        gpus=gpus,
    )
    op.job_id = job.id
    services.audit(db, user.id, via, f"model.{kind}", model.id, name=model.name, version=model.version)
    return op


def start_eval(
    db: Session, user: User, model_id: str, release_id: str, split: str = "val", gpus: int | None = None, via: str = "api"
) -> ModelJob:
    """Measure a model on a release: the way to compare models trained on different data.

    Like training, it runs where the release's images are and takes a GPU when that server has one
    (``gpus=0`` forces the CPU).
    """
    model = training.get_model(db, model_id)
    release = catalog.get_release(db, release_id)
    catalog.require_ready(release)
    dataset = db.get(Dataset, release.dataset_id)
    if dataset.project_id != model.project_id:
        raise ServiceError(400, "다른 프로젝트의 버전으로는 평가할 수 없습니다.")
    if split not in ("val", "test", "train"):
        raise ServiceError(400, "split은 val, test, train 중 하나여야 합니다.")
    task = catalog.dataset_task(dataset)
    if model.task and not task_names.same(model.task, task):
        short = task_names.short
        raise ServiceError(400, f"이 모델은 {short(model.task)}용이고 이 데이터셋은 {short(task)}입니다.")
    if not (((release.stats or {}).get("splits") or {}).get(split) or {}).get("items"):
        raise ServiceError(400, f"이 버전의 {split}에는 이미지가 없습니다. 다른 split이나 버전을 고르세요.")
    space = db.get(LabelSpace, release.label_space_id)
    ids = {str(c["name"]): int(c["id"]) for c in space.classes}
    missing = [name for name in model.classes or [] if name not in ids]
    if missing:
        raise ServiceError(400, f"이 버전에는 모델의 클래스가 없습니다: {', '.join(missing[:5])}")
    class_ids = [ids[name] for name in model.classes or []]
    if class_ids != sorted(class_ids):
        raise ServiceError(400, "이 버전의 클래스 순서가 모델과 달라 평가할 수 없습니다.")
    node = training.data_node(db, release.id)
    gpus = (1 if node.info.get("gpus") else 0) if gpus is None else gpus
    extra = {"release_id": release.id, "split": split, "class_ids": class_ids}
    return _start(db, user, model, "eval", node, {"split": split}, release_id=release.id, extra=extra, gpus=gpus, via=via)


ALIAS_ORDER = ("production", "champion", "staging", "challenger")


def eval_problem(db: Session, model: ModelVersion, release: Release) -> str:
    """Why the model cannot evaluate (and so check the labels of) this version, or "" when it can:
    a model of another task, classes the version does not have or has in another order, no eval code."""
    task = catalog.release_task(db, release)
    if not task_names.same(model.task or "detect", task):
        return f"{task_names.short(model.task)} 모델이라 {task_names.short(task)} 데이터셋에는 쓸 수 없습니다"
    space = db.get(LabelSpace, release.label_space_id)
    ids = {str(c["name"]): int(c["id"]) for c in (space.classes if space else [])}
    missing = [name for name in model.classes or [] if name not in ids]
    if missing:
        more = f" 외 {len(missing) - 3}개" if len(missing) > 3 else ""
        return f"이 버전에 없는 클래스가 있습니다: {', '.join(missing[:3])}{more}"
    class_ids = [ids[name] for name in model.classes or []]
    if class_ids != sorted(class_ids):
        return "클래스 순서가 이 버전과 다릅니다"
    if "eval" not in actions(db, model):
        return "이 모델의 템플릿에는 평가 코드가 없습니다"
    return ""


def eval_choices(db: Session, release: Release) -> dict[str, Any]:
    """The project's models for checking this version's labels: the ones that fit first (an alias, then the
    best), each with why it does not when it does not. A work in progress is evaluated as its base version."""
    dataset = db.get(Dataset, release.dataset_id)
    target = drafts.latest_ready(db, dataset.id) if release.status == "draft" else release
    splits: dict[str, int] = {}
    for name, part in ((target.stats or {}).get("splits") or {}).items() if target else []:
        key = name or "train"  # images without a split are trained on
        splits[key] = splits.get(key, 0) + int(part.get("items") or 0)
    rows = []
    for model in db.scalars(
        select(ModelVersion)
        .where(ModelVersion.project_id == dataset.project_id, ModelVersion.deleted_at.is_(None))
        .order_by(ModelVersion.created_at.desc())
    ):
        why = eval_problem(db, model, target or release)
        metric = training.PRIMARY_METRIC.get(model.task or "detect", "map50_95")
        value = (model.metrics or {}).get(metric)
        rows.append(
            {
                **checks.model_ref(model),
                "arch": model.arch,
                "metric": metric,
                "value": value if isinstance(value, int | float) else None,
                "created_at": iso(model.created_at),
                "trained_on": model.release_id == (target.id if target else None),  # its train split was learned
                "fits": not why,
                "why": why,
            }
        )

    def rank(row: dict[str, Any]) -> tuple:
        known = [ALIAS_ORDER.index(a) for a in row["aliases"] if a in ALIAS_ORDER]
        return (not row["fits"], min(known) if known else len(ALIAS_ORDER), -(row["value"] or -1))

    rows.sort(key=rank)
    return {
        "release_id": release.id,
        # where the evaluation runs: the version itself, or a work in progress's base
        "evaluate": {"id": target.id, "version": target.version} if target else None,
        "task": catalog.dataset_task(dataset),
        "splits": splits,
        "models": rows,
    }


def start_export(db: Session, user: User, model_id: str, fmt: str = "onnx", via: str = "api") -> ModelJob:
    model = training.get_model(db, model_id)
    if fmt not in EXPORT_FORMATS:
        raise ServiceError(400, f"형식은 {', '.join(EXPORT_FORMATS)} 중 하나여야 합니다.")
    return _start(db, user, model, "export", _training_node(db, model), {"format": fmt}, via=via)


def start_predict(
    db: Session,
    user: User,
    model_id: str,
    images: list[dict[str, Any]],
    storage: Storage,
    conf: float = 0.25,
    via: str = "api",
) -> ModelJob:
    """Run the model on a few uploaded images and keep what it found."""
    model = training.get_model(db, model_id)
    if not 1 <= len(images) <= MAX_TEST_IMAGES:
        raise ServiceError(400, f"이미지는 1~{MAX_TEST_IMAGES}장을 올려 주세요.")
    inputs = []
    for n, image in enumerate(images):
        blob, name = str(image.get("blob", "")), Path(str(image.get("name") or f"image{n}.jpg")).name
        if not blob or not storage.has_blob(blob):
            raise ServiceError(400, f"올린 이미지를 찾을 수 없습니다: {name}")
        inputs.append({"name": f"{n:02d}_{name}", "blob": blob})
    if not 0 <= conf <= 1:
        raise ServiceError(400, "conf는 0~1이어야 합니다.")
    return _start(db, user, model, "predict", _training_node(db, model), {"conf": conf, "images": inputs}, via=via)


# ---- node protocol -----------------------------------------------------------------


def _running(db: Session, node: Node, job_id: str) -> tuple[ModelJob, ModelVersion]:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.type != "model.run" or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 모델 작업이 아닙니다.")
    op = db.scalar(select(ModelJob).where(ModelJob.job_id == job.id))
    model = db.get(ModelVersion, op.model_id) if op else None
    if op is None or model is None:
        raise ServiceError(404, "이 작업에 연결된 모델이 없습니다.")
    return op, model


def manifest(db: Session, node: Node, job_id: str, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    op, _ = _running(db, node, job_id)
    if op.kind == "autolabel":
        from . import autolabel

        return autolabel.manifest(db, node, op, after, limit)
    release = db.get(Release, op.release_id) if op.release_id else None
    if release is None:
        raise ServiceError(404, "이 작업은 버전을 쓰지 않습니다.")
    return training.release_manifest(db, node, release, after, limit)


def blob_for(db: Session, node: Node, job_id: str, what: str, index: int = 0) -> str:
    """The storage key of something the node needs: the model's weights, the template's code, an input image."""
    op, model = _running(db, node, job_id)
    if what == "weights":
        return model.blob
    if what == "template":
        return _template_of(db, model).blob or ""
    images = (op.params or {}).get("images") or []
    if not 0 <= index < len(images):
        raise ServiceError(404, "그런 입력 이미지가 없습니다.")
    return images[index]["blob"]


def add_file(
    db: Session,
    node: Node,
    job_id: str,
    name: str,
    blob: str,
    size: int,
    media: str = "",
    caption: str = "",
    storage: Storage | None = None,
) -> dict[str, Any]:
    """A file the job recorded. ``media``: ``table`` (bm.log_table), ``table-image`` (a picture a table draws on)
    or ``predictions`` (bm.log_predictions: every evaluated image's predictions, stored per image)."""
    op, _ = _running(db, node, job_id)
    if media == "predictions":
        kept = op.kind in ("eval", "autolabel")  # per image, by image: a test's few uploads are not
        matched = checks.ingest(db, op, storage.get_blob(blob)) if storage is not None and kept else 0
        files = [f for f in (op.files or []) if f.get("kind") != "predictions"]
        entry = {"name": "predictions.json", "blob": blob, "bytes": size, "kind": "predictions", "images": matched}
        op.files = [*files, entry]
        return {"name": "predictions.json", "images": matched}
    name = training.clean_file_name(name)
    picture = name.lower().endswith(IMAGE_SUFFIXES)
    kind = "image" if picture else "file"
    if media == "table" or (media == "table-image" and picture):
        kind = media
    entry = {"name": name, "blob": blob, "bytes": size, "kind": kind, "caption": caption[:255]}
    files = [f for f in (op.files or []) if f["name"] != name]
    if len(files) >= MAX_RESULT_ITEMS:
        raise ServiceError(400, "기록할 수 있는 파일 수를 넘었습니다.")
    op.files = [*files, entry]
    return {"name": name}


def finish(db: Session, node: Node, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
    op, _ = _running(db, node, job_id)
    predictions = result.get("predictions")
    if isinstance(predictions, list):
        result = {**result, "predictions": predictions[:MAX_RESULT_ITEMS]}
    if op.kind == "autolabel":
        from . import autolabel

        result = {**result, "autolabel": autolabel.apply(db, op)}
    op.result = result
    return {"ok": True}


# ---- views -------------------------------------------------------------------------


def get_op(db: Session, op_id: str) -> ModelJob:
    op = db.get(ModelJob, op_id)
    if op is None:
        raise ServiceError(404, "기록을 찾을 수 없습니다.")
    return op


def op_view(db: Session, settings: HubSettings, op: ModelJob, node_names: dict[str, str] | None = None) -> dict[str, Any]:
    job = db.get(Job, op.job_id) if op.job_id else None
    data = services.job_view(db, settings, job, node_names) if job else None
    listed = [(f, catalog.media_url(settings, "op", f"{op.id}-{n}")) for n, f in enumerate(op.files or [])]
    release = db.get(Release, op.release_id) if op.release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    return {
        "id": op.id,
        "kind": op.kind,
        "status": data["status"] if data else "failed",
        "error": data["error"] if data else None,
        "wait_reason": data["wait_reason"] if data else None,
        "duration_s": data["duration_s"] if data else None,
        "node_name": data["node_name"] if data else None,
        "job_id": op.job_id,
        "created_at": iso(op.created_at),
        "release": {"id": release.id, "dataset": dataset.name, "version": release.version} if release and dataset else None,
        "params": {k: v for k, v in (op.params or {}).items() if k != "images"},
        "result": op.result or {},
        "files": [
            {**{k: f[k] for k in ("name", "kind", "bytes")}, "url": url}
            for f, url in listed
            if f["kind"] not in ("table", "table-image", "predictions")
        ],
        # images whose predictions were kept: the dataset screen can sort them by likely wrong labels
        "predicted": sum(int(f.get("images") or 0) for f in op.files or [] if f["kind"] == "predictions"),
        # Tables the code recorded (per-image results), and the pictures their cells draw on
        "tables": [
            {"name": f["name"], "kind": "table", "caption": f.get("caption", ""), "url": url}
            for f, url in listed
            if f["kind"] == "table"
        ],
        "table_images": {f["name"]: url for f, url in listed if f["kind"] == "table-image"},
    }


def list_ops(db: Session, settings: HubSettings, model_id: str, limit: int = 50) -> list[dict[str, Any]]:
    training.get_model(db, model_id)
    ops = db.scalars(
        select(ModelJob).where(ModelJob.model_id == model_id).order_by(ModelJob.created_at.desc()).limit(limit)
    ).all()
    names = services.node_name_map(db)
    return [op_view(db, settings, op, names) for op in ops]


def file_of(db: Session, key: str) -> dict[str, Any]:
    """``<op id>-<index>`` → the recorded file."""
    op_id, _, index = key.rpartition("-")
    op = get_op(db, op_id)
    files = op.files or []
    if not index.isdigit() or int(index) >= len(files):
        raise ServiceError(404, "파일을 찾을 수 없습니다.")
    return files[int(index)]
