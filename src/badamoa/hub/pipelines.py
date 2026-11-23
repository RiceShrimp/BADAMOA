"""Pipelines (Phase 5 · Automate): steps the Hub runs one after another, each when the one before succeeded.

A step is something the Hub already does, started as a person would start it, by the person who started the run:

    train    a training run on the run's version (``training.start_run``) → the model it registers
    eval     that model measured on a version's split (``modelops.start_eval``); with ``review`` also every split of
             the version it trained on, so 검수 sorts that version's images by likely wrong labels (its model check)
    promote  a gate: an alias (SOTA) moves to the model only if it beats the version holding it, both measured on
             the same version and split; otherwise the alias stays and the run ends ``held``
    export   the model exported (ONNX)

They run in that order, each at most once. Without ``train`` a pipeline is one ``eval`` of the model holding an alias
(re-measuring the SOTA on each new version: how it holds up, and a model check for 검수).

A step that fails stops the run and says why; the steps after it are skipped. A run follows one version and is
started by a person, by a new version of the dataset (``new_version``), or by a new training run ("학습 끝나면").
The Hub's background loop advances runs (``advance_all``): soon after a job ends, otherwise every 15 s.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from pathlib import PurePath
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
from badamoa.core.protocol import TERMINAL_STATUSES

from . import catalog, modelops, registry, services, training
from .catalog_models import Dataset, Project, Release
from .config import HubSettings
from .db import iso, utcnow, write_lock
from .models import Job, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Pipeline, PipelineRun, Run, Template

log = logging.getLogger("badamoa.hub.pipelines")

KINDS = ("train", "eval", "promote", "export")
LABEL = {"train": "학습", "eval": "평가", "promote": "승격", "export": "내보내기"}
TRIGGERS = ("manual", "new_version")
SPLITS = ("auto", "val", "test", "train")
GOING = ("pending", "running")
ENDED = {"failed": "failed", "stopped": "stopped", "held": "held"}  # a step's end → the run's
MAX_NAME = 160
MAX_RUNS_SHOWN = 50
# What a training step may carry: the new-run dialog's settings, without the version (the run's own)
TRAIN_KEYS = (
    "template", "environment", "params", "name", "model_name", "classes", "unlabelled", "val_fraction", "gpus",
    "gpu_indices", "min_vram_mb", "base_model_id", "tags", "note", "nodes", "data_after",
)  # fmt: skip
METRIC_LABEL = {
    "map50_95": "mAP50-95",
    "map50": "mAP50",
    "mask_map50_95": "mask mAP50-95",
    "pose_map50_95": "pose mAP50-95",
    "top1": "Top-1",
    "top5": "Top-5",
}

# One advancing pass at a time (two would start the same step twice); requests that start or stop a run hold it
# until they have committed, as hyperparameter searches do.
lock = threading.Lock()
_dirty = threading.Event()


def wake() -> None:
    _dirty.set()


def pending() -> bool:
    return _dirty.is_set()


def on_job_finished(job: Job) -> None:
    if job.type in ("train.run", "model.run", "data.sync"):
        wake()


# ---- what a pipeline is made of -----------------------------------------------------------


def _has(release: Release, split: str) -> bool:
    return bool((((release.stats or {}).get("splits") or {}).get(split) or {}).get("items"))


def _split_of(release: Release, wanted: str) -> str:
    """``auto``: the version's test split when it has one (data no model trained on), else val."""
    if wanted != "auto":
        return wanted
    return "test" if _has(release, "test") else "val"


def _check_alias(alias: str) -> str:
    alias = alias.strip()
    if not training.ALIAS_RE.match(alias) or alias.lower() == "latest" or re.fullmatch(r"[vV]\d+", alias):
        raise ServiceError(400, "alias는 32자 이내의 글자·숫자·._- 로 짓고, latest나 v1 같은 버전 표기는 쓸 수 없습니다.")
    return alias


def _clean_train(db: Session, dataset: Dataset, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    if config.get("attached"):  # follows a training run started on its own: repeated as "다시 실행" would
        return {"attached": True, "run_id": str(config.get("run_id") or "")}
    out = {k: v for k, v in config.items() if k in TRAIN_KEYS and v not in (None, "")}
    if not isinstance(out.get("params", {}), dict):
        raise ServiceError(400, "params는 설정 이름과 값입니다.")
    for key in ("classes", "gpu_indices"):
        if key in out:
            out[key] = [int(v) for v in out[key] or []]
    for key in ("nodes", "tags"):
        if key in out:
            out[key] = [str(v) for v in out[key] or []]
    return out


def _clean_eval(db: Session, dataset: Dataset, config: dict[str, Any], train: bool = True, **_: Any) -> dict[str, Any]:
    release_id = str(config.get("release_id") or "")
    split = str(config.get("split") or "auto")
    if split not in SPLITS:
        raise ServiceError(400, "평가 split은 auto(test가 있으면 test, 없으면 val) · val · test · train 중 하나입니다.")
    if release_id:  # a fixed version (a test set): every run is measured on the same data
        release = catalog.get_release(db, release_id)
        catalog.require_ready(release)
        other = db.get(Dataset, release.dataset_id)
        if not task_names.same(catalog.dataset_task(other), catalog.dataset_task(dataset)):
            raise ServiceError(400, "평가할 버전은 학습하는 데이터셋과 태스크가 같아야 합니다.")
        if not _has(release, _split_of(release, split)):
            raise ServiceError(400, f"{other.name} {release.version}의 {_split_of(release, split)}에는 이미지가 없습니다.")
    out: dict[str, Any] = {"release_id": release_id, "split": split, "review": bool(config.get("review"))}
    if config.get("gpus") is not None:  # else as many as the training took (none: one when the server has any)
        if not isinstance(config["gpus"], int) or config["gpus"] < 0:
            raise ServiceError(400, "평가 gpus는 0(CPU) 이상의 정수입니다.")
        out["gpus"] = config["gpus"]
    if not train:
        model = config.get("model") or {}
        name, alias = str(model.get("name") or "").strip(), str(model.get("alias") or "").strip()
        if not name or not alias:
            raise ServiceError(400, "평가할 모델을 고르세요: 모델 이름과 alias(예: SOTA).")
        out["model"] = {"name": name, "alias": _check_alias(alias)}
    return out


def _clean_promote(db: Session, dataset: Dataset, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    try:
        min_delta = float(config.get("min_delta") or 0)
    except (TypeError, ValueError) as e:
        raise ServiceError(400, "min_delta는 숫자입니다.") from e
    if min_delta < 0:
        raise ServiceError(400, "min_delta(이만큼 더 나아야 승격)는 0 이상입니다.")
    return {
        "alias": _check_alias(str(config.get("alias") or "SOTA")),
        "metric": str(config.get("metric") or "").strip(),
        "min_delta": min_delta,
    }


def _clean_export(db: Session, dataset: Dataset, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    fmt = str(config.get("format") or "onnx")
    if fmt not in modelops.EXPORT_FORMATS:
        raise ServiceError(400, f"내보내기 형식은 {', '.join(modelops.EXPORT_FORMATS)} 중 하나입니다.")
    return {"format": fmt}


CLEAN = {"train": _clean_train, "eval": _clean_eval, "promote": _clean_promote, "export": _clean_export}


def clean_steps(db: Session, dataset: Dataset, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The steps as stored: known kinds, in order, each once, with complete settings."""
    if not steps:
        raise ServiceError(400, "단계를 하나 이상 넣으세요.")
    kinds = [str((s or {}).get("kind") or "") for s in steps]
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ServiceError(400, f"모르는 단계입니다: {unknown[0]}. 단계는 train · eval · promote · export입니다.")
    if len(set(kinds)) != len(kinds):
        raise ServiceError(400, "같은 단계는 한 번만 넣을 수 있습니다.")
    if kinds != sorted(kinds, key=KINDS.index):
        raise ServiceError(400, "단계는 학습 → 평가 → 승격 → 내보내기 순서입니다.")
    if "promote" in kinds and "eval" not in kinds:
        raise ServiceError(400, "승격은 평가한 결과로 정합니다. 평가 단계를 앞에 넣으세요.")
    if "train" not in kinds and kinds != ["eval"]:
        raise ServiceError(400, "학습 없이는 alias가 붙은 모델을 평가하는 단계 하나만 넣을 수 있습니다.")
    train = "train" in kinds
    return [
        {"kind": k, "config": CLEAN[k](db, dataset, dict(s.get("config") or {}), train=train)}
        for k, s in zip(kinds, steps, strict=True)
    ]


def _check_code(db: Session, dataset: Dataset, steps: list[dict[str, Any]]) -> None:
    """The model kind the training step uses has code for the steps after it: refused now, not after training."""
    train = next((s for s in steps if s["kind"] == "train"), None)
    if train is None:
        return
    task = catalog.dataset_task(dataset)
    if train["config"].get("attached"):  # the kind the followed run trained with
        followed = db.get(Run, train["config"].get("run_id") or "")
        row = db.get(Template, (followed.template or {}).get("id", "")) if followed else None
        if row is None:
            return
    else:
        project = catalog.get_project(db, dataset.project_id)
        row = registry.template_for(db, project.settings, task, str(train["config"].get("template") or ""))
    entrypoints = registry.manifest_of(row, task).entrypoints
    for step in steps:
        if step["kind"] in ("eval", "export") and step["kind"] not in entrypoints:
            raise ServiceError(400, f"{row.title}에는 {LABEL[step['kind']]} 코드가 없어 그 단계를 넣을 수 없습니다.")


def train_settings_of(db: Session, run: Run) -> dict[str, Any]:
    """A training run's settings as a training step's (what "다시 실행" repeats): to keep its way as a pipeline."""
    c = run.config or {}
    row = db.get(Template, (run.template or {}).get("id", ""))
    declared = registry.manifest_of(row).params if row is not None else {}
    env = run.environment or {}
    settings = {
        "template": (row.name if row is not None and row.builtin else row.id) if row is not None else run.trainer,
        "environment": env.get("id") or ("auto" if env.get("kind") == "auto" else ""),
        "params": {k: c[k] for k in declared if k in c},
        "model_name": c.get("model_name", ""),
        "classes": c.get("classes") or [],
        "unlabelled": c.get("unlabelled", "exclude"),
        "val_fraction": c.get("val_fraction", 0.2),
        "gpus": c.get("gpus"),
        "gpu_indices": c.get("gpu_indices") or [],
        "min_vram_mb": c.get("min_vram_mb", 0),
        "base_model_id": (c.get("base_model") or {}).get("id", ""),
        "nodes": c.get("nodes") or [],
        "data_after": c.get("data_after", ""),
    }
    return {k: v for k, v in settings.items() if v not in (None, "", [], {})}


# ---- saved pipelines ------------------------------------------------------------------------


def _dataset_of(db: Session, dataset_id: str) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise ServiceError(404, "데이터셋을 찾을 수 없습니다.")
    return dataset


def newest_release(db: Session, dataset_id: str) -> Release | None:
    return db.scalar(
        select(Release)
        .where(Release.dataset_id == dataset_id, Release.status == "ready", Release.deleted_at.is_(None))
        .order_by(Release.created_at.desc())
        .limit(1)
    )


def _unique_name(db: Session, project_id: str, name: str, pipeline_id: str = "") -> str:
    taken = set(
        db.scalars(
            select(Pipeline.name).where(
                Pipeline.project_id == project_id, Pipeline.deleted_at.is_(None), Pipeline.id != pipeline_id
            )
        )
    )
    found, n = name, 2
    while found in taken:
        found, n = f"{name} {n}", n + 1
    return found


def _default_name(dataset: Dataset, steps: list[dict[str, Any]]) -> str:
    return f"{dataset.name} {' → '.join(LABEL[s['kind']] for s in steps)}"


def create_pipeline(
    db: Session,
    user: User,
    *,
    dataset_id: str,
    steps: list[dict[str, Any]],
    name: str = "",
    trigger: str = "manual",
    note: str = "",
    via: str = "api",
) -> Pipeline:
    """Save a pipeline for a dataset. Its training settings are checked against the dataset's newest version now."""
    dataset = _dataset_of(db, dataset_id)
    cleaned = clean_steps(db, dataset, steps)
    if any(s["config"].get("attached") for s in cleaned):
        raise ServiceError(400, "저장하는 파이프라인의 학습 단계에는 설정이 있어야 합니다.")
    _check_code(db, dataset, cleaned)
    if trigger not in TRIGGERS:
        raise ServiceError(400, "trigger는 manual(직접 실행) 또는 new_version(새 버전이 생기면)입니다.")
    train = next((s for s in cleaned if s["kind"] == "train"), None)
    newest = newest_release(db, dataset.id)
    if train is not None and newest is not None:
        c = train["config"]
        found = training.check_run(
            db,
            release_id=newest.id,
            template=str(c.get("template") or ""),
            params=c.get("params") or {},
            classes=c.get("classes"),
            nodes=c.get("nodes"),
            val_fraction=float(c.get("val_fraction", 0.2)),
            environment=str(c.get("environment") or ""),
            gpus=c.get("gpus"),
            user=user,
        )
        if found["problems"]:
            raise ServiceError(400, " ".join(found["problems"]))
    pipeline = Pipeline(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        name=_unique_name(db, dataset.project_id, name.strip()[:MAX_NAME] or _default_name(dataset, cleaned)),
        steps=cleaned,
        trigger=trigger,
        watch_since=utcnow() if trigger == "new_version" else None,
        note=note.strip(),
        created_by=user.id,
    )
    db.add(pipeline)
    db.flush()
    services.audit(db, user.id, via, "pipeline.create", pipeline.id, name=pipeline.name, trigger=trigger)
    return pipeline


def get_pipeline(db: Session, pipeline_id: str) -> Pipeline:
    pipeline = db.get(Pipeline, pipeline_id)
    if pipeline is None or pipeline.deleted_at is not None:
        raise ServiceError(404, "파이프라인을 찾을 수 없습니다.")
    return pipeline


def update_pipeline(
    db: Session,
    user: User,
    pipeline_id: str,
    *,
    name: str | None = None,
    trigger: str | None = None,
    note: str | None = None,
    steps: list[dict[str, Any]] | None = None,
    via: str = "api",
) -> Pipeline:
    pipeline = get_pipeline(db, pipeline_id)
    if name is not None and name.strip():
        pipeline.name = _unique_name(db, pipeline.project_id, name.strip()[:MAX_NAME], pipeline.id)
    if trigger is not None:
        if trigger not in TRIGGERS:
            raise ServiceError(400, "trigger는 manual(직접 실행) 또는 new_version(새 버전이 생기면)입니다.")
        if trigger == "new_version" and pipeline.trigger != "new_version":
            pipeline.watch_since = utcnow()  # versions from now on: not the ones made while it was off
        pipeline.trigger = trigger
    if note is not None:
        pipeline.note = note.strip()
    if steps is not None:
        dataset = _dataset_of(db, pipeline.dataset_id)
        cleaned = clean_steps(db, dataset, steps)
        _check_code(db, dataset, cleaned)
        pipeline.steps = cleaned
    services.audit(db, user.id, via, "pipeline.update", pipeline.id, name=pipeline.name, trigger=pipeline.trigger)
    return pipeline


def delete_pipeline(db: Session, user: User, pipeline_id: str, via: str = "api") -> None:
    """It stops watching for new versions; its runs stay listed (a running one goes on)."""
    pipeline = get_pipeline(db, pipeline_id)
    pipeline.deleted_at = utcnow()
    services.audit(db, user.id, via, "pipeline.delete", pipeline.id, name=pipeline.name)


# ---- runs -----------------------------------------------------------------------------------


def _fresh(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": step["kind"],
        "config": step["config"],
        "status": "pending",
        "message": "",
        "refs": {},
        "started_at": None,
        "finished_at": None,
    }


def start_run(
    db: Session,
    user: User,
    *,
    steps: list[dict[str, Any]],
    release_id: str = "",
    pipeline: Pipeline | None = None,
    follow_run_id: str = "",
    name: str = "",
    trigger: str = "manual",
    settings: HubSettings | None = None,
    via: str = "api",
    strict: bool = True,
) -> PipelineRun:
    """Run steps on a version. The first step starts now: with ``strict`` what stops it is refused (nothing is
    made), else it is the run's failure. ``follow_run_id``: an existing training run is the first step (the
    steps after a run started on its own, or to finish what it began)."""
    followed = None
    if follow_run_id:
        followed = training.get_run(db, follow_run_id)
        if followed.deleted_at is not None:
            raise ServiceError(404, "학습 실행을 찾을 수 없습니다.")
        if following(db, followed.id, going_only=True) is not None:
            raise ServiceError(409, "이 학습에는 이미 이어서 할 일이 있습니다. 파이프라인 화면에서 보세요.")
        job = db.get(Job, followed.job_id) if followed.job_id else None
        if job is None or job.status in ("failed", "canceled"):
            raise ServiceError(409, "실패했거나 중단한 학습에는 이어서 할 수 없습니다. 다시 실행한 뒤에 붙이세요.")
        release_id = followed.release_id
        steps = [{"kind": "train", "config": {"attached": True, "run_id": followed.id}}, *steps]
    release = catalog.get_release(db, release_id)
    dataset = db.get(Dataset, release.dataset_id)
    if pipeline is not None and pipeline.dataset_id != dataset.id:
        raise ServiceError(400, "이 파이프라인은 다른 데이터셋의 버전에서 돕니다.")
    cleaned = clean_steps(db, dataset, steps)
    _check_code(db, dataset, cleaned)
    states = [_fresh(s) for s in cleaned]
    run = PipelineRun(
        project_id=dataset.project_id,
        pipeline_id=pipeline.id if pipeline else None,
        name=(name.strip() or (pipeline.name if pipeline else ""))[:MAX_NAME],
        release_id=release.id,
        steps=states,
        trigger=trigger if not followed else "training",
        created_by=user.id,
    )
    if followed is not None:  # the training is already there: the run picks it up where it stands
        run.train_run_id = followed.id
        run.name = run.name or followed.name
        states[0].update(
            status="running", refs={"run_id": followed.id, "job_id": followed.job_id}, started_at=iso(followed.created_at)
        )
    db.add(run)
    db.flush()
    _advance(db, run, settings, strict=strict, via=via)
    services.audit(
        db, user.id, via, "pipeline.run", run.id, name=run.name, release=f"{dataset.name} {release.version}", trigger=trigger
    )
    return run


def run_pipeline(
    db: Session,
    user: User,
    pipeline_id: str,
    release_id: str = "",
    settings: HubSettings | None = None,
    via: str = "api",
) -> PipelineRun:
    """A saved pipeline on a version of its dataset (the newest when none is named)."""
    pipeline = get_pipeline(db, pipeline_id)
    if not release_id:
        newest = newest_release(db, pipeline.dataset_id)
        if newest is None:
            raise ServiceError(400, "이 데이터셋에는 아직 쓸 수 있는 버전이 없습니다.")
        release_id = newest.id
    return start_run(db, user, steps=pipeline.steps, release_id=release_id, pipeline=pipeline, settings=settings, via=via)


def get_run(db: Session, run_id: str) -> PipelineRun:
    run = db.get(PipelineRun, run_id)
    if run is None:
        raise ServiceError(404, "파이프라인 실행을 찾을 수 없습니다.")
    return run


def following(db: Session, training_run_id: str, going_only: bool = False) -> PipelineRun | None:
    """The pipeline run a training run is the first step of (the newest)."""
    query = select(PipelineRun).where(PipelineRun.train_run_id == training_run_id)
    if going_only:
        query = query.where(PipelineRun.status == "running")
    return db.scalar(query.order_by(PipelineRun.created_at.desc()).limit(1))


def _jobs_of(step: dict[str, Any]) -> list[str]:
    refs = step.get("refs") or {}
    ids = [refs["job_id"]] if refs.get("job_id") else []
    return ids + [op_job for op_job in refs.get("op_jobs", []) if op_job]


def stop_run(db: Session, user: User, run_id: str, via: str = "api") -> PipelineRun:
    """Stop the step going (its training or evaluation is canceled) and skip the rest. What finished stays."""
    run = get_run(db, run_id)
    if run.status != "running":
        raise ServiceError(409, "이미 끝난 파이프라인 실행입니다.")
    steps = copy.deepcopy(run.steps)
    index = next((i for i, s in enumerate(steps) if s["status"] in GOING), len(steps) - 1)
    for job_id in _jobs_of(steps[index]):
        job = db.get(Job, job_id)
        if job is not None and job.status not in TERMINAL_STATUSES:
            services.cancel_job(db, user, job.id, via=via)
    _end(run, steps, index, "stopped", f"{user.username}이(가) 멈췄습니다.")
    run.steps = steps
    services.audit(db, user.id, via, "pipeline.stop", run.id, name=run.name)
    return run


def retry_run(db: Session, user: User, run_id: str, settings: HubSettings | None = None, via: str = "api") -> PipelineRun:
    """From the step that failed or was stopped: what succeeded before it stays (its training, its model)."""
    run = get_run(db, run_id)
    if run.status not in ("failed", "stopped"):
        raise ServiceError(409, "실패했거나 멈춘 실행만 이어서 다시 할 수 있습니다.")
    steps = copy.deepcopy(run.steps)
    index = next((i for i, s in enumerate(steps) if s["status"] in ("failed", "stopped")), 0)
    for step in steps[index:]:
        step.update(_fresh(step))
    if steps[0]["kind"] == "train" and index == 0 and run.train_run_id:  # again with what that training ran
        steps[0]["config"] = {**steps[0]["config"], "again": run.train_run_id}
    run.steps, run.status, run.error, run.finished_at = steps, "running", "", None
    run.created_by = user.id  # the steps from here on are started as this person
    _advance(db, run, settings, strict=True, via=via)
    services.audit(db, user.id, via, "pipeline.retry", run.id, name=run.name, step=steps[index]["kind"])
    return run


# ---- moving a run along ---------------------------------------------------------------------


def _model(db: Session, model_id: str | None) -> ModelVersion | None:
    model = db.get(ModelVersion, model_id) if model_id else None
    return model if model is not None and model.deleted_at is None else None


def _holder(db: Session, project_id: str, name: str, alias: str) -> ModelVersion | None:
    """The version of a model family carrying ``alias`` (one at a time)."""
    family = db.scalars(
        select(ModelVersion).where(
            ModelVersion.project_id == project_id, ModelVersion.name == name, ModelVersion.deleted_at.is_(None)
        )
    )
    return next((m for m in family if alias in training._aliases(m)), None)


def _numbers(op: ModelJob, split: str) -> dict[str, Any] | None:
    """An evaluation's numbers on one of its splits."""
    splits = modelops._splits_of(op)
    if split not in splits:
        return None
    result = op.result or {}
    return ((result.get("splits") or {}).get(split) if len(splits) > 1 else result.get("metrics")) or None


def _measured(db: Session, model_id: str, release_id: str, split: str) -> tuple[ModelJob, dict[str, Any]] | None:
    """The model's newest finished evaluation on this version's split."""
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.model_id == model_id, ModelJob.release_id == release_id)
        .order_by(ModelJob.created_at.desc())
    )
    for op in ops:
        job = db.get(Job, op.job_id) if op.job_id else None
        numbers = _numbers(op, split) if job is not None and job.status == "succeeded" else None
        if numbers:
            return op, numbers
    return None


def _measuring(db: Session, model_id: str, release_id: str, split: str) -> ModelJob | None:
    """An evaluation of the model on this version's split still waiting or going."""
    ops = db.scalars(
        select(ModelJob)
        .join(Job, Job.id == ModelJob.job_id)
        .where(
            ModelJob.kind == "eval",
            ModelJob.model_id == model_id,
            ModelJob.release_id == release_id,
            Job.status.in_(("queued", "running")),
        )
    )
    return next((op for op in ops if split in modelops._splits_of(op)), None)


def _step(steps: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((s for s in steps if s["kind"] == kind), None)


def _subject(db: Session, run: PipelineRun, steps: list[dict[str, Any]], step: dict[str, Any]) -> ModelVersion:
    """The model the steps after training act on: the one it registered, or (no training) the alias holder."""
    train = _step(steps, "train")
    if train is not None:
        model = _model(db, (train.get("refs") or {}).get("model_id"))
        if model is None:
            raise ServiceError(409, "학습이 등록한 모델이 지워졌습니다.")
        return model
    wanted = step["config"].get("model") or {}
    holder = _holder(db, run.project_id, wanted.get("name", ""), wanted.get("alias", ""))
    if holder is None:
        raise ServiceError(409, f"지금 {wanted.get('alias')}인 {wanted.get('name')} 모델이 없습니다.")
    return holder


def _gpus(db: Session, steps: list[dict[str, Any]]) -> int | None:
    """An evaluation takes as many GPUs as the training did (a run on the CPU is measured on the CPU); without
    training, one when the server has any."""
    evaluation = _step(steps, "eval")
    if evaluation is not None and evaluation["config"].get("gpus") is not None:
        return int(evaluation["config"]["gpus"])
    train = _step(steps, "train")
    made = db.get(Run, (train.get("refs") or {}).get("run_id") or "") if train else None
    return int((made.config or {}).get("gpus") or 0) if made is not None else None


def _metric_key(model: ModelVersion, config: dict[str, Any] | None = None) -> str:
    return (config or {}).get("metric") or training.PRIMARY_METRIC.get(model.task or "detect", "map50_95")


def _begin(
    db: Session,
    owner: User,
    run: PipelineRun,
    steps: list[dict[str, Any]],
    index: int,
    settings: HubSettings | None,
    via: str = "pipeline",
) -> None:
    """Start a step: what it runs is started as ``owner`` would start it by hand (``via``: the request that started
    it, else the pipeline)."""
    step = steps[index]
    kind, c = step["kind"], step["config"]
    refs: dict[str, Any] = {}
    if kind == "train":
        again = c.get("again") or (c.get("run_id") if c.get("attached") else "")
        if again:  # the training it followed, again with the same settings
            made = training.rerun(db, owner, again, via=via)
        else:
            made = training.start_run(
                db, owner, release_id=run.release_id, via=via, **{k: v for k, v in c.items() if k in TRAIN_KEYS}
            )
        refs = {"run_id": made.id, "job_id": made.job_id}
        run.train_run_id = made.id
        run.name = run.name or made.name
    elif kind == "eval":
        model = _subject(db, run, steps, step)
        release = catalog.get_release(db, c.get("release_id") or run.release_id)
        split = _split_of(release, c.get("split") or "auto")
        if not _has(release, split):
            raise ServiceError(400, f"{release.version}의 {split}에는 이미지가 없습니다.")
        splits = [split]
        if c.get("review") and release.id == run.release_id:  # one evaluation: the gate's split and 검수's images
            splits += [s for s in ("train", "val", "test") if s != split and _has(release, s)]
        gpus = _gpus(db, steps)
        refs = {"model_id": model.id, "release_id": release.id, "split": split, "op_jobs": []}
        gate = _step(steps[index + 1 :], "promote")
        if gate is not None:  # the alias holder measured on the same data, alongside (and before, so the newest
            # evaluation 검수 offers is the new model's)
            holder = _holder(db, model.project_id, model.name, gate["config"]["alias"])
            if holder is not None and holder.id != model.id and _measured(db, holder.id, release.id, split) is None:
                if _measuring(db, holder.id, release.id, split) is None:
                    try:
                        with _savepoint(db):
                            op = modelops.start_eval(db, owner, holder.id, release.id, split, gpus, via="pipeline")
                        refs.update(holder_op_id=op.id)
                        refs["op_jobs"].append(op.job_id)
                    except ServiceError as e:  # the gate says so when it compares
                        refs["holder_problem"] = e.message
        op = modelops.start_eval(db, owner, model.id, release.id, split, gpus, via="pipeline", splits=splits)
        refs.update(op_id=op.id)
        refs["op_jobs"].append(op.job_id)
        if c.get("review") and release.id != run.release_id:  # 검수 looks at the version it trained on
            own = catalog.get_release(db, run.release_id)
            parts = [s for s in ("train", "val", "test") if _has(own, s)]
            review = modelops.start_eval(db, owner, model.id, own.id, parts[0], gpus, via="pipeline", splits=parts)
            refs.update(review_op_id=review.id)
            refs["op_jobs"].append(review.job_id)
    elif kind == "promote":
        refs = {}  # decided when the evaluations are in (``_check``)
    elif kind == "export":
        model = _subject(db, run, steps, step)
        op = modelops.start_export(db, owner, model.id, c.get("format", "onnx"), via="pipeline", settings=settings)
        refs = {"model_id": model.id, "op_id": op.id, "op_jobs": [op.job_id]}
    step.update(status="running", refs=refs, started_at=iso(utcnow()), message="")


def _savepoint(db: Session) -> Any:
    """A savepoint inside a real transaction (SQLite: take the write lock first, so it is not a transaction itself)."""
    write_lock(db)
    return db.begin_nested()


def _job_end(db: Session, job_id: str | None) -> tuple[str, str] | None:
    """A job that ended: (status, error); None while it waits or runs."""
    job = db.get(Job, job_id) if job_id else None
    if job is None:
        return "failed", "작업 기록을 찾을 수 없습니다."
    if job.status not in TERMINAL_STATUSES:
        return None
    return job.status, job.error or ""


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "–"


def _check(db: Session, owner: User, run: PipelineRun, steps: list[dict[str, Any]], index: int) -> tuple[str, str] | None:
    """Whether a running step is over: (status, what to say), or None while it goes on."""
    step = steps[index]
    kind, refs = step["kind"], step["refs"]
    if kind == "train":
        ended = _job_end(db, refs.get("job_id"))
        if ended is None:
            return None
        status, error = ended
        if status == "canceled":
            return "stopped", "학습을 중단했습니다."
        if status != "succeeded":
            return "failed", error or "학습이 실패했습니다."
        model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == refs.get("run_id")))
        if model is None:
            return "failed", "학습은 끝났지만 등록된 모델이 없습니다 (코드가 가중치를 올리지 않았습니다)."
        refs["model_id"] = model.id
        return "succeeded", f"{model.name} v{model.version} 등록"
    if kind == "eval":
        mine = [refs.get("op_id"), refs.get("review_op_id")]
        ends = {}
        for op_id in [*mine, refs.get("holder_op_id")]:
            op = db.get(ModelJob, op_id) if op_id else None
            if op is None:
                continue
            ended = _job_end(db, op.job_id)
            if ended is None:
                return None  # the holder's evaluation too: the gate right after needs it
            ends[op_id] = ended
        for op_id in mine:
            status, error = ends.get(op_id, ("succeeded", ""))
            if status == "canceled":
                return "stopped", "평가를 중단했습니다."
            if status != "succeeded":
                return "failed", error or "평가가 실패했습니다."
        op = db.get(ModelJob, refs.get("op_id"))
        numbers = _numbers(op, refs["split"]) if op is not None else None
        if not numbers:
            return "failed", "평가는 끝났지만 지표가 없습니다."
        model = _model(db, refs.get("model_id"))
        gate = _step(steps[index + 1 :], "promote")
        key = _metric_key(model, gate["config"] if gate else None) if model else "map50_95"
        refs["metrics"] = {k: v for k, v in numbers.items() if isinstance(v, int | float)}
        refs["metric"], refs["value"] = key, numbers.get(key)
        release = db.get(Release, refs["release_id"])
        said = (
            f"{METRIC_LABEL.get(key, key)} {_fmt(numbers.get(key))} ({release.version if release else '?'} {refs['split']})"
        )
        if _step(steps, "train") is None and model is not None:  # the SOTA on a new version: how it holds up
            before = _before(db, model, release, refs["split"], key)
            if before is not None:
                refs["before"] = before
                change = (numbers.get(key) or 0) - before["value"]
                said += f" · {before['version']}보다 {'+' if change >= 0 else ''}{change:.3f}"
        return "succeeded", said
    if kind == "promote":
        return _decide(db, owner, steps, index)
    if kind == "export":
        op = db.get(ModelJob, refs.get("op_id"))
        ended = _job_end(db, op.job_id if op else None)
        if ended is None:
            return None
        status, error = ended
        if status == "canceled":
            return "stopped", "내보내기를 중단했습니다."
        if status != "succeeded":
            return "failed", error or "내보내기가 실패했습니다."
        return "succeeded", f"{(op.result or {}).get('file') or step['config'].get('format', 'onnx')} 만듦"
    return "failed", f"모르는 단계입니다: {kind}"


def _before(db: Session, model: ModelVersion, release: Release | None, split: str, key: str) -> dict[str, Any] | None:
    """The model's newest finished evaluation on another version of the same dataset (same split)."""
    if release is None:
        return None
    others = select(Release.id).where(Release.dataset_id == release.dataset_id, Release.id != release.id)
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.model_id == model.id, ModelJob.release_id.in_(others))
        .order_by(ModelJob.created_at.desc())
    )
    for op in ops:
        job = db.get(Job, op.job_id) if op.job_id else None
        numbers = _numbers(op, split) if job is not None and job.status == "succeeded" else None
        if numbers and isinstance(numbers.get(key), int | float):
            other = db.get(Release, op.release_id)
            return {"release_id": op.release_id, "version": other.version if other else "?", "value": numbers[key]}
    return None


def _decide(db: Session, owner: User, steps: list[dict[str, Any]], index: int) -> tuple[str, str] | None:
    """The gate: move the alias here when this model measured better than the version holding it."""
    step = steps[index]
    c, refs = step["config"], step["refs"]
    measured = (_step(steps, "eval") or {}).get("refs") or {}
    model = _model(db, measured.get("model_id"))
    if model is None:
        return "failed", "평가한 모델이 지워졌습니다."
    alias, key = c["alias"], _metric_key(model, c)
    mine = (measured.get("metrics") or {}).get(key)
    if not isinstance(mine, int | float):
        return "failed", f"평가 결과에 {METRIC_LABEL.get(key, key)}이(가) 없어 비교할 수 없습니다."
    label = METRIC_LABEL.get(key, key)
    holder = _holder(db, model.project_id, model.name, alias)
    refs.update(alias=alias, metric=key, mine=mine, model_id=model.id)
    if holder is not None and holder.id == model.id:
        refs["outcome"] = "promoted"
        return "succeeded", f"이미 {alias}입니다."
    if holder is None:
        training.set_alias(db, owner, model.id, alias, via="pipeline")
        refs.update(outcome="promoted", holder=None)
        return "succeeded", f"{alias}이(가) 없어 {model.name} v{model.version}에 붙였습니다."
    release_id, split = measured["release_id"], measured["split"]
    found = _measured(db, holder.id, release_id, split)
    if found is None:
        if _measuring(db, holder.id, release_id, split) is not None:
            return None
        earlier = db.get(ModelJob, measured["holder_op_id"]) if measured.get("holder_op_id") else None
        if refs.get("holder_tried") == holder.id or (earlier is not None and earlier.model_id == holder.id):
            why = measured.get("holder_problem") or "평가가 실패했습니다"
            return "failed", f"지금 {alias}인 v{holder.version}을(를) 같은 데이터로 재지 못해 비교하지 못했습니다: {why}"
        refs["holder_tried"] = holder.id  # the holder changed since, or was never measured: measure it now
        try:
            with _savepoint(db):
                op = modelops.start_eval(db, owner, holder.id, release_id, split, _gpus(db, steps), via="pipeline")
        except ServiceError as e:
            return (
                "failed",
                f"지금 {alias}인 v{holder.version}을(를) 같은 데이터로 평가할 수 없어 비교하지 못했습니다: {e.message}",
            )
        refs["op_jobs"] = [op.job_id]
        return None
    theirs = found[1].get(key)
    refs["holder"] = {"id": holder.id, "name": holder.name, "version": holder.version, "op_id": found[0].id}
    refs["theirs"] = theirs
    if not isinstance(theirs, int | float):
        theirs = None
    lower = "loss" in key
    better = theirs is None or (mine < theirs - c["min_delta"] if lower else mine > theirs + c["min_delta"])
    if better:
        training.set_alias(db, owner, model.id, alias, via="pipeline")
        refs["outcome"] = "promoted"
        moved = f"{alias}을(를) v{model.version}(으)로 옮김"
        return "succeeded", f"{label} {_fmt(mine)} > v{holder.version} {_fmt(theirs)}: {moved}"
    refs["outcome"] = "held"
    return "held", f"{label} {_fmt(mine)} ≤ v{holder.version} {_fmt(theirs)}: {alias}은(는) v{holder.version} 그대로"


def _end(run: PipelineRun, steps: list[dict[str, Any]], index: int, status: str, message: str) -> None:
    now = utcnow()
    steps[index].update(status=status, message=message, finished_at=iso(now))
    for later in steps[index + 1 :]:
        if later["status"] in GOING:
            later["status"] = "skipped"
    run.status, run.finished_at = ENDED.get(status, "failed"), now
    run.error = f"{LABEL[steps[index]['kind']]} 단계: {message}" if status in ("failed", "stopped") else ""


def _advance(
    db: Session, run: PipelineRun, settings: HubSettings | None, strict: bool = False, via: str = "pipeline"
) -> None:
    """Move a run as far as it goes now: a step that ended hands over to the next, which starts at once."""
    if run.status != "running":
        return
    steps = copy.deepcopy(run.steps)  # a new value (the stored JSON is compared to it when saved)
    try:
        owner = db.get(User, run.created_by) if run.created_by else None
        for index, step in enumerate(steps):
            if step["status"] not in GOING:
                continue
            if owner is None:
                _end(run, steps, index, "failed", "실행을 시작한 사람의 계정이 없어 멈췄습니다.")
                return
            if step["status"] == "pending":
                try:
                    with _savepoint(db):  # a start that fails leaves nothing half-made
                        _begin(db, owner, run, steps, index, settings, via if strict else "pipeline")
                except ServiceError as e:
                    if strict:
                        raise
                    _end(run, steps, index, "failed", e.message)
                    return
            outcome = _check(db, owner, run, steps, index)
            if outcome is None:
                return
            status, message = outcome
            if status != "succeeded":
                _end(run, steps, index, status, message)
                return
            step.update(status="succeeded", message=message, finished_at=iso(utcnow()))
        run.status, run.finished_at, run.error = "succeeded", utcnow(), ""
    finally:
        run.steps = steps


def _watch(db: Session, pipeline: Pipeline, settings: HubSettings | None) -> None:
    """A version of the dataset that became ready since the pipeline last looked starts a run (the newest only)."""
    since = pipeline.watch_since or pipeline.created_at
    newest = db.scalar(
        select(Release)
        .where(
            Release.dataset_id == pipeline.dataset_id,
            Release.status == "ready",
            Release.deleted_at.is_(None),
            Release.created_at > since,
        )
        .order_by(Release.created_at.desc())
        .limit(1)
    )
    if newest is None:
        return
    write_lock(db)
    pipeline.watch_since = newest.created_at
    owner = db.get(User, pipeline.created_by) if pipeline.created_by else None
    if owner is None:
        return
    try:
        with _savepoint(db):
            start_run(
                db, owner, steps=pipeline.steps, release_id=newest.id, pipeline=pipeline, trigger="new_version",
                settings=settings, via="pipeline", strict=True,
            )  # fmt: skip
    except ServiceError as e:  # nothing started: a run that says why, so the screen shows it
        steps = [{**_fresh(s), "status": "skipped"} for s in pipeline.steps]
        if steps:
            steps[0].update(status="failed", message=e.message, finished_at=iso(utcnow()))
        db.add(
            PipelineRun(
                project_id=pipeline.project_id,
                pipeline_id=pipeline.id,
                name=pipeline.name,
                release_id=newest.id,
                steps=steps,
                status="failed",
                error=f"{LABEL[steps[0]['kind']] if steps else ''} 단계: {e.message}",
                trigger="new_version",
                created_by=owner.id,
                finished_at=utcnow(),
            )
        )


def advance_all(database: Any, settings: HubSettings | None = None) -> None:
    """One pass: watched datasets' new versions, then every running run (the Hub's background loop)."""
    with lock:
        _dirty.clear()
        with database.session() as db:
            watched = select(Pipeline).where(Pipeline.trigger == "new_version", Pipeline.deleted_at.is_(None))
            for pipeline in db.scalars(watched).all():
                try:
                    _watch(db, pipeline, settings)
                except Exception:  # one pipeline's trouble does not hold up the others
                    log.exception("pipeline %s: watching for new versions failed", pipeline.id)
            for run in db.scalars(select(PipelineRun).where(PipelineRun.status == "running")).all():
                try:
                    _advance(db, run, settings)
                except Exception:
                    log.exception("pipeline run %s: advancing failed", run.id)


# ---- views ----------------------------------------------------------------------------------


def _release_ref(db: Session, release_id: str | None) -> dict[str, Any] | None:
    release = db.get(Release, release_id) if release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    if release is None or dataset is None:
        return None
    return {
        "id": release.id,
        "version": release.version,
        "dataset": dataset.name,
        "dataset_id": dataset.id,
        "task": catalog.dataset_task(dataset),
        "deleted": release.deleted_at is not None,
    }


def _model_ref(model: ModelVersion | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "aliases": training._aliases(model),
        "deleted": model.deleted_at is not None,
    }


SPLIT_LABEL = {"auto": "test (없으면 val)", "val": "val", "test": "test", "train": "train"}


def _summary(db: Session, project: Project | None, dataset: Dataset | None, step: dict[str, Any]) -> str:
    """What a step does, in a few words (the cards and the steps not started yet)."""
    kind, c = step["kind"], step["config"]
    if kind == "train":
        if c.get("attached"):
            return "따로 시작한 학습"
        task = catalog.dataset_task(dataset) if dataset else "detect"
        try:
            row = registry.template_for(db, project.settings if project else {}, task, str(c.get("template") or ""))
        except ServiceError:
            return "삭제된 모델 종류"
        manifest = registry.manifest_of(row, task)
        params = c.get("params") or {}
        parts = [row.title]
        if manifest.variant:
            spec = manifest.params.get(manifest.variant)
            value = params.get(manifest.variant, spec.default if spec else "")
            if value:
                parts.append(PurePath(str(value)).stem)
        epochs = params.get("epochs", manifest.params["epochs"].default if "epochs" in manifest.params else None)
        if epochs:
            parts.append(f"epoch {epochs}")
        return " · ".join(parts)
    if kind == "eval":
        where = "그 버전" if c.get("model") else "학습한 버전"  # without training: the version the run is on
        if c.get("release_id"):
            ref = _release_ref(db, c["release_id"])
            where = f"{ref['dataset']} {ref['version']}" if ref else "지운 버전"
        said = f"{where} · {SPLIT_LABEL.get(c.get('split', 'auto'), c.get('split'))}"
        if c.get("model"):
            said = f"{c['model']['name']} {c['model']['alias']} → {said}"
        return said + (" · 검수에도" if c.get("review") else "")
    if kind == "promote":
        gain = f" (+{c['min_delta']} 이상)" if c.get("min_delta") else ""
        return f"더 나으면 {c['alias']}{gain}"
    if kind == "export":
        return str(c.get("format", "onnx")).upper()
    return ""


def _job_line(db: Session, settings: HubSettings, job_id: str | None) -> dict[str, Any] | None:
    job = db.get(Job, job_id) if job_id else None
    if job is None:
        return None
    line = {"id": job.id, "status": job.status}
    if job.status == "queued":
        line["wait_reason"] = services.explain_wait(db, settings, job)
    return line


def _step_view(
    db: Session, settings: HubSettings, run: PipelineRun, step: dict[str, Any], names: tuple[Any, Any]
) -> dict[str, Any]:
    kind, refs = step["kind"], step.get("refs") or {}
    view: dict[str, Any] = {
        "kind": kind,
        "label": LABEL[kind] + (" · 검수" if kind == "eval" and step["config"].get("review") else ""),
        "status": step["status"],
        "message": step.get("message") or "",
        "summary": _summary(db, names[0], names[1], step),
        "config": step["config"],
        "started_at": step.get("started_at"),
        "finished_at": step.get("finished_at"),
    }
    if kind == "train" and refs.get("run_id"):
        made = db.get(Run, refs["run_id"])
        view["run"] = {"id": made.id, "name": made.name, "epoch": made.epoch, "epochs": made.epochs} if made else None
        view["job"] = _job_line(db, settings, refs.get("job_id")) if step["status"] == "running" else None
        view["model"] = _model_ref(db.get(ModelVersion, refs["model_id"])) if refs.get("model_id") else None
    elif kind == "eval" and refs:
        view["model"] = _model_ref(db.get(ModelVersion, refs.get("model_id")))
        view["release"] = _release_ref(db, refs.get("release_id"))
        view["split"] = refs.get("split")
        view["op_id"] = refs.get("op_id")
        view["metric"] = {"key": refs.get("metric"), "label": METRIC_LABEL.get(refs.get("metric") or "", refs.get("metric")),
                          "value": refs.get("value")} if refs.get("metric") else None  # fmt: skip
        view["before"] = refs.get("before")
        # 검수 shows this model's check on the version it trained on (or the version measured, without training)
        if step["config"].get("review"):
            view["review_release_id"] = run.release_id
        if step["status"] == "running":
            op = db.get(ModelJob, refs.get("op_id")) if refs.get("op_id") else None
            view["job"] = _job_line(db, settings, op.job_id if op else None)
    elif kind == "promote" and refs:
        view["alias"] = refs.get("alias") or step["config"].get("alias")
        view["outcome"] = refs.get("outcome")
        view["mine"], view["theirs"] = refs.get("mine"), refs.get("theirs")
        view["metric"] = {"key": refs.get("metric"), "label": METRIC_LABEL.get(refs.get("metric") or "", refs.get("metric"))}
        view["holder"] = refs.get("holder")
        view["model"] = _model_ref(db.get(ModelVersion, refs["model_id"])) if refs.get("model_id") else None
    elif kind == "export" and refs:
        view["op_id"] = refs.get("op_id")
        view["model"] = _model_ref(db.get(ModelVersion, refs.get("model_id")))
        if step["status"] == "running":
            op = db.get(ModelJob, refs.get("op_id")) if refs.get("op_id") else None
            view["job"] = _job_line(db, settings, op.job_id if op else None)
    return view


def run_view(db: Session, settings: HubSettings, run: PipelineRun) -> dict[str, Any]:
    pipeline = db.get(Pipeline, run.pipeline_id) if run.pipeline_id else None
    release = _release_ref(db, run.release_id)
    dataset = db.get(Dataset, release["dataset_id"]) if release else None
    project = db.get(Project, run.project_id)
    owner = db.get(User, run.created_by) if run.created_by else None
    end = run.finished_at or utcnow()
    return {
        "id": run.id,
        "name": run.name or (pipeline.name if pipeline else ""),
        "status": run.status,
        "error": run.error or None,
        "trigger": run.trigger,
        "project": {"id": run.project_id, "name": project.name if project else ""},
        "pipeline": {"id": pipeline.id, "name": pipeline.name, "deleted": pipeline.deleted_at is not None}
        if pipeline
        else None,
        "release": release,
        "train_run_id": run.train_run_id,
        "created_by": owner.username if owner else None,
        "created_at": iso(run.created_at),
        "finished_at": iso(run.finished_at),
        "duration_s": (end - run.created_at).total_seconds(),
        "steps": [_step_view(db, settings, run, s, (project, dataset)) for s in run.steps],
    }


def pipeline_view(db: Session, settings: HubSettings, pipeline: Pipeline, detail: bool = False) -> dict[str, Any]:
    dataset = db.get(Dataset, pipeline.dataset_id)
    project = db.get(Project, pipeline.project_id)
    owner = db.get(User, pipeline.created_by) if pipeline.created_by else None
    runs = db.scalars(
        select(PipelineRun).where(PipelineRun.pipeline_id == pipeline.id).order_by(PipelineRun.created_at.desc())
    ).all()
    newest = newest_release(db, pipeline.dataset_id)
    view: dict[str, Any] = {
        "id": pipeline.id,
        "name": pipeline.name,
        "project_id": pipeline.project_id,
        "dataset": {"id": dataset.id, "name": dataset.name, "task": catalog.dataset_task(dataset)} if dataset else None,
        "steps": [
            {
                "kind": s["kind"],
                "label": LABEL[s["kind"]] + (" · 검수" if s["kind"] == "eval" and s["config"].get("review") else ""),
                "summary": _summary(db, project, dataset, s),
                "config": s["config"],
            }
            for s in pipeline.steps
        ],  # fmt: skip
        "trigger": pipeline.trigger,
        "watch_since": iso(pipeline.watch_since),
        "note": pipeline.note,
        "created_by": owner.username if owner else None,
        "created_at": iso(pipeline.created_at),
        "runs": len(runs),
        "going": sum(1 for r in runs if r.status == "running"),
        "last": run_view(db, settings, runs[0]) if runs else None,
        "newest_release": {"id": newest.id, "version": newest.version} if newest else None,
    }
    if detail:
        view["history"] = [run_view(db, settings, r) for r in runs[:MAX_RUNS_SHOWN]]
    return view


def board(db: Session, settings: HubSettings, project_id: str) -> dict[str, Any]:
    """The 파이프라인 screen for a project: its saved pipelines, and the runs (theirs and the ones without)."""
    catalog.get_project(db, project_id)
    pipelines = db.scalars(
        select(Pipeline)
        .where(Pipeline.project_id == project_id, Pipeline.deleted_at.is_(None))
        .order_by(Pipeline.created_at.desc())
    ).all()
    runs = db.scalars(
        select(PipelineRun)
        .where(PipelineRun.project_id == project_id)
        .order_by(PipelineRun.created_at.desc())
        .limit(MAX_RUNS_SHOWN)
    ).all()
    counts = dict(
        db.execute(
            select(PipelineRun.status, func.count()).where(PipelineRun.project_id == project_id).group_by(PipelineRun.status)
        ).all()
    )
    promoted = sum(
        1
        for r in runs
        for s in r.steps
        if s["kind"] == "promote" and (s.get("refs") or {}).get("outcome") == "promoted" and s["status"] == "succeeded"
    )
    return {
        "pipelines": [pipeline_view(db, settings, p) for p in pipelines],
        "runs": [run_view(db, settings, r) for r in runs],
        "totals": {
            "pipelines": len(pipelines),
            "watching": sum(1 for p in pipelines if p.trigger == "new_version"),
            "runs": sum(counts.values()),
            "going": counts.get("running", 0),
            "succeeded": counts.get("succeeded", 0),
            "held": counts.get("held", 0),
            "failed": counts.get("failed", 0) + counts.get("stopped", 0),
            "promoted": promoted,
        },
    }


def follow_view(db: Session, settings: HubSettings, run: Run) -> dict[str, Any]:
    """For a training run's screen: the pipeline run it is the first step of, and whether its model was evaluated
    (a finished run with no evaluation and no pipeline is nudged to go on)."""
    found = following(db, run.id)
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id))
    evaluations = (
        db.scalar(select(func.count()).select_from(ModelJob).where(ModelJob.kind == "eval", ModelJob.model_id == model.id))
        if model is not None
        else 0
    )
    return {"pipeline": run_view(db, settings, found) if found else None, "evaluations": evaluations or 0}
