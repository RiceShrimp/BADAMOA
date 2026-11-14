"""Hyperparameter search (W&B Sweeps): one release and template, many settings, the best run found.

A sweep names the settings to try (``space``) and how to pick them:

    grid    every combination of the listed values
    random  ``max_runs`` draws: a listed value, or a number between ``min`` and ``max``
            (on a log scale with ``log``, the usual choice for learning rates)

The Hub starts the runs itself, ``parallel`` at a time, and starts the next as one ends
(``advance_all``, from the Hub's background loop). Every run is an ordinary run with the
sweep's id in its config, so it shows up, compares and registers its model like any other.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
import threading
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from badamoa.core.protocol import TERMINAL_STATUSES
from badamoa.core.templates import RESERVED, TemplateError, resolve_params

from . import catalog, registry, services, training
from .catalog_models import Dataset
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, User
from .services import ServiceError
from .train_models import Run, Sweep, Template

log = logging.getLogger("badamoa.hub.sweeps")

METHODS = ("grid", "random")
MAX_RUNS = 100
MAX_PARALLEL = 8
MAX_VALUES = 50

# One advancing pass at a time (two would start the same run twice); requests that start or stop a
# search hold it until they have committed, so a pass never works from a stale picture.
lock = threading.Lock()
_dirty = threading.Event()  # a training job ended: advance soon rather than at the next regular tick


# ---- the settings each run gets ---------------------------------------------------------


def _check_space(manifest: Any, params: dict[str, Any], space: dict[str, Any], method: str) -> dict[str, dict[str, Any]]:
    if not space:
        raise ServiceError(400, "시험할 설정을 하나 이상 고르세요.")
    clean: dict[str, dict[str, Any]] = {}
    for name, spec in space.items():
        declared = manifest.params.get(name)
        if declared is None or name in RESERVED:
            raise ServiceError(400, f"{manifest.title}에 없는 설정입니다: {name}")
        if not isinstance(spec, dict):
            raise ServiceError(400, f"{name}: values 목록이나 min · max 범위로 적어 주세요.")
        if "values" in spec:
            values = list(spec["values"] or [])
            if not values or len(values) > MAX_VALUES:
                raise ServiceError(400, f"{name}: 시험할 값은 1~{MAX_VALUES}개여야 합니다.")
            checked = []
            for value in values:
                try:
                    checked.append(resolve_params(manifest, {**params, name: value})[name])
                except TemplateError as e:
                    raise ServiceError(400, str(e)) from e
            clean[name] = {"values": list(dict.fromkeys(checked))}
        elif "min" in spec and "max" in spec:
            if method != "random":
                raise ServiceError(400, f"{name}: 범위는 무작위 탐색에서만 쓸 수 있습니다. 모든 조합에는 값을 나열하세요.")
            if declared.type not in ("integer", "number"):
                raise ServiceError(400, f"{name}: 범위는 숫자 설정에만 쓸 수 있습니다.")
            try:
                low, high = float(spec["min"]), float(spec["max"])
            except (TypeError, ValueError) as e:
                raise ServiceError(400, f"{name}: min · max는 숫자여야 합니다.") from e
            log_scale = bool(spec.get("log"))
            if not low < high or (log_scale and low <= 0):
                raise ServiceError(400, f"{name}: min < max여야 하고, 로그 범위는 0보다 커야 합니다.")
            for end in (low, high):
                try:
                    resolve_params(manifest, {**params, name: int(round(end)) if declared.type == "integer" else end})
                except TemplateError as e:
                    raise ServiceError(400, str(e)) from e
            clean[name] = {"min": low, "max": high, "log": log_scale}
        else:
            raise ServiceError(400, f"{name}: values 목록이나 min · max 범위로 적어 주세요.")
    return clean


def _check_used(manifest: Any, fixed: dict[str, Any], space: dict[str, dict[str, Any]]) -> None:
    """Refuse to try values of a setting the code would ignore in every run (identical runs, wasted GPU time)."""
    for name in space:
        for other, values in manifest.params[name].unused_when.items():
            tried = space.get(other, {}).get("values")
            always = all(v in values for v in tried) if tried else (other not in space and fixed.get(other) in values)
            if always:
                spec, condition = manifest.params[name], manifest.params[other]
                raise ServiceError(
                    400,
                    f"{condition.label or other} 설정이 {fixed.get(other) if not tried else tried[0]}이면 "
                    f"{spec.label or name} 설정은 쓰이지 않아 모든 실행이 같아집니다. "
                    f"세부 설정에서 {condition.label or other} 설정을 다른 값으로 정하거나 함께 시험하세요.",
                )


def total_of(sweep: Sweep) -> int:
    if sweep.method == "grid":
        return math.prod(len(spec["values"]) for spec in sweep.space.values())
    return sweep.max_runs


def settings_for(sweep: Sweep, index: int, types: dict[str, str]) -> dict[str, Any]:
    """The settings of the sweep's ``index``-th run. The same index always gives the same settings."""
    if sweep.method == "grid":
        names = list(sweep.space)
        combo = next(itertools.islice(itertools.product(*(sweep.space[n]["values"] for n in names)), index, None))
        return dict(zip(names, combo, strict=True))
    rng = random.Random(f"{sweep.id}:{index}")
    out: dict[str, Any] = {}
    for name, spec in sweep.space.items():
        if "values" in spec:
            out[name] = rng.choice(spec["values"])
            continue
        low, high = spec["min"], spec["max"]
        value = math.exp(rng.uniform(math.log(low), math.log(high))) if spec.get("log") else rng.uniform(low, high)
        out[name] = int(round(value)) if types.get(name) == "integer" else float(f"{value:.4g}")
    return out


# ---- start, advance, stop -------------------------------------------------------------------


def start_sweep(
    db: Session,
    user: User,
    *,
    release_id: str,
    space: dict[str, Any],
    template: str = "",
    environment: str = "",
    params: dict[str, Any] | None = None,
    method: str = "grid",
    metric: str = "",
    goal: str = "",
    max_runs: int = 10,
    parallel: int = 1,
    name: str = "",
    classes: list[int] | None = None,
    unlabelled: str = "exclude",
    val_fraction: float = 0.2,
    gpus: int | None = None,
    min_vram_mb: int = 0,
    base_model_id: str = "",
    tags: list[str] | None = None,
    via: str = "api",
) -> Sweep:
    if method not in METHODS:
        raise ServiceError(400, "method는 grid(모든 조합) 또는 random(무작위)입니다.")
    release = catalog.get_release(db, release_id)
    dataset = db.get(Dataset, release.dataset_id)
    project = catalog.get_project(db, dataset.project_id)
    fallback = registry.DEFAULT_BY_TASK.get(project.task, registry.DEFAULT_TEMPLATE)
    row = registry.get_template(db, template or (project.settings or {}).get("template") or fallback)
    manifest = registry.manifest_of(row)
    fixed = {k: v for k, v in (params or {}).items() if k not in space}
    try:
        resolve_params(manifest, fixed)
    except TemplateError as e:
        raise ServiceError(400, str(e)) from e
    clean = _check_space(manifest, fixed, space, method)
    _check_used(manifest, resolve_params(manifest, fixed), clean)
    if not 1 <= max_runs <= MAX_RUNS:
        raise ServiceError(400, f"실행 수는 1~{MAX_RUNS}이어야 합니다.")
    metric = metric.strip() or training.PRIMARY_METRIC.get(project.task, "map50_95")
    goal = goal or ("minimize" if "loss" in metric else "maximize")
    if goal not in ("maximize", "minimize"):
        raise ServiceError(400, "goal은 maximize 또는 minimize입니다.")
    taken = set(db.scalars(select(Sweep.name).where(Sweep.project_id == project.id)))
    sweep = Sweep(
        project_id=project.id,
        name=training._unique(training._slug(name) or training._slug(f"{dataset.name}-{release.version}-탐색"), taken),
        release_id=release.id,
        method=method,
        metric=metric,
        goal=goal,
        space=clean,
        max_runs=max_runs,
        parallel=max(1, min(parallel, MAX_PARALLEL)),
        settings={
            "template": row.id,
            "environment": environment,
            "params": fixed,
            "classes": classes or [],
            "unlabelled": unlabelled,
            "val_fraction": val_fraction,
            "gpus": gpus,
            "min_vram_mb": min_vram_mb,
            "base_model_id": base_model_id,
            "tags": training.clean_tags(tags or []),
        },
        created_by=user.id,
    )
    total = total_of(sweep)
    if total > MAX_RUNS:
        raise ServiceError(400, f"조합이 {total}개입니다. {MAX_RUNS}개 이하가 되게 값을 줄이거나 무작위 탐색을 쓰세요.")
    db.add(sweep)
    db.flush()
    _start_next(db, sweep, via)  # the first runs right away: the person sees them start, mistakes show at once
    services.audit(db, user.id, via, "sweep.create", sweep.id, name=sweep.name, runs=total, method=method)
    return sweep


def _template_of(db: Session, sweep: Sweep) -> Template | None:
    """The template the search runs; a built-in one moves on with the installed version, like "다시 실행"."""
    row = db.get(Template, sweep.settings.get("template", ""))
    if row is not None and row.deleted_at is not None and row.builtin:
        try:
            row = registry.get_template(db, row.name)
        except ServiceError:
            return None
    return row if row is not None and row.deleted_at is None else None


def _runs_of(db: Session, sweep_id: str) -> list[Run]:
    return list(db.scalars(select(Run).where(Run.sweep_id == sweep_id).order_by(Run.created_at)))


def _status(db: Session, run: Run) -> str:
    job = db.get(Job, run.job_id) if run.job_id else None
    return job.status if job else "failed"


def _start_next(db: Session, sweep: Sweep, via: str = "sweep") -> None:
    """Start runs until ``parallel`` are going or all have been started; finish the sweep when all are done."""
    if sweep.status != "running":
        return
    runs = _runs_of(db, sweep.id)
    active = sum(1 for r in runs if _status(db, r) not in TERMINAL_STATUSES)
    started, total = len(runs), total_of(sweep)
    owner = db.get(User, sweep.created_by) if sweep.created_by else None
    if owner is None:
        sweep.status, sweep.error = "stopped", "탐색을 시작한 사람의 계정이 없어 멈췄습니다."
        return
    row = _template_of(db, sweep)
    if row is None:
        sweep.status, sweep.error = "stopped", "이 탐색이 쓰던 모델 종류가 삭제되어 멈췄습니다."
        return
    manifest = registry.manifest_of(row)
    types = {name: spec.type for name, spec in manifest.params.items()}
    s = sweep.settings
    while active < sweep.parallel and started < total:
        chosen = settings_for(sweep, started, types)
        try:
            training.start_run(
                db,
                owner,
                release_id=sweep.release_id,
                template=row.id,
                environment=s.get("environment", ""),
                params={**s.get("params", {}), **chosen},
                name=f"{sweep.name}-{started + 1}",
                model_name=sweep.name,  # the sweep's models form their own family, apart from the main one
                classes=s.get("classes"),
                unlabelled=s.get("unlabelled", "exclude"),
                val_fraction=s.get("val_fraction", 0.2),
                gpus=s.get("gpus"),
                min_vram_mb=s.get("min_vram_mb", 0),
                base_model_id=s.get("base_model_id", ""),
                tags=s.get("tags"),
                sweep={"id": sweep.id, "name": sweep.name, "index": started},
                via=via,
            )
        except ServiceError as e:  # the release or template went away, or no server holds the data
            sweep.status, sweep.error = "stopped", e.detail
            return
        started += 1
        active += 1
    if started >= total and active == 0:
        sweep.status, sweep.finished_at = "finished", utcnow()


def wake() -> None:
    _dirty.set()


def pending() -> bool:
    return _dirty.is_set()


def advance_all(database: Any) -> None:
    """One pass over the running sweeps (the Hub's background loop calls this every few seconds)."""
    with lock:
        _dirty.clear()
        with database.session() as db:
            for sweep in db.scalars(select(Sweep).where(Sweep.status == "running")).all():
                _start_next(db, sweep)


def on_job_finished(job: Job) -> None:
    if job.type == "train.run":
        wake()


def stop_sweep(db: Session, user: User, sweep_id: str, via: str = "api") -> Sweep:
    """Start no more runs and stop the ones going. What finished stays, with its models."""
    sweep = get_sweep(db, sweep_id)
    if sweep.status == "running":
        sweep.status, sweep.finished_at = "stopped", utcnow()
    for run in _runs_of(db, sweep.id):
        job = db.get(Job, run.job_id) if run.job_id else None
        if job is not None and job.status not in TERMINAL_STATUSES:
            services.cancel_job(db, user, job.id, via=via)
    services.audit(db, user.id, via, "sweep.stop", sweep.id, name=sweep.name)
    return sweep


# ---- views ------------------------------------------------------------------------------


def get_sweep(db: Session, sweep_id: str) -> Sweep:
    sweep = db.get(Sweep, sweep_id)
    if sweep is None:
        raise ServiceError(404, "탐색을 찾을 수 없습니다.")
    return sweep


def sweep_view(db: Session, settings: HubSettings, sweep: Sweep, detail: bool = False) -> dict[str, Any]:
    runs = [r for r in _runs_of(db, sweep.id) if r.deleted_at is None]
    statuses = {r.id: _status(db, r) for r in runs}
    scored = [r for r in runs if isinstance((r.summary or {}).get(sweep.metric), int | float)]
    pick = max if sweep.goal == "maximize" else min
    best = pick(scored, key=lambda r: r.summary[sweep.metric]) if scored else None
    release = catalog.get_release(db, sweep.release_id) if sweep.release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    template = db.get(Template, sweep.settings.get("template", ""))  # also after it was replaced or removed
    view: dict[str, Any] = {
        "id": sweep.id,
        "name": sweep.name,
        "project_id": sweep.project_id,
        "task": (project.task if (project := catalog.get_project(db, sweep.project_id)) else "detect"),
        "status": sweep.status,
        "error": sweep.error or None,
        "method": sweep.method,
        "metric": sweep.metric,
        "goal": sweep.goal,
        "space": sweep.space,
        "fixed": sweep.settings.get("params", {}),
        "parallel": sweep.parallel,
        "total": total_of(sweep),
        "started": len(runs),
        "active": sum(1 for s in statuses.values() if s not in TERMINAL_STATUSES),
        "finished": sum(1 for s in statuses.values() if s == "succeeded"),
        "failed": sum(1 for s in statuses.values() if s in ("failed", "canceled")),
        "best": (
            {
                "run_id": best.id,
                "name": best.name,
                "value": best.summary[sweep.metric],
                "settings": {k: best.config.get(k) for k in sweep.space},
            }
            if best
            else None
        ),
        "release": {"id": release.id, "dataset": dataset.name, "version": release.version} if release and dataset else None,
        "template": (
            {"name": template.name, "title": template.title, "version": template.version}
            if template
            else {"name": "", "title": "삭제된 모델 종류", "version": ""}
        ),
        "created_at": iso(sweep.created_at),
        "finished_at": iso(sweep.finished_at),
    }
    if detail:
        view["points"] = [
            {
                "run_id": r.id,
                "name": r.name,
                "status": statuses[r.id],
                "settings": {k: r.config.get(k) for k in sweep.space},
                "value": (r.summary or {}).get(sweep.metric),
            }
            for r in runs
        ]
    return view


def list_sweeps(db: Session, settings: HubSettings, project_id: str) -> list[dict[str, Any]]:
    catalog.get_project(db, project_id)
    rows = db.scalars(select(Sweep).where(Sweep.project_id == project_id).order_by(Sweep.created_at.desc()).limit(50))
    return [sweep_view(db, settings, s) for s in rows]
