"""Hyperparameter search (W&B Sweeps): one release and template, many settings, the best run found.

A sweep names the settings to try (``space``) and how to pick them:

    grid    every combination of the listed values
    random  ``max_runs`` draws: a listed value, or a number between ``min`` and ``max``
            (on a log scale with ``log``, the usual choice for learning rates)
    bayes   like random for the first few runs, then each next setting is chosen from the
            results so far by Optuna's TPE sampler (Tree-structured Parzen Estimator)

The Hub starts the runs itself, ``parallel`` at a time, and starts the next as one ends
(``advance_all``, from the Hub's background loop). Every run is an ordinary run with the
sweep's id in its config, so it shows up, compares and registers its model like any other.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import threading
import warnings
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

METHODS = ("grid", "random", "bayes")
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
            if method == "grid":
                raise ServiceError(
                    400, f"{name}: 범위는 무작위 · 베이지안 탐색에서만 쓸 수 있습니다. 모든 조합에는 값을 나열하세요."
                )
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


def _combinations(sweep: Sweep) -> int | None:
    """How many different settings there are, when every setting lists its values (None with a range)."""
    if all("values" in spec for spec in sweep.space.values()):
        return math.prod(len(spec["values"]) for spec in sweep.space.values())
    return None


def total_of(sweep: Sweep) -> int:
    combinations = _combinations(sweep)
    if sweep.method == "grid":
        return combinations or 0
    # Draws among listed values only: never more runs than there are different settings.
    return min(sweep.max_runs, combinations) if combinations else sweep.max_runs


def startup_runs(sweep: Sweep) -> int:
    """Bayes: how many results there must be before they guide the choice; until then the draws are random."""
    return max(3, min(10, sweep.max_runs // 4))


def _combination(sweep: Sweep, number: int) -> dict[str, Any]:
    """The ``number``-th of all combinations, in the order of ``itertools.product``."""
    out: dict[str, Any] = {}
    for name in reversed(list(sweep.space)):
        values = sweep.space[name]["values"]
        number, at = divmod(number, len(values))
        out[name] = values[at]
    return {name: out[name] for name in sweep.space}


def _number(value: float, kind: str | None) -> int | float:
    return int(round(value)) if kind == "integer" else float(f"{value:.4g}")


def settings_for(
    sweep: Sweep,
    index: int,
    types: dict[str, str],
    history: list[tuple[dict[str, Any], float]] | None = None,
    tried: list[dict[str, Any]] | None = None,
    running: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The settings of the sweep's ``index``-th run.

    Grid and random: the same index always gives the same settings. Bayes: the same results so
    far (``history``: settings and the value of the sweep's metric; ``running``: settings of the
    runs going) give the same next settings; ``tried`` are not chosen again when it can be helped.
    """
    if sweep.method == "grid":
        return _combination(sweep, index)
    rng = random.Random(f"{sweep.id}:{index}")
    if sweep.method == "bayes" and len(history or []) >= startup_runs(sweep):
        return _suggest(sweep, index, types, history or [], tried or [], running or [])
    combinations = _combinations(sweep)
    if combinations is not None:  # listed values only: a random order of all combinations, none twice
        order = random.Random(sweep.id).sample(range(combinations), total_of(sweep))
        return _combination(sweep, order[index % len(order)])
    # A range is cut into as many equal parts as there are random draws and each part is drawn
    # once (a Latin hypercube): a few draws spread over the range instead of landing together.
    draws = startup_runs(sweep) if sweep.method == "bayes" else total_of(sweep)
    out: dict[str, Any] = {}
    for name, spec in sweep.space.items():
        if "values" in spec:
            out[name] = rng.choice(spec["values"])
            continue
        part = random.Random(f"{sweep.id}:{name}").sample(range(draws), draws)[index % draws]
        share = (part + rng.random()) / draws
        low, high = (math.log(spec["min"]), math.log(spec["max"])) if spec.get("log") else (spec["min"], spec["max"])
        value = low + share * (high - low)
        out[name] = _number(math.exp(value) if spec.get("log") else value, types.get(name))
    return out


# ---- bayes: Optuna's TPE ------------------------------------------------------------------
#
# The Hub keeps no Optuna storage: each time a run is to start, a study is rebuilt in memory
# from the sweep's runs (finished ones with their value, running ones as running) and asked for
# the next settings. TPE splits the results into good and the rest and draws where the good ones
# are dense and the rest are not; with ``constant_liar`` the runs going count as not-so-good, so
# runs started side by side spread out instead of trying the same place.


def _same(a: Any, b: Any) -> bool:
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _distributions(sweep: Sweep, types: dict[str, str]) -> dict[str, Any]:
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

    out: dict[str, Any] = {}
    for name, spec in sweep.space.items():
        if "values" in spec:
            out[name] = CategoricalDistribution(tuple(spec["values"]))
        elif types.get(name) == "integer":
            low, high = math.ceil(spec["min"]), math.floor(spec["max"])
            out[name] = IntDistribution(low, max(low, high), log=bool(spec.get("log")) and low > 0)
        else:
            out[name] = FloatDistribution(spec["min"], spec["max"], log=bool(spec.get("log")))
    return out


def _as_params(settings: dict[str, Any], space: dict[str, Any]) -> dict[str, Any] | None:
    """A run's settings as Optuna params: numbers kept inside their range, None when a listed value is not listed."""
    from optuna.distributions import CategoricalDistribution, IntDistribution

    params: dict[str, Any] = {}
    for name, dist in space.items():
        value = settings.get(name)
        if isinstance(dist, CategoricalDistribution):
            match = next((c for c in dist.choices if _same(c, value)), None)
            if match is None and not any(_same(c, value) for c in dist.choices):
                return None
            params[name] = match
        else:
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            value = min(dist.high, max(dist.low, value))
            params[name] = int(round(value)) if isinstance(dist, IntDistribution) else float(value)
    return params


def _suggest(
    sweep: Sweep,
    index: int,
    types: dict[str, str],
    history: list[tuple[dict[str, Any], float]],
    tried: list[dict[str, Any]],
    running: list[dict[str, Any]],
) -> dict[str, Any]:
    import optuna
    from optuna.trial import TrialState, create_trial

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = _distributions(sweep, types)
    seed = int(hashlib.sha256(f"{sweep.id}:{index}".encode()).hexdigest()[:8], 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # multivariate and constant_liar are marked experimental
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=startup_runs(sweep), multivariate=True, constant_liar=True
        )
        study = optuna.create_study(direction=sweep.goal, sampler=sampler)
        for settings, value in history:
            if (params := _as_params(settings, space)) is not None:
                study.add_trial(create_trial(params=params, distributions=space, value=value))
        for settings in running:
            if (params := _as_params(settings, space)) is not None:
                study.add_trial(create_trial(params=params, distributions=space, state=TrialState.RUNNING))
        chosen: dict[str, Any] = {}
        for _ in range(5):  # settings already tried would only repeat a result: ask again (the ask counts as running)
            trial = study.ask(space)
            chosen = {
                name: _number(value, types.get(name)) if isinstance(value, float) else value
                for name, value in trial.params.items()
            }
            if not any(all(_same(t.get(k), v) for k, v in chosen.items()) for t in tried):
                break
    return chosen


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
        raise ServiceError(400, "method는 grid(모든 조합), random(무작위) 또는 bayes(베이지안)입니다.")
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
        raise ServiceError(
            400, f"조합이 {total}개입니다. {MAX_RUNS}개 이하가 되게 값을 줄이거나 무작위 · 베이지안 탐색을 쓰세요."
        )
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
    history: list[tuple[dict[str, Any], float]] = []
    tried: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    if sweep.method == "bayes":  # what was tried so far, how the finished ones did, which are going
        for r in runs:
            settings = {k: (r.config or {}).get(k) for k in sweep.space}
            tried.append(settings)
            status, value = _status(db, r), (r.summary or {}).get(sweep.metric)
            if status == "succeeded" and isinstance(value, int | float) and math.isfinite(value):
                history.append((settings, float(value)))
            elif status not in TERMINAL_STATUSES:
                running.append(settings)
    while active < sweep.parallel and started < total:
        chosen = settings_for(sweep, started, types, history, tried, running)
        tried.append(chosen)
        running.append(chosen)
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
        "startup": startup_runs(sweep) if sweep.method == "bayes" else None,  # drawn at random before results guide
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
