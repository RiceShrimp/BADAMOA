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

from . import asks, catalog, drafts, labeling, modelops, registry, services, training, transfers
from .catalog_models import (
    Dataset,
    ItemPrediction,
    LabelSet,
    LabelSetItem,
    LabelSpace,
    Project,
    Release,
    ReleaseItem,
)
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion

SINCE = (0, 52, 0)  # node versions from here on run a template's predict in autolabel mode
RECEIVED_SINCE = (0, 93, 0)  # and from here on over images received from another server (kept in its cache)
KEEP = 0.25  # predictions below this are dropped
ALIASES = ("production", "SOTA")


def _names(db: Session, release: Release) -> dict[str, int]:
    space = db.get(LabelSpace, release.label_space_id)
    return {str(c["name"]): int(c["id"]) for c in (space.classes if space else [])}


def _problem(db: Session, model: ModelVersion, task: str, names: dict[str, int]) -> str:
    """Why this model cannot label the set's images, or "" when it can."""
    if model.task and not task_names.same(model.task, task):
        return f"{task_names.short(model.task)}용 모델입니다 (이 데이터셋은 {task_names.short(task)})"
    if "predict" not in modelops.actions(db, model):
        return "이 모델 종류에는 예측 코드가 없습니다"
    if not modelops.autolabels(db, model):
        return "이 모델 종류는 오토라벨을 하지 않습니다"
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
    return {"todo": _todo(db, found), "keep": KEEP, "models": _models(db, found.project_id, release)}


def _models(db: Session, project_id: str, release: Release) -> list[dict[str, Any]]:
    """The project's models, each with why it cannot label this release's images ("" when it can), the ones that
    can and have an alias (production, SOTA) first, then the best by the task's metric."""
    task = catalog.release_task(db, release)
    names = _names(db, release)
    models = []
    for model in db.scalars(
        select(ModelVersion)
        .where(ModelVersion.project_id == project_id, ModelVersion.deleted_at.is_(None))
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
    return models


def start(
    db: Session,
    user: User,
    set_id: str,
    model_id: str,
    keep: float = KEEP,
    accept: float | None = None,
    gpus: int | None = None,
    via: str = "ui",
    node_id: str = "",
    gpu_indices: list[int] | None = None,
    data_after: str = "",
) -> ModelJob:
    """Run the model over the set's images not labelled yet, on the server that has them or on ``node_id``, which
    receives them first when it has none (only the set's images still to label, like a training's or an evaluation's
    transfer). On a GPU when the server has one, unless ``gpus=0``."""
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
    holder: Node = training.data_node(db, release.id)
    node, after = holder, None
    if node_id and node_id != holder.id:
        node = modelops._picked(db, node_id)
        if node.id not in transfers.holders(db, release.id):  # the set's images come to it first
            if not asks.answers_questions(node.agent_version or "", RECEIVED_SINCE):
                need = ".".join(str(n) for n in RECEIVED_SINCE)
                raise ServiceError(400, f"{node.name}에서 다른 서버의 이미지로 오토라벨하려면 {need} 이상이어야 합니다.")
            transfers.check_route(holder, node)
            after = transfers.sync_job(db, user, release.id, node, via, set_id=found.id).id
    if not asks.answers_questions(node.agent_version or "", SINCE):
        need = ".".join(str(n) for n in SINCE)
        raise ServiceError(409, f"{node.name}의 MOA를 {need} 이상으로 업데이트해야 오토라벨을 돌릴 수 있습니다.")
    # the model numbers its classes 0..n-1: each one's id in this dataset, by name (None: not in it, dropped)
    class_ids = [names.get(name) for name in model.classes or []]
    picked = transfers.picked_gpus(node, gpu_indices, gpus)
    data_after = transfers.data_after_of(data_after, node.id, holder.id)
    if picked:
        gpus = len(picked)
    else:
        gpus = (1 if (node.info or {}).get("gpus") else 0) if gpus is None else max(0, min(int(gpus), 1))
    params = {"conf": keep, "keep": keep, "accept": accept, "set_id": found.id}
    # the version too: its copy on that server is spared while it runs, and kept or removed as asked once over
    extra = {"class_ids": class_ids, "release_id": release.id}
    op = modelops._start(
        db, user, model, "autolabel", node, params, release_id=release.id, extra=extra, gpus=gpus, via=via, after=after,
        gpu_indices=picked, data_after=data_after,
    )  # fmt: skip
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
        "op_id": info.get("op_id"),  # the run on the 오토라벨 screen
        "job_id": job.id if job is not None else None,
        "status": state,
        "error": (job.error or "")[:500] if job is not None and state == "failed" else None,
    }


# ---- the 오토라벨 screen: every run across projects, the sets waiting, starting from a dataset's images --------------

RECENT = 4  # runs listed on a project's card
SHOWN_SETS = 3  # sets waiting, listed on a project's card
MAX_RUNS = 2000
GOING = ("queued", "running")


def _ops(db: Session, project_id: str = "", set_id: str = "", limit: int = MAX_RUNS) -> list[ModelJob]:
    query = select(ModelJob).where(ModelJob.kind == "autolabel")
    if project_id:
        query = query.join(ModelVersion, ModelVersion.id == ModelJob.model_id).where(ModelVersion.project_id == project_id)
    ops = list(db.scalars(query.order_by(ModelJob.created_at.desc()).limit(limit)))
    if set_id:
        ops = [op for op in ops if (op.params or {}).get("set_id") == set_id]
    return ops


def _rows(db: Session, settings: HubSettings, ops: list[ModelJob]) -> list[dict[str, Any]]:
    """Runs as the screen lists them: where each is, which model on which set and images, and what came of it."""
    if not ops:
        return []
    jobs = {j.id: j for j in db.scalars(select(Job).where(Job.id.in_([op.job_id for op in ops if op.job_id])))}
    models = {m.id: m for m in db.scalars(select(ModelVersion).where(ModelVersion.id.in_({op.model_id for op in ops})))}
    set_ids = {str((op.params or {}).get("set_id") or "") for op in ops} - {""}
    sets = {s.id: s for s in db.scalars(select(LabelSet).where(LabelSet.id.in_(set_ids)))} if set_ids else {}
    release_ids = {op.release_id for op in ops if op.release_id}
    releases = {r.id: r for r in db.scalars(select(Release).where(Release.id.in_(release_ids)))}
    dataset_ids = {r.dataset_id for r in releases.values()}
    datasets = {d.id: d for d in db.scalars(select(Dataset).where(Dataset.id.in_(dataset_ids)))}
    project_ids = {m.project_id for m in models.values()}
    projects = {p.id: p for p in db.scalars(select(Project).where(Project.id.in_(project_ids)))}
    starters = {op.created_by for op in ops if op.created_by}
    users = dict(db.execute(select(User.id, User.username).where(User.id.in_(starters))).all()) if starters else {}
    nodes = services.node_name_map(db)
    now = utcnow()
    out = []
    for op in ops:
        params = op.params or {}
        job = jobs.get(op.job_id or "")
        state = job.status if job is not None else "failed"
        model = models.get(op.model_id)
        found = sets.get(str(params.get("set_id") or ""))
        release = releases.get(op.release_id or "")
        dataset = datasets.get(release.dataset_id) if release else None
        project = projects.get(model.project_id) if model else None
        result = (op.result or {}).get("autolabel") or None
        current = found is not None and (found.autolabel or {}).get("op_id") == op.id
        took = None
        if job is not None and job.started_at and (job.finished_at or state == "running"):
            took = ((job.finished_at or now) - job.started_at).total_seconds()
        out.append(
            {
                "id": op.id,
                "job_id": op.job_id,
                "status": state,
                "error": (job.error or "")[:500] if job is not None and state == "failed" else None,
                "wait_reason": services.explain_wait(db, settings, job) if job is not None and state == "queued" else None,
                "duration_s": round(took, 1) if took is not None else None,
                "node_name": nodes.get(job.node_id or job.target_node_id or "") if job is not None else None,
                "gpus": int((job.requirements or {}).get("gpus") or 0) if job is not None else 0,
                "created_at": iso(op.created_at),
                "finished_at": iso(job.finished_at) if job is not None and job.finished_at else None,
                "started_by": users.get(op.created_by or ""),
                "via": job.created_via if job is not None else None,
                "project": {"id": project.id, "name": project.name} if project else None,
                "model": modelops._model_ref(model),
                # the set it labelled (None once the set is deleted); a set's newest run is where its editor starts
                "set": {"id": found.id, "name": found.name, "review": found.review} if found else None,
                "set_gone": found is None,
                "current": current,
                "dataset": {"id": dataset.id, "name": dataset.name} if dataset else None,
                "release": (
                    {"id": release.id, "version": release.version, "draft": release.status == "draft"} if release else None
                ),
                "keep": params.get("keep"),
                "accept": params.get("accept"),
                # images it was asked to label: what its set had to do then (its newest run), else what it predicted
                "items": (found.autolabel or {}).get("items") if current else (result or {}).get("images"),
                "result": result,
            }
        )
    return out


def runs(db: Session, settings: HubSettings, project_id: str = "", set_id: str = "") -> list[dict[str, Any]]:
    """Every 오토라벨 run (of a project, of a set), newest first."""
    return _rows(db, settings, _ops(db, project_id, set_id))


def run_view(db: Session, settings: HubSettings, op_id: str) -> dict[str, Any]:
    """One run on its own screen: the row, which code ran it and how much it logged."""
    op = db.get(ModelJob, op_id)
    if op is None or op.kind != "autolabel":
        raise ServiceError(404, "오토라벨 기록을 찾을 수 없습니다.")
    (row,) = _rows(db, settings, [op])
    job = db.get(Job, op.job_id) if op.job_id else None
    return {
        **row,
        "code": modelops._code_of(job),
        "started_at": iso(job.started_at) if job is not None and job.started_at else None,
        "log_count": job.log_count if job is not None else 0,
        "gpu_indices": list(job.gpu_indices or []) if job is not None else [],
    }


def _waiting(db: Session) -> dict[str, int]:
    """Each set's images nobody labelled yet."""
    return dict(
        db.execute(
            select(LabelSetItem.set_id, func.count()).where(LabelSetItem.state == "todo").group_by(LabelSetItem.set_id)
        ).all()
    )


TOTALS = ("runs", "going", "succeeded", "failed", "images", "labels", "accepted", "sets_waiting", "todo")


def board(db: Session, settings: HubSettings) -> dict[str, Any]:
    """오토라벨's first screen: the runs going on now, then each project: its runs and what came of them, the sets
    with images still to label, and how many of its models can label."""
    tasks = catalog.project_tasks(db)
    projects = {p.id: p for p in db.scalars(select(Project).where(Project.deleted_at.is_(None)).order_by(Project.name))}
    rows = [r for r in _rows(db, settings, _ops(db)) if r["project"] and r["project"]["id"] in projects]
    todo = _waiting(db)
    sets: dict[str, list[LabelSet]] = {}
    for found in db.scalars(select(LabelSet).order_by(LabelSet.created_at.desc())):
        if found.project_id in projects:
            sets.setdefault(found.project_id, []).append(found)
    able: Counter = Counter()
    for model in db.scalars(select(ModelVersion).where(ModelVersion.deleted_at.is_(None))):
        if model.project_id in projects and modelops.autolabels(db, model):
            able[model.project_id] += 1
    names = {d.id: d.name for d in db.scalars(select(Dataset).where(Dataset.project_id.in_(list(projects))))}
    cards = []
    totals: Counter = Counter()
    for pid, project in projects.items():
        mine = [r for r in rows if r["project"]["id"] == pid]
        waiting = sorted((s for s in sets.get(pid, []) if todo.get(s.id)), key=lambda s: -todo[s.id])
        if not mine and not waiting and not tasks.get(pid):
            continue  # no dataset version in it yet: nothing to label
        done = [r["result"] for r in mine if r["status"] == "succeeded" and r["result"]]
        counts = Counter(r["status"] for r in mine)
        card = {
            "project": {"id": pid, "name": project.name, "folder": project.folder, "tasks": tasks.get(pid, [])},
            "runs": len(mine),
            "going": counts["queued"] + counts["running"],
            "succeeded": counts["succeeded"],
            "failed": counts["failed"],
            "images": sum(int(r.get("images") or 0) for r in done),
            "labels": sum(int(r.get("labels") or 0) for r in done),
            "accepted": sum(int(r.get("accepted") or 0) for r in done),
            "recent": mine[:RECENT],
            "last_at": mine[0]["created_at"] if mine else None,
            "sets": [
                {
                    "id": s.id,
                    "name": s.name,
                    "dataset": names.get(s.dataset_id, ""),
                    "todo": todo[s.id],
                    "autolabeled": bool((s.autolabel or {}).get("op_id")),
                }
                for s in waiting[:SHOWN_SETS]
            ],
            "sets_waiting": len(waiting),
            "todo": sum(todo[s.id] for s in waiting),
            "models": able[pid],
        }
        cards.append(card)
        for key in TOTALS:
            totals[key] += card[key]
    # the projects worked on most recently first, then the ones with nothing run yet by name
    cards.sort(key=lambda c: c["project"]["name"])
    cards.sort(key=lambda c: c["last_at"] or "", reverse=True)
    return {
        "going": [r for r in rows if r["status"] in GOING],
        "projects": cards,
        "totals": {k: totals[k] for k in TOTALS},
    }


def targets(db: Session, project_id: str) -> dict[str, Any]:
    """What a project's 오토라벨 can run over: its labeling sets (with the images nobody labelled yet), and each
    dataset's images where labels go now (its work in progress, else its newest version), counted by split and
    label state, to make a set of first."""
    todo = _waiting(db)
    sizes = dict(
        db.execute(
            select(LabelSetItem.set_id, func.count())
            .join(LabelSet, LabelSet.id == LabelSetItem.set_id)
            .where(LabelSet.project_id == project_id)
            .group_by(LabelSetItem.set_id)
        ).all()
    )
    sets = []
    for found in db.scalars(select(LabelSet).where(LabelSet.project_id == project_id).order_by(LabelSet.created_at.desc())):
        dataset = db.get(Dataset, found.dataset_id)
        sets.append(
            {
                "id": found.id,
                "name": found.name,
                "dataset": {"id": dataset.id, "name": dataset.name} if dataset else None,
                "review": found.review,
                "todo": todo.get(found.id, 0),
                "items": sizes.get(found.id, 0),
                "autolabel": status(db, found),
            }
        )
    datasets = []
    for dataset in db.scalars(select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.name)):
        release = drafts.draft_of(db, dataset.id) or drafts.latest_ready(db, dataset.id)
        if release is None:
            continue
        counts: dict[str, dict[str, int]] = {}
        for split, state, n in db.execute(
            select(ReleaseItem.split, ReleaseItem.state, func.count())
            .where(ReleaseItem.release_id == release.id)
            .group_by(ReleaseItem.split, ReleaseItem.state)
        ):
            counts.setdefault(split or "", {})[state] = n
        datasets.append(
            {
                "id": dataset.id,
                "name": dataset.name,
                "task": catalog.dataset_task(dataset),
                "release": {"id": release.id, "version": release.version, "draft": release.status == "draft"},
                "counts": counts,  # split → label state → images
            }
        )
    return {"sets": sets, "datasets": datasets}


def _servers(db: Session, settings: HubSettings, release: Release, set_id: str = "") -> dict[str, Any]:
    """The servers it can run on, as for an evaluation: the one holding the images, or another one, which receives
    the set's images first. Each with whether it can and why not; the one picked unless another is: the holder when it
    can, else the first that can (one that has the images first)."""
    try:
        holder = training.data_node(db, release.id)
    except ServiceError as e:
        return {"default": "", "holder": None, "nodes": [], "problem": e.message}
    from . import resources  # noqa: PLC0415

    have = transfers.holders(db, release.id)
    assigned = resources.assignments(db)
    # what a server without the images receives: the set's images still to label (a dataset's: its version's)
    only = select(LabelSetItem.item_id).where(LabelSetItem.set_id == set_id, LabelSetItem.state == "todo")
    items, size = transfers.release_bytes(db, release.id, only if set_id else None)
    found = []
    for node in db.scalars(select(Node).where(Node.enabled.is_(True)).order_by(Node.name)):
        why: list[str] = []
        status_now = services.node_status(node, settings)
        if status_now != "online":
            why.append("응답 없음")
        if not registry.has_role(node, registry.AUTOLABELS):
            why.append("오토라벨용 서버가 아님 (노드 설정에서 '오토라벨용'을 켭니다)")
        if not asks.answers_questions(node.agent_version or "", SINCE):
            why.append(f"MOA {'.'.join(str(n) for n in SINCE)} 이상 필요")
        transfer = node.id not in have
        if transfer:
            if not asks.answers_questions(node.agent_version or "", RECEIVED_SINCE):
                need = ".".join(str(n) for n in RECEIVED_SINCE)
                why.append(f"다른 서버의 이미지로 오토라벨하려면 MOA {need} 이상 필요")
            else:
                try:
                    transfers.check_route(holder, node)
                except ServiceError as e:
                    why.append(e.message)
        found.append(
            {
                "id": node.id,
                "name": node.name,
                "status": status_now,
                "gpus": len((node.info or {}).get("gpus") or []),
                "fits": not why,
                "why": " · ".join(dict.fromkeys(why)),
                "transfer": transfer,  # the set's images are sent there first
                "holds_images": node.id == holder.id,
                **transfers.place_info(node, settings, status_now, assigned.get(node.id), holder.id, have),
            }
        )
    fitting = [f for f in found if f["fits"]]
    first = next((f for f in fitting if f["id"] == holder.id), None) or next(
        (f for f in fitting if not f["transfer"]), None
    )
    default = (first or (fitting[0] if fitting else None) or {"id": holder.id})["id"]
    found.sort(key=lambda f: (f["id"] != default, not f["fits"], f["name"]))
    return {
        "default": default,
        "holder": {"id": holder.id, "name": holder.name},
        "items": items,
        "bytes": size,
        "nodes": found,
        "problem": "",
    }


def offer(db: Session, settings: HubSettings, set_id: str = "", release_id: str = "") -> dict[str, Any]:
    """For starting one: the models that can label a set's images (or a dataset's, before its set is made) and the
    server it runs on."""
    if set_id:
        found = labeling.get_set(db, set_id)
        release, project_id, todo = _open(db, found), found.project_id, _todo(db, found)
    else:
        release = catalog.get_release(db, release_id)
        _, why = drafts.edit_place(db, release)
        if why:
            raise ServiceError(409, why.replace("라벨은", "오토라벨은").replace("고칩니다", "돌립니다"))
        project_id, todo = db.get(Dataset, release.dataset_id).project_id, None
    return {
        "todo": todo,
        "keep": KEEP,
        "models": _models(db, project_id, release),
        "servers": _servers(db, settings, release, set_id),
    }


def start_on_images(
    db: Session,
    user: User,
    release_id: str,
    model_id: str,
    *,
    name: str = "",
    split: str = "",
    part_size: int = 50,
    review: bool = True,
    keep: float = KEEP,
    accept: float | None = None,
    gpus: int | None = None,
    via: str = "ui",
    node_id: str = "",
    parts: int = 0,
    instructions: str = "",
    gpu_indices: list[int] | None = None,
    data_after: str = "",
) -> tuple[LabelSet, ModelJob]:
    """A dataset's images nobody labelled yet (of one split, or all): a labeling set of them, labelled by the model
    first. People take its parts as with any set (``parts``, ``instructions``: as ``labeling.create_set``)."""
    release = catalog.get_release(db, release_id)
    dataset = db.get(Dataset, release.dataset_id)
    model = training.get_model(db, model_id)
    if model.project_id != dataset.project_id:
        raise ServiceError(400, "다른 프로젝트의 모델입니다.")
    if split and split not in ("train", "val", "test"):
        raise ServiceError(400, "split은 train, val, test 중에서 고르세요.")
    match: dict[str, Any] = {"state": "unlabelled", **({"split": split} if split else {})}
    name = name.strip() or f"{dataset.name} 오토라벨 {utcnow():%m/%d}"
    try:
        found = labeling.create_set(
            db, user, release_id=release.id, name=name, match=match, part_size=part_size, review=review, via=via,
            parts=parts, instructions=instructions,
        )  # fmt: skip
    except ServiceError as e:
        if e.status == 400 and "맞는 이미지가 없습니다" in e.message:
            where = f"{split}에 " if split else ""
            raise ServiceError(400, f"이 데이터셋의 {where}아직 라벨링하지 않은 이미지가 없습니다.") from e
        raise
    return found, start(
        db, user, found.id, model_id, keep=keep, accept=accept, gpus=gpus, via=via, node_id=node_id,
        gpu_indices=gpu_indices, data_after=data_after,
    )  # fmt: skip
