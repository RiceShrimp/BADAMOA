"""The first screens of 라벨링, 검수, 학습 and 모델: every project at a glance, like the dataset board.

    labeling   the parts of labeling sets given to me (to label or to review) with what is left, and
               each project's sets: a few of their images, how far they got, parts per board column
    review     the newest labels of each dataset (its work in progress, else its newest version): how
               many images the rule checks flag as missing, wrong or odd boxes, and what the newest
               evaluation of a model adds; the most problems first
    training   the runs going on now (how far, about how long is left, node and GPUs) and each
               project's runs: how many, how many failed, the best by the task's metric, the latest
    models     each project's models: the versions with an alias (champion, production …) and their
               metric, how many there are, the best and the newest

Everything comes from what the Hub keeps already: nothing is stored for these screens.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from . import catalog, checks, drafts, labeling, overview, qa, services, training
from .catalog_models import (
    Dataset,
    Item,
    LabelRevision,
    LabelSet,
    LabelSetItem,
    LabelSetPart,
    Project,
    QaResult,
    Release,
    ReleaseItem,
)
from .config import HubSettings
from .db import iso
from .models import Job, User
from .train_models import ModelVersion, Run, RunMetric

SHOTS = 4  # thumbnails on a card
QA_ON_BOARD = 20_000  # images: a bigger version whose checks are out of date is checked when it is opened
RECENT = 4  # runs listed on a project's card
TREND = 12  # finished runs drawn on a project's card, oldest first
ALIAS_ORDER = ("production", "champion", "staging", "challenger")
SHOWN_ALIASED = 4


# ---- pieces ------------------------------------------------------------------------


def _projects(db: Session) -> list[Project]:
    return list(db.scalars(select(Project).where(Project.deleted_at.is_(None)).order_by(Project.name)))


def _project(project: Project, tasks: dict[str, list[str]]) -> dict[str, Any]:
    return {"id": project.id, "name": project.name, "folder": project.folder, "tasks": tasks.get(project.id, [])}


def _shots(db: Session, settings: HubSettings, release: Release | None, item_ids: list[int]) -> list[dict[str, Any]]:
    """Thumbnails of these images with their labels in the release, in the order given."""
    if release is None or not item_ids:
        return []
    rows = db.execute(
        select(Item, LabelRevision)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id)
        .where(ReleaseItem.release_id == release.id, Item.id.in_(item_ids), Item.thumb_key.is_not(None))
    ).all()
    found = {item.id: (item, revision) for item, revision in rows}
    task = catalog.release_task(db, release)
    return [
        {
            "url": catalog.media_url(settings, "thumb", item.thumb_key),
            "width": item.width,
            "height": item.height,
            "boxes": [catalog.light_shape(b) for b in (revision.boxes or [])[: overview.COLLAGE_BOXES]],
            "task": task,
        }
        for item, revision in (found[i] for i in item_ids if i in found)
    ]


def _number(value: Any) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


# ---- 라벨링 ----------------------------------------------------------------------------


def labeling_board(db: Session, settings: HubSettings, user: User) -> dict[str, Any]:
    """My parts first (what is left to label or review), then every project's labeling sets."""
    tasks = catalog.project_tasks(db)
    projects = {p.id: p for p in _projects(db)}
    sets = [s for s in db.scalars(select(LabelSet).order_by(LabelSet.created_at.desc())) if s.project_id in projects]
    mine: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    totals = Counter()
    for found in sets:
        counts = labeling._counts(db, found.id)
        card = labeling.set_summary(db, found, user, counts)
        opened = labeling._open_release(db, found)
        size = card["items"]
        positions = sorted({size * n // SHOTS for n in range(SHOTS)}) if size else []
        spread = list(
            db.scalars(
                select(LabelSetItem.item_id)
                .where(LabelSetItem.set_id == found.id, LabelSetItem.position.in_(positions))
                .order_by(LabelSetItem.position)
            )
        )
        labelled, reviewed = db.execute(
            select(func.max(LabelSetItem.labelled_at), func.max(LabelSetItem.reviewed_at)).where(
                LabelSetItem.set_id == found.id
            )
        ).one()
        card["collage"] = _shots(db, settings, opened, spread)
        card["open_release_id"] = opened.id if opened else None
        card["updated_at"] = iso(max(m for m in (found.created_at, labelled, reviewed) if m is not None))
        groups.setdefault(found.project_id, []).append(card)

        states = card["states"]
        totals["sets"] += 1
        totals["items"] += size
        totals["todo"] += states["todo"] + states["rejected"]
        totals["review"] += states["labelled"] if found.review else 0
        totals["done"] += states["approved"] + (0 if found.review else states["labelled"])

        given = db.scalars(
            select(LabelSetPart)
            .where(LabelSetPart.set_id == found.id, or_(LabelSetPart.assignee == user.id, LabelSetPart.reviewer == user.id))
            .order_by(LabelSetPart.part)
        )
        for part in given:
            c = counts.get(part.part, Counter())
            column = labeling.column_of(c, found.review, part.assignee)
            if column == "done":
                continue
            labels = part.assignee == user.id
            reviews = found.review and part.reviewer == user.id
            to_label = c["todo"] + c["rejected"] if labels else 0
            to_review = c["labelled"] if reviews else 0
            mine.append(
                {
                    "set": {"id": found.id, "name": found.name, "review": found.review},
                    "project": {"id": found.project_id, "name": projects[found.project_id].name},
                    "dataset": card["dataset"],
                    "part": part.part,
                    "size": sum(c.values()),
                    "states": {s: c[s] for s in labeling.STATES},
                    "column": column,
                    "roles": [r for r, yes in (("label", labels), ("review", reviews)) if yes],
                    "to_label": to_label,
                    "rejected": c["rejected"] if labels else 0,
                    "to_review": to_review,
                    # where to open it: the first image to label (or sent back), the first to review
                    "next_label": labeling._first(db, found.id, part.part, ("todo", "rejected")) if to_label else None,
                    "next_review": labeling._first(db, found.id, part.part, ("labelled",)) if to_review else None,
                }
            )
    mine.sort(key=lambda m: 0 if m["to_label"] or m["to_review"] else 1)  # something to do now first
    out = [
        {"project": _project(projects[pid], tasks), "sets": cards, "updated_at": max(c["updated_at"] for c in cards)}
        for pid, cards in groups.items()
    ]
    out.sort(key=lambda g: g["updated_at"], reverse=True)
    return {
        "me": {"id": user.id, "name": user.username},
        "mine": mine,
        "projects": out,
        # projects with a dataset version but no set yet: where a set could be made
        "without": [_project(p, tasks) for p in projects.values() if p.id not in groups and tasks.get(p.id)],
        "totals": {k: totals[k] for k in ("sets", "items", "todo", "review", "done")},
    }


# ---- 검수 ------------------------------------------------------------------------------


def review_board(db: Session, settings: HubSettings) -> dict[str, Any]:
    """Each dataset's newest labels with the problems the checks find, the most first."""
    tasks = catalog.project_tasks(db)
    projects = {p.id: p for p in _projects(db)}
    cards = []
    for dataset in db.scalars(select(Dataset).where(Dataset.project_id.in_(list(projects))).order_by(Dataset.name)):
        latest = drafts.latest_ready(db, dataset.id)
        if latest is None:
            continue
        draft = drafts.draft_of(db, dataset.id)
        target = draft or latest  # where fixes go: what is checked on the 검수 screen first
        items = (latest.stats or {}).get("items", 0)
        card: dict[str, Any] = {
            "project": _project(projects[dataset.project_id], tasks),
            "dataset": {"id": dataset.id, "name": dataset.name},
            "task": catalog.dataset_task(dataset),
            "release": {
                "id": target.id,
                "version": drafts.DRAFT if draft else target.version,
                "draft": draft is not None,
                "base": latest.version,
                "edited": drafts.edited_count(db, draft) if draft else 0,
            },
            "items": items,
            "checked": False,
            "computed_at": None,
            "groups": None,
            "checks": [],
            "flagged_shots": False,
        }
        if items <= QA_ON_BOARD or qa.is_fresh(db, target):
            done = qa.run(db, target)
            open_counts = qa.check_counts(db, target.id)  # what someone let be is not a problem
            flagged = list(
                db.scalars(
                    select(QaResult.item_id)
                    .where(QaResult.release_id == target.id)
                    .distinct()
                    .order_by(QaResult.item_id)
                    .limit(SHOTS)
                )
            )
            shots = _shots(db, settings, target, flagged)
            card.update(
                checked=True,
                computed_at=iso(done.computed_at),
                groups=qa.groups(db, target.id),
                checks=sorted(
                    (
                        {"id": cid, "title": meta["title"], "group": meta["group"], "images": open_counts[cid][0]}
                        for cid, meta in qa.CHECKS.items()
                        if open_counts.get(cid, (0, 0))[0]
                    ),
                    key=lambda c: -c["images"],
                ),
                flagged_shots=bool(shots),
                collage=shots,
            )
        if not card.get("collage"):
            card["collage"] = overview._collage(db, settings, latest)
        card["evaluation"] = checks.newest(db, target)  # what a model adds: extra objects, labels it missed
        cards.append(card)

    def rank(c: dict[str, Any]) -> tuple:
        if not c["checked"]:
            return (1, 0, c["dataset"]["name"])
        model = c["evaluation"]
        found = sum(model.get(kind, 0) for kind in ("extra", "missed", "reclass")) if model else 0
        return (0, -(c["groups"]["any"] + found), c["dataset"]["name"])

    cards.sort(key=rank)
    checked = [c for c in cards if c["checked"]]
    return {
        "datasets": cards,
        "totals": {
            "datasets": len(cards),
            "checked": len(checked),
            "flagged": sum(c["groups"]["any"] for c in checked),
            "missing": sum(c["groups"]["missing"] for c in checked),
            "wrong": sum(c["groups"]["wrong"] for c in checked),
            "boxes": sum(c["groups"]["boxes"] for c in checked),
            "evaluated": sum(1 for c in cards if c["evaluation"]),
        },
    }


# ---- 학습 ------------------------------------------------------------------------------


def _eta(db: Session, run: Run, elapsed: float | None) -> int | None:
    """About how many seconds are left: the time the reported epochs took, times the epochs to go.

    From the steps' timestamps when they came one by one; when they came together (a node reports in
    batches), from the time the job has run so far (setting up included: an estimate on the long side)."""
    if not run.epochs or run.epoch < 1 or run.epoch >= run.epochs:
        return None
    first, last = db.execute(select(func.min(RunMetric.ts), func.max(RunMetric.ts)).where(RunMetric.run_id == run.id)).one()
    if run.epoch >= 2 and first is not None and last is not None and (last - first).total_seconds() >= 1:
        per_epoch = (last - first).total_seconds() / (run.epoch - 1)  # the first report comes after epoch 1
    elif elapsed:
        per_epoch = elapsed / run.epoch
    else:
        return None
    return int(per_epoch * (run.epochs - run.epoch))


def training_board(db: Session, settings: HubSettings) -> dict[str, Any]:
    """The runs going on now, then each project's runs summed up (projects with runs going on first)."""
    tasks = catalog.project_tasks(db)
    projects = _projects(db)
    names = {p.id: p.name for p in projects}
    rows = [
        (run, status or "failed")
        for run, status in db.execute(
            select(Run, Job.status)
            .outerjoin(Job, Job.id == Run.job_id)
            .where(Run.deleted_at.is_(None))
            .order_by(Run.created_at.desc())
        ).all()
        if run.project_id in names
    ]
    nodes = services.node_name_map(db)
    going = []
    for run, status in rows:
        if status not in ("queued", "running"):
            continue
        view = training.run_view(db, settings, run, node_names=nodes)
        metric = training.PRIMARY_METRIC.get(view["task"], "map50_95")
        going.append(
            {
                **{k: view[k] for k in ("id", "name", "status", "stage", "task", "epoch", "epochs", "release")},
                **{k: view[k] for k in ("node_name", "gpu_indices", "wait_reason", "duration_s", "sync")},
                "project": {"id": run.project_id, "name": names[run.project_id]},
                "metric": metric,
                "value": _number(view["metrics"].get(metric)),
                "eta_s": _eta(db, run, view["duration_s"]) if status == "running" else None,
            }
        )
    going.sort(key=lambda r: (r["status"] != "running", r["project"]["name"]))

    by_project: dict[str, list[tuple[Run, str]]] = {}
    for run, status in rows:
        by_project.setdefault(run.project_id, []).append((run, status))
    cards = []
    for project in projects:
        mine = by_project.get(project.id, [])
        metric = training.PRIMARY_METRIC.get((tasks.get(project.id) or ["detect"])[0], "map50_95")
        statuses = Counter(status for _, status in mine)
        scored = [(run, v, s) for run, s in mine if (v := _number((run.summary or {}).get(metric))) is not None]
        best = max(scored, key=lambda rvs: rvs[1], default=None)  # a run going on counts with its best step so far
        finished = [v for _, v, s in scored if s == "succeeded"]  # the line: only runs that ended
        cards.append(
            {
                "project": _project(project, tasks),
                "metric": metric,
                "runs": len(mine),
                **{s: statuses[s] for s in ("running", "queued", "succeeded", "failed", "canceled")},
                "best": {"id": best[0].id, "name": best[0].name, "value": best[1]} if best else None,
                "trend": [round(v, 4) for v in finished[:TREND]][::-1],
                "recent": [
                    {
                        "id": run.id,
                        "name": run.name,
                        "status": status,
                        "epoch": run.epoch,
                        "epochs": run.epochs,
                        "value": _number((run.summary or {}).get(metric)),
                        "created_at": iso(run.created_at),
                    }
                    for run, status in mine[:RECENT]
                ],
                "last_at": iso(mine[0][0].created_at) if mine else None,
            }
        )
    cards.sort(key=lambda c: c["last_at"] or "", reverse=True)  # the latest first, no runs last
    cards.sort(key=lambda c: 0 if c["running"] or c["queued"] else 1)  # runs going on before all
    statuses = Counter(status for _, status in rows)
    return {
        "going": going,
        "projects": cards,
        "totals": {
            "runs": len(rows),
            **{s: statuses[s] for s in ("running", "queued", "succeeded", "failed")},
            "gpus": sum(len(r["gpu_indices"] or []) for r in going if r["status"] == "running"),
        },
    }


# ---- 모델 ------------------------------------------------------------------------------


def _model_row(model: ModelVersion) -> dict[str, Any]:
    metric = training.PRIMARY_METRIC.get(model.task or "detect", "map50_95")
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "aliases": training._aliases(model),
        "task": model.task or "detect",
        "arch": model.arch,
        "metric": metric,
        "value": _number((model.metrics or {}).get(metric)),
        "created_at": iso(model.created_at),
    }


def _alias_rank(row: dict[str, Any]) -> tuple:
    known = [ALIAS_ORDER.index(a) for a in row["aliases"] if a in ALIAS_ORDER]
    return (min(known) if known else len(ALIAS_ORDER), row["name"], -row["version"])


def models_board(db: Session, settings: HubSettings) -> dict[str, Any]:
    """Each project's models: the ones with an alias, the best, the newest (latest registered first)."""
    tasks = catalog.project_tasks(db)
    projects = _projects(db)
    live = {p.id for p in projects}
    by_project: dict[str, list[ModelVersion]] = {}
    for model in db.scalars(
        select(ModelVersion).where(ModelVersion.deleted_at.is_(None)).order_by(ModelVersion.created_at.desc())
    ):
        if model.project_id in live:
            by_project.setdefault(model.project_id, []).append(model)
    cards = []
    for project in projects:
        rows = [_model_row(m) for m in by_project.get(project.id, [])]
        metric = training.PRIMARY_METRIC.get((tasks.get(project.id) or ["detect"])[0], "map50_95")
        scored = [r for r in rows if r["metric"] == metric and r["value"] is not None]
        aliased = sorted((r for r in rows if r["aliases"]), key=_alias_rank)
        cards.append(
            {
                "project": _project(project, tasks),
                "metric": metric,
                "models": len(rows),
                "families": len({r["name"] for r in rows}),
                "aliased": aliased[:SHOWN_ALIASED],
                "aliased_count": len(aliased),
                "best": max(scored, key=lambda r: r["value"], default=None),
                "latest": rows[0] if rows else None,
            }
        )
    cards.sort(key=lambda c: c["latest"]["created_at"] if c["latest"] else "", reverse=True)
    return {
        "projects": cards,
        "totals": {
            "models": sum(c["models"] for c in cards),
            "families": sum(c["families"] for c in cards),
            "aliased": sum(c["aliased_count"] for c in cards),
            "production": sum(1 for models in by_project.values() for m in models if "production" in training._aliases(m)),
            "projects": sum(1 for c in cards if c["models"]),
        },
    }
