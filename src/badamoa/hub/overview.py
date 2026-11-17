"""What the card screens show: projects and datasets as cards, and a release's charts, classes and models.

    project card   collage of its newest images (with their boxes), task, counts of datasets,
                   releases, images, runs and models, the best model by the task's metric
    dataset card   the newest release of a dataset family: collage, classes, size, splits, label states
    release charts split and label state, classes, image sizes (histogram and width × height grid),
                   where the boxes sit (centre grid) and how big they are
    home           how much there is in all, the projects touched last, the latest runs

Large releases are summarized from a sample of their items so a chart never reads half a million
label revisions; the answer says how many it looked at.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import catalog, training
from .catalog_models import Dataset, Item, LabelRevision, LabelSpace, Project, Release, ReleaseItem
from .config import HubSettings
from .db import iso
from .models import Job
from .services import ServiceError
from .train_models import ModelVersion, Run, Sweep

COLLAGE = 4
COLLAGE_BOXES = 12
CHART_SAMPLE = 5000
GRID = 20
RECENT_PROJECTS = 4
RECENT_RUNS = 6


# ---- pieces ------------------------------------------------------------------------


def _latest(db: Session, dataset_id: str) -> Release | None:
    return db.scalar(
        select(Release)
        .where(Release.dataset_id == dataset_id, Release.deleted_at.is_(None), Release.status == "ready")
        .order_by(Release.created_at.desc())
        .limit(1)
    )


def _collage(db: Session, settings: HubSettings, release: Release | None, count: int = COLLAGE) -> list[dict[str, Any]]:
    """A few labelled images of a release, spread over it, with their boxes (fractions of the image)."""
    if release is None:
        return []
    total = (release.stats or {}).get("items") or 0
    step = max(1, total // (count * 2)) if total else 1
    rows = db.execute(
        select(Item, LabelRevision)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id)
        .where(ReleaseItem.release_id == release.id, ReleaseItem.box_count > 0, Item.thumb_key.is_not(None))
        .order_by(Item.id)
        .limit(count * step)
    ).all()
    picked = rows[::step][:count] or rows[:count]
    task = catalog.release_task(db, release)  # how to draw what follows each box
    return [
        {
            "url": catalog.media_url(settings, "thumb", item.thumb_key),
            "width": item.width,
            "height": item.height,
            "boxes": [catalog.light_shape(b) for b in (revision.boxes or [])[:COLLAGE_BOXES]],
            "task": task,
        }
        for item, revision in picked
    ]


def _class_names(db: Session, release: Release | None) -> dict[int, dict[str, Any]]:
    space = db.get(LabelSpace, release.label_space_id) if release else None
    return {int(c["id"]): c for c in (space.classes if space else [])}


def dataset_card(db: Session, settings: HubSettings, dataset: Dataset) -> dict[str, Any] | None:
    releases = db.scalar(
        select(func.count(Release.id)).where(Release.dataset_id == dataset.id, Release.deleted_at.is_(None))
    )
    latest = _latest(db, dataset.id)
    if latest is None:
        return None
    stats = latest.stats or {}
    classes = _class_names(db, latest)
    ranked = sorted(stats.get("classes", {}).items(), key=lambda kv: -kv[1].get("boxes", 0))
    return {
        "id": dataset.id,
        "name": dataset.name,
        "task": catalog.dataset_task(dataset),
        "release": {"id": latest.id, "version": latest.version, "created_at": iso(latest.created_at), "note": latest.note},
        "releases": releases or 0,
        "items": stats.get("items", 0),
        "bytes": stats.get("bytes", 0),
        "boxes": stats.get("boxes", 0),
        "classes": len(classes),
        "top_classes": [
            {
                "id": int(cid),
                "name": classes.get(int(cid), {}).get("name", cid),
                "color": classes.get(int(cid), {}).get("color"),
            }
            for cid, _ in ranked[:4]
        ],
        "splits": {name: s.get("items", 0) for name, s in (stats.get("splits") or {}).items()},
        "states": stats.get("states", {}),
        "collage": _collage(db, settings, latest),
    }


def _model_groups(db: Session, project: Project, tasks: list[str]) -> list[dict[str, Any]]:
    """The project's models by task, best first by that task's own main metric.

    A Top-1 and an mAP do not compare, so a project with detection and classification datasets
    ranks each on its own. Tasks come in the order of the project's datasets, then any others.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for model in db.scalars(
        select(ModelVersion).where(ModelVersion.project_id == project.id, ModelVersion.deleted_at.is_(None))
    ):
        task = model.task or "detect"
        value = (model.metrics or {}).get(training.PRIMARY_METRIC.get(task, "map50_95"))
        groups.setdefault(task, []).append(
            {
                "id": model.id,
                "name": f"{model.name} v{model.version}",
                "value": value if isinstance(value, int | float) else None,
                "aliases": model.aliases or [],
            }
        )
    order = [t for t in tasks if t in groups] + sorted(t for t in groups if t not in tasks)
    return [
        {
            "task": task,
            "metric": training.PRIMARY_METRIC.get(task, "map50_95"),
            "models": sorted(groups[task], key=lambda m: -(m["value"] if m["value"] is not None else -1)),
        }
        for task in order
    ]


def _best_model(db: Session, project: Project, tasks: list[str]) -> dict[str, Any] | None:
    """The best model of the project's main task (its first dataset's), or of the first task that has one."""
    for group in _model_groups(db, project, tasks):
        top = group["models"][0]
        if top["value"] is not None:
            return {"id": top["id"], "name": top["name"], "metric": group["metric"], "value": top["value"]}
    return None


def _labelled_classes(db: Session, project: Project) -> list[dict[str, Any]]:
    """Classes that have labels in the newest version of each dataset, by name, most labels first.

    Declared but unused classes (a COCO label space with 80 names) do not count: the board's
    class filter finds the projects that actually hold examples of a class.
    """
    found: dict[str, dict[str, Any]] = {}
    for dataset in db.scalars(select(Dataset).where(Dataset.project_id == project.id)):
        latest = _latest(db, dataset.id)
        if latest is None:
            continue
        names = _class_names(db, latest)
        for cid, numbers in ((latest.stats or {}).get("classes") or {}).items():
            name = str(names.get(int(cid), {}).get("name", cid))
            entry = found.setdefault(name.lower(), {"name": name, "labels": 0, "images": 0})
            entry["labels"] += int(numbers.get("boxes") or 0)
            entry["images"] += int(numbers.get("items") or 0)
    return sorted((e for e in found.values() if e["labels"] > 0), key=lambda e: (-e["labels"], e["name"]))


def _project_card(db: Session, settings: HubSettings, project: Project, counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    newest = db.scalar(
        select(Release)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(Dataset.project_id == project.id, Release.deleted_at.is_(None), Release.status == "ready")
        .order_by(Release.created_at.desc())
        .limit(1)
    )
    moments = [project.created_at, newest.created_at if newest else None]
    moments.append(db.scalar(select(func.max(Run.created_at)).where(Run.project_id == project.id)))
    return {
        "id": project.id,
        "name": project.name,
        "tasks": (tasks := catalog.project_tasks(db).get(project.id, [])),
        "task": (tasks or [""])[0],  # the first dataset's task, for older readers
        "folder": project.folder,
        "description": project.description,
        "created_at": iso(project.created_at),
        "updated_at": iso(max(m for m in moments if m is not None)),
        "datasets": counts["datasets"].get(project.id, 0),
        "releases": counts["releases"].get(project.id, 0),
        "items": counts["items"].get(project.id, 0),
        "runs": counts["runs"].get(project.id, 0),
        "models": counts["models"].get(project.id, 0),
        "best": _best_model(db, project, tasks or ["detect"]),  # the same order as the project page
        "collage": _collage(db, settings, newest),
        "classes": _labelled_classes(db, project),
    }


def _counts(db: Session) -> dict[str, dict[str, int]]:
    alive = Release.deleted_at.is_(None)
    return {
        "datasets": dict(
            db.execute(
                select(Dataset.project_id, func.count(func.distinct(Dataset.id)))
                .join(Release, Release.dataset_id == Dataset.id)
                .where(alive)
                .group_by(Dataset.project_id)
            ).all()
        ),
        "releases": dict(
            db.execute(
                select(Dataset.project_id, func.count(Release.id))
                .join(Release, Release.dataset_id == Dataset.id)
                .where(alive)
                .group_by(Dataset.project_id)
            ).all()
        ),
        "items": dict(db.execute(select(Item.project_id, func.count(Item.id)).group_by(Item.project_id)).all()),
        "runs": dict(
            db.execute(
                select(Run.project_id, func.count(Run.id)).where(Run.deleted_at.is_(None)).group_by(Run.project_id)
            ).all()
        ),
        "models": dict(
            db.execute(
                select(ModelVersion.project_id, func.count(ModelVersion.id))
                .where(ModelVersion.deleted_at.is_(None))
                .group_by(ModelVersion.project_id)
            ).all()
        ),
    }


# ---- screens -------------------------------------------------------------------------


def project_cards(db: Session, settings: HubSettings) -> list[dict[str, Any]]:
    projects = db.scalars(select(Project).where(Project.deleted_at.is_(None)).order_by(Project.name)).all()
    counts = _counts(db)
    return [_project_card(db, settings, p, counts) for p in projects]


def project_page(db: Session, settings: HubSettings, project_id: str) -> dict[str, Any]:
    project = catalog.get_project(db, project_id)
    card = _project_card(db, settings, project, _counts(db))
    datasets = db.scalars(select(Dataset).where(Dataset.project_id == project.id).order_by(Dataset.name)).all()
    cards = [c for c in (dataset_card(db, settings, d) for d in datasets) if c is not None]
    tasks = card["tasks"] or ["detect"]
    groups = _model_groups(db, project, tasks)
    shown = 5 if len(groups) <= 1 else 3  # several tasks share the card: a few of each
    groups = [{**g, "models": g["models"][:shown]} for g in groups]
    metric = groups[0]["metric"] if groups else training.PRIMARY_METRIC.get(tasks[0], "map50_95")
    sweeps = db.scalar(select(func.count(Sweep.id)).where(Sweep.project_id == project.id)) or 0
    return {
        **card,
        "metric": metric,  # of the first group: what top_models is ranked by
        "dataset_cards": cards,
        "model_groups": groups,
        "top_models": groups[0]["models"] if groups else [],  # for older readers: the first task only
        "sweeps": sweeps,
    }


def home(db: Session, settings: HubSettings) -> dict[str, Any]:
    """The first screen: totals over live projects, the projects touched last, the latest runs."""
    cards = project_cards(db, settings)
    live = [c["id"] for c in cards]
    dataset_ids = db.scalars(
        select(func.distinct(Release.dataset_id))
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(Release.deleted_at.is_(None), Dataset.project_id.in_(live))
    ).all()
    newest = [r for r in (_latest(db, d) for d in dataset_ids) if r is not None]
    totals = {key: sum(c[key] for c in cards) for key in ("datasets", "releases", "items", "runs", "models")}
    totals["projects"] = len(cards)
    totals["folders"] = len({c["folder"] for c in cards if c["folder"]})
    totals["boxes"] = sum((r.stats or {}).get("boxes", 0) for r in newest)  # in the newest release of each dataset
    cards.sort(key=lambda c: c["updated_at"], reverse=True)

    rows = db.execute(
        select(Run, Job.status, Project)
        .join(Project, Project.id == Run.project_id)
        .outerjoin(Job, Job.id == Run.job_id)
        .where(Run.deleted_at.is_(None), Project.deleted_at.is_(None))
        .order_by(Run.created_at.desc())
        .limit(RECENT_RUNS)
    ).all()
    runs = []
    for run, status, project in rows:
        metric = training.PRIMARY_METRIC.get(catalog.release_task(db, run.release_id), "map50_95")
        runs.append(
            {
                "id": run.id,
                "name": run.name,
                "project": {"id": project.id, "name": project.name},
                "status": status or "failed",
                "epoch": run.epoch,
                "epochs": run.epochs,
                "metric": metric,
                "value": (run.summary or {}).get(metric),
                "created_at": iso(run.created_at),
            }
        )
    return {"totals": totals, "projects": cards[:RECENT_PROJECTS], "runs": runs}


def release_models(db: Session, settings: HubSettings, release_id: str) -> list[dict[str, Any]]:
    """The runs that trained on a release, newest first, with the model each one made."""
    release = catalog.get_release(db, release_id)
    metric = training.PRIMARY_METRIC.get(catalog.release_task(db, release), "map50_95")
    runs = db.scalars(
        select(Run).where(Run.release_id == release.id, Run.deleted_at.is_(None)).order_by(Run.created_at.desc()).limit(100)
    ).all()
    models = (
        {
            m.run_id: m
            for m in db.scalars(
                select(ModelVersion).where(ModelVersion.run_id.in_([r.id for r in runs]), ModelVersion.deleted_at.is_(None))
            )
        }
        if runs
        else {}
    )
    out = []
    for run in runs:
        model = models.get(run.id)
        out.append(
            {
                "run_id": run.id,
                "name": run.name,
                "template": (run.template or {}).get("variant") or run.trainer,
                "epochs": run.epochs,
                "created_at": iso(run.created_at),
                "metric": metric,
                "value": (run.summary or {}).get(metric),
                "model": {"id": model.id, "name": f"{model.name} v{model.version}", "aliases": model.aliases or []}
                if model
                else None,
            }
        )
    return out


def _histogram(values: list[int], bins: int = 12) -> dict[str, Any]:
    if not values:
        return {"edges": [], "counts": []}
    low, high = min(values), max(values)
    width = max(1, -(-(high - low + 1) // bins))
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, (v - low) // width)] += 1
    return {"edges": [low + i * width for i in range(bins + 1)], "counts": counts}


def release_charts(db: Session, release_id: str) -> dict[str, Any]:
    release = catalog.get_release(db, release_id)
    stats = release.stats or {}
    classes = _class_names(db, release)
    # image sizes: every item, grouped by size in SQL
    sizes = db.execute(
        select(Item.width, Item.height, func.count())
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release.id)
        .group_by(Item.width, Item.height)
    ).all()
    widths = [w for w, _, n in sizes for _ in range(min(n, 100000))]
    heights = [h for _, h, n in sizes for _ in range(min(n, 100000))]
    images = sum(n for _, _, n in sizes)
    top = max([w for w, _, _ in sizes] + [h for _, h, _ in sizes] + [1])
    grid = [[0] * GRID for _ in range(GRID)]
    for w, h, n in sizes:
        grid[min(GRID - 1, int(h / top * GRID))][min(GRID - 1, int(w / top * GRID))] += n
    # boxes: from a sample of the labelled items
    total = stats.get("items") or 0
    step = max(1, total // CHART_SAMPLE)
    revisions = db.execute(
        select(LabelRevision.boxes)
        .join(ReleaseItem, ReleaseItem.revision_id == LabelRevision.id)
        .where(ReleaseItem.release_id == release.id, ReleaseItem.box_count > 0)
        .order_by(ReleaseItem.item_id)
    )
    centres = [[0] * GRID for _ in range(GRID)]
    shapes = [[0] * GRID for _ in range(GRID)]
    sampled = boxes = 0
    for n, (rows,) in enumerate(revisions):
        if n % step:
            continue
        sampled += 1
        for b in rows or []:
            if len(b) < 5:
                continue  # a classification label has no box
            cx, cy, bw, bh = (float(v) for v in b[1:5])
            centres[min(GRID - 1, max(0, int(cy * GRID)))][min(GRID - 1, max(0, int(cx * GRID)))] += 1
            shapes[min(GRID - 1, max(0, int(bh * GRID)))][min(GRID - 1, max(0, int(bw * GRID)))] += 1
            boxes += 1
        if sampled >= CHART_SAMPLE:
            break
    return {
        "splits": {name: s.get("items", 0) for name, s in (stats.get("splits") or {}).items()},
        "states": stats.get("states", {}),
        "classes": [
            {
                "id": cid,
                "name": c.get("name"),
                "color": c.get("color"),
                "boxes": stats.get("classes", {}).get(str(cid), {}).get("boxes", 0),
                "images": stats.get("classes", {}).get(str(cid), {}).get("items", 0),
            }
            for cid, c in sorted(classes.items())
        ],
        "sizes": {
            "images": images,
            "mean_width": round(sum(w * n for w, _, n in sizes) / images, 1) if images else None,
            "mean_height": round(sum(h * n for _, h, n in sizes) / images, 1) if images else None,
            "width": _histogram(widths),
            "height": _histogram(heights),
            "grid": grid,
            "max": top,
        },
        "boxes": {"sampled_items": sampled, "boxes": boxes, "centres": centres, "shapes": shapes},
    }


def set_class_color(db: Session, release_id: str, class_id: int, color: str) -> dict[str, Any]:
    """A class's colour on screen (kept with the label space; labels and releases do not change)."""
    release = catalog.get_release(db, release_id)
    space = db.get(LabelSpace, release.label_space_id)
    color = color.strip()
    if color and not (len(color) == 7 and color.startswith("#") and all(ch in "0123456789abcdefABCDEF" for ch in color[1:])):
        raise ServiceError(400, "색은 #rrggbb 형식이어야 합니다.")
    classes = [dict(c) for c in space.classes]
    found = next((c for c in classes if int(c["id"]) == class_id), None)
    if found is None:
        raise ServiceError(404, "그 클래스가 이 버전의 라벨 공간에 없습니다.")
    if color:
        found["color"] = color.lower()
    else:
        found.pop("color", None)  # back to the automatic colour
    space.classes = classes
    return {"id": class_id, "color": found.get("color")}
