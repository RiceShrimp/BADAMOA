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

import json
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from . import catalog, drafts, training
from .catalog_models import Dataset, Item, LabelRevision, LabelSpace, Project, Release, ReleaseItem
from .config import HubSettings
from .db import iso
from .models import Job
from .services import ServiceError
from .train_models import ModelVersion, Run, Sweep

COLLAGE = 4
COLLAGE_BOXES = 12
GRID = 20
RECENT_PROJECTS = 4
RECENT_RUNS = 6
RECENT_DATASETS = 3  # lines under a project's card: its datasets with the newest version first


# ---- pieces ------------------------------------------------------------------------


def _latest(db: Session, dataset_id: str) -> Release | None:
    return db.scalar(
        select(Release)
        .where(Release.dataset_id == dataset_id, Release.deleted_at.is_(None), Release.status == "ready")
        .order_by(Release.created_at.desc())
        .limit(1)
    )


def _collage(db: Session, settings: HubSettings, release: Release | None, count: int = COLLAGE) -> list[dict[str, Any]]:
    """A few labelled images of a release, spread over it, with their boxes (fractions of the image).

    Each is the first labelled image at or after an evenly spaced image number: a jump in the release's own
    index, so a version of a million images costs what one of ten does (reading the release up to the spot
    took half a minute for 684,049 images and held every screen that shows project cards)."""
    if release is None:
        return []
    lo, hi = db.execute(
        select(func.min(ReleaseItem.item_id), func.max(ReleaseItem.item_id)).where(ReleaseItem.release_id == release.id)
    ).one()
    if lo is None:
        return []
    spots: list[tuple[int, int]] = []
    for n in range(count * 2):  # twice as many tries as pictures: some spots land on images without a thumbnail
        start = lo + (hi - lo + 1) * n // (count * 2)
        found = db.execute(
            select(ReleaseItem.item_id, ReleaseItem.revision_id)
            .where(ReleaseItem.release_id == release.id, ReleaseItem.item_id >= start, ReleaseItem.box_count > 0)
            .order_by(ReleaseItem.item_id)
            .limit(1)
        ).first()
        if found is not None and (found[0], found[1]) not in spots:
            spots.append((found[0], found[1]))
    shown = select(Item).where(Item.id.in_([i for i, _ in spots]), Item.thumb_key.is_not(None))
    items = {i.id: i for i in db.scalars(shown)}
    revisions = {r.id: r for r in db.scalars(select(LabelRevision).where(LabelRevision.id.in_([r for _, r in spots])))}
    rows = [(items[i], revisions[r]) for i, r in spots if i in items and r in revisions]
    picked = rows[:: max(1, len(rows) // count)][:count]
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
        select(func.count(Release.id)).where(
            Release.dataset_id == dataset.id, Release.deleted_at.is_(None), Release.status != "draft"
        )
    )
    latest = _latest(db, dataset.id)
    draft = drafts.draft_of(db, dataset.id)
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
        # fixed labels not yet made a version
        "draft": {"id": draft.id, "edited": drafts.edited_count(db, draft)} if draft is not None else None,
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


def _recent_datasets(db: Session, project: Project) -> tuple[list[dict[str, Any]], int]:
    """The project's datasets by their newest version, the newest first: the first few as lines, and how many
    datasets have a version. Only the columns a line needs are read (a version's stats can be large)."""
    rows = db.execute(
        select(Release.id, Release.dataset_id, Release.version, Release.created_at, Dataset.name, Dataset.task)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(Dataset.project_id == project.id, Release.deleted_at.is_(None), Release.status == "ready")
        .order_by(Release.created_at.desc(), Release.id.desc())
    ).all()
    newest: dict[str, Any] = {}
    for row in rows:
        newest.setdefault(row.dataset_id, row)  # the first seen is the newest version
    shown = list(newest.values())[:RECENT_DATASETS]
    stats = dict(db.execute(select(Release.id, Release.stats).where(Release.id.in_([r.id for r in shown]))).all())
    return [
        {
            "id": row.dataset_id,
            "name": row.name,
            "task": row.task or "detect",
            "release": {"id": row.id, "version": row.version, "created_at": iso(row.created_at)},
            "items": (stats.get(row.id) or {}).get("items", 0),
        }
        for row in shown
    ], len(newest)


def _project_card(db: Session, settings: HubSettings, project: Project, counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    newest = db.scalar(
        select(Release)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(Dataset.project_id == project.id, Release.deleted_at.is_(None), Release.status == "ready")
        .order_by(Release.created_at.desc())
        .limit(1)
    )
    moments = [project.created_at, newest.created_at if newest else None]
    last_run = db.scalar(select(func.max(Run.created_at)).where(Run.project_id == project.id, Run.deleted_at.is_(None)))
    moments.append(last_run)
    recent, with_version = _recent_datasets(db, project)
    return {
        "id": project.id,
        "name": project.name,
        "tasks": (tasks := catalog.project_tasks(db).get(project.id, [])),
        "task": (tasks or [""])[0],  # the first dataset's task, for older readers
        "folder": project.folder,
        "description": project.description,
        "created_at": iso(project.created_at),
        "updated_at": iso(max(m for m in moments if m is not None)),
        "version_at": iso(newest.created_at) if newest else None,  # its newest version
        "run_at": iso(last_run),  # its newest training
        "datasets": counts["datasets"].get(project.id, 0),
        "releases": counts["releases"].get(project.id, 0),
        "items": counts["items"].get(project.id, 0),
        "runs": counts["runs"].get(project.id, 0),
        "models": counts["models"].get(project.id, 0),
        "best": _best_model(db, project, tasks or ["detect"]),  # the same order as the project page
        "collage": _collage(db, settings, newest),
        "classes": _labelled_classes(db, project),
        "recent": recent,  # its datasets, the newest version first: a line each under the card
        "recent_total": with_version,  # how many datasets have a version (the lines say "N개 더" past them)
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
                .where(alive, Release.status != "draft")  # fixes not made a version yet are not a version
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


def _charts_key(db: Session, release: Release) -> str:
    """What a version's kept statistics were computed for: a version never changes once made; a work in
    progress does as labels are fixed (each fix is a new label record, so the newest record tells)."""
    if release.status == "ready":
        return "ready"
    count, newest = db.execute(
        select(func.count(), func.max(ReleaseItem.revision_id)).where(ReleaseItem.release_id == release.id)
    ).one()
    return f"{release.status}:{count}:{newest}"


def release_charts(db: Session, release_id: str) -> dict[str, Any]:
    """The statistics tab. Image sizes and the spread of boxes read the whole version (684,049 images took
    7 s), so they are kept with it once computed (``Release.charts``); the counts come from its stats."""
    release = catalog.get_release(db, release_id)
    stats = release.stats or {}
    classes = _class_names(db, release)
    key = _charts_key(db, release)
    kept = release.charts or {}
    if kept.get("key") != key or "sizes" not in kept or "boxes" not in kept:
        kept = {"key": key, **_measure(db, release)}
        release.charts = kept
        db.commit()
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
        "sizes": kept["sizes"],
        "boxes": kept["boxes"],
    }


def _measure(db: Session, release: Release) -> dict[str, Any]:
    """Image sizes (every image, grouped in SQL) and where boxes sit and how big they are (a sample)."""
    # image sizes: every item, grouped by size in SQL
    sizes = db.execute(
        select(Item.width, Item.height, func.count())
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release.id)
        .group_by(Item.width, Item.height)
    ).all()
    widths = [w for w, _, n in sizes for _ in range(min(n, 100000))]
    heights = [h for _, h, n in sizes for _ in range(min(n, 100000))]
    counted = sum(n for _, _, n in sizes)
    top = max([w for w, _, _ in sizes] + [h for _, h, _ in sizes] + [1])
    grid = [[0] * GRID for _ in range(GRID)]
    for w, h, n in sizes:
        grid[min(GRID - 1, int(h / top * GRID))][min(GRID - 1, int(w / top * GRID))] += n
    # boxes: every label of every labelled image, where it sits and how big it is (5,079,439 boxes of 684,049
    # images in 4.7 s). The labels are read as text and parsed here: faster than SQLite's json_each, and than
    # SQLAlchemy turning each into a Python value. Kept once computed, so it is read once per version.
    centres = [[0] * GRID for _ in range(GRID)]
    shapes = [[0] * GRID for _ in range(GRID)]
    images = boxes = 0
    rows = db.execute(
        text(
            "SELECT lr.boxes FROM release_items ri JOIN label_revisions lr ON lr.id = ri.revision_id "
            "WHERE ri.release_id = :release AND ri.box_count > 0"
        ),
        {"release": release.id},
    )
    for (raw,) in rows:
        images += 1
        for b in (json.loads(raw) if isinstance(raw, str) else raw) or []:
            if len(b) < 5:
                continue  # a classification label has no box
            cx, cy, bw, bh = (float(v) for v in b[1:5])
            centres[min(GRID - 1, max(0, int(cy * GRID)))][min(GRID - 1, max(0, int(cx * GRID)))] += 1
            shapes[min(GRID - 1, max(0, int(bh * GRID)))][min(GRID - 1, max(0, int(bw * GRID)))] += 1
            boxes += 1
    return {
        "sizes": {
            "images": counted,
            "mean_width": round(sum(w * n for w, _, n in sizes) / counted, 1) if counted else None,
            "mean_height": round(sum(h * n for _, h, n in sizes) / counted, 1) if counted else None,
            "width": _histogram(widths),
            "height": _histogram(heights),
            "grid": grid,
            "max": top,
        },
        # "sampled_items": the images the boxes come from (every labelled one; the name stays for the screen)
        "boxes": {"sampled_items": images, "boxes": boxes, "centres": centres, "shapes": shapes},
    }


def fill_charts(db: Session, limit: int = 1) -> int:
    """Statistics of finished versions computed ahead (the Hub's background loop, one at a time), so the
    statistics tab of even a large one opens at once."""
    missing = db.scalars(
        select(Release)
        .where(Release.status == "ready", Release.deleted_at.is_(None))
        .where((Release.charts.is_(None)) | (func.json_extract(Release.charts, "$.key").is_(None)))
        .order_by(Release.created_at.desc())
        .limit(limit)
    ).all()
    for release in missing:
        release.charts = {"key": "ready", **_measure(db, release)}
        db.commit()
    return len(missing)


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
