"""Dataset catalog logic: projects, in-place imports, releases, item queries, comparison."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from badamoa.core.protocol import ImportBegin, ImportChunk
from badamoa.core.storage import Storage

from . import services
from .catalog_models import (
    Batch,
    Dataset,
    Item,
    LabelRevision,
    LabelSpace,
    Project,
    Release,
    ReleaseItem,
    RevisionClass,
    StorageLocation,
)
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError

TASKS = ("detect", "segment", "classify")
SPLITS = ("train", "val", "test", "")
STATES = ("labelled", "background", "unlabelled")
MAX_PAGE = 200


# ---- projects ------------------------------------------------------------------


def create_project(
    db: Session, user: User, name: str, task: str = "detect", folder: str = "", description: str = ""
) -> Project:
    name = name.strip()
    if not 1 <= len(name) <= 128:
        raise ServiceError(400, "프로젝트 이름은 1~128자여야 합니다.")
    if task not in TASKS:
        raise ServiceError(400, f"task는 {', '.join(TASKS)} 중 하나여야 합니다.")
    if db.scalar(select(Project).where(Project.name == name)):
        raise ServiceError(409, "같은 이름의 프로젝트가 이미 있습니다.")
    project = Project(name=name, task=task, folder=folder.strip(), description=description, created_by=user.id)
    db.add(project)
    db.flush()
    services.audit(db, user.id, "api", "project.create", project.id, name=name)
    return project


def get_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None or project.deleted_at is not None:
        raise ServiceError(404, "프로젝트를 찾을 수 없습니다.")
    return project


def list_projects(db: Session) -> list[dict[str, Any]]:
    projects = db.scalars(select(Project).where(Project.deleted_at.is_(None)).order_by(Project.folder, Project.name)).all()
    counts = dict(
        db.execute(
            select(Dataset.project_id, func.count(Release.id))
            .join(Release, Release.dataset_id == Dataset.id)
            .where(Release.deleted_at.is_(None))
            .group_by(Dataset.project_id)
        ).all()
    )
    items = dict(db.execute(select(Item.project_id, func.count(Item.id)).group_by(Item.project_id)).all())
    return [
        {
            "id": p.id,
            "name": p.name,
            "task": p.task,
            "folder": p.folder,
            "description": p.description,
            "settings": p.settings or {},
            "created_at": iso(p.created_at),
            "releases": counts.get(p.id, 0),
            "items": items.get(p.id, 0),
        }
        for p in projects
    ]


def update_project_settings(db: Session, user: User, project_id: str, settings: dict[str, Any]) -> Project:
    """Per-project defaults an admin sets once so dialogs open already filled in."""
    services.require_admin(user)
    project = get_project(db, project_id)
    project.settings = {**(project.settings or {}), **settings}
    services.audit(db, user.id, "api", "project.settings", project.id, **settings)
    return project


def tree(db: Session, project_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
    """Dataset families of a project with their releases, newest release first."""
    get_project(db, project_id)
    spaces = {s.id: s.name for s in db.scalars(select(LabelSpace).where(LabelSpace.project_id == project_id))}
    result = []
    for dataset in db.scalars(select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.name)):
        query = select(Release).where(Release.dataset_id == dataset.id).order_by(Release.created_at.desc())
        releases = [r for r in db.scalars(query) if include_deleted or r.deleted_at is None]
        if not releases:
            continue
        result.append(
            {
                "id": dataset.id,
                "name": dataset.name,
                "releases": [
                    {
                        "id": r.id,
                        "version": r.version,
                        "status": r.status,
                        "items": (r.stats or {}).get("items", 0),
                        "label_space": spaces.get(r.label_space_id, ""),
                        "created_at": iso(r.created_at),
                        "deleted_at": iso(r.deleted_at),
                    }
                    for r in releases
                ],
            }
        )
    return result


# ---- import --------------------------------------------------------------------


def start_import(
    db: Session,
    user: User,
    *,
    project_id: str,
    node_id: str,
    path: str,
    dataset: str,
    version: str,
    fmt: str = "auto",
    label_space: str = "",
    site: str = "",
    camera: str = "",
    note: str = "",
    base_release_id: str = "",
    via: str = "api",
) -> Job:
    get_project(db, project_id)
    dataset, version, path = dataset.strip(), version.strip(), path.strip()
    if not dataset or not version or not path:
        raise ServiceError(400, "데이터셋 이름, 버전, 경로는 비워 둘 수 없습니다.")
    if fmt not in ("auto", "yolo", "coco", "imagefolder"):
        raise ServiceError(400, "형식은 auto, yolo, coco, imagefolder 중 하나여야 합니다.")
    existing = db.scalar(select(Dataset).where(Dataset.project_id == project_id, Dataset.name == dataset))
    if existing and db.scalar(select(Release).where(Release.dataset_id == existing.id, Release.version == version)):
        raise ServiceError(409, f"{dataset} {version} 릴리스가 이미 있습니다. 다른 버전 이름을 쓰세요.")
    params = {
        "project_id": project_id,
        "dataset": dataset,
        "version": version,
        "path": path,
        "format": fmt,
        "label_space": label_space.strip(),
        "site": site.strip(),
        "camera": camera.strip(),
        "note": note,
        "base_release_id": base_release_id,
    }
    return services.create_job(db, user, "data.import", params, node_id, via=via)


def _import_job(db: Session, node: Node, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.type != "data.import" or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 가져오기 작업이 아닙니다.")
    return job


def _label_space(db: Session, project_id: str, classes: list[str], wanted: str) -> LabelSpace:
    wanted_classes = [{"id": i, "name": name} for i, name in enumerate(classes)]
    spaces = db.scalars(select(LabelSpace).where(LabelSpace.project_id == project_id)).all()
    for space in spaces:
        if space.classes == wanted_classes and (not wanted or space.name == wanted):
            return space
    name = wanted or f"{len(classes)}cls"
    taken = {s.name for s in spaces}
    candidate, n = name, 2
    while candidate in taken:  # same name, different classes: keep both
        candidate, n = f"{name}-{n}", n + 1
    space = LabelSpace(project_id=project_id, name=candidate, classes=wanted_classes)
    db.add(space)
    db.flush()
    return space


def import_begin(db: Session, node: Node, job_id: str, body: ImportBegin) -> dict[str, Any]:
    job = _import_job(db, node, job_id)
    release = db.scalar(select(Release).where(Release.job_id == job.id))
    if release is not None:
        return {"release_id": release.id}
    p = job.params
    project = get_project(db, p["project_id"])
    found = "classify" if body.format == "imagefolder" else "detect"
    if project.task != found and not (project.task == "segment" and found == "detect"):
        kinds = {"detect": "객체 탐지", "classify": "분류", "segment": "분할"}
        mismatch = f"이 폴더는 {kinds[found]} 데이터인데 프로젝트는 {kinds[project.task]}용입니다."
        raise ServiceError(400, f"{mismatch} {kinds[found]} 프로젝트에 가져오세요.")
    location = db.scalar(
        select(StorageLocation).where(StorageLocation.node_id == node.id, StorageLocation.root == p["path"])
    )
    if location is None:
        location = StorageLocation(node_id=node.id, root=p["path"])
        db.add(location)
        db.flush()
    batch = Batch(
        project_id=project.id,
        name=f"{p['dataset']} {p['version']}",
        source=body.format,
        job_id=job.id,
        location_id=location.id,
        created_by=job.created_by,
    )
    db.add(batch)
    space = _label_space(db, project.id, body.classes, p.get("label_space", ""))
    dataset = db.scalar(select(Dataset).where(Dataset.project_id == project.id, Dataset.name == p["dataset"]))
    if dataset is None:
        dataset = Dataset(project_id=project.id, name=p["dataset"])
        db.add(dataset)
        db.flush()
    base_id = p.get("base_release_id") or db.scalar(
        select(Release.id)
        .where(Release.dataset_id == dataset.id, Release.status == "ready", Release.deleted_at.is_(None))
        .order_by(Release.created_at.desc())
        .limit(1)
    )
    db.flush()
    release = Release(
        dataset_id=dataset.id,
        version=p["version"],
        label_space_id=space.id,
        base_release_id=base_id,
        note=p.get("note", ""),
        batch_id=batch.id,
        job_id=job.id,
        created_by=job.created_by,
    )
    db.add(release)
    db.flush()
    return {"release_id": release.id}


def _digest(space_id: str, state: str, boxes: list[list[float]]) -> str:
    payload = json.dumps([space_id, state, sorted(boxes)], separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def import_items(db: Session, node: Node, job_id: str, chunk: ImportChunk, storage: Storage) -> dict[str, Any]:
    job = _import_job(db, node, job_id)
    release = db.scalar(select(Release).where(Release.job_id == job.id))
    if release is None:
        raise ServiceError(409, "가져오기를 먼저 시작해야 합니다 (begin).")
    batch = db.get(Batch, release.batch_id)
    space = db.get(LabelSpace, release.label_space_id)
    project_id = job.params["project_id"]
    site, camera = job.params.get("site", ""), job.params.get("camera", "")
    class_ids = [c["id"] for c in space.classes]
    incoming = {item.sha256: item for item in chunk.items}  # a file listed twice counts once

    items = {
        i.sha256: i for i in db.scalars(select(Item).where(Item.project_id == project_id, Item.sha256.in_(list(incoming))))
    }
    for sha, data in incoming.items():
        if sha in items:
            continue
        thumb_key = None
        if data.thumb_b64:
            try:
                thumb_key = storage.put_blob(base64.b64decode(data.thumb_b64))
            except (binascii.Error, ValueError):
                thumb_key = None
        items[sha] = Item(
            project_id=project_id,
            sha256=sha,
            file_name=data.file_name[:512],
            width=data.width,
            height=data.height,
            bytes=data.bytes,
            site=site,
            camera=camera,
            location_id=batch.location_id,
            rel_path=data.rel_path,
            thumb_key=thumb_key,
            batch_id=batch.id,
        )
        db.add(items[sha])
    db.flush()

    item_ids = [i.id for i in items.values()]
    revisions: dict[int, list[LabelRevision]] = {}
    for rev in db.scalars(select(LabelRevision).where(LabelRevision.item_id.in_(item_ids)).order_by(LabelRevision.rev)):
        revisions.setdefault(rev.item_id, []).append(rev)
    members = set(
        db.scalars(
            select(ReleaseItem.item_id).where(ReleaseItem.release_id == release.id, ReleaseItem.item_id.in_(item_ids))
        )
    )

    pinned: dict[str, LabelRevision] = {}
    created: list[LabelRevision] = []
    for sha, data in incoming.items():
        item = items[sha]
        # A box is [class, cx, cy, w, h]; a classification label is just [class].
        boxes = [[int(b[0]), *b[1:5]] for b in data.boxes if len(b) in (1, 5) and int(b[0]) in class_ids]
        state = data.state if boxes or data.state != "labelled" else "background"
        digest = _digest(space.id, state, boxes)
        history = revisions.get(item.id, [])
        same = next((r for r in history if r.digest == digest), None)
        if same is None:  # labels differ from every earlier revision: add one, never overwrite
            same = LabelRevision(
                item_id=item.id,
                rev=(history[-1].rev + 1) if history else 1,
                label_space_id=space.id,
                source=f"import:{batch.name}",
                state=state,
                box_count=len(boxes),
                boxes=boxes,
                labelled_class_ids=class_ids if state != "unlabelled" else [],
                digest=digest,
                batch_id=batch.id,
            )
            db.add(same)
            created.append(same)
            revisions.setdefault(item.id, []).append(same)
        pinned[sha] = same
    db.flush()

    for rev in created:
        counts: dict[int, int] = {}
        for box in rev.boxes:
            counts[int(box[0])] = counts.get(int(box[0]), 0) + 1
        db.add_all(RevisionClass(revision_id=rev.id, class_id=cid, count=n) for cid, n in counts.items())
    for sha, data in incoming.items():
        item, rev = items[sha], pinned[sha]
        if item.id in members:
            continue
        split = data.split if data.split in SPLITS else ""
        db.add(
            ReleaseItem(
                release_id=release.id,
                item_id=item.id,
                split=split,
                revision_id=rev.id,
                state=rev.state,
                box_count=rev.box_count,
            )
        )
    return {"accepted": len(incoming)}


def compute_stats(db: Session, release: Release) -> dict[str, Any]:
    rows = db.execute(
        select(ReleaseItem.split, ReleaseItem.state, func.count(), func.sum(Item.bytes), func.sum(ReleaseItem.box_count))
        .join(Item, Item.id == ReleaseItem.item_id)
        .where(ReleaseItem.release_id == release.id)
        .group_by(ReleaseItem.split, ReleaseItem.state)
    ).all()
    stats: dict[str, Any] = {"items": 0, "bytes": 0, "boxes": 0, "splits": {}, "states": {s: 0 for s in STATES}}
    for split, state, count, size, boxes in rows:
        bucket = stats["splits"].setdefault(split, {"items": 0, "bytes": 0})
        bucket["items"] += count
        bucket["bytes"] += size or 0
        stats["states"][state] = stats["states"].get(state, 0) + count
        stats["items"] += count
        stats["bytes"] += size or 0
        stats["boxes"] += boxes or 0
    class_rows = db.execute(
        select(RevisionClass.class_id, func.sum(RevisionClass.count), func.count())
        .join(ReleaseItem, ReleaseItem.revision_id == RevisionClass.revision_id)
        .where(ReleaseItem.release_id == release.id)
        .group_by(RevisionClass.class_id)
    ).all()
    stats["classes"] = {str(cid): {"boxes": boxes, "items": images} for cid, boxes, images in class_rows}
    for field in ("site", "camera"):
        column = getattr(Item, field)
        values = db.scalars(
            select(column)
            .join(ReleaseItem, ReleaseItem.item_id == Item.id)
            .where(ReleaseItem.release_id == release.id)
            .distinct()
        ).all()
        stats[f"{field}s"] = sorted(v for v in values if v)
    return stats


def import_finish(db: Session, node: Node, job_id: str) -> dict[str, Any]:
    job = _import_job(db, node, job_id)
    release = db.scalar(select(Release).where(Release.job_id == job.id))
    if release is None:
        raise ServiceError(409, "가져오기를 먼저 시작해야 합니다 (begin).")
    db.flush()
    release.stats = compute_stats(db, release)
    release.status = "ready"
    services.audit(
        db, job.created_by, "agent", "release.create", release.id, name=job.params["dataset"], version=release.version
    )
    return {"release_id": release.id, "version": release.version, "items": release.stats["items"]}


# ---- releases --------------------------------------------------------------------


def get_release(db: Session, release_id: str, allow_deleted: bool = False) -> Release:
    release = db.get(Release, release_id)
    if release is None or (release.deleted_at is not None and not allow_deleted):
        raise ServiceError(404, "릴리스를 찾을 수 없습니다.")
    return release


def release_view(db: Session, release_id: str) -> dict[str, Any]:
    release = get_release(db, release_id)
    dataset = db.get(Dataset, release.dataset_id)
    space = db.get(LabelSpace, release.label_space_id)
    base = db.get(Release, release.base_release_id) if release.base_release_id else None
    siblings = db.scalars(
        select(Release).where(Release.dataset_id == dataset.id, Release.deleted_at.is_(None), Release.status == "ready")
    ).all()
    return {
        "id": release.id,
        "version": release.version,
        "status": release.status,
        "note": release.note,
        "created_at": iso(release.created_at),
        "project_id": dataset.project_id,
        "dataset": {"id": dataset.id, "name": dataset.name},
        "label_space": {"id": space.id, "name": space.name, "classes": space.classes},
        "base": {"id": base.id, "version": base.version} if base and base.deleted_at is None else None,
        "stats": release.stats or {},
        "siblings": [{"id": r.id, "version": r.version} for r in siblings if r.id != release.id],
    }


def _item_filter(release_id: str, split, state, class_id, site, camera, q):
    conditions = [ReleaseItem.release_id == release_id]
    if split is not None:
        conditions.append(ReleaseItem.split == split)
    if state:
        conditions.append(ReleaseItem.state == state)
    if site:
        conditions.append(Item.site == site)
    if camera:
        conditions.append(Item.camera == camera)
    if q:
        conditions.append(Item.file_name.contains(q, autoescape=True))
    query = select(ReleaseItem, Item).join(Item, Item.id == ReleaseItem.item_id)
    if class_id is not None:
        query = query.join(
            RevisionClass, and_(RevisionClass.revision_id == ReleaseItem.revision_id, RevisionClass.class_id == class_id)
        )
    return query.where(*conditions)


def list_items(
    db: Session,
    settings: HubSettings,
    release_id: str,
    *,
    split: str | None = None,
    state: str = "",
    class_id: int | None = None,
    site: str = "",
    camera: str = "",
    q: str = "",
    offset: int = 0,
    limit: int = 60,
) -> dict[str, Any]:
    get_release(db, release_id)
    query = _item_filter(release_id, split, state, class_id, site, camera, q.strip())
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    limit = max(1, min(limit, MAX_PAGE))
    rows = db.execute(query.order_by(ReleaseItem.item_id).offset(max(0, offset)).limit(limit)).all()
    return {
        "total": total,
        "offset": max(0, offset),
        "limit": limit,
        "items": [
            {
                "id": item.id,
                "file_name": item.file_name,
                "width": item.width,
                "height": item.height,
                "split": member.split,
                "state": member.state,
                "box_count": member.box_count,
                "thumb_url": media_url(settings, "thumb", item.thumb_key) if item.thumb_key else None,
            }
            for member, item in rows
        ],
    }


def _revision_view(rev: LabelRevision, spaces: dict[str, LabelSpace], with_boxes: bool = False) -> dict[str, Any]:
    space = spaces.get(rev.label_space_id)
    names = {c["id"]: c["name"] for c in (space.classes if space else [])}
    data = {
        "id": rev.id,
        "rev": rev.rev,
        "state": rev.state,
        "box_count": rev.box_count,
        "source": rev.source,
        "created_at": iso(rev.created_at),
        "label_space": {"id": rev.label_space_id, "name": space.name if space else "", "classes": len(names)},
        "labelled_classes": [names.get(i, str(i)) for i in rev.labelled_class_ids or []],
    }
    if with_boxes:
        data["boxes"] = rev.boxes
        data["classes"] = space.classes if space else []
    return data


def item_detail(db: Session, settings: HubSettings, release_id: str, item_id: int) -> dict[str, Any]:
    release = get_release(db, release_id)
    member = db.get(ReleaseItem, (release_id, item_id))
    item = db.get(Item, item_id)
    if member is None or item is None:
        raise ServiceError(404, "이 릴리스에 없는 항목입니다.")
    spaces = {s.id: s for s in db.scalars(select(LabelSpace).where(LabelSpace.project_id == item.project_id))}
    revisions = db.scalars(select(LabelRevision).where(LabelRevision.item_id == item_id).order_by(LabelRevision.rev)).all()
    rev_numbers = {r.id: r.rev for r in revisions}
    memberships = db.execute(
        select(ReleaseItem, Release, Dataset)
        .join(Release, Release.id == ReleaseItem.release_id)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(ReleaseItem.item_id == item_id, Release.deleted_at.is_(None))
        .order_by(Release.created_at)
    ).all()
    return {
        "id": item.id,
        "file_name": item.file_name,
        "width": item.width,
        "height": item.height,
        "bytes": item.bytes,
        "sha256": item.sha256,
        "site": item.site,
        "camera": item.camera,
        "rel_path": item.rel_path,
        "split": member.split,
        "revision_id": member.revision_id,
        "release_label_space_id": release.label_space_id,
        "thumb_url": media_url(settings, "thumb", item.thumb_key) if item.thumb_key else None,
        "image_url": media_url(settings, "item", str(item.id)) if item_file(db, item) else None,
        "revisions": [_revision_view(r, spaces, with_boxes=True) for r in revisions],
        "releases": [
            {
                "id": rel.id,
                "dataset": ds.name,
                "version": rel.version,
                "split": m.split,
                "rev": rev_numbers.get(m.revision_id),
                "current": rel.id == release_id,
            }
            for m, rel, ds in memberships
        ],
    }


def compare(db: Session, release_id: str, other_id: str) -> dict[str, Any]:
    """What changed going from ``other`` to ``release``."""
    current, other = get_release(db, release_id), get_release(db, other_id)
    mine = select(ReleaseItem.item_id).where(ReleaseItem.release_id == release_id)
    theirs = select(ReleaseItem.item_id).where(ReleaseItem.release_id == other_id)

    def count(*conditions) -> int:
        return db.scalar(select(func.count()).select_from(ReleaseItem).where(*conditions)) or 0

    added = count(ReleaseItem.release_id == release_id, ReleaseItem.item_id.not_in(theirs))
    removed = count(ReleaseItem.release_id == other_id, ReleaseItem.item_id.not_in(mine))
    a = ReleaseItem.__table__.alias("a")
    b = ReleaseItem.__table__.alias("b")
    both = (a.c.item_id == b.c.item_id) & (a.c.release_id == release_id) & (b.c.release_id == other_id)
    relabelled = db.scalar(select(func.count()).select_from(a.join(b, both)).where(a.c.revision_id != b.c.revision_id)) or 0
    moved = db.scalar(select(func.count()).select_from(a.join(b, both)).where(a.c.split != b.c.split)) or 0

    space_a, space_b = db.get(LabelSpace, current.label_space_id), db.get(LabelSpace, other.label_space_id)
    names_a = {str(c["id"]): c["name"] for c in space_a.classes}
    names_b = {str(c["id"]): c["name"] for c in space_b.classes}
    boxes_a = {names_a.get(k, k): v["boxes"] for k, v in (current.stats or {}).get("classes", {}).items()}
    boxes_b = {names_b.get(k, k): v["boxes"] for k, v in (other.stats or {}).get("classes", {}).items()}
    classes = [
        {
            "name": name,
            "boxes": boxes_a.get(name, 0),
            "other": boxes_b.get(name, 0),
            "delta": boxes_a.get(name, 0) - boxes_b.get(name, 0),
        }
        for name in sorted(set(boxes_a) | set(boxes_b))
    ]
    stats_a, stats_b = current.stats or {}, other.stats or {}
    return {
        "release": {
            "id": current.id,
            "version": current.version,
            "items": stats_a.get("items", 0),
            "boxes": stats_a.get("boxes", 0),
        },
        "other": {
            "id": other.id,
            "version": other.version,
            "items": stats_b.get("items", 0),
            "boxes": stats_b.get("boxes", 0),
        },
        "added": added,
        "removed": removed,
        "relabelled": relabelled,
        "moved_split": moved,
        "label_space_changed": current.label_space_id != other.label_space_id,
        "classes": classes,
    }


def set_release_deleted(db: Session, user: User, release_id: str, deleted: bool) -> Release:
    services.require_admin(user)
    release = get_release(db, release_id, allow_deleted=True)
    release.deleted_at = utcnow() if deleted else None
    services.audit(
        db, user.id, "api", "release.delete" if deleted else "release.restore", release.id, version=release.version
    )
    return release


# ---- media -----------------------------------------------------------------------
# <img> tags cannot send an Authorization header, so image URLs are signed and
# expire. They are rounded to the hour so the browser can cache them.


@lru_cache(maxsize=4)
def storage_for(root: str) -> Storage:
    return Storage(Path(root))


@lru_cache(maxsize=4)
def _media_secret(home: str) -> bytes:
    path = Path(home) / "media.key"
    if not path.is_file():
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip().encode()


def _sign(settings: HubSettings, kind: str, key: str, exp: int) -> str:
    message = f"{kind}:{key}:{exp}".encode()
    return hmac.new(_media_secret(str(settings.home)), message, hashlib.sha256).hexdigest()[:32]


def media_url(settings: HubSettings, kind: str, key: str) -> str:
    exp = (int(time.time()) // 3600 + 2) * 3600
    return f"/media/{kind}/{key}?exp={exp}&sig={_sign(settings, kind, key, exp)}"


def verify_media(settings: HubSettings, kind: str, key: str, exp: int, sig: str) -> bool:
    return exp >= time.time() and hmac.compare_digest(_sign(settings, kind, key, exp), sig)


def item_file(db: Session, item: Item) -> Path | None:
    """The image file, if this machine can read it (Hub-local data or a shared mount)."""
    location = db.get(StorageLocation, item.location_id) if item.location_id else None
    if location is None:
        return None
    root = Path(location.root).expanduser()
    try:
        path = (root / item.rel_path).resolve()
        # The size check guards against a same-named path on this machine holding other data.
        if root.resolve() not in path.parents or not path.is_file() or path.stat().st_size != item.bytes:
            return None
    except OSError:
        return None
    return path
