"""Dataset catalog logic: projects, in-place imports, releases, item queries, comparison."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
import time
from collections import Counter
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import and_, func, insert, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
from badamoa.core.protocol import ImportBegin, ImportChunk, ImportProgress
from badamoa.core.storage import Storage

from . import (  # these import this module too: each uses the other only when called
    checks,
    drafts,
    hub_storage,
    labeling,
    qa,
    review,
    services,
)
from .catalog_models import (
    Batch,
    Dataset,
    Item,
    ItemPrediction,
    LabelRevision,
    LabelSetItem,
    LabelSpace,
    Project,
    QaResult,
    Release,
    ReleaseItem,
    RevisionClass,
    StorageLocation,
)
from .config import HubSettings
from .db import insert_new, iso, utcnow, write_lock
from .models import HubSetting, Job, Node, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run, Sweep

TASKS = task_names.BUILTIN
SPLITS = ("train", "val", "test", "")
# labelled; background: looked at, nothing to find in it; unlabelled: nobody has labelled it yet (0.38.0).
# Whether an image without labels is background or not labelled yet is decided when it is imported.
STATES = ("labelled", "background", "unlabelled")
UNLABELLED_RULES = ("auto", "background", "unlabelled")
# Data that may go into a dataset of another task: outlines kept as boxes, keypoints kept as boxes.
_FITS = {("segment", "detect"), ("detect", "segment"), ("detect", "pose")}
MAX_PAGE = 200
GRID_BOXES = 30  # boxes drawn over a grid thumbnail; the item view shows all
OUTLINE_POINTS = 48  # points of an outline drawn over a thumbnail


def light_shape(box: list[float]) -> list[float]:
    """A label for drawing over a thumbnail: the box, and its outline or keypoints kept small.

    Keypoints (triples) are kept whole; a long outline (pairs) is thinned to OUTLINE_POINTS points.
    Which of the two follows the box is for the screen to tell from the project's task.
    """
    rest = box[5:]
    if len(rest) <= OUTLINE_POINTS * 2 or len(rest) % 2:
        return list(box)
    pairs = len(rest) // 2
    step = pairs / OUTLINE_POINTS
    picked = [int(i * step) for i in range(OUTLINE_POINTS)]
    return [*box[:5], *(v for i in picked for v in rest[i * 2 : i * 2 + 2])]


# ---- projects ------------------------------------------------------------------


def create_project(db: Session, user: User, name: str, folder: str = "", description: str = "") -> Project:
    """A project groups datasets that share a class system; each dataset has its own task."""
    name = name.strip()
    if not 1 <= len(name) <= 128:
        raise ServiceError(400, "프로젝트 이름은 1~128자여야 합니다.")
    _name_free(db, name)
    project = Project(name=name, task="", folder=folder.strip()[:128], description=description, created_by=user.id)
    db.add(project)
    db.flush()
    _remember_folder(db, project.folder)
    services.audit(db, user.id, "api", "project.create", project.id, name=name)
    return project


def _name_free(db: Session, name: str, other_than: str = "") -> None:
    taken = db.scalar(select(Project).where(Project.name == name, Project.id != other_than))
    if taken is not None and taken.deleted_at is not None:
        raise ServiceError(409, "휴지통에 같은 이름의 프로젝트가 있습니다. 되살리거나 다른 이름을 쓰세요.")
    if taken is not None:
        raise ServiceError(409, "같은 이름의 프로젝트가 이미 있습니다.")


# ---- folders ------------------------------------------------------------------------
#
# A folder is the name in ``Project.folder``. The names are also kept on their own (a Hub
# setting), so a folder can be made before anything is in it and stays when it empties;
# it goes only when someone removes it.

FOLDERS_KEY = "catalog.folders"


def _kept_folders(db: Session) -> list[str]:
    row = db.get(HubSetting, FOLDERS_KEY)
    return list((row.value or {}).get("names", [])) if row else []


def _set_folders(db: Session, names: set[str]) -> None:
    value = {"names": sorted(n for n in names if n)}
    row = db.get(HubSetting, FOLDERS_KEY)
    if row is None:
        db.add(HubSetting(key=FOLDERS_KEY, value=value))
    else:
        row.value = value


def _remember_folder(db: Session, name: str) -> None:
    if name and name not in _kept_folders(db):
        _set_folders(db, {*_kept_folders(db), name})


def _folder_name(name: str) -> str:
    name = name.strip()
    if not 1 <= len(name) <= 128:
        raise ServiceError(400, "폴더 이름은 1~128자여야 합니다.")
    return name


def list_folders(db: Session) -> list[dict[str, Any]]:
    """Every folder with how many projects (not in the trash) are in it, empty ones included."""
    counts = dict(
        db.execute(
            select(Project.folder, func.count(Project.id))
            .where(Project.deleted_at.is_(None), Project.folder != "")
            .group_by(Project.folder)
        ).all()
    )
    names = set(_kept_folders(db)) | set(counts)
    return [{"name": n, "projects": counts.get(n, 0)} for n in sorted(names)]


def _projects_in(db: Session, name: str) -> list[Project]:
    return list(db.scalars(select(Project).where(Project.folder == name)))


def create_folder(db: Session, user: User, name: str, projects: list[str] | None = None) -> dict[str, Any]:
    """A new folder, empty or with the chosen projects moved into it."""
    name = _folder_name(name)
    if any(f["name"] == name for f in list_folders(db)):
        raise ServiceError(409, "같은 이름의 폴더가 이미 있습니다.")
    moving = [get_project(db, pid) for pid in dict.fromkeys(projects or [])]
    for project in moving:
        _may_edit(project, user)
    _remember_folder(db, name)
    for project in moving:
        project.folder = name
    services.audit(db, user.id, "ui", "folder.create", "", name=name, projects=len(moving))
    return {"name": name, "projects": len(moving)}


def rename_folder(db: Session, user: User, name: str, new_name: str) -> dict[str, Any]:
    """Rename a folder; the projects in it move along. Needs the right to change each of them."""
    if not any(f["name"] == name for f in list_folders(db)):
        raise ServiceError(404, "폴더를 찾을 수 없습니다.")
    new_name = _folder_name(new_name)
    if new_name != name and any(f["name"] == new_name for f in list_folders(db)):
        raise ServiceError(409, "같은 이름의 폴더가 이미 있습니다.")
    inside = _projects_in(db, name)
    for project in inside:
        _may_edit(project, user)
    for project in inside:
        project.folder = new_name
    _set_folders(db, (set(_kept_folders(db)) - {name}) | {new_name})
    services.audit(db, user.id, "ui", "folder.rename", "", name=new_name, old=name)
    return {"name": new_name, "projects": sum(1 for p in inside if p.deleted_at is None)}


def delete_folder(db: Session, user: User, name: str) -> dict[str, Any]:
    """Remove a folder. Its projects are not touched beyond leaving the folder ("no folder")."""
    if not any(f["name"] == name for f in list_folders(db)):
        raise ServiceError(404, "폴더를 찾을 수 없습니다.")
    inside = _projects_in(db, name)
    for project in inside:
        _may_edit(project, user)
    for project in inside:
        project.folder = ""
    _set_folders(db, set(_kept_folders(db)) - {name})
    services.audit(db, user.id, "ui", "folder.delete", "", name=name, projects=len(inside))
    return {"name": name, "projects": sum(1 for p in inside if p.deleted_at is None)}


def _may_edit(project: Project, user: User) -> None:
    if user.role != "admin" and project.created_by != user.id:
        raise ServiceError(403, "프로젝트를 만든 사람이나 관리자만 바꿀 수 있습니다.")


def get_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None or project.deleted_at is not None:
        raise ServiceError(404, "프로젝트를 찾을 수 없습니다.")
    return project


def dataset_task(dataset: Dataset | None) -> str:
    return (dataset.task if dataset is not None else "") or "detect"


def release_task(db: Session, release: Release | str | None) -> str:
    """The task of the dataset a release belongs to."""
    if isinstance(release, str):
        release = db.get(Release, release)
    return dataset_task(db.get(Dataset, release.dataset_id) if release is not None else None)


def project_tasks(db: Session) -> dict[str, list[str]]:
    """The tasks of each project's datasets that have a version out of the trash, oldest dataset first."""
    rows = db.execute(
        select(Dataset.project_id, Dataset.task)
        .where(Dataset.id.in_(select(Release.dataset_id).where(Release.deleted_at.is_(None))))
        .order_by(Dataset.created_at)
    ).all()
    found: dict[str, list[str]] = {}
    for project_id, task in rows:
        listed = found.setdefault(project_id, [])
        if (task or "detect") not in listed:
            listed.append(task or "detect")
    return found


def list_projects(db: Session) -> list[dict[str, Any]]:
    projects = db.scalars(select(Project).where(Project.deleted_at.is_(None)).order_by(Project.folder, Project.name)).all()
    tasks = project_tasks(db)
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
            "tasks": tasks.get(p.id, []),
            "task": (tasks.get(p.id) or [""])[0],  # the first dataset's task, for older readers
            "folder": p.folder,
            "description": p.description,
            "settings": p.settings or {},
            "created_at": iso(p.created_at),
            "releases": counts.get(p.id, 0),
            "items": items.get(p.id, 0),
        }
        for p in projects
    ]


def update_project(
    db: Session,
    user: User,
    project_id: str,
    *,
    name: str | None = None,
    folder: str | None = None,
    description: str | None = None,
) -> Project:
    """Rename a project, move it into a folder (a new folder is just a new name), or change its description."""
    project = get_project(db, project_id)
    _may_edit(project, user)
    if name is not None:
        name = name.strip()
        if not 1 <= len(name) <= 128:
            raise ServiceError(400, "프로젝트 이름은 1~128자여야 합니다.")
        _name_free(db, name, other_than=project.id)
        project.name = name
    if folder is not None:
        project.folder = folder.strip()[:128]
        _remember_folder(db, project.folder)
    if description is not None:
        project.description = description.strip()[:2000]
    services.audit(db, user.id, "api", "project.update", project.id, name=name, folder=folder)
    return project


def rename_dataset(db: Session, user: User, dataset_id: str, name: str) -> Dataset:
    """Give a dataset (every version of it) another name. Runs, models and records keep pointing at it; names
    already made from the old one (a run's, a model's) stay as they were. Importing under the old name later starts a
    new dataset."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise ServiceError(404, "데이터셋을 찾을 수 없습니다.")
    _may_edit(get_project(db, dataset.project_id), user)
    name = name.strip()
    if not 1 <= len(name) <= 128:
        raise ServiceError(400, "데이터셋 이름은 1~128자여야 합니다.")
    if name == dataset.name:
        return dataset
    taken = db.scalar(
        select(Dataset.id).where(Dataset.project_id == dataset.project_id, Dataset.name == name, Dataset.id != dataset.id)
    )
    if taken:
        raise ServiceError(409, f"이 프로젝트에 {name} 데이터셋이 이미 있습니다(휴지통에 있는 것 포함). 다른 이름을 쓰세요.")
    services.audit(db, user.id, "api", "dataset.rename", dataset.id, old=dataset.name, name=name)
    dataset.name = name
    return dataset


def rename_release(db: Session, user: User, release_id: str, version: str) -> Release:
    """Give a version another name within its dataset. A work in progress has none until it is made a version."""
    release = db.get(Release, release_id)
    if release is None:
        raise ServiceError(404, "버전을 찾을 수 없습니다.")
    dataset = db.get(Dataset, release.dataset_id)
    _may_edit(get_project(db, dataset.project_id), user)
    if release.status == "draft":
        raise ServiceError(400, "작업 중에는 이름이 없습니다. 버전으로 확정할 때 이름을 정합니다.")
    version = version.strip()
    if not 1 <= len(version) <= 64:
        raise ServiceError(400, "버전 이름은 1~64자여야 합니다.")
    if version == drafts.DRAFT:
        raise ServiceError(400, f"'{version}'은 버전 이름으로 쓸 수 없습니다.")
    if version == release.version:
        return release
    taken = db.scalar(
        select(Release.id).where(
            Release.dataset_id == release.dataset_id, Release.version == version, Release.id != release.id
        )
    )
    if taken:
        raise ServiceError(409, f"{dataset.name} {version} 버전이 이미 있습니다(휴지통에 있는 것 포함). 다른 이름을 쓰세요.")
    services.audit(
        db, user.id, "api", "release.rename", release.id, dataset=dataset.name, old=release.version, version=version
    )
    release.version = version
    return release


def update_project_settings(db: Session, user: User, project_id: str, settings: dict[str, Any]) -> Project:
    """Per-project defaults an admin sets once so dialogs open already filled in."""
    services.require_admin(user)
    project = get_project(db, project_id)
    project.settings = {**(project.settings or {}), **settings}
    services.audit(db, user.id, "api", "project.settings", project.id, **settings)
    return project


ACTIVE = ("queued", "running")


def _busy(db: Session, project_id: str) -> list[str]:
    """What is still under way in a project, e.g. ["학습 2개", "가져오기 1개"]."""
    runs = db.scalar(
        select(func.count(Run.id))
        .join(Job, Job.id == Run.job_id)
        .where(Run.project_id == project_id, Job.status.in_(ACTIVE))
    )
    sweeps = db.scalar(select(func.count(Sweep.id)).where(Sweep.project_id == project_id, Sweep.status == "running"))
    imports = sum(
        1
        for params in db.scalars(select(Job.params).where(Job.type == "data.import", Job.status.in_(ACTIVE)))
        if (params or {}).get("project_id") == project_id
    )
    found = [("학습", runs or 0), ("탐색", sweeps or 0), ("가져오기", imports)]
    return [f"{what} {n}개" for what, n in found if n]


def set_project_deleted(db: Session, user: User, project_id: str, deleted: bool) -> Project:
    """Move a project to the trash (with its datasets, runs and models) or bring it back. Nothing is erased."""
    project = db.get(Project, project_id)
    if project is None:
        raise ServiceError(404, "프로젝트를 찾을 수 없습니다.")
    _may_edit(project, user)
    if deleted and project.deleted_at is None:
        busy = _busy(db, project.id)
        if busy:
            raise ServiceError(409, f"이 프로젝트에서 {', '.join(busy)}가 진행 중입니다. 끝나거나 멈춘 뒤에 지우세요.")
        project.deleted_at = utcnow()
        hub_storage.mark_changed()
    elif not deleted:
        project.deleted_at = None
        hub_storage.mark_changed()
    services.audit(
        db, user.id, "api", "project.delete" if deleted else "project.restore", project.id, name=project.name
    )
    return project


def release_users(db: Session, release_id: str) -> list[str]:
    """What still points at a version, e.g. ["학습 2개", "모델 1개"]: such a version is not removed for good."""
    counts = [
        ("학습", select(func.count(Run.id)).where(Run.release_id == release_id)),
        ("모델", select(func.count(ModelVersion.id)).where(ModelVersion.release_id == release_id)),
        ("평가", select(func.count(ModelJob.id)).where(ModelJob.release_id == release_id)),
        ("탐색", select(func.count(Sweep.id)).where(Sweep.release_id == release_id)),
    ]
    found = [(what, db.scalar(query) or 0) for what, query in counts]
    return [f"{what} {n}개" for what, n in found if n]


def trash_count(db: Session) -> dict[str, int]:
    """How many projects and versions are in the trash (what ``trash`` lists), for the sidebar."""
    projects = db.scalar(select(func.count(Project.id)).where(Project.deleted_at.is_not(None))) or 0
    releases = (
        db.scalar(
            select(func.count(Release.id))
            .join(Dataset, Dataset.id == Release.dataset_id)
            .join(Project, Project.id == Dataset.project_id)
            .where(Release.deleted_at.is_not(None), Project.deleted_at.is_(None))
        )
        or 0
    )
    return {"projects": projects, "releases": releases, "total": projects + releases}


def trash(db: Session) -> dict[str, Any]:
    """Deleted projects, and deleted versions of projects that are still there."""
    projects = db.scalars(select(Project).where(Project.deleted_at.is_not(None)).order_by(Project.deleted_at.desc())).all()
    ids = [p.id for p in projects]
    releases_of = dict(
        db.execute(
            select(Dataset.project_id, func.count(Release.id))
            .join(Release, Release.dataset_id == Dataset.id)
            .where(Dataset.project_id.in_(ids), Release.deleted_at.is_(None))
            .group_by(Dataset.project_id)
        ).all()
    )
    items_of = dict(
        db.execute(
            select(Item.project_id, func.count(Item.id)).where(Item.project_id.in_(ids)).group_by(Item.project_id)
        ).all()
    )
    rows = db.execute(
        select(Release, Dataset.name, Project)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .join(Project, Project.id == Dataset.project_id)
        .where(Release.deleted_at.is_not(None), Project.deleted_at.is_(None))
        .order_by(Release.deleted_at.desc())
    ).all()
    return {
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "folder": p.folder,
                "created_by": p.created_by,
                "deleted_at": iso(p.deleted_at),
                "releases": releases_of.get(p.id, 0),
                "items": items_of.get(p.id, 0),
            }
            for p in projects
        ],
        "releases": [
            {
                "id": r.id,
                "dataset": name,
                "version": r.version,
                "project": {"id": p.id, "name": p.name},
                "deleted_at": iso(r.deleted_at),
                "items": (r.stats or {}).get("items", 0),
                "in_use": release_users(db, r.id),
            }
            for r, name, p in rows
        ],
    }


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
        releases.sort(key=lambda r: r.status != "draft")  # the work in progress first: newer than every version
        result.append(
            {
                "id": dataset.id,
                "name": dataset.name,
                "task": dataset_task(dataset),
                "releases": [
                    {
                        "id": r.id,
                        "version": r.version,
                        "status": r.status,
                        "items": (r.stats or {}).get("items", 0),
                        "label_space": spaces.get(r.label_space_id, ""),
                        "created_at": iso(r.created_at),
                        "deleted_at": iso(r.deleted_at),
                        **({"edited": drafts.edited_count(db, r)} if r.status == "draft" else {}),
                        # 검수: a record kept when it was confirmed, the flagged images left (once checked)
                        **({"review": review.badge(db, r)} if r.status == "ready" and not r.deleted_at else {}),
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
    note: str = "",
    base_release_id: str = "",
    task: str = "",
    unlabelled: str = "auto",
    classes: list[str] | None = None,
    via: str = "api",
) -> Job:
    """Import a node's folder as a dataset version. ``task``: Det/Seg/Kpt/Cls, a named task, or "" (from the data).

    ``unlabelled``: what an image without a label file is — "background", "unlabelled" (to be labelled), or
    "auto": not labelled yet when the folder has no label files at all, else background (the YOLO custom).
    ``classes``: class names for a folder that names none (images to label); otherwise the folder's are used.
    """
    get_project(db, project_id)
    dataset, version, path = dataset.strip(), version.strip(), path.strip()
    if not dataset or not version or not path:
        raise ServiceError(400, "데이터셋 이름, 버전, 경로는 비워 둘 수 없습니다.")
    try:
        task = task_names.normalize(task)
    except ValueError as e:
        raise ServiceError(400, str(e)) from e
    if fmt not in ("auto", "yolo", "coco", "imagefolder"):
        raise ServiceError(400, "형식은 auto, yolo, coco, imagefolder 중 하나여야 합니다.")
    if unlabelled not in UNLABELLED_RULES:
        raise ServiceError(400, "unlabelled는 auto, background, unlabelled 중 하나여야 합니다.")
    if version == drafts.DRAFT:
        raise ServiceError(400, f"'{version}'은 버전 이름으로 쓸 수 없습니다.")
    existing = db.scalar(select(Dataset).where(Dataset.project_id == project_id, Dataset.name == dataset))
    if existing and db.scalar(select(Release).where(Release.dataset_id == existing.id, Release.version == version)):
        raise ServiceError(409, f"{dataset} {version} 버전이 이미 있습니다. 다른 이름을 쓰세요.")
    if existing and task and not task_names.same(task, dataset_task(existing)):
        raise ServiceError(
            400,
            f"데이터셋 {dataset}의 태스크는 {task_names.short(dataset_task(existing))}입니다. "
            "다른 태스크로 가져오려면 새 데이터셋 이름을 쓰세요.",
        )
    params = {
        # a new version keeps its dataset's task; the node reads a named task's labels as they are
        "task": task or (existing.task if existing else ""),
        "project_id": project_id,
        "dataset": dataset,
        "version": version,
        "path": path,
        "format": fmt,
        "label_space": label_space.strip(),
        "note": note,
        "base_release_id": base_release_id,
        "unlabelled": unlabelled,
        "classes": list(dict.fromkeys(c.strip() for c in classes or [] if c.strip())),
    }
    return services.create_job(db, user, "data.import", params, node_id, via=via)


def _import_job(db: Session, node: Node, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.type != "data.import" or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 가져오기 작업이 아닙니다.")
    return job


def _label_space(
    db: Session, project_id: str, classes: list[str], wanted: str, keypoints: dict[str, Any] | None = None
) -> LabelSpace:
    """Keypoint datasets give every class the points' names and the limbs that join them."""
    extra: dict[str, Any] = {}
    if keypoints:
        extra = {"keypoints": list(keypoints.get("names") or []), "skeleton": keypoints.get("skeleton") or []}
    return _space_for(db, project_id, [{"id": i, "name": name, **extra} for i, name in enumerate(classes)], wanted)


def _space_for(db: Session, project_id: str, classes: list[dict[str, Any]], wanted: str = "") -> LabelSpace:
    """The project's label space with these classes (ids and names; colours do not count), or a new one."""
    def plain(entries: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        return [(int(c["id"]), c["name"], tuple(c.get("keypoints") or ())) for c in entries]

    wanted_plain = plain(classes)
    spaces = db.scalars(select(LabelSpace).where(LabelSpace.project_id == project_id)).all()
    for space in spaces:
        if plain(space.classes) == wanted_plain and (not wanted or space.name == wanted):
            return space
    name = wanted or f"{len(classes)}cls"
    taken = {s.name for s in spaces}
    candidate, n = name, 2
    while candidate in taken:  # same name, different classes: keep both
        candidate, n = f"{name}-{n}", n + 1
    space = LabelSpace(project_id=project_id, name=candidate, classes=classes)
    db.add(space)
    db.flush()
    return space


IMPORT_STEPS = ("find", "labels", "read")  # finding images, reading their labels, reading and sending the images


def import_progress(db: Session, node: Node, job_id: str, body: ImportProgress) -> None:
    """How far finding images and reading labels got, kept on the job until the version exists (``begin``)."""
    job = _import_job(db, node, job_id)
    before = (job.result or {}).get("progress") or {}
    since = before.get("since") if before.get("step") == body.step else iso(utcnow())
    job.result = {
        "progress": {"step": body.step, "done": max(0, body.done), "total": max(0, body.total), "since": since}
    }


def import_begin(
    db: Session, node: Node, job_id: str, body: ImportBegin, settings: HubSettings | None = None
) -> dict[str, Any]:
    """The version an import fills, once the node has found the images. ``settings``: the Hub checks first that
    their thumbnails and records fit on its disk (refused with how much it needs and has, before a single read)."""
    job = _import_job(db, node, job_id)
    release = db.scalar(select(Release).where(Release.job_id == job.id))
    if release is not None:
        return {"release_id": release.id}
    room = hub_storage.check_room(db, settings, int(body.total or 0)) if settings is not None else {}
    # Imports into one project begin one at a time: two at once would both find no dataset or label space
    # with these classes and both make one.
    write_lock(db)
    p = job.params
    project = get_project(db, p["project_id"])
    dataset = db.scalar(select(Dataset).where(Dataset.project_id == project.id, Dataset.name == p["dataset"]))
    found = "classify" if body.format == "imagefolder" else "pose" if body.keypoints else "detect"
    found = "segment" if found == "detect" and body.outlines else found
    # The dataset's task when it has versions already, else the one chosen, else what the folder holds.
    task = (dataset.task if dataset is not None and dataset.task else "") or p.get("task") or found
    # A named task takes the labels as they are; a built-in one only data it can use.
    if task in TASKS and task != found and (task, found) not in _FITS:
        short = task_names.short
        raise ServiceError(
            400,
            f"태스크가 맞지 않습니다: 폴더는 {short(found)} 데이터, 가져오려는 태스크는 {short(task)}. "
            f"태스크를 자동이나 {short(found)}에 두고 가져오세요.",
        )
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
    keypoints = body.keypoints if task == "pose" else None
    classes = list(body.classes)
    if not classes and p.get("classes"):  # a folder of images to label: the classes given when importing
        classes = list(p["classes"])
    if not classes and dataset is not None:  # or those of the dataset's newest version
        newest = db.scalar(
            select(Release)
            .where(Release.dataset_id == dataset.id, Release.status == "ready", Release.deleted_at.is_(None))
            .order_by(Release.created_at.desc())
            .limit(1)
        )
        if newest is not None:
            space = db.get(LabelSpace, newest.label_space_id)
            classes = [c["name"] for c in space.classes] if space else []
    space = _label_space(db, project.id, classes, p.get("label_space", ""), keypoints)
    if dataset is None:
        dataset = Dataset(project_id=project.id, name=p["dataset"], task=task)
        db.add(dataset)
        db.flush()
    elif not dataset.task:
        dataset.task = task
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
        # How far the import got, for the progress shown on every screen; replaced by the stats at the end.
        stats={"import": {"total": int(body.total or 0), "done": 0, "skipped": 0}},
    )
    db.add(release)
    db.flush()
    return {"release_id": release.id, **room}


def _digest(space_id: str, state: str, boxes: list[list[float]]) -> str:
    payload = json.dumps([space_id, state, sorted(boxes)], separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def import_items(
    db: Session, node: Node, job_id: str, chunk: ImportChunk, storage: Storage, settings: HubSettings | None = None
) -> dict[str, Any]:
    """``settings``: an import stops when the Hub's disk gets close to full."""
    job = _import_job(db, node, job_id)
    if settings is not None:
        hub_storage.check_running(settings)
    release = db.scalar(select(Release).where(Release.job_id == job.id))
    if release is None:
        raise ServiceError(409, "가져오기를 먼저 시작해야 합니다 (begin).")
    batch = db.get(Batch, release.batch_id)
    space = db.get(LabelSpace, release.label_space_id)
    project_id = job.params["project_id"]
    # Imports before 0.50.1 could set site and camera (fixed, or read from file names by a rule).
    site, camera = job.params.get("site", ""), job.params.get("camera", "")
    class_ids = [c["id"] for c in space.classes]
    # Outlines (Seg), keypoints (Kpt) or whatever a named task's labels hold follow the box; Det keeps the box.
    whole = release_task(db, release) not in ("detect", "classify")
    incoming = {item.sha256: item for item in chunk.items}  # a file listed twice counts once

    def in_project():
        return select(Item).where(Item.project_id == project_id, Item.sha256.in_(list(incoming)))

    # Thumbnails of the images new to the project are stored before taking the database's write lock (the disk
    # can be slow); images already there keep theirs.
    known = set(db.scalars(in_project().with_only_columns(Item.sha256)))
    rows = []
    for sha, data in incoming.items():
        if sha in known:
            continue
        thumb_key, thumb_bytes, thumb_v = None, 0, 0
        if data.thumb_b64:
            try:
                thumb = base64.b64decode(data.thumb_b64)
                thumb_key, thumb_bytes = storage.put_blob(thumb), len(thumb)
                thumb_v = 1 if hub_storage.thumb_current(thumb) else 0  # a node before 0.61: shrunk here later
            except (binascii.Error, ValueError):
                thumb_key = None
        rows.append(
            {
                "project_id": project_id,
                "sha256": sha,
                "file_name": data.file_name[:512],
                "width": data.width,
                "height": data.height,
                "bytes": data.bytes,
                "site": site,
                "camera": camera,
                "location_id": batch.location_id,
                "rel_path": data.rel_path,
                "thumb_key": thumb_key,
                "thumb_bytes": thumb_bytes,
                "thumb_v": thumb_v,
                "batch_id": batch.id,
            }
        )
    # Imports into one project at once (a folder and a copy of it, Det and Seg of the same images) share their
    # images: from here one chunk at a time, so an image another import registered meanwhile is used, not
    # inserted again, and its label revisions are numbered after the ones that import added.
    write_lock(db)
    insert_new(db, Item, rows, ["project_id", "sha256"])
    items = {i.sha256: i for i in db.scalars(in_project())}

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
        # A box is [class, cx, cy, w, h]; a classification label is just [class]; a segmentation
        # label is a box followed by its outline (x1, y1, x2, y2, …), kept unless the dataset is Det.
        boxes = [
            [int(b[0]), *(b[1:] if whole else b[1:5])]
            for b in data.boxes
            if (len(b) == 1 or len(b) >= 5) and int(b[0]) in class_ids
        ]
        # No labels: background (nothing to find in it), or not labelled yet when the node says so.
        state = "labelled" if boxes else ("unlabelled" if data.state == "unlabelled" else "background")
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
    stats = dict(release.stats or {})
    progress = dict(stats.get("import") or {})
    total = int(progress.get("total") or 0)
    done = int(progress.get("done") or 0) + len(chunk.items)
    progress["done"] = min(done, total) if total else done
    progress["skipped"] = int(progress.get("skipped") or 0) + chunk.skipped
    progress["bytes"] = int(progress.get("bytes") or 0) + sum(i.bytes for i in chunk.items)
    release.stats = {**stats, "import": progress}
    return {"accepted": len(incoming)}


# ---- imports under way (every screen shows them) -----------------------------------------

RECENT_IMPORTS_S = 30 * 60  # finished ones stay listed this long


def imports(db: Session) -> list[dict[str, Any]]:
    """Dataset imports that are queued or running, and those that ended in the last half hour."""
    since = utcnow() - timedelta(seconds=RECENT_IMPORTS_S)
    jobs = db.scalars(
        select(Job)
        .where(Job.type == "data.import")
        .where((Job.status.in_(ACTIVE)) | (Job.finished_at >= since))
        .order_by(Job.created_at.desc())
        .limit(20)
    ).all()
    # a waiting import has no server yet: the one it was sent to
    targets = [j.node_id or j.target_node_id for j in jobs if j.node_id or j.target_node_id]
    nodes = {n.id: n.name for n in db.scalars(select(Node).where(Node.id.in_(targets)))}
    out = []
    for job in jobs:
        params = job.params or {}
        project = db.get(Project, params.get("project_id") or "")
        if project is None or project.deleted_at is not None:
            continue
        release = db.scalar(select(Release).where(Release.job_id == job.id))
        stats = (release.stats if release else None) or {}
        progress = stats.get("import") or {}
        status = {"succeeded": "done"}.get(job.status, job.status)  # queued | running | done | failed | canceled
        total = int(progress.get("total") or stats.get("items") or 0)
        done = total if status == "done" else int(progress.get("done") or 0)
        # The step it is on: the first two (finding images, reading labels) come before the version exists.
        step, since = "", None
        if status == "running" and release is not None:
            step, since = "read", iso(release.created_at)
        elif status == "running":
            early = (job.result or {}).get("progress") or {}
            step = early.get("step") if early.get("step") in IMPORT_STEPS else "find"
            since = early.get("since") or iso(job.started_at)
            done, total = int(early.get("done") or 0), int(early.get("total") or 0)
        out.append(
            {
                "job_id": job.id,
                "status": status,
                "project": {"id": project.id, "name": project.name},
                "dataset": params.get("dataset", ""),
                "version": params.get("version", ""),
                "release_id": release.id if release else None,
                "node": nodes.get(job.node_id or job.target_node_id or "", ""),
                "done": done,
                "total": total,
                "skipped": int(progress.get("skipped") or (job.result or {}).get("skipped") or 0),
                "step": step,  # "" (waiting or ended) | find | labels | read
                "step_started_at": since,
                "bytes": int(progress.get("bytes") or 0),  # of the images read and sent so far
                "created_at": iso(job.created_at),
                "started_at": iso(job.started_at),
                "finished_at": iso(job.finished_at),
                "error": job.error or "",
            }
        )
    return out


def drop_unfinished_imports(db: Session, settings: HubSettings) -> int:
    """Versions an import began but never finished (stopped, failed, or its server went away): removed with the
    image records only they held and those thumbnails, so the name can be used again and the dataset deleted."""
    from . import purge

    stale = db.scalars(
        select(Release)
        .join(Job, Job.id == Release.job_id)
        .where(Release.status == "building", Job.status.in_(("failed", "canceled")))
    ).all()
    thumbs: set[str] = set()
    for release in stale:
        thumbs |= purge._drop_release(db, release)
    if stale:
        db.flush()
        purge._remove_unused(settings, db, thumbs)
    return len(stale)


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
        state = state if state in STATES else "background"
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
    hub_storage.mark_changed()  # the storage screen counts the new images
    job = _import_job(db, node, job_id)
    release = db.scalar(select(Release).where(Release.job_id == job.id))
    if release is None:
        raise ServiceError(409, "가져오기를 먼저 시작해야 합니다 (begin).")
    db.flush()
    release.stats = compute_stats(db, release)
    release.status = "ready"
    stamp_digest(db, release)
    services.audit(
        db, job.created_by, "agent", "release.create", release.id, name=job.params["dataset"], version=release.version
    )
    return {"release_id": release.id, "version": release.version, "items": release.stats["items"]}


# ---- releases --------------------------------------------------------------------


def get_release(db: Session, release_id: str, allow_deleted: bool = False) -> Release:
    release = db.get(Release, release_id)
    if release is None or (release.deleted_at is not None and not allow_deleted):
        raise ServiceError(404, "버전을 찾을 수 없습니다.")
    return release


def require_ready(release: Release, then: str = "") -> None:
    """Training, export, evaluation and new versions take only finished versions."""
    if release.status == "draft":
        raise ServiceError(409, "작업 중은 아직 버전이 아닙니다. '버전으로 확정'한 뒤에 쓰세요.")
    if release.status != "ready":
        raise ServiceError(409, "아직 등록 중인 버전입니다." + (f" {then}" if then else ""))


# ---- a version's fingerprint, like a commit hash --------------------------------------------

SHORT_DIGEST = 12  # how much of it screens show
_DIGEST_REF = re.compile(r"^[0-9a-f]{7,64}$")


def release_digest(db: Session, release: Release) -> str:
    """SHA-256 of what a version is: its task and classes, then every image with its split and pinned labels.

    Images go in the order of their own hash, one line each (image hash, split, label state, the label
    revision's digest), so the same content gives the same fingerprint whatever its name, note or
    version name, and in whatever order it was put together.
    """
    dataset = db.get(Dataset, release.dataset_id)
    space = db.get(LabelSpace, release.label_space_id)
    h = hashlib.sha256()
    head = {"task": dataset_task(dataset), "classes": space.classes if space else []}
    h.update(json.dumps(head, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
    rows = db.execute(
        select(Item.sha256, ReleaseItem.split, ReleaseItem.state, LabelRevision.digest)
        .join(Item, Item.id == ReleaseItem.item_id)
        .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id, isouter=True)
        .where(ReleaseItem.release_id == release.id)
        .order_by(Item.sha256)
        .execution_options(yield_per=5000)
    )
    for sha, split, state, labels in rows:
        h.update(f"\n{sha} {split or '-'} {state} {labels or '-'}".encode())
    return h.hexdigest()


def stamp_digest(db: Session, release: Release) -> str:
    """Give a finished version its fingerprint (once: a finished version does not change)."""
    if release.status == "ready" and not release.digest:
        db.flush()
        release.digest = release_digest(db, release)
    return release.digest


def fill_digests(db: Session, limit: int = 20) -> int:
    """Fingerprints for versions made before there were any (the Hub's background loop, a few at a time)."""
    missing = db.scalars(select(Release).where(Release.status == "ready", Release.digest == "").limit(limit)).all()
    for release in missing:
        stamp_digest(db, release)
    return len(missing)


def find_release(db: Session, ref: str) -> Release:
    """A version by id, by the start of its fingerprint (7 characters or more), or as ``dataset@version``."""
    ref = (ref or "").strip()
    release = db.get(Release, ref)
    if release is not None and release.deleted_at is None:
        return release
    found: list[Release] = []
    if "@" in ref:
        name, _, version = ref.rpartition("@")
        found = list(
            db.scalars(
                select(Release)
                .join(Dataset, Dataset.id == Release.dataset_id)
                .where(Dataset.name == name.strip(), Release.version == version.strip(), Release.deleted_at.is_(None))
            )
        )
        what = f"{name.strip()} {version.strip()}"
    elif _DIGEST_REF.match(ref.lower()):
        found = list(
            db.scalars(select(Release).where(Release.digest.startswith(ref.lower()), Release.deleted_at.is_(None)))
        )
        what = f"지문 {ref}"
    else:
        raise ServiceError(404, "버전을 찾을 수 없습니다. id, 지문(7자 이상) 또는 '데이터셋@버전'으로 가리키세요.")
    if not found:
        raise ServiceError(404, f"{what}에 맞는 버전이 없습니다.")
    if len(found) > 1:
        projects = {db.get(Dataset, r.dataset_id).project_id for r in found}
        why = "여러 프로젝트에 같은 이름이 있습니다" if len(projects) > 1 else "지문이 같은 버전이 여럿입니다"
        raise ServiceError(409, f"{what}에 맞는 버전이 {len(found)}개입니다({why}). id나 더 긴 지문으로 가리키세요.")
    return found[0]


def release_view(db: Session, release_id: str) -> dict[str, Any]:
    release = get_release(db, release_id)
    drafts.fresh_stats(db, release)
    dataset = db.get(Dataset, release.dataset_id)
    space = db.get(LabelSpace, release.label_space_id)
    base = db.get(Release, release.base_release_id) if release.base_release_id else None
    siblings = db.scalars(
        select(Release).where(Release.dataset_id == dataset.id, Release.deleted_at.is_(None), Release.status == "ready")
    ).all()
    # Other versions of the project with exactly this content (the same fingerprint)
    twins = (
        db.execute(
            select(Release.id, Release.version, Dataset.name)
            .join(Dataset, Dataset.id == Release.dataset_id)
            .where(
                Release.digest == release.digest,
                Release.id != release.id,
                Release.deleted_at.is_(None),
                Dataset.project_id == dataset.project_id,
            )
        ).all()
        if release.digest
        else []
    )
    return {
        "id": release.id,
        "version": release.version,
        "status": release.status,
        "note": release.note,
        "digest": release.digest,
        "same_as": [{"id": i, "version": v, "dataset": name} for i, v, name in twins],
        "created_at": iso(release.created_at),
        "project_id": dataset.project_id,
        "dataset": {"id": dataset.id, "name": dataset.name},
        "task": dataset_task(dataset),
        "label_space": {"id": space.id, "name": space.name, "classes": space.classes},
        "base": {"id": base.id, "version": base.version} if base and base.deleted_at is None else None,
        "stats": release.stats or {},
        "siblings": [{"id": r.id, "version": r.version} for r in siblings if r.id != release.id],
        # the dataset's work in progress (fixed labels not yet a version), whichever version is open
        "draft": drafts.draft_info(db, dataset.id),
        # labels are fixed (and labeling sets made) on the newest version or the work in progress
        "fixable": not drafts.edit_place(db, release)[1],
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
    edited: bool = False,
    suspect: str = "",
    label_set: str = "",
    part: int | None = None,
    check: str = "",
    mistake: str = "",
    changed: str = "",
    let_be: bool = False,
    offset: int = 0,
    limit: int = 60,
    pred: str = "",
    classes: list[int] | None = None,
) -> dict[str, Any]:
    """A page of a release's items. ``pred``: a model run's id (``checks.prediction_runs``): each image also carries
    what that run predicted for it, to lay over the thumbnail.
    ``edited``: only the images the dataset's work in progress changed.
    ``suspect``: an evaluation's id: only the images it measured, those with the most mistakes first.
    ``label_set`` (and ``part``): only a labeling set's images, in the set's order.
    ``check``: a label check's id (``qa.CHECKS``): only the images it flagged. ``mistake`` with ``suspect``:
    ``extra`` (objects the model found that no label has) or ``missed`` (labels it did not find).
    ``changed``: with ``edited``, only the images the work in progress changed that way (``drafts.KINDS``).
    ``classes`` (검수): with ``check``, only what it flagged on a label of these classes (an image of other classes
    flagged is not one, though it holds one of them); with ``suspect`` and ``mistake``, only the images where the
    model's extra or missed objects are of these classes; otherwise the images holding any of them.
    With ``check`` (or ``suspect`` and ``mistake``), what someone judged right ("문제 없음") is left out;
    ``let_be`` lists only that instead."""
    release = get_release(db, release_id)
    draft = drafts.fixing(db, release)
    if edited and draft is None:
        return {"total": 0, "offset": 0, "limit": max(1, min(limit, MAX_PAGE)), "items": []}
    query = _item_filter(release_id, split, state, class_id, site, camera, q.strip())
    if edited:
        query = query.where(ReleaseItem.item_id.in_(drafts.edited_items(draft)))
        if changed:
            if changed not in drafts.KINDS:
                raise ServiceError(400, "changed는 added, removed, moved, reclassed 중 하나여야 합니다.")
            query = query.where(ReleaseItem.item_id.in_(drafts.items_changed(db, draft, changed)))
    limit = max(1, min(limit, MAX_PAGE))
    order = [ReleaseItem.item_id]
    if suspect:
        op = checks.check_of(db, release, suspect)
        query = query.join(
            ItemPrediction, and_(ItemPrediction.op_id == op.id, ItemPrediction.item_id == ReleaseItem.item_id)
        )
        order = [ItemPrediction.wrong.desc(), ItemPrediction.missed.desc(), ReleaseItem.item_id]
        if mistake in ("extra", "missed"):
            column = getattr(ItemPrediction, mistake)
            query = query.where(column > 0)
            order = [column.desc(), ReleaseItem.item_id]
    picked = {int(c) for c in classes or []}
    of_class: dict[int, list[int]] = {}
    if check:
        if check not in qa.CHECKS:
            raise ServiceError(400, "그런 검사가 없습니다.")
        qa.run(db, release)
        joined = and_(QaResult.release_id == release.id, QaResult.check == check, QaResult.item_id == ReleaseItem.item_id)
        query = query.join(QaResult, joined)
        if picked:
            of_class = qa.of_classes(db, release.id, check, picked)
            query = query.where(ReleaseItem.item_id.in_(list(of_class)))
    elif picked and suspect and mistake in checks.MISTAKES:
        query = query.where(ReleaseItem.item_id.in_(list(checks.mistakes_of_classes(db, release, suspect, mistake, picked))))
    elif picked:
        holding = select(RevisionClass.revision_id).where(RevisionClass.class_id.in_(picked))
        query = query.where(ReleaseItem.revision_id.in_(holding))
    judged = check or (mistake if suspect and mistake in checks.MISTAKES else "")  # whose "문제 없음" counts
    if judged:
        kept = checks.let_be(judged, ReleaseItem.item_id, ReleaseItem.revision_id)
        query = query.where(kept if let_be else ~kept)
    if label_set:
        labeling.get_set(db, label_set)
        joined = and_(LabelSetItem.set_id == label_set, LabelSetItem.item_id == ReleaseItem.item_id)
        query = query.join(LabelSetItem, joined)
        if part is not None:
            query = query.where(LabelSetItem.part == part)
        order = [LabelSetItem.position]
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.execute(query.order_by(*order).offset(max(0, offset)).limit(limit)).all()
    mistakes = (
        {
            p.item_id: {"found": p.found, "missed": p.missed, "extra": p.extra}
            for p in db.scalars(
                select(ItemPrediction).where(
                    ItemPrediction.op_id == suspect, ItemPrediction.item_id.in_([item.id for _, item in rows])
                )
            )
        }
        if suspect and rows
        else {}
    )
    fixed = drafts.edited_ids(db, draft, [item.id for _, item in rows]) if draft is not None and rows else set()
    change = drafts.change_counts(db, draft, list(fixed)) if fixed else {}
    in_set = labeling.item_states(db, label_set, [item.id for _, item in rows]) if label_set else {}
    found = qa.flagged(db, release.id, check, [item.id for _, item in rows]) if check else {}
    for item_id, mine in of_class.items():  # only the labels of the chosen classes are marked
        if item_id in found:
            found[item_id] = {**found[item_id], "boxes": mine}
    kept_as = qa.judgements(db, release.id, judged, [item.id for _, item in rows]) if judged and let_be else {}
    revisions = {
        rev.id: rev.boxes or []
        for rev in db.scalars(select(LabelRevision).where(LabelRevision.id.in_([m.revision_id for m, _ in rows])))
    }
    predicted: dict[int, dict[str, Any]] = {}
    if pred:  # one model run's predictions, laid over the thumbnails
        shown = checks.page_predictions(db, release, pred, [item.id for _, item in rows], GRID_BOXES)
        predicted = {k: {"labels": [light_shape(b) for b in v["labels"]], "scores": v["scores"]} for k, v in shown.items()}
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
                "boxes": [light_shape(b) for b in revisions.get(member.revision_id, [])[:GRID_BOXES]],
                "fixed": item.id in fixed,  # changed in the work in progress
                # how: labels added, removed, moved, reclassed (against the base version)
                **({"change": change[item.id]} if item.id in change else {}),
                # the evaluation's found / missed / extra (sorting by likely wrong labels)
                **({"check": mistakes[item.id]} if item.id in mistakes else {}),
                # in a labeling set: how far it got (todo / labelled / approved / rejected) and why it was sent back
                **({"set": in_set[item.id]} if item.id in in_set else {}),
                # flagged by a label check: which labels (indices) and why
                **({"qa": found[item.id]} if item.id in found else {}),
                # judged right for the check in play ("문제 없음"): by whom, why
                **({"let_be": kept_as[item.id]} if item.id in kept_as else {}),
                # what the run asked for (``pred``) predicted for it: labels (light) and scores
                **({"pred": predicted[item.id]} if item.id in predicted else {}),
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
        raise ServiceError(404, "이 버전에 없는 항목입니다.")
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
        # The original: read here, or asked of the node that has it when the view opens.
        "image_url": media_url(settings, "item", str(item.id)) if item.location_id else None,
        "image_local": item_file(db, item) is not None,
        "edit": drafts.item_edit_view(db, release, item.id),
        "revisions": [_revision_view(r, spaces, with_boxes=True) for r in revisions],
        "releases": [
            {
                "id": rel.id,
                "dataset": ds.name,
                "version": rel.version,
                "split": m.split,
                "rev": rev_numbers.get(m.revision_id),
                "current": rel.id == release_id,
                "draft": rel.status == "draft",
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


# ---- new releases from existing ones ---------------------------------------------------
#
# A release is a list of (item, split, label revision). A new one is made on the Hub alone:
# pick items of one or more releases (all, those matching a filter, or all but those), and
# keep their splits or deal them out again. Nothing on the nodes changes; the images stay put.

SPLIT_GROUPS = ("", "site", "camera")


def _share(seed: str) -> float:
    return int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) / 0x100000000


def _split_of(item: Item, ratios: list[tuple[str, float]], group_by: str) -> str:
    """Train, val or test for an item, decided by a hash: the same item (or group) always lands in the same split.

    So a later release that adds items keeps the earlier ones where they were, and with
    ``group_by`` every image of one camera (or site) lands together, keeping near-identical
    frames out of both train and val.
    """
    group = getattr(item, group_by, "") if group_by else ""
    point = _share(f"{group_by}:{group}") if group else _share(item.sha256)
    edge = 0.0
    for name, ratio in ratios:
        edge += ratio
        if point < edge:
            return name
    return ratios[-1][0]


def _ratios(split: dict[str, Any] | None) -> list[tuple[str, float]] | None:
    if not split:
        return None
    try:
        values = [(name, float(split.get(name) or 0)) for name in ("train", "val", "test")]
    except (TypeError, ValueError) as e:
        raise ServiceError(400, "split 비율은 숫자여야 합니다.") from e
    if any(v < 0 for _, v in values) or sum(v for _, v in values) <= 0:
        raise ServiceError(400, "split 비율은 0 이상이고, 합이 0보다 커야 합니다.")
    total = sum(v for _, v in values)
    return [(name, v / total) for name, v in values if v > 0]


def _class_plan(space: LabelSpace, wanted: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[int, int]] | None:
    """New classes from the old: ``wanted`` maps an old class id to its new name. The same name merges, "" drops.

    Returns the new classes (numbered from 0 in their old order) and old id → new id, or None
    when nothing changes.
    """
    if not wanted:
        return None
    classes: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    mapping: dict[int, int] = {}
    for c in space.classes:
        old = int(c["id"])
        name = str(wanted.get(str(old), c["name"]) or "").strip()[:128]
        if not name:
            continue
        if name not in index:
            index[name] = len(classes)
            classes.append({"id": len(classes), "name": name, **({"color": c["color"]} if c.get("color") else {})})
        mapping[old] = index[name]
    if not classes:
        raise ServiceError(400, "클래스를 하나 이상 남기세요.")
    if [(c["id"], c["name"]) for c in classes] == [(int(c["id"]), c["name"]) for c in space.classes]:
        return None
    return classes, mapping


def _without_dropped(db: Session, members: dict[int, ReleaseItem], kept: list[int]) -> dict[int, ReleaseItem]:
    """Classification: an image whose class was dropped leaves the new release (it has nothing left to say)."""
    labelled = [row.revision_id for row in members.values() if row.state == "labelled"]
    keep: set[int] = set()
    for start in range(0, len(labelled), 5000):
        chunk = labelled[start : start + 5000]
        keep.update(
            db.scalars(
                select(RevisionClass.revision_id).where(
                    RevisionClass.revision_id.in_(chunk), RevisionClass.class_id.in_(kept)
                )
            )
        )
    return {i: row for i, row in members.items() if row.state != "labelled" or row.revision_id in keep}


def _relabel(
    db: Session, members: dict[int, ReleaseItem], space: LabelSpace, mapping: dict[int, int], source: str
) -> dict[int, LabelRevision]:
    """Each item's labels under the new classes: an existing revision when one says the same, else a new one."""
    olds_of: dict[int, set[int]] = {}
    for old, new in mapping.items():
        olds_of.setdefault(new, set()).add(old)
    out: dict[int, LabelRevision] = {}
    ids = list(members)
    for start in range(0, len(ids), 2000):
        chunk = ids[start : start + 2000]
        wanted = [members[i].revision_id for i in chunk]
        source_revs = {r.id: r for r in db.scalars(select(LabelRevision).where(LabelRevision.id.in_(wanted)))}
        history: dict[int, list[LabelRevision]] = {}
        for r in db.scalars(select(LabelRevision).where(LabelRevision.item_id.in_(chunk))):
            history.setdefault(r.item_id, []).append(r)
        created: list[LabelRevision] = []
        for item_id in chunk:
            src = source_revs[members[item_id].revision_id]
            boxes = [[mapping[int(b[0])], *b[1:]] for b in src.boxes if int(b[0]) in mapping]
            if boxes and all(len(b) == 1 for b in boxes):  # classification: two merged classes are one label
                boxes = [list(b) for b in dict.fromkeys(tuple(b) for b in boxes)]
            state = src.state if boxes or src.state != "labelled" else "background"
            covered = set(src.labelled_class_ids or [])
            # A merged class was looked for everywhere only if every class merged into it was.
            labelled = [new for new, olds in sorted(olds_of.items()) if olds <= covered] if state != "unlabelled" else []
            digest = _digest(space.id, state, boxes)
            past = history.setdefault(item_id, [])
            same = next((r for r in past if r.digest == digest), None)
            if same is None:
                same = LabelRevision(
                    item_id=item_id,
                    rev=max((r.rev for r in past), default=0) + 1,
                    label_space_id=space.id,
                    source=source,
                    state=state,
                    box_count=len(boxes),
                    boxes=boxes,
                    labelled_class_ids=labelled,
                    digest=digest,
                )
                db.add(same)
                created.append(same)
                past.append(same)
            out[item_id] = same
        db.flush()
        for rev in created:
            counts = Counter(int(b[0]) for b in rev.boxes)
            db.add_all(RevisionClass(revision_id=rev.id, class_id=cid, count=n) for cid, n in counts.items())
    return out


def new_release(
    db: Session,
    user: User,
    *,
    sources: list[str],
    version: str,
    dataset: str = "",
    note: str = "",
    match: dict[str, Any] | None = None,
    exclude: bool = False,
    split: dict[str, Any] | None = None,
    group_by: str = "",
    classes: dict[str, Any] | None = None,
    dry_run: bool = False,
    via: str = "api",
) -> dict[str, Any]:
    """Make a release from the items of existing ones; ``dry_run`` only counts what it would hold.

    ``match`` takes the release screen's filters (split, state, class_id, site, camera, q): only
    the items matching it, or with ``exclude`` all but those. With several sources an item that is
    in more than one keeps the split and labels of the first. ``split``: ratios such as
    {"train": 80, "val": 10, "test": 10} to deal the items out again (otherwise they keep theirs).
    ``classes``: {old class id: new name} to rename classes, merge them (the same new name) or
    drop them (""); the labels are rewritten as new revisions, the old versions stay as they were.
    """
    if not sources:
        raise ServiceError(400, "새 버전에 담을 버전을 하나 이상 고르세요.")
    releases = [get_release(db, rid) for rid in dict.fromkeys(sources)]
    for release in releases:
        require_ready(release, "끝난 뒤에 다시 시도하세요.")
    first = releases[0]
    first_dataset = db.get(Dataset, first.dataset_id)
    project_id = first_dataset.project_id
    for other in releases[1:]:
        if db.get(Dataset, other.dataset_id).project_id != project_id:
            raise ServiceError(400, "다른 프로젝트의 버전은 합칠 수 없습니다.")
        if other.label_space_id != first.label_space_id:
            raise ServiceError(400, "클래스 목록(라벨 공간)이 다른 버전은 합칠 수 없습니다.")
        if not task_names.same(release_task(db, other), dataset_task(first_dataset)):
            raise ServiceError(400, "태스크가 다른 버전은 합칠 수 없습니다.")
    if group_by not in SPLIT_GROUPS:
        raise ServiceError(400, "split을 묶는 기준은 site 또는 camera입니다.")
    old_space = db.get(LabelSpace, first.label_space_id)
    plan = _class_plan(old_space, classes)
    ratios = _ratios(split)
    wanted = match or {}
    class_id = wanted.get("class_id")
    try:
        class_id = int(class_id) if class_id not in (None, "") else None
    except (TypeError, ValueError) as e:
        raise ServiceError(400, "class_id는 숫자여야 합니다.") from e

    members: dict[int, ReleaseItem] = {}
    for release in releases:
        rows = db.scalars(select(ReleaseItem).where(ReleaseItem.release_id == release.id)).all()
        if match:
            matched_query = _item_filter(
                release.id,
                wanted.get("split"),
                wanted.get("state") or "",
                class_id,
                wanted.get("site") or "",
                wanted.get("camera") or "",
                str(wanted.get("q") or "").strip(),
            ).with_only_columns(ReleaseItem.item_id)
            matched = set(db.scalars(matched_query))
            rows = [r for r in rows if (r.item_id in matched) != exclude]
        for row in rows:
            members.setdefault(row.item_id, row)
    if plan and dataset_task(first_dataset) == "classify":
        members = _without_dropped(db, members, list(plan[1]))
    items = {}
    if ratios:
        ids = list(members)
        for start in range(0, len(ids), 5000):
            for item in db.scalars(select(Item).where(Item.id.in_(ids[start : start + 5000]))):
                items[item.id] = item

    splits = {
        item_id: _split_of(items[item_id], ratios, group_by) if ratios else row.split for item_id, row in members.items()
    }
    counts: dict[str, int] = {}
    for value in splits.values():
        counts[value] = counts.get(value, 0) + 1
    summary = {"items": len(members), "splits": counts, "classes": len(plan[0] if plan else old_space.classes)}
    if dry_run:
        return summary
    if not members:
        raise ServiceError(400, "고른 조건에 맞는 항목이 없어 버전을 만들 수 없습니다.")

    version, name = version.strip(), (dataset.strip() or first_dataset.name)
    if not version:
        raise ServiceError(400, "버전 이름을 적으세요.")
    if version == drafts.DRAFT:
        raise ServiceError(400, f"'{version}'은 버전 이름으로 쓸 수 없습니다.")
    target = db.scalar(select(Dataset).where(Dataset.project_id == project_id, Dataset.name == name))
    taken = select(Release).where(Release.dataset_id == target.id, Release.version == version) if target else None
    if taken is not None and db.scalar(taken):
        raise ServiceError(409, f"{name} {version} 버전이 이미 있습니다. 다른 이름을 쓰세요.")
    if target is not None and not task_names.same(dataset_task(target), dataset_task(first_dataset)):
        raise ServiceError(
            400, f"데이터셋 {name}의 태스크는 {task_names.short(dataset_task(target))}입니다. 다른 데이터셋 이름을 쓰세요."
        )
    if target is None:
        target = Dataset(project_id=project_id, name=name, task=dataset_task(first_dataset))
        db.add(target)
        db.flush()
    space = _space_for(db, project_id, plan[0]) if plan else old_space
    release = Release(
        dataset_id=target.id,
        version=version,
        label_space_id=space.id,
        base_release_id=first.id,
        note=note.strip(),
        status="building",
        created_by=user.id,
    )
    db.add(release)
    db.flush()
    labels = {
        item_id: (row.revision_id, row.state, row.box_count) for item_id, row in members.items()
    }
    if plan:
        for item_id, rev in _relabel(db, members, space, plan[1], f"classes:{name} {version}").items():
            labels[item_id] = (rev.id, rev.state, rev.box_count)
    rows = [
        {
            "release_id": release.id,
            "item_id": item_id,
            "split": splits[item_id],
            "revision_id": labels[item_id][0],
            "state": labels[item_id][1],
            "box_count": labels[item_id][2],
        }
        for item_id in members
    ]
    for start in range(0, len(rows), 5000):
        db.execute(insert(ReleaseItem), rows[start : start + 5000])
    db.flush()
    release.stats = compute_stats(db, release)
    release.status = "ready"
    stamp_digest(db, release)
    services.audit(
        db,
        user.id,
        via,
        "release.create",
        release.id,
        name=name,
        version=version,
        sources=[r.id for r in releases],
        items=len(members),
        resplit=bool(ratios),
        classes=summary["classes"] if plan else None,
    )
    return {**summary, "release_id": release.id, "dataset": name, "version": version}


def delete_dataset(db: Session, user: User, dataset_id: str) -> int:
    """Every version of a dataset to the trash (each comes back on its own). Returns how many went."""
    services.require_admin(user)
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise ServiceError(404, "데이터셋을 찾을 수 없습니다.")
    get_project(db, dataset.project_id)
    live = list(db.scalars(select(Release).where(Release.dataset_id == dataset.id, Release.deleted_at.is_(None))))
    if any(r.status == "building" for r in live):
        raise ServiceError(409, f"{dataset.name}에 가져오는 중인 버전이 있습니다. 끝난 뒤에 지우세요.")
    for draft in [r for r in live if r.status == "draft"]:  # fixes not made a version go; their labels stay
        drafts.discard(db, user, draft.id, via="api")
        live.remove(draft)
    for release in live:
        release.deleted_at = utcnow()
    services.audit(db, user.id, "api", "dataset.delete", dataset.id, name=dataset.name, versions=len(live))
    return len(live)


def set_release_deleted(db: Session, user: User, release_id: str, deleted: bool) -> Release:
    services.require_admin(user)
    release = get_release(db, release_id, allow_deleted=True)
    if release.status == "draft":
        raise ServiceError(400, "작업 중은 휴지통으로 옮기지 않습니다. '모두 되돌리기'로 지우세요.")
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
