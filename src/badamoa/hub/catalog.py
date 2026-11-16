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

from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
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
from .models import HubSetting, Job, Node, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run, Sweep

TASKS = task_names.BUILTIN
SPLITS = ("train", "val", "test", "")
# An image without labels is background: nothing to find in it. ("unlabelled" from older nodes is read as that.)
STATES = ("labelled", "background")
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
    elif not deleted:
        project.deleted_at = None
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
    name_pattern: str = "",
    note: str = "",
    base_release_id: str = "",
    task: str = "",
    via: str = "api",
) -> Job:
    """Import a node's folder as a dataset version. ``task``: Det/Seg/Kpt/Cls, a named task, or "" (from the data)."""
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
    if name_pattern.strip():
        name_rule(name_pattern)  # a wrong rule is turned away now, not after the node has read every file
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
        "task": task,
        "project_id": project_id,
        "dataset": dataset,
        "version": version,
        "path": path,
        "format": fmt,
        "label_space": label_space.strip(),
        "site": site.strip(),
        "camera": camera.strip(),
        "name_pattern": name_pattern.strip(),
        "note": note,
        "base_release_id": base_release_id,
    }
    return services.create_job(db, user, "data.import", params, node_id, via=via)


# ---- site and camera from file names ---------------------------------------------------
#
# A rule such as ``{site}_{camera}_*`` is matched against the file name without its extension;
# with folders in it (``{site}/{camera}/*``) against as many trailing folders of the path.

NAME_FIELDS = ("site", "camera")


def name_rule(pattern: str) -> tuple[re.Pattern[str], int, list[str]]:
    """The rule as a regular expression, how many path parts it covers, and the fields it fills."""
    pattern = pattern.strip().replace("\\", "/").strip("/")
    if not pattern:
        raise ServiceError(400, "파일 이름 규칙을 적으세요. 예: {site}_{camera}_*")
    regex, fields = [], []
    for part in re.split(r"(\{[^{}]*\}|\*)", pattern):
        if not part:
            continue
        if part == "*":
            regex.append(r"[^/]*")
        elif part.startswith("{") and part.endswith("}"):
            name = part[1:-1].strip()
            if name not in NAME_FIELDS:
                raise ServiceError(400, f"규칙에는 {{site}}와 {{camera}}만 쓸 수 있습니다. ({part})")
            if name in fields:
                raise ServiceError(400, f"{part}가 두 번 나옵니다.")
            fields.append(name)
            regex.append(f"(?P<{name}>[^/]+?)")
        else:
            regex.append(re.escape(part))
    if not fields:
        raise ServiceError(400, "규칙에 {site}나 {camera}를 하나 이상 넣으세요.")
    return re.compile("".join(regex)), pattern.count("/") + 1, fields


def names_from(rule: tuple[re.Pattern[str], int, list[str]], path: str) -> dict[str, str] | None:
    """{"site": …, "camera": …} read from a file path by the rule, or None when it does not fit."""
    regex, depth, _ = rule
    parts = path.replace("\\", "/").strip("/").split("/")
    if len(parts) < depth:
        return None
    stem, dot, _ext = parts[-1].rpartition(".")
    parts[-1] = stem if dot and stem else parts[-1]
    found = regex.fullmatch("/".join(parts[-depth:]))
    if found is None:
        return None
    values = {k: v.strip()[:64] for k, v in found.groupdict().items()}
    return values if all(values.values()) else None


def _places(db: Session, release: Release) -> None:
    """Refresh the sites and cameras a release's filters offer."""
    stats = dict(release.stats or {})
    for field in NAME_FIELDS:
        values = db.scalars(
            select(getattr(Item, field))
            .join(ReleaseItem, ReleaseItem.item_id == Item.id)
            .where(ReleaseItem.release_id == release.id)
            .distinct()
        ).all()
        stats[f"{field}s"] = sorted(v for v in values if v)
    release.stats = stats


def fill_names(
    db: Session, user: User, release_id: str, pattern: str, dry_run: bool = False, via: str = "api"
) -> dict[str, Any]:
    """Read site and camera from the file names of a release's images; ``dry_run`` shows what the rule finds.

    Site and camera belong to the image, so every version holding it sees the new values.
    Images the rule does not fit keep what they had.
    """
    release = get_release(db, release_id)
    project = get_project(db, db.get(Dataset, release.dataset_id).project_id)
    rule = name_rule(pattern)
    fields = rule[2]
    rows = db.execute(
        select(Item.id, Item.rel_path, Item.file_name, Item.site, Item.camera)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release.id)
        .order_by(Item.id)
    ).all()
    found: dict[int, dict[str, str]] = {}
    changed: dict[tuple[str, ...], list[int]] = {}
    examples: list[dict[str, str]] = []
    misses: list[str] = []
    counts = {field: Counter() for field in fields}
    for item_id, rel_path, file_name, site, camera in rows:
        path = rel_path or file_name
        values = names_from(rule, path)
        if values is None:
            if len(misses) < 4:
                misses.append(path)
            continue
        found[item_id] = values
        for field in fields:
            counts[field][values[field]] += 1
        if len(examples) < 6:
            examples.append({"path": path, **values})
        if any(values[field] != now for field, now in (("site", site), ("camera", camera)) if field in fields):
            changed.setdefault(tuple(values[f] for f in fields), []).append(item_id)
    summary = {
        "items": len(rows),
        "matched": len(found),
        "changed": sum(len(ids) for ids in changed.values()),
        "fields": fields,
        "examples": examples,
        "misses": misses,
        "values": {
            field: {"count": len(counts[field]), "top": [{"value": v, "items": n} for v, n in counts[field].most_common(6)]}
            for field in fields
        },
    }
    if dry_run:
        return summary
    _may_edit(project, user)
    if not found:
        raise ServiceError(400, "규칙에 맞는 파일이 없습니다. 예시의 경로를 보고 규칙을 고치세요.")
    for values, ids in changed.items():
        new = dict(zip(fields, values, strict=True))
        for start in range(0, len(ids), 900):
            db.execute(update(Item).where(Item.id.in_(ids[start : start + 900])).values(new))
    db.flush()
    touched = db.scalars(
        select(Release)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(Dataset.project_id == project.id, Release.status == "ready")
    ).all()
    for each in touched:
        _places(db, each)
    services.audit(db, user.id, via, "items.names", release.id, pattern=pattern.strip(), changed=summary["changed"])
    return summary


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


def import_begin(db: Session, node: Node, job_id: str, body: ImportBegin) -> dict[str, Any]:
    job = _import_job(db, node, job_id)
    release = db.scalar(select(Release).where(Release.job_id == job.id))
    if release is not None:
        return {"release_id": release.id}
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
    space = _label_space(db, project.id, body.classes, p.get("label_space", ""), keypoints)
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
    rule = name_rule(job.params["name_pattern"]) if job.params.get("name_pattern") else None
    class_ids = [c["id"] for c in space.classes]
    # Outlines (Seg), keypoints (Kpt) or whatever a named task's labels hold follow the box; Det keeps the box.
    whole = release_task(db, release) not in ("detect", "classify")
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
        named = (names_from(rule, data.rel_path or data.file_name) if rule else None) or {}
        items[sha] = Item(
            project_id=project_id,
            sha256=sha,
            file_name=data.file_name[:512],
            width=data.width,
            height=data.height,
            bytes=data.bytes,
            site=named.get("site", site),
            camera=named.get("camera", camera),
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
        # A box is [class, cx, cy, w, h]; a classification label is just [class]; a segmentation
        # label is a box followed by its outline (x1, y1, x2, y2, …), kept unless the dataset is Det.
        boxes = [
            [int(b[0]), *(b[1:] if whole else b[1:5])]
            for b in data.boxes
            if (len(b) == 1 or len(b) >= 5) and int(b[0]) in class_ids
        ]
        state = "labelled" if boxes else "background"  # no labels: nothing to find in it
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
    nodes = {n.id: n.name for n in db.scalars(select(Node).where(Node.id.in_([j.node_id for j in jobs if j.node_id])))}
    out = []
    for job in jobs:
        params = job.params or {}
        project = db.get(Project, params.get("project_id") or "")
        if project is None or project.deleted_at is not None:
            continue
        release = db.scalar(select(Release).where(Release.job_id == job.id))
        stats = (release.stats if release else None) or {}
        progress = stats.get("import") or {}
        status = {"succeeded": "done", "failed": "failed", "canceled": "failed"}.get(job.status, job.status)
        total = int(progress.get("total") or stats.get("items") or 0)
        done = total if status == "done" else int(progress.get("done") or 0)
        out.append(
            {
                "job_id": job.id,
                "status": status,  # queued | running | done | failed
                "project": {"id": project.id, "name": project.name},
                "dataset": params.get("dataset", ""),
                "version": params.get("version", ""),
                "release_id": release.id if release else None,
                "node": nodes.get(job.node_id or "", ""),
                "done": done,
                "total": total,
                "skipped": int(progress.get("skipped") or (job.result or {}).get("skipped") or 0),
                "created_at": iso(job.created_at),
                "started_at": iso(job.started_at),
                "finished_at": iso(job.finished_at),
                "error": job.error or "",
            }
        )
    return out


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
        raise ServiceError(404, "버전을 찾을 수 없습니다.")
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
        "task": dataset_task(dataset),
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
    revisions = {
        rev.id: rev.boxes or []
        for rev in db.scalars(select(LabelRevision).where(LabelRevision.id.in_([m.revision_id for m, _ in rows])))
    }
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
    if any(r.status != "ready" for r in releases):
        raise ServiceError(409, "아직 등록 중인 버전이 있습니다. 끝난 뒤에 다시 시도하세요.")
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
