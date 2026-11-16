"""Dataset catalog endpoints: user API, node import protocol, signed media URLs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from badamoa.core.protocol import ImportBegin, ImportChunk

from . import catalog, originals, overview, purge, services
from .catalog_models import Item
from .deps import Hub, admin_user, current_node, current_user, db_session, get_hub
from .models import Node, User

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/import")
media_router = APIRouter(prefix="/media")


# ---- projects & tree -------------------------------------------------------------


class NewProject(BaseModel):
    name: str
    task: str = ""  # ignored since 0.34.0: each dataset has its own task, chosen when importing
    folder: str = ""
    description: str = ""


@router.get("/projects")
def list_projects(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    return catalog.list_projects(db)


@router.post("/projects")
def create_project(body: NewProject, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    project = catalog.create_project(db, user, body.name, body.folder, body.description)
    db.commit()
    return next(p for p in catalog.list_projects(db) if p["id"] == project.id)


class ProjectPatch(BaseModel):
    name: str | None = None
    folder: str | None = None  # "" takes it out of its folder
    description: str | None = None


@router.patch("/projects/{project_id}")
def update_project(
    project_id: str, body: ProjectPatch, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    catalog.update_project(db, user, project_id, **body.model_dump(exclude_unset=True))
    db.commit()
    return next(p for p in catalog.list_projects(db) if p["id"] == project_id)


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """To the trash, with its datasets, runs and models; the person who made it or an admin."""
    catalog.set_project_deleted(db, user, project_id, True)
    db.commit()
    return {"ok": True}


@router.post("/projects/{project_id}/restore")
def restore_project(project_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    catalog.set_project_deleted(db, user, project_id, False)
    db.commit()
    return {"ok": True}


@router.get("/trash")
def trash(_: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """Deleted projects and versions, to bring back."""
    return catalog.trash(db)


class EmptyTrash(BaseModel):
    projects: list[str] | None = None  # both left out: everything in the trash
    releases: list[str] | None = None


@router.post("/trash/empty")
def empty_trash(
    body: EmptyTrash, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Remove deleted projects and versions for good, and the stored files only they used. Cannot be undone."""
    result = purge.empty_trash(db, hub.settings, user, projects=body.projects, releases=body.releases)
    db.commit()
    return result


class NewFolder(BaseModel):
    name: str
    projects: list[str] = []  # moved into it; none: an empty folder


class FolderRename(BaseModel):
    name: str
    new_name: str


@router.get("/folders")
def list_folders(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    """Every folder, empty ones included, with how many projects are in it."""
    return catalog.list_folders(db)


@router.post("/folders")
def create_folder(body: NewFolder, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    result = catalog.create_folder(db, user, body.name, body.projects)
    db.commit()
    return result


@router.patch("/folders")
def rename_folder(body: FolderRename, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    result = catalog.rename_folder(db, user, body.name, body.new_name)
    db.commit()
    return result


@router.delete("/folders")
def delete_folder(name: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """The folder goes; its projects stay, outside any folder. (The name is a query parameter: any text works.)"""
    result = catalog.delete_folder(db, user, name)
    db.commit()
    return result


@router.get("/project-cards")
def project_cards(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    """Every project as a card: collage, task, counts, the best model."""
    return overview.project_cards(db, hub.settings)


@router.get("/home")
def home(_: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Totals, the projects touched last and the latest runs, for the home screen."""
    return overview.home(db, hub.settings)


@router.get("/projects/{project_id}/overview")
def project_page(
    project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """A project with its datasets as cards and its best models."""
    return overview.project_page(db, hub.settings, project_id)


@router.get("/projects/{project_id}/datasets")
def project_tree(
    project_id: str, trash: bool = False, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    return catalog.tree(db, project_id, include_deleted=trash)


# ---- import ------------------------------------------------------------------------


@router.get("/imports")
def imports(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    """Dataset imports under way and those that ended in the last half hour, with how far they got."""
    return catalog.imports(db)


class ImportRequest(BaseModel):
    node_id: str
    path: str
    dataset: str
    version: str
    format: str = "auto"
    label_space: str = ""
    site: str = ""
    camera: str = ""
    name_pattern: str = ""  # e.g. "{site}_{camera}_*": site and camera read from each file name
    note: str = ""
    base_release_id: str = ""
    task: str = ""  # Det · Seg · Kpt · Cls, a named task, or "" to read it from the folder


@router.post("/projects/{project_id}/imports")
def start_import(
    project_id: str,
    body: ImportRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    job = catalog.start_import(
        db,
        user,
        project_id=project_id,
        node_id=body.node_id,
        path=body.path,
        dataset=body.dataset,
        version=body.version,
        fmt=body.format,
        label_space=body.label_space,
        site=body.site,
        camera=body.camera,
        name_pattern=body.name_pattern,
        note=body.note,
        base_release_id=body.base_release_id,
        task=body.task,
        via="ui",
    )
    db.commit()
    return services.job_view(db, hub.settings, job)


@agent_router.post("/begin")
def import_begin(
    job_id: str, body: ImportBegin, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = catalog.import_begin(db, node, job_id, body)
    db.commit()
    return result


@agent_router.post("/items")
def import_items(
    job_id: str,
    body: ImportChunk,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    result = catalog.import_items(db, node, job_id, body, storage)
    db.commit()
    return result


@agent_router.post("/finish")
def import_finish(job_id: str, node: Node = Depends(current_node), db: Session = Depends(db_session)) -> dict[str, Any]:
    result = catalog.import_finish(db, node, job_id)
    db.commit()
    return result


# ---- releases ----------------------------------------------------------------------


@router.get("/releases/{release_id}")
def get_release(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return catalog.release_view(db, release_id)


@router.get("/releases/{release_id}/items")
def list_items(
    release_id: str,
    split: str | None = None,
    state: str = "",
    class_id: int | None = None,
    site: str = "",
    camera: str = "",
    q: str = "",
    offset: int = 0,
    limit: int = 60,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    return catalog.list_items(
        db,
        hub.settings,
        release_id,
        split=split,
        state=state,
        class_id=class_id,
        site=site,
        camera=camera,
        q=q,
        offset=offset,
        limit=limit,
    )


@router.get("/releases/{release_id}/items/{item_id}")
def get_item(
    release_id: str,
    item_id: int,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    return catalog.item_detail(db, hub.settings, release_id, item_id)


@router.get("/releases/{release_id}/compare")
def compare(
    release_id: str, other: str, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    return catalog.compare(db, release_id, other)


class NewRelease(BaseModel):
    sources: list[str]
    version: str = ""
    dataset: str = ""  # empty: the first source's dataset (a new version of it)
    note: str = ""
    filter: dict[str, Any] | None = None  # the release screen's filters: split, state, class_id, site, camera, q
    exclude: bool = False  # all items but those matching the filter
    split: dict[str, float] | None = None  # deal out again, e.g. {"train": 80, "val": 10, "test": 10}
    group_by: str = ""  # "", "site" or "camera": one group, one split
    classes: dict[str, str] | None = None  # {old class id: new name}: same name merges, "" drops
    dry_run: bool = False


@router.post("/releases")
def new_release(body: NewRelease, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """A release made from the items of existing ones, or with ``dry_run`` how many items it would hold per split."""
    result = catalog.new_release(
        db,
        user,
        sources=body.sources,
        version=body.version,
        dataset=body.dataset,
        note=body.note,
        match=body.filter,
        exclude=body.exclude,
        split=body.split,
        group_by=body.group_by,
        classes=body.classes,
        dry_run=body.dry_run,
        via="ui",
    )
    if not body.dry_run:
        db.commit()
    return result


class NameRule(BaseModel):
    pattern: str  # "{site}_{camera}_*", "{site}/{camera}/*" …
    dry_run: bool = False


@router.post("/releases/{release_id}/names")
def fill_names(
    release_id: str, body: NameRule, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Site and camera read from the file names of a release's images; ``dry_run`` only shows what the rule finds."""
    result = catalog.fill_names(db, user, release_id, body.pattern, dry_run=body.dry_run, via="ui")
    if not body.dry_run:
        db.commit()
    return result


@router.get("/releases/{release_id}/charts")
def release_charts(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """Split and label states, classes, image sizes, where the boxes sit and how big they are."""
    return overview.release_charts(db, release_id)


@router.get("/releases/{release_id}/models")
def release_models(
    release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    """The runs that trained on this release and the models they made."""
    return overview.release_models(db, hub.settings, release_id)


class ClassColor(BaseModel):
    color: str = ""  # #rrggbb; empty: back to the automatic colour


@router.put("/releases/{release_id}/classes/{class_id}")
def class_color(
    release_id: str, class_id: int, body: ClassColor, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = overview.set_class_color(db, release_id, class_id, body.color)
    db.commit()
    return result


@router.delete("/releases/{release_id}")
def delete_release(release_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    catalog.set_release_deleted(db, user, release_id, True)
    db.commit()
    return {"ok": True}


@router.post("/releases/{release_id}/restore")
def restore_release(release_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    catalog.set_release_deleted(db, user, release_id, False)
    db.commit()
    return {"ok": True}


# ---- media (signed, no bearer header) ------------------------------------------------

_CACHE = {"Cache-Control": "private, max-age=3600"}


@media_router.get("/thumb/{key}")
def thumb(key: str, exp: int, sig: str, hub: Hub = Depends(get_hub)) -> Response:
    if not catalog.verify_media(hub.settings, "thumb", key, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(key):
        raise HTTPException(status_code=404, detail="not found")
    return Response(storage.get_blob(key), media_type="image/jpeg", headers=_CACHE)


@media_router.get("/item/{item_id}")
def item_image(
    item_id: int, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> FileResponse:
    if not catalog.verify_media(hub.settings, "item", str(item_id), exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="not found")
    path = catalog.item_file(db, item)
    if path is None:  # on another server: that node sends it (kept in the cache afterwards)
        try:
            path = originals.fetch(db, hub.settings, item)
        except services.ServiceError as e:
            raise HTTPException(status_code=e.status, detail=e.message) from e
    return FileResponse(path, headers=_CACHE)
