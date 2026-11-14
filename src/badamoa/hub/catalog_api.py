"""Dataset catalog endpoints: user API, node import protocol, signed media URLs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from badamoa.core.protocol import ImportBegin, ImportChunk

from . import catalog, services
from .catalog_models import Item
from .deps import Hub, admin_user, current_node, current_user, db_session, get_hub
from .models import Node, User

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/import")
media_router = APIRouter(prefix="/media")


# ---- projects & tree -------------------------------------------------------------


class NewProject(BaseModel):
    name: str
    task: str = "detect"
    folder: str = ""
    description: str = ""


@router.get("/projects")
def list_projects(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    return catalog.list_projects(db)


@router.post("/projects")
def create_project(body: NewProject, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    project = catalog.create_project(db, user, body.name, body.task, body.folder, body.description)
    db.commit()
    return next(p for p in catalog.list_projects(db) if p["id"] == project.id)


@router.get("/projects/{project_id}/datasets")
def project_tree(
    project_id: str, trash: bool = False, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    return catalog.tree(db, project_id, include_deleted=trash)


# ---- import ------------------------------------------------------------------------


class ImportRequest(BaseModel):
    node_id: str
    path: str
    dataset: str
    version: str
    format: str = "auto"
    label_space: str = ""
    site: str = ""
    camera: str = ""
    note: str = ""
    base_release_id: str = ""


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
        note=body.note,
        base_release_id=body.base_release_id,
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
    path = catalog.item_file(db, item) if item else None
    if path is None:
        raise HTTPException(status_code=404, detail="이 머신에서 읽을 수 없는 파일입니다.")
    return FileResponse(path, headers=_CACHE)
