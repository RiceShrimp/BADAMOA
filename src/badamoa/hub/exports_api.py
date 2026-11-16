"""Release downloads: start an export, list a release's exports, the node's upload, the signed download."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import catalog, exports, services, training
from .deps import Hub, bearer_token, current_node, current_user, db_session, get_hub
from .models import Node, User
from .training_api import blob_response

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/export")
media_router = APIRouter(prefix="/media")


class ExportRequest(BaseModel):
    format: str = "yolo"  # yolo | coco | imagefolder
    images: bool = True  # False: the labels only


@router.post("/releases/{release_id}/exports")
def start_export(
    release_id: str,
    body: ExportRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    job = exports.start_export(db, user, release_id, body.format, body.images, via="ui")
    db.commit()
    return services.job_view(db, hub.settings, job)


@router.get("/releases/{release_id}/exports")
def list_exports(
    release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    return exports.list_exports(db, hub.settings, release_id)


@agent_router.get("/manifest")
def manifest(
    job_id: str, after: int = 0, limit: int = 2000, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    return exports.manifest(db, node, job_id, after, limit)


@agent_router.put("/file")
async def upload(job_id: str, request: Request, name: str = "release.zip", hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """The finished zip, streamed to disk and kept by content hash. Its hash goes into the job's result."""
    token = bearer_token(request)

    def check() -> None:
        with hub.db.session() as db:
            node = services.authenticate_node(db, token)
            if node is None:
                raise HTTPException(status_code=401, detail="노드 인증 실패")
            exports.check_upload(db, node, job_id)

    await anyio.to_thread.run_sync(check)
    incoming = hub.settings.storage_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=incoming, suffix=".part")
    size = 0
    try:
        with os.fdopen(handle, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > exports.MAX_EXPORT_BYTES:
                    raise HTTPException(status_code=413, detail="내보낸 파일이 너무 큽니다.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        storage = catalog.storage_for(str(hub.settings.storage_dir))
        blob = await anyio.to_thread.run_sync(storage.put_blob_file, temp)
    finally:
        Path(temp).unlink(missing_ok=True)
    return {"blob": blob, "bytes": size, "file": training.clean_file_name(name).rsplit("/", 1)[-1]}


@media_router.get("/export/{job_id}")
def download(
    job_id: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> StreamingResponse:
    if not catalog.verify_media(hub.settings, "export", job_id, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    blob, name = exports.download(db, job_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(blob):
        raise HTTPException(status_code=410, detail="파일이 정리되어 없습니다. 다시 내보내세요.")
    return blob_response(storage, blob, name, False)
