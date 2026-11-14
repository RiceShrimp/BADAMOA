"""Endpoints for model actions (evaluate, export, test): users start them, nodes run them."""

from __future__ import annotations

import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import catalog, modelops, services
from .deps import Hub, bearer_token, current_node, current_user, db_session, get_hub
from .models import Node, User

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/model")
media_router = APIRouter(prefix="/media")

MAX_UPLOAD_BYTES = 30 * 1024**2
MAX_FILE_BYTES = 2 * 1024**3  # an exported model can be large


def _storage(hub: Hub):
    return catalog.storage_for(str(hub.settings.storage_dir))


# ---- users ---------------------------------------------------------------------------


class EvalRequest(BaseModel):
    release_id: str
    split: str = "val"
    gpus: int | None = None  # empty: one GPU when the server has any


class ExportRequest(BaseModel):
    format: str = "onnx"


class PredictRequest(BaseModel):
    images: list[dict[str, Any]] = Field(default_factory=list)  # [{"name", "blob"}] from /api/uploads
    conf: float = 0.25


@router.post("/uploads")
async def upload(request: Request, name: str = "image.jpg", hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Keep a small file (a sample image to test a model on). The body is the file itself."""
    token = bearer_token(request)
    data = await request.body()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413 if data else 400, detail="파일이 비었거나 너무 큽니다 (30 MB까지).")

    def work() -> dict[str, Any]:
        with hub.db.session() as db:
            if services.authenticate(db, token) is None:
                raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
        return {"blob": _storage(hub).put_blob(data), "bytes": len(data), "name": Path(name).name}

    return await anyio.to_thread.run_sync(work)


@router.get("/models/{model_id}/ops")
def list_ops(
    model_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    return modelops.list_ops(db, hub.settings, model_id)


@router.post("/models/{model_id}/eval")
def start_eval(
    model_id: str,
    body: EvalRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    op = modelops.start_eval(db, user, model_id, body.release_id, body.split, body.gpus, via="ui")
    db.commit()
    return modelops.op_view(db, hub.settings, op)


@router.post("/models/{model_id}/export")
def start_export(
    model_id: str,
    body: ExportRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    op = modelops.start_export(db, user, model_id, body.format, via="ui")
    db.commit()
    return modelops.op_view(db, hub.settings, op)


@router.post("/models/{model_id}/predict")
def start_predict(
    model_id: str,
    body: PredictRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    op = modelops.start_predict(db, user, model_id, body.images, _storage(hub), body.conf, via="ui")
    db.commit()
    return modelops.op_view(db, hub.settings, op)


# ---- nodes ---------------------------------------------------------------------------


@agent_router.get("/manifest")
def manifest(
    job_id: str, after: int = 0, limit: int = 2000, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    return modelops.manifest(db, node, job_id, after, limit)


@agent_router.get("/blob/{what}")
def blob(
    job_id: str,
    what: str,
    index: int = 0,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> Response:
    """``weights`` (the model), ``template`` (its code) or ``input`` (the index-th image to test on)."""
    if what not in ("weights", "template", "input"):
        raise HTTPException(status_code=404, detail="not found")
    key = modelops.blob_for(db, node, job_id, what, index)
    storage = _storage(hub)
    if not key or not storage.has_blob(key):
        raise HTTPException(status_code=404, detail="파일이 저장소에 없습니다.")
    return Response(storage.get_blob(key), media_type="application/octet-stream")


@agent_router.put("/file")
async def put_file(job_id: str, request: Request, name: str = "file", hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """A result file (an exported model, an annotated image), streamed to disk and stored by content hash."""
    token = bearer_token(request)
    incoming = hub.settings.storage_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=incoming, suffix=".part")
    size = 0
    try:
        with os.fdopen(handle, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다.")
                out.write(chunk)
        key = await anyio.to_thread.run_sync(_storage(hub).put_blob_file, temp)
    finally:
        Path(temp).unlink(missing_ok=True)

    def record() -> dict[str, Any]:
        with hub.db.session() as db:
            node = services.authenticate_node(db, token)
            if node is None:
                raise HTTPException(status_code=401, detail="노드 인증 실패")
            return modelops.add_file(db, node, job_id, name, key, size)

    return await anyio.to_thread.run_sync(record)


@agent_router.post("/finish")
def finish(
    job_id: str, body: dict[str, Any], node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = modelops.finish(db, node, job_id, body)
    db.commit()
    return result


# ---- result files (signed URL) -------------------------------------------------------


@media_router.get("/op/{key}")
def op_file(key: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)) -> Response:
    if not catalog.verify_media(hub.settings, "op", key, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    entry = modelops.file_of(db, key)
    storage = _storage(hub)
    if not storage.has_blob(entry["blob"]):
        raise HTTPException(status_code=404, detail="파일이 저장소에 없습니다.")
    kind = mimetypes.guess_type(entry["name"])[0] or "application/octet-stream"
    inline = entry["kind"] == "image" and kind.startswith("image/") and kind != "image/svg+xml"
    headers = {
        "Content-Disposition": f"{'inline' if inline else 'attachment'}; filename*=UTF-8''{quote(entry['name'])}",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=3600",
    }
    media_type = kind if inline else "application/octet-stream"
    return Response(storage.get_blob(entry["blob"]), media_type=media_type, headers=headers)
