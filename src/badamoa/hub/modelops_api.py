"""Endpoints for model actions (evaluate, export, test): users start them, nodes run them."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import catalog, modelops, services
from .deps import Hub, bearer_token, current_node, current_user, db_session, get_hub
from .models import Node, User
from .training_api import blob_response

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/model")
media_router = APIRouter(prefix="/media")

MAX_UPLOAD_BYTES = 30 * 1024**2
MAX_FILE_BYTES = 8 * 1024**3  # an exported model can be large


def _storage(hub: Hub):
    return catalog.storage_for(str(hub.settings.storage_dir))


# ---- users ---------------------------------------------------------------------------


class EvalRequest(BaseModel):
    release_id: str
    split: str = "val"
    splits: list[str] | None = None  # several in one evaluation (train and val …); ``split`` when empty
    gpus: int | None = None  # empty: one GPU when the server has any
    node_id: str = ""  # empty: the server holding the version's images


class ExportRequest(BaseModel):
    format: str = "onnx"
    node_id: str = ""  # empty: the server that trained the model (or another that is up)


class PredictRequest(BaseModel):
    images: list[dict[str, Any]] = Field(default_factory=list)  # [{"name", "blob"}] from /api/uploads
    conf: float = 0.25
    node_id: str = ""


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
    op = modelops.start_eval(
        db, user, model_id, body.release_id, body.split, body.gpus, via="ui", node_id=body.node_id, splits=body.splits
    )
    db.commit()
    return modelops.op_view(db, hub.settings, op)


@router.get("/models/{model_id}/places")
def model_places(
    model_id: str,
    kind: str,
    release_id: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """The servers a model's evaluation (on ``release_id``), export or test can run on, and the default one."""
    return modelops.places(db, hub.settings, model_id, kind, release_id)


@router.get("/model-ops/{op_id}")
def op_detail(
    op_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """One evaluation (export, test): status, server, time, numbers per class and split, log size."""
    return modelops.op_detail(db, hub.settings, op_id)


@router.delete("/model-ops/{op_id}")
def delete_op(op_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """Remove an evaluation (export, test) with its numbers, files and per-image predictions."""
    modelops.delete_op(db, user, op_id, via="ui")
    db.commit()
    return {"ok": True}


@router.get("/models-find")
def find_models(q: str = "", limit: int = 30, _: User = Depends(current_user), db: Session = Depends(db_session)) -> list:
    """Models of every project, for picking ones to compare."""
    return modelops.find_models(db, q, max(1, min(limit, 100)))


@router.get("/models-compare")
def compare_models(
    ids: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Models from any project (``ids`` comma separated, up to six) on their own validation and shared evaluations."""
    return modelops.compare(db, hub.settings, [i for i in ids.split(",") if i])


class RerunRequest(BaseModel):
    node_id: str = ""  # empty: the server it ran on (or the default when that one is gone)


@router.post("/model-ops/{op_id}/rerun")
def rerun_op(
    op_id: str,
    body: RerunRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """The same evaluation (export, test) again, with the same settings."""
    op = modelops.rerun(db, user, hub.settings, op_id, body.node_id, via="ui")
    db.commit()
    return modelops.op_view(db, hub.settings, op)


@router.get("/releases/{release_id}/evaluations")
def release_evaluations(
    release_id: str,
    split: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """Every model evaluated on this version (and split): to compare them on the same data."""
    return modelops.release_evaluations(db, hub.settings, release_id, split)


@router.get("/releases/{release_id}/eval-models")
def eval_models(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """The project's models for checking this version's labels, the ones that fit first, with why others do not."""
    return modelops.eval_choices(db, catalog.get_release(db, release_id))


@router.post("/models/{model_id}/export")
def start_export(
    model_id: str,
    body: ExportRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    op = modelops.start_export(db, user, model_id, body.format, via="ui", node_id=body.node_id, settings=hub.settings)
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
    op = modelops.start_predict(
        db, user, model_id, body.images, _storage(hub), body.conf, via="ui", node_id=body.node_id, settings=hub.settings
    )
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
async def put_file(
    job_id: str,
    request: Request,
    name: str = "file",
    media: str = "",
    caption: str = "",
    key: str = "",
    step: int | None = None,
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
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
        blob = await anyio.to_thread.run_sync(_storage(hub).put_blob_file, temp)
    finally:
        Path(temp).unlink(missing_ok=True)

    def record() -> dict[str, Any]:
        with hub.db.session() as db:
            node = services.authenticate_node(db, token)
            if node is None:
                raise HTTPException(status_code=401, detail="노드 인증 실패")
            return modelops.add_file(db, node, job_id, name, blob, size, media, caption, storage=_storage(hub))

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
    return blob_response(storage, entry["blob"], entry["name"], entry["kind"] in ("image", "table-image"))
