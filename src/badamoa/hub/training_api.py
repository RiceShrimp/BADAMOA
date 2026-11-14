"""Training endpoints: runs and models for users, the training protocol for nodes, weights download."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from badamoa.core.protocol import TrainFinish, TrainReport

from . import catalog, services, training
from .deps import Hub, admin_user, bearer_token, current_node, current_user, db_session, get_hub
from .models import Node, User

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/train")
media_router = APIRouter(prefix="/media")

MAX_WEIGHTS_BYTES = 8 * 1024**3
_CHUNK = 1024 * 1024


# ---- runs ----------------------------------------------------------------------------


class NewRun(BaseModel):
    release_id: str
    template: str = ""  # id or name; empty = the project's default
    environment: str = ""  # id or name; empty = the template's default
    params: dict[str, Any] = Field(default_factory=dict)  # the template's own settings
    name: str = ""
    model_name: str = ""
    classes: list[int] = Field(default_factory=list)
    unlabelled: str = "exclude"
    val_fraction: float = 0.2
    gpus: int | None = None
    min_vram_mb: int = 0


@router.get("/projects/{project_id}/runs")
def list_runs(
    project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    return training.list_runs(db, hub.settings, project_id)


@router.post("/runs")
def start_run(
    body: NewRun, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    run = training.start_run(db, user, via="ui", **body.model_dump())
    db.commit()
    return training.run_view(db, hub.settings, run)


@router.get("/runs/{run_id}")
def get_run(
    run_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return training.run_view(db, hub.settings, training.get_run(db, run_id), detail=True)


@router.post("/runs/{run_id}/rerun")
def rerun(
    run_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    run = training.rerun(db, user, run_id, via="ui")
    db.commit()
    return training.run_view(db, hub.settings, run)


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    training.delete_run(db, user, run_id)
    db.commit()
    return {"ok": True}


# ---- models --------------------------------------------------------------------------


class AliasRequest(BaseModel):
    alias: str = ""


@router.get("/projects/{project_id}/models")
def list_models(
    project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    return training.list_models(db, hub.settings, project_id)


@router.get("/models/{model_id}")
def get_model(
    model_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return training.model_view(db, hub.settings, training.get_model(db, model_id), detail=True)


@router.post("/models/{model_id}/alias")
def set_alias(
    model_id: str,
    body: AliasRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    model = training.set_alias(db, user, model_id, body.alias, via="ui")
    db.commit()
    return training.model_view(db, hub.settings, model, detail=True)


@router.delete("/models/{model_id}")
def delete_model(model_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    training.delete_model(db, user, model_id)
    db.commit()
    return {"ok": True}


# ---- node protocol -------------------------------------------------------------------


@agent_router.get("/manifest")
def manifest(
    job_id: str, after: int = 0, limit: int = 2000, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    return training.manifest(db, node, job_id, after, limit)


@agent_router.get("/template")
def template_bundle(
    job_id: str, node: Node = Depends(current_node), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> Response:
    """The code of the run's template, as the zip it was registered with."""
    blob = training.template_bundle(db, node, job_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(blob):
        raise HTTPException(status_code=404, detail="템플릿 묶음이 저장소에 없습니다.")
    return Response(storage.get_blob(blob), media_type="application/zip")


@agent_router.post("/report")
def report(
    job_id: str, body: TrainReport, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = training.report(db, node, job_id, body)
    db.commit()
    return result


@agent_router.put("/artifact")
async def artifact(job_id: str, request: Request, name: str = "best.pt", hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Receive the weights as a raw request body, streamed to disk, and store them by content hash."""
    token = bearer_token(request)

    def check() -> None:
        with hub.db.session() as db:
            node = services.authenticate_node(db, token)
            if node is None:
                raise HTTPException(status_code=401, detail="노드 인증 실패")
            training._running(db, node, job_id)

    await anyio.to_thread.run_sync(check)
    incoming = hub.settings.storage_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=incoming, suffix=".part")
    size = 0
    try:
        with os.fdopen(handle, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_WEIGHTS_BYTES:
                    raise HTTPException(status_code=413, detail="가중치 파일이 너무 큽니다.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        storage = catalog.storage_for(str(hub.settings.storage_dir))
        blob = await anyio.to_thread.run_sync(storage.put_blob_file, temp)
    finally:
        Path(temp).unlink(missing_ok=True)
    return {"blob": blob, "bytes": size, "file_name": Path(name).name}


@agent_router.post("/finish")
def finish(
    job_id: str, body: TrainFinish, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = training.finish(db, node, job_id, body)
    db.commit()
    return result


# ---- weights download (signed URL, like images) ---------------------------------------


@media_router.get("/model/{model_id}")
def download_model(
    model_id: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> StreamingResponse:
    if not catalog.verify_media(hub.settings, "model", model_id, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    model = training.get_model(db, model_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(model.blob):
        raise HTTPException(status_code=404, detail="가중치 파일이 저장소에 없습니다.")
    suffix = Path(model.file_name).suffix or ".pt"

    def stream():
        with storage.open(storage.blob_key(model.blob)) as source:
            while chunk := source.read(_CHUNK):
                yield chunk

    # Model names may be Korean; headers are latin-1, so the real name goes in the RFC 5987 form.
    file_name = quote(f"{model.name}-v{model.version}{suffix}")
    return StreamingResponse(
        stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=\"model{suffix}\"; filename*=UTF-8''{file_name}",
            "Content-Length": str(storage.blob_size(model.blob)),
        },
    )
