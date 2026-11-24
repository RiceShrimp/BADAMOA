"""Image embeddings and 이미지 탐색: how far a version's images have vectors, searching a project's images by words or
by another image, and the node's side of making them (the vectors, and once the text encoder)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import catalog, embeddings, services
from .deps import Hub, bearer_token, current_node, current_user, db_session, get_hub
from .models import Node, User
from .services import ServiceError

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/embed")
MAX_TEXT_FILE = 2 * 1024 * 1024 * 1024  # the text encoder, before it is made smaller (about 0.8 GB)


def _storage(hub: Hub):
    return catalog.storage_for(str(hub.settings.storage_dir))


@router.get("/releases/{release_id}/embeddings")
def release_embeddings(
    release_id: str, model: str = "", _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """How many of the version's images have a vector by ``model`` (the default one) already, and the models."""
    catalog.get_release(db, release_id)
    return {**embeddings.coverage(db, release_id, model), "models": embeddings.models_view()}


@router.get("/projects/{project_id}/explore")
def explore_status(
    project_id: str, model: str = "", _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """How much of the project 이미지 탐색 can find, and whether 글 검색 is ready."""
    return embeddings.explore_status(db, project_id, model)


@router.get("/projects/{project_id}/explore/search")
def explore_search(
    project_id: str,
    text: str = "",
    like: int = 0,
    release_id: str = "",
    limit: int = 60,
    model: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """The project's images (or one version's) nearest to the words or to the image ``like``, the nearest first."""
    return embeddings.search(
        db, hub.settings, _storage(hub), project_id, text=text, like=like, release_id=release_id, limit=limit, model=model
    )


@router.get("/projects/{project_id}/explore/versions")
def explore_versions(
    project_id: str, model: str = "", _: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    """Every version of the project with how far its images have vectors."""
    return embeddings.project_versions(db, project_id, model)


@router.get("/releases/{release_id}/embeddings/places")
def embedding_places(
    release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """The servers the version's vectors can be made on, each with why not."""
    return embeddings.places(db, hub.settings, catalog.get_release(db, release_id))


class StartEmbedding(BaseModel):
    node_id: str = ""  # empty: where the images are, on a server with the 임베딩용 role
    gpus: int | None = None  # None: a GPU when the server has one; 0: the CPU


@router.post("/releases/{release_id}/embeddings")
def start_embedding(
    release_id: str,
    body: StartEmbedding,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """Make the version's missing vectors without a training (임베딩 만들기)."""
    job = embeddings.start_alone(db, hub.settings, user, release_id, body.node_id, via="ui", gpus=body.gpus)
    db.commit()
    return {"job_id": job.id, **embeddings.coverage(db, release_id)}


@agent_router.get("/manifest")
def manifest(
    job_id: str, after: int = 0, limit: int = 2000, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    return embeddings.manifest(db, node, job_id, after, limit)


@agent_router.put("/part")
async def put_part(
    job_id: str,
    request: Request,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, int]:
    data = await request.body()
    if len(data) > embeddings.MAX_PART:
        raise ServiceError(413, "임베딩 파일이 너무 큽니다. 더 작게 나눠 보내세요.")
    stored = embeddings.ingest(db, _storage(hub), node, job_id, data)
    db.commit()
    return stored


@agent_router.put("/text")
async def put_text(job_id: str, request: Request, kind: str, hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """A file of the model's text encoder (``onnx``, ``tokenizer``, ``checks``), streamed to disk: 글 검색 on the Hub."""
    token = bearer_token(request)
    incoming = hub.settings.storage_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=incoming, suffix=".part")
    size = 0
    try:
        with os.fdopen(handle, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_TEXT_FILE:
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
            found = embeddings.text_file(db, hub.settings, _storage(hub), node, job_id, kind, blob)
            db.commit()
            return found

    return await anyio.to_thread.run_sync(record)
