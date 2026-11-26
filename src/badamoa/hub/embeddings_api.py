"""Image embeddings and 이미지 탐색: how far a version's images have vectors, searching a project's images by words or
by another image, and the node's side of making them (the vectors, and once the text encoder)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from . import catalog, embeddings, item_lists, services, vector_store
from .deps import Hub, admin_user, bearer_token, current_node, current_user, db_session, get_hub
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


@router.get("/explore/board")
def explore_board(
    model: str = "", _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """이미지 탐색's first screen: the embedding jobs going now and each project's images and vectors."""
    return embeddings.board(db, hub.settings, model)


@router.get("/embeddings/jobs")
def embedding_jobs(
    project_id: str = "",
    release_id: str = "",
    limit: int = 100,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> list[dict[str, Any]]:
    """임베딩 작업 기록: a project's (or a version's) embedding jobs, newest first."""
    return embeddings.history(db, hub.settings, project_id=project_id, release_id=release_id, limit=limit)


@router.get("/embeddings/jobs/{job_id}")
def embedding_job(
    job_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """One embedding job: what it makes, where, how far."""
    return embeddings.job_detail(db, hub.settings, job_id)


@router.get("/projects/{project_id}/explore")
def explore_status(
    project_id: str,
    model: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """How much of the project 이미지 탐색 can find, and whether 글 검색 is ready."""
    found = embeddings.explore_status(db, hub.settings, project_id, model)
    db.commit()  # images the vector DB took in for the project
    return found


@router.get("/projects/{project_id}/explore/search")
def explore_search(
    project_id: str,
    text: str = "",
    like: int = 0,
    release_id: str = "",
    limit: int = 60,
    model: str = "",
    min_score: float = 0.0,  # only the images at least this near (cosine)
    image: str = "",  # an image put in (``/explore/query-image``'s id)
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """The project's images (or one version's) nearest to the words or to the image ``like``, the nearest first."""
    found = embeddings.search(
        db, hub.settings, _storage(hub), project_id, text=text, like=like, release_id=release_id, limit=limit,
        model=model, min_score=min_score, image=image,
    )  # fmt: skip
    db.commit()
    return found


@router.post("/projects/{project_id}/explore/query-image")
async def explore_query_image(
    project_id: str,
    request: Request,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """An image to find the like of (the request's body): kept for a while by its content; its id goes to the search
    as ``image``."""
    catalog.get_project(db, project_id)
    data = await request.body()
    return await anyio.to_thread.run_sync(embeddings.keep_query_image, hub.settings, data)


@router.get("/projects/{project_id}/explore/search/list")
def explore_search_file(
    project_id: str,
    format: str = "csv",  # noqa: A002  (the query's name)
    text: str = "",
    like: int = 0,
    release_id: str = "",
    limit: int = 60,
    model: str = "",
    min_score: float = 0.0,
    image: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> FileResponse:
    """The same search as a file: csv (a row per image, the nearest first, with its score) or txt (the paths)."""
    fmt = item_lists.check_format(format)
    found = embeddings.search(
        db, hub.settings, _storage(hub), project_id, text=text, like=like, release_id=release_id, limit=limit,
        model=model, min_score=min_score, image=image,
    )  # fmt: skip
    db.commit()
    path, count = item_lists.write(item_lists.search_rows(db, found["items"]), fmt, item_lists.SEARCH_COLUMNS)
    asked = text.strip()[:40] if text.strip() else (f"like-{like}" if like else ("image" if image else ""))
    return FileResponse(
        path,
        media_type="text/csv; charset=utf-8" if fmt == "csv" else "text/plain; charset=utf-8",
        filename=item_lists.file_name("search", asked, fmt=fmt),
        headers={"X-Images": str(count)},
        background=BackgroundTask(os.unlink, path),
    )


@router.get("/vector-store")
def vector_store_view(
    _: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Where the vector DB is, how big, its tables, and a move in progress (관리자)."""
    return vector_store.view(db, hub.settings)


class MoveVectors(BaseModel):
    path: str  # an empty or new folder on the Hub PC


@router.post("/vector-store/move")
def move_vector_store(
    body: MoveVectors, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Move the vector DB to another folder of the Hub PC (관리자): copied in the background, then used from there."""
    found = vector_store.move(db, hub.settings, body.path)
    services.audit(db, user.id, "ui", "vectors.move", "", to=body.path)
    db.commit()
    return found


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
    gpu_indices: list[int] | None = None  # that GPU of the server (one)
    data_after: str = ""  # keep | delete: the copy the server received, once it is over


@router.post("/releases/{release_id}/embeddings")
def start_embedding(
    release_id: str,
    body: StartEmbedding,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """Make the version's missing vectors without a training (임베딩 만들기)."""
    job = embeddings.start_alone(
        db,
        hub.settings,
        user,
        release_id,
        body.node_id,
        via="ui",
        gpus=body.gpus,
        gpu_indices=body.gpu_indices,
        data_after=body.data_after,
    )
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
    stored = embeddings.ingest(db, hub.settings, _storage(hub), node, job_id, data)
    db.commit()
    return stored


@agent_router.put("/vision")
async def put_vision(job_id: str, request: Request, kind: str, hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """A file of the model's image encoder (``onnx``, ``checks``), streamed to disk: 이미지로 찾기 on the Hub."""
    return await _encoder_file(job_id, request, kind, hub, embeddings.vision_file)


@agent_router.put("/text")
async def put_text(job_id: str, request: Request, kind: str, hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """A file of the model's text encoder (``onnx``, ``tokenizer``, ``checks``), streamed to disk: 글 검색 on the Hub."""
    return await _encoder_file(job_id, request, kind, hub, embeddings.text_file)


async def _encoder_file(job_id: str, request: Request, kind: str, hub: Hub, keep: Any) -> dict[str, Any]:
    """A file of an encoder the Hub reads queries with, streamed to disk and stored, then handed to ``keep``
    (``embeddings.text_file`` or ``vision_file``)."""
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
            found = keep(db, hub.settings, _storage(hub), node, job_id, kind, blob)
            db.commit()
            return found

    return await anyio.to_thread.run_sync(record)
