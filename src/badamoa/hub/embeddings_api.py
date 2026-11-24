"""Image embeddings (이미지 탐색): how far a version's images have vectors, and the node's side of making them."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from . import catalog, embeddings
from .deps import Hub, current_node, current_user, db_session, get_hub
from .models import Node, User
from .services import ServiceError

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/embed")


@router.get("/releases/{release_id}/embeddings")
def release_embeddings(
    release_id: str, model: str = "", _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """How many of the version's images have a vector by ``model`` (the default one) already, and the models."""
    catalog.get_release(db, release_id)
    return {**embeddings.coverage(db, release_id, model), "models": embeddings.models_view()}


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
    stored = embeddings.ingest(db, catalog.storage_for(str(hub.settings.storage_dir)), node, job_id, data)
    db.commit()
    return stored
