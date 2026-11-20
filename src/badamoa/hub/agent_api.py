"""Node Agent protocol endpoints (``/api/agent/*``). Nodes always call in; the Hub never dials out."""

from __future__ import annotations

import time
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from badamoa.core.protocol import (
    ClaimedJob,
    ClaimRequest,
    ClaimResponse,
    CompleteRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    LogAck,
    LogBatch,
    RegisterRequest,
    RegisterResponse,
)

from . import (
    asks,
    catalog,
    cleanup,
    migration,
    node_data,
    node_storage,
    originals,
    registry,
    services,
    sweeps,
    template_checks,
    transfers,
)
from .db import utcnow
from .deps import Hub, bearer_token, current_node, db_session, get_hub
from .models import Node

router = APIRouter(prefix="/api/agent")

_CLAIM_POLL_S = 0.5


@router.post("/register", response_model=RegisterResponse)
def register(body: RegisterRequest, db: Session = Depends(db_session)) -> RegisterResponse:
    node, token = services.register_node(db, body)
    db.commit()
    return RegisterResponse(node_id=node.id, node_token=token, roles=node.roles)


@router.post("/heartbeat", response_model=HeartbeatResponse)
def heartbeat(
    body: HeartbeatRequest, request: Request, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> HeartbeatResponse:
    seen = request.client.host if request.client else ""
    cancel_ids = services.heartbeat(db, node, body, seen_from=seen)
    if body.storage:
        node_storage.save(node, body.storage)
    db.commit()
    days = cleanup.get_policy(db)["workdir_days"]
    return HeartbeatResponse(
        cancel_job_ids=cancel_ids,
        hub_version=services.HUB_VERSION,
        workdir_days=days,
        cache_dir=node.cache_dir or "",
        work_dir=node.work_dir or "",
        hub_url=migration.moved_to(db),
    )


# ---- images from node to node (the Hub only arranges it) -------------------------------------


@router.get("/jobs/{job_id}/sync/plan")
def sync_plan(
    job_id: str,
    after: int = 0,
    limit: int = transfers.PLAN_MAX,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
) -> dict[str, Any]:
    page = transfers.plan(db, node, job_id, after, limit)
    db.commit()
    return page


class SyncProgress(BaseModel):
    progress: dict[str, Any]


@router.post("/jobs/{job_id}/sync/progress")
def sync_progress(
    job_id: str, body: SyncProgress, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, bool]:
    transfers.report(db, node, job_id, body.progress)
    db.commit()
    return {"ok": True}


class Resolve(BaseModel):
    grant: str
    ids: list[int] = Field(default_factory=list)


@router.post("/transfers/resolve")
def resolve_transfer(body: Resolve, node: Node = Depends(current_node), db: Session = Depends(db_session)) -> dict:
    """The holding node asks which of the asked-for images a grant lets it send, and where they lie."""
    return transfers.resolve(db, node, body.grant, body.ids)


@router.get("/jobs/{job_id}/check/expected")
def check_expected(
    job_id: str,
    release: str = "",
    after: int = 0,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
) -> dict[str, Any]:
    """A checking node asks which copies it should hold, then the images of each, page by page."""
    return node_data.expected(db, node, job_id, release, after)


class QuestionsWait(BaseModel):
    wait_seconds: float = Field(default=20.0, ge=0, le=60)


class Answer(BaseModel):
    result: Any = None
    error: str | None = None


@router.post("/requests")
async def questions(body: QuestionsWait, node: Node = Depends(current_node)) -> dict[str, Any]:
    """Long-poll: the questions the Hub has for this node (a folder listing …), as soon as there are any."""
    deadline = time.monotonic() + body.wait_seconds
    while True:
        waiting = asks.take(node.id)
        if waiting or time.monotonic() >= deadline:
            return {"requests": waiting}
        await anyio.sleep(0.2)


@router.put("/content/{key}")
async def upload_content(
    key: str, request: Request, _: Node = Depends(current_node), hub: Hub = Depends(get_hub)
) -> dict[str, int]:
    """A file a question asked for (an original image), sent before the answer."""
    content = await request.body()
    await anyio.to_thread.run_sync(originals.store, hub.settings, key, content)
    return {"bytes": len(content)}


@router.post("/requests/{question_id}")
def answer(question_id: str, body: Answer, _: Node = Depends(current_node)) -> dict[str, bool]:
    return {"taken": asks.answer(question_id, body.model_dump())}


def _claim_once(hub: Hub, token: str) -> dict[str, Any] | None | bool:
    """Returns False on auth failure, None when nothing to do, else the claimed job."""
    with hub.db.session() as db:
        node = services.authenticate_node(db, token)
        if node is None:
            return False
        node.last_seen_at = utcnow()
        if migration.read_only(hub):  # being moved: a job's results would not reach the new Hub
            db.commit()
            return None
        job = services.try_claim(db, node, hub.settings)
        db.commit()
        if job is None:
            return None
        return ClaimedJob(
            id=job.id, type=job.type, params=job.params, created_at=job.created_at, gpu_indices=job.gpu_indices or []
        ).model_dump()


@router.post("/claim", response_model=ClaimResponse)
async def claim(body: ClaimRequest, request: Request, hub: Hub = Depends(get_hub)) -> ClaimResponse:
    """Long-poll: hold the request open until a job is available or ``wait_seconds`` elapse."""
    token = bearer_token(request)
    deadline = time.monotonic() + body.wait_seconds
    while True:
        result = await anyio.to_thread.run_sync(_claim_once, hub, token)
        if result is False:
            raise HTTPException(status_code=401, detail="노드 인증 실패")
        if result:
            return ClaimResponse(job=ClaimedJob(**result))
        if time.monotonic() >= deadline or await request.is_disconnected():
            return ClaimResponse(job=None)
        await anyio.sleep(_CLAIM_POLL_S)


@router.post("/jobs/{job_id}/logs", response_model=LogAck)
def post_logs(
    job_id: str, body: LogBatch, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> LogAck:
    cancel = services.append_logs(db, node, job_id, body.lines)
    db.commit()
    return LogAck(cancel_requested=cancel)


@router.post("/jobs/{job_id}/complete")
def complete(
    job_id: str,
    body: CompleteRequest,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    job = services.complete_job(db, node, job_id, body)
    if job.type == "data.import" and job.status in ("failed", "canceled"):  # stopped or failed: nothing left half-made
        catalog.drop_unfinished_imports(db, hub.settings)
    registry.on_job_finished(db, job)
    transfers.on_job_finished(db, job)
    node_data.on_job_finished(db, job)
    node_storage.on_job_finished(db, job)
    template_checks.on_job_finished(db, job)
    db.commit()
    sweeps.on_job_finished(job)  # a search may start its next run
    return {"status": job.status}
