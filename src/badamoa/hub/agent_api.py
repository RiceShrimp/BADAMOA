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

from . import asks, cleanup, registry, services, sweeps
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
    body: HeartbeatRequest, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> HeartbeatResponse:
    cancel_ids = services.heartbeat(db, node, body)
    db.commit()
    days = cleanup.get_policy(db)["workdir_days"]
    return HeartbeatResponse(cancel_job_ids=cancel_ids, hub_version=services.HUB_VERSION, workdir_days=days)


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
    job_id: str, body: CompleteRequest, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    job = services.complete_job(db, node, job_id, body)
    registry.on_job_finished(db, job)
    db.commit()
    sweeps.on_job_finished(job)  # a search may start its next run
    return {"status": job.status}
