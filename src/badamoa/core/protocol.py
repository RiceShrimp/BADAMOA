"""Wire models for the Node Agent ↔ Hub protocol (``/api/agent/*``).

Nodes always connect outbound to the Hub, so remote machines behind NAT or a
firewall work without opening inbound ports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

NodeRole = Literal["compute", "storage", "inference"]
ALL_ROLES: list[str] = ["compute", "storage", "inference"]

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


class RegisterRequest(BaseModel):
    join_token: str
    agent_version: str
    info: dict[str, Any]
    name: str | None = None


class RegisterResponse(BaseModel):
    node_id: str
    node_token: str
    roles: list[str]


class HeartbeatRequest(BaseModel):
    agent_version: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    running_job_ids: list[str] = Field(default_factory=list)
    info: dict[str, Any] | None = None


class HeartbeatResponse(BaseModel):
    cancel_job_ids: list[str] = Field(default_factory=list)
    hub_version: str


class ClaimRequest(BaseModel):
    wait_seconds: float = Field(default=20.0, ge=0, le=60)


class ClaimedJob(BaseModel):
    id: str
    type: str
    params: dict[str, Any]
    created_at: datetime


class ClaimResponse(BaseModel):
    job: ClaimedJob | None = None


class LogLine(BaseModel):
    ts: datetime
    stream: Literal["stdout", "stderr", "system"] = "stdout"
    text: str


class LogBatch(BaseModel):
    lines: list[LogLine]


class LogAck(BaseModel):
    cancel_requested: bool = False


class CompleteRequest(BaseModel):
    status: Literal["succeeded", "failed", "canceled"]
    result: dict[str, Any] | None = None
    error: str | None = None
