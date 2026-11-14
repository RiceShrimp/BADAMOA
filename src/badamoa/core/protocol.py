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
    gpu_indices: list[int] = Field(default_factory=list)


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


class ImportBegin(BaseModel):
    """First call of a dataset import: the class names, in id order."""

    classes: list[str]
    total: int = 0
    format: str = ""


class ImportItem(BaseModel):
    sha256: str
    file_name: str
    rel_path: str  # relative to the import root on the node
    width: int
    height: int
    bytes: int
    split: str = ""  # train | val | test | ""
    state: Literal["labelled", "background", "unlabelled"]
    # [class_id, cx, cy, w, h] normalised to 0..1 (YOLO convention)
    boxes: list[list[float]] = Field(default_factory=list)
    thumb_b64: str | None = None


class ImportChunk(BaseModel):
    items: list[ImportItem]


class TrainReport(BaseModel):
    """Progress of a training job: ``export`` (dataset written), ``info`` (environment), ``epoch`` (metrics)."""

    event: Literal["export", "info", "epoch"]
    data: dict[str, Any] = Field(default_factory=dict)


class TrainFinish(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    # Per class: {"id", "name", "precision", "recall", "map50", "map50_95"}
    classes: list[dict[str, Any]] = Field(default_factory=list)
    # The uploaded weights: {"blob", "bytes", "file_name"}
    weights: dict[str, Any] = Field(default_factory=dict)


class CompleteRequest(BaseModel):
    status: Literal["succeeded", "failed", "canceled"]
    result: dict[str, Any] | None = None
    error: str | None = None
