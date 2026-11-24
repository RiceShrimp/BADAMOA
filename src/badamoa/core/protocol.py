"""Wire models for the Node Agent ↔ Hub protocol (``/api/agent/*``).

Nodes always connect outbound to the Hub, so remote machines behind NAT or a
firewall work without opening inbound ports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

NodeRole = Literal["compute", "autolabel", "inference", "storage", "embedding"]
ALL_ROLES: list[str] = ["compute", "autolabel", "inference", "storage", "embedding"]
# What each role is called on screen. compute: training runs here. autolabel: a labeling set's images are
# auto-labelled on the server holding them only when it has this role. Templates' environments go to servers
# with either. storage: datasets are imported only there. embedding: image vectors for 이미지 탐색 are made there
# (with a training on the copy it received). inference: shown on the server, nothing is sent by it yet.
ROLE_LABELS: dict[str, str] = {
    "compute": "학습용",
    "autolabel": "오토라벨용",
    "inference": "추론용",
    "storage": "데이터 서버용",
    "embedding": "임베딩용",
}

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
    # What BADAMOA keeps on the node, by kind (``node.storage``): sent after each measurement, not every beat
    storage: dict[str, Any] | None = None


class HeartbeatResponse(BaseModel):
    cancel_job_ids: list[str] = Field(default_factory=list)
    hub_version: str
    # Work folders older than this many days that no job is using go (0: keep them).
    workdir_days: int = 0
    # Where to keep images copied from other nodes, as an admin set it ("" = the node chooses).
    cache_dir: str = ""
    # Where to keep environments, work folders and fetched files, as an admin set it ("" = the node chooses).
    work_dir: str = ""
    # The Hub moved to another PC: the node switches there once that Hub knows it ("" = stay).
    hub_url: str = ""


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
    keypoints: dict[str, Any] | None = None  # pose data: {"count", "names", "skeleton"}
    outlines: bool = False  # some labels carry an outline (segmentation data)


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
    mtime_ns: int = 0  # the file's modification time (0: not known), to know it unchanged on a later import


class ImportSame(BaseModel):
    """Files of an import's folder ([path in the folder, size, modification time ns]): which the project holds
    unchanged already."""

    files: list[list[Any]] = Field(default_factory=list)


class ImportChunk(BaseModel):
    items: list[ImportItem]
    skipped: int = 0  # files in this stretch that were not readable images


class ImportProgress(BaseModel):
    """How far an import got before its version exists: "find" (images found so far), "labels" (label files
    read of ``total``) or "same" (files looked up of ``total``: which the project holds unchanged). Reading and
    sending the images (the last step) the Hub counts from what it is sent."""

    step: str = Field(max_length=20)
    done: int = 0
    total: int = 0


class TrainReport(BaseModel):
    """Progress of a training job: ``stage`` (what it is doing now), ``export`` (dataset written),
    ``info`` (library versions, device), ``epoch`` (metrics)."""

    event: Literal["stage", "export", "info", "epoch"]
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
