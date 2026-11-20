from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, utcnow


def _uuid() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="member")  # admin | member
    # A member an admin lets upload model kinds (as drafts, with proof they ran on their own PC)
    template_upload: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # A stopped account cannot sign in and its tokens stop working; nothing it made is removed.
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # The order this person dragged cards and lists into, per screen: {"review.projects": [id, …], …}
    orders: Mapped[dict[str, list[str]]] = mapped_column(JSON, default=dict)


class ApiToken(Base):
    """Bearer tokens for humans: browser sessions and long-lived API/MCP tokens."""

    __tablename__ = "api_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # session | api
    name: Mapped[str] = mapped_column(String(128), default="")
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prefix: Mapped[str] = mapped_column(String(16))  # first chars, for display
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship()


class JoinToken(Base):
    __tablename__ = "join_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    roles: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    used_by_node_id: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    roles: Mapped[list[str]] = mapped_column(JSON, default=list)
    labels: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    os: Mapped[str] = mapped_column(String(16), default="")
    info: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    agent_version: Mapped[str] = mapped_column(String(32), default="")
    is_local: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # GPU indices kept for people: BADAMOA never places a job on them
    reserved_gpus: Mapped[list[int]] = mapped_column(JSON, default=list)
    # Where images copied from other nodes are kept ("" = the node picks the disk with the most room).
    cache_dir: Mapped[str] = mapped_column(String(512), default="")
    # The last check of its received copies against the Hub's records (``node_data``): when, what it
    # found that no version needs, what it removed, or why it failed.
    data_check: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # What BADAMOA keeps there, by kind, as the node last measured it, and what the last clean did
    # (``node_storage``).
    storage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TransferGrant(Base):
    """Leave for one node to take a version's images straight from the node that holds them, while a job runs."""

    __tablename__ = "transfer_grants"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    release_id: Mapped[str] = mapped_column(String(32))
    source_node_id: Mapped[str] = mapped_column(String(32))
    dest_node_id: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class NodeRelease(Base):
    """A version whose images a node received in full: a copy in its cache (``node_data``)."""

    __tablename__ = "node_releases"

    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), primary_key=True)
    release_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    items: Mapped[int] = mapped_column(Integer, default=0)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # "" kept · "remove": goes once nothing on the node uses it · "held": a run asked to remove it but did
    # not succeed, so it stays for continuing · "removing": the check job ``job_id`` is taking it away
    state: Mapped[str] = mapped_column(String(12), default="")
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # What the node's last check found of it (None: not checked since it arrived)
    present_items: Mapped[int | None] = mapped_column(Integer, nullable=True)
    present_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    type: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    # Placement: a specific node, or any node that satisfies the requirements.
    target_node_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    requirements: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    node_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_via: Mapped[str] = mapped_column(String(32), default="api")  # api | ui | mcp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_count: Mapped[int] = mapped_column(Integer, default=0)
    # GPUs (indices on the node) reserved for this job while it runs.
    gpu_indices: Mapped[list[int]] = mapped_column(JSON, default=list)


class JobLog(Base):
    __tablename__ = "job_logs"
    __table_args__ = (Index("ix_job_logs_job_seq", "job_id", "seq", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)
    ts: Mapped[datetime] = mapped_column(DateTime)
    stream: Mapped[str] = mapped_column(String(8), default="stdout")
    text: Mapped[str] = mapped_column(Text)


class NodeMetric(Base):
    """Rolled-up node metrics. One row per (node, resolution, bucket start).

    ``data`` maps a series key (e.g. ``cpu``, ``gpu0_util``) to ``[sum, count, max]``
    so buckets can be extended as samples arrive and averaged when read.
    """

    __tablename__ = "node_metrics"
    __table_args__ = (Index("ix_node_metrics_key", "node_id", "res", "ts", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"))
    res: Mapped[str] = mapped_column(String(4))  # 1m | 10m | 1h
    ts: Mapped[datetime] = mapped_column(DateTime)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class GpuSpan(Base):
    """A stretch of time a person's processes held a GPU (BADAMOA's own jobs are known from the jobs table).

    Written as heartbeats arrive: the open span of the same people is stretched, anything else
    starts a new one. Kept as long as the coarsest resource history.
    """

    __tablename__ = "gpu_spans"
    __table_args__ = (Index("ix_gpu_spans_node_end", "node_id", "ended_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(32))
    gpu_index: Mapped[int] = mapped_column(Integer)
    who: Mapped[str] = mapped_column(String(255), default="")  # user names, "" when the processes' owner is unknown
    programs: Mapped[str] = mapped_column(String(255), default="")  # process names
    memory_mb: Mapped[int] = mapped_column(Integer, default=0)  # the most it held
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime] = mapped_column(DateTime)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    via: Mapped[str] = mapped_column(String(32), default="api")
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class HubSetting(Base):
    """A Hub-wide setting an admin changes in the screen (e.g. how long outputs are kept)."""

    __tablename__ = "hub_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

