"""Training tables: runs, their metric history, and the model versions they produce.

A run pins everything needed to repeat it (release, trainer, settings) and
points at the job that executes it; its status is the job's status. A model
version points back at its run, so lineage is model → run → release → items.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, utcnow


def _uuid() -> str:
    return uuid.uuid4().hex


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    release_id: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    trainer: Mapped[str] = mapped_column(String(32))  # yolo | dummy
    # What the user asked for; enough to start the same run again.
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    epochs: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[int] = mapped_column(Integer, default=0)  # last epoch reported
    export: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # what the node wrote as training data
    info: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # library versions, device
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # final metrics of the best weights
    class_metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    source_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)  # the run this one repeats
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RunMetric(Base):
    """Metrics of one epoch: ``{"map50_95": 0.41, "train_box_loss": 1.2, …}``."""

    __tablename__ = "run_metrics"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    step: Mapped[int] = mapped_column(Integer, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ModelVersion(Base):
    """One trained model. Versions of the same ``name`` form a family; an alias marks one of them."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("project_id", "name", "version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer)
    alias: Mapped[str] = mapped_column(String(16), default="")  # "" | champion | challenger
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    release_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    task: Mapped[str] = mapped_column(String(16), default="detect")
    arch: Mapped[str] = mapped_column(String(64), default="")  # e.g. yolo11n
    trainer: Mapped[str] = mapped_column(String(32), default="")
    blob: Mapped[str] = mapped_column(String(64))  # sha256 of the weights in Hub storage
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    file_name: Mapped[str] = mapped_column(String(128), default="")
    imgsz: Mapped[int] = mapped_column(Integer, default=0)
    classes: Mapped[list[str]] = mapped_column(JSON, default=list)  # names in the model's own order
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    class_metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
