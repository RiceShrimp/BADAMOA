"""Training tables: templates and environments, runs with their metric history, model versions.

A run pins everything needed to repeat it (release, template version, environment,
settings) and points at the job that executes it; its status is the job's status.
A model version points back at its run, so lineage is model → run → (release,
template version, environment).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, utcnow


def _uuid() -> str:
    return uuid.uuid4().hex


class Template(Base):
    """One version of a model template (``badamoa.core.templates``). Built-in ones have no bundle."""

    __tablename__ = "templates"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(128), default="")
    task: Mapped[str] = mapped_column(String(16), default="detect")
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    blob: Mapped[str | None] = mapped_column(String(64), nullable=True)  # sha256 of the zip in Hub storage
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Environment(Base):
    """A named environment an admin prepared on the servers, e.g. an existing conda environment.

    ``nodes`` maps a node id to ``{"path", "ok", "detail", "checked_at", "job_id"}``.
    The automatically installed environment needs no row: it is derived from the template.
    """

    __tablename__ = "environments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(16), default="python")  # python (docker comes later)
    description: Mapped[str] = mapped_column(Text, default="")
    nodes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    release_id: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    trainer: Mapped[str] = mapped_column(String(64))  # template name
    # Snapshots, so the run stays readable after a template or environment is changed or removed.
    template: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # id, name, title, version, sha256, variant
    environment: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # kind, name, requirements | path
    stage: Mapped[str] = mapped_column(String(16), default="")  # data | env | train | upload
    # The latest checkpoint of an unfinished run, kept on the Hub so any server can pick the run up:
    # {"blob", "bytes", "file_name", "epoch"}. Dropped when the run finishes.
    checkpoint: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # What the user asked for; enough to start the same run again.
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    epochs: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[int] = mapped_column(Integer, default=0)  # last epoch reported
    export: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # what the node wrote as training data
    info: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # library versions, device
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # final metrics of the best weights
    class_metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    source_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)  # the run this one repeats
    note: Mapped[str] = mapped_column(Text, default="")  # what the person tried, in their words (W&B notes)
    sweep_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)  # the search it is part of
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Sweep(Base):
    """A hyperparameter search: many runs of one template on one release, settings picked by ``method``."""

    __tablename__ = "sweeps"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    release_id: Mapped[str] = mapped_column(String(32))
    method: Mapped[str] = mapped_column(String(16), default="grid")  # grid | random
    metric: Mapped[str] = mapped_column(String(64))  # what "best" means
    goal: Mapped[str] = mapped_column(String(8), default="maximize")  # maximize | minimize
    # {"lr0": {"values": [0.01, 0.001]}, "momentum": {"min": 0.8, "max": 0.98, "log": false}}
    space: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Everything else every run gets: template id, environment, fixed params, classes, tags …
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    max_runs: Mapped[int] = mapped_column(Integer, default=10)
    parallel: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | finished | stopped
    error: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Tag(Base):
    """A label on a run or a model version, to find and group them (W&B tags, MLflow tags)."""

    __tablename__ = "tags"
    __table_args__ = (UniqueConstraint("kind", "object_id", "name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(8))  # run | model
    object_id: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(48), index=True)


class RunMetric(Base):
    """Metrics of one epoch: ``{"map50_95": 0.41, "train_box_loss": 1.2, …}``."""

    __tablename__ = "run_metrics"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    step: Mapped[int] = mapped_column(Integer, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ModelJob(Base):
    """Something done with a model after training: an evaluation, an export, a test on sample images."""

    __tablename__ = "model_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # eval | export | predict
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    release_id: Mapped[str | None] = mapped_column(String(32), nullable=True)  # what an evaluation measured on
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # metrics, per-class numbers, predictions
    files: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # [{"name", "blob", "bytes", "kind"}]
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RunFile(Base):
    """A file the training code recorded with the run: a plot, sample predictions, a report."""

    __tablename__ = "run_files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(8), default="file")  # image | table | file
    blob: Mapped[str] = mapped_column(String(64))
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    # A per-step record (``bm.log_image(…, key=, step=)``, ``bm.log_table``): shown by key, step by step.
    key: Mapped[str] = mapped_column(String(64), default="")
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caption: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ModelVersion(Base):
    """One trained model. Versions of the same ``name`` form a family; an alias marks one of them."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("project_id", "name", "version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer)
    # Names that point at this version within its family, like MLflow's aliases ("champion", "production").
    # An alias sits on one version of a family at a time; a version can carry several.
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    alias: Mapped[str] = mapped_column(String(32), default="")  # the first of ``aliases``, for older readers
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    release_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    task: Mapped[str] = mapped_column(String(16), default="detect")
    arch: Mapped[str] = mapped_column(String(64), default="")  # e.g. yolo11n
    trainer: Mapped[str] = mapped_column(String(64), default="")
    template: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # the template version that made it
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
