"""Dataset catalog tables (see docs/SCREEN_SPECS.md, section 2).

Project → label spaces, items (images identified by sha256), label revisions
(labels are never overwritten; each import or edit adds a revision), and
dataset families whose immutable releases point at one revision per item.

Kept in SQLite for now: filters and paging over a few hundred thousand items
are index lookups. A columnar store comes in when embeddings do (Phase 2).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, utcnow


def _uuid() -> str:
    return uuid.uuid4().hex


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    # Before 0.34.0 a project had one task; it is now each dataset's (``Dataset.task``) and this stays empty.
    task: Mapped[str] = mapped_column(String(16), default="")
    folder: Mapped[str] = mapped_column(String(128), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # dialog defaults, e.g. {"template": "yolo"}
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LabelSpace(Base):
    """A versioned set of classes, e.g. ``od11``. ``classes`` is ``[{"id": 0, "name": "person"}, …]``."""

    __tablename__ = "label_spaces"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    classes: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class StorageLocation(Base):
    """Where image files live: a folder on a node. Files are registered in place, never copied."""

    __tablename__ = "storage_locations"
    __table_args__ = (UniqueConstraint("node_id", "root"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    node_id: Mapped[str] = mapped_column(String(32), index=True)
    root: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Batch(Base):
    """One import. Every item and label revision remembers the batch it came from."""

    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(32), default="")  # yolo | coco
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    location_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Item(Base):
    __tablename__ = "items"
    __table_args__ = (UniqueConstraint("project_id", "sha256"), Index("ix_items_project_name", "project_id", "file_name"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    sha256: Mapped[str] = mapped_column(String(64))
    file_name: Mapped[str] = mapped_column(String(512))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    bytes: Mapped[int] = mapped_column(Integer)
    site: Mapped[str] = mapped_column(String(64), default="", index=True)
    camera: Mapped[str] = mapped_column(String(64), default="")
    location_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rel_path: Mapped[str] = mapped_column(String(1024), default="")
    thumb_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class LabelRevision(Base):
    """Labels of one item at one point in time. ``boxes`` is ``[[class_id, cx, cy, w, h], …]`` (0..1)."""

    __tablename__ = "label_revisions"
    __table_args__ = (UniqueConstraint("item_id", "rev"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), index=True)
    rev: Mapped[int] = mapped_column(Integer)
    label_space_id: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(255), default="")  # import:<batch name> | human | autolabel:<model>
    state: Mapped[str] = mapped_column(String(12))  # labelled | background (no labels: nothing to find)
    box_count: Mapped[int] = mapped_column(Integer, default=0)
    boxes: Mapped[list[list[float]]] = mapped_column(JSON, default=list)
    # Classes that were labelled exhaustively: a missing box of these means "not there",
    # of any other class it means "nobody looked".
    labelled_class_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    digest: Mapped[str] = mapped_column(String(40))
    batch_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RevisionClass(Base):
    """Boxes per class in a revision; makes "items containing class X" an index lookup."""

    __tablename__ = "revision_classes"
    __table_args__ = (Index("ix_revision_classes_class", "class_id", "revision_id"),)

    revision_id: Mapped[int] = mapped_column(ForeignKey("label_revisions.id", ondelete="CASCADE"), primary_key=True)
    class_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    count: Mapped[int] = mapped_column(Integer)


class Dataset(Base):
    """A dataset family; its content lives in releases."""

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    # What its labels are for: detect | segment | pose | classify, or a task a team named (badamoa.core.tasks)
    task: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Release(Base):
    __tablename__ = "releases"
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version: Mapped[str] = mapped_column(String(64))
    label_space_id: Mapped[str] = mapped_column(String(32))
    base_release_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(12), default="building")  # building | ready
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    batch_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ReleaseItem(Base):
    """Membership of an item in a release: its split and the label revision the release pins."""

    __tablename__ = "release_items"
    __table_args__ = (
        Index("ix_release_items_split", "release_id", "split", "item_id"),
        Index("ix_release_items_state", "release_id", "state", "item_id"),
        Index("ix_release_items_item", "item_id"),
    )

    release_id: Mapped[str] = mapped_column(ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), primary_key=True)
    split: Mapped[str] = mapped_column(String(8), default="")
    revision_id: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(12))
    box_count: Mapped[int] = mapped_column(Integer, default=0)
