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

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
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
    __table_args__ = (
        UniqueConstraint("project_id", "sha256"),
        Index("ix_items_project_name", "project_id", "file_name"),
        Index("ix_items_project_path", "project_id", "rel_path"),  # a later import finds unchanged files by path
    )

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
    # the file's modification time when registered (ns; 0 not known): with the path and size, a later import of a
    # copy (rsync -a, cp -p) or the same folder knows the file unchanged and does not read it again
    mtime_ns: Mapped[int] = mapped_column(Integer, default=0)
    thumb_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The thumbnail's size on the Hub: 0 not known yet (filled in the background), -1 the file is gone
    thumb_bytes: Mapped[int] = mapped_column(Integer, default=0)
    # 1: the thumbnail is of today's kind (256 px WebP); 0: an older one the Hub shrinks in the background
    thumb_v: Mapped[int] = mapped_column(Integer, default=0)
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
    # import:<batch> | human:<name> (fixed on screen) | classes:<version> | autolabel:<model>
    source: Mapped[str] = mapped_column(String(255), default="")
    # labelled | background (looked at, nothing to find) | unlabelled (nobody has looked yet)
    state: Mapped[str] = mapped_column(String(12))
    box_count: Mapped[int] = mapped_column(Integer, default=0)
    boxes: Mapped[list[list[float]]] = mapped_column(JSON, default=list)
    # Classes that were labelled exhaustively: a missing box of these means "not there",
    # of any other class it means "nobody looked".
    labelled_class_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    digest: Mapped[str] = mapped_column(String(40))
    batch_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # the revision a person fixed
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
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
    # building | ready | draft: the dataset's work in progress, fixed labels not yet made a version (one per dataset)
    status: Mapped[str] = mapped_column(String(12), default="building")
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # A fingerprint of the content, like a commit hash (``catalog.release_digest``); "" until it is ready
    digest: Mapped[str] = mapped_column(String(64), default="")
    # The statistics tab's image sizes and box spread, kept once computed (``overview.release_charts``):
    # {"key": what they were computed for, "sizes": …, "boxes": …}
    charts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # 검수 when a work in progress became this version: problems before and after, what was fixed or let be
    review: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
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


class EmbeddingPart(Base):
    """Image vectors a node made with a model and sent in one file (``blob`` in the Hub's storage): a JSON header
    line ``{"model", "dim", "shas"}``, then a float16 vector a row, L2-normalised, in ``shas``' order."""

    __tablename__ = "embedding_parts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model: Mapped[str] = mapped_column(String(64), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    count: Mapped[int] = mapped_column(Integer)
    blob: Mapped[str] = mapped_column(String(64))
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    release_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ImageEmbedding(Base):
    """An image's vector by a model, found by the image's content (the same picture in any version, dataset or
    project has one): which part file holds it and at which row."""

    __tablename__ = "image_embeddings"

    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    part_id: Mapped[str] = mapped_column(ForeignKey("embedding_parts.id", ondelete="CASCADE"), index=True)
    row: Mapped[int] = mapped_column(Integer)


class ItemPrediction(Base):
    """What a model predicted for one image in one evaluation (``model_jobs.id``), and how it compared with the
    labels then: found, missed (labels it did not find) and extra (predictions no label backs). Many mistakes
    on an image often mean its labels are wrong; the dataset screen sorts by them.

    ``labels`` are in the evaluated release's label space: ``[[class_id, cx, cy, w, h, …]]`` like a revision.
    """

    __tablename__ = "item_predictions"
    __table_args__ = (
        Index("ix_item_predictions_wrong", "op_id", "wrong", "item_id"),
        Index("ix_item_predictions_item", "item_id"),  # an image deleted: its rows found at once
    )

    op_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), primary_key=True)
    found: Mapped[int] = mapped_column(Integer, default=0)
    missed: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[int] = mapped_column(Integer, default=0)
    reclass: Mapped[int] = mapped_column(Integer, default=0)  # labels whose class the model is sure is another
    wrong: Mapped[int] = mapped_column(Integer, default=0)  # missed + extra + reclass
    # The same before any threshold (``checks.judge``): the surest prediction no label has, the surest of another
    # class than its label, the biggest label nothing touches (a share of the image); 0 when none. Counting at any
    # threshold is then one query (검수's 확신도 · 최소 크기)
    extra_top: Mapped[float] = mapped_column(Float, default=0.0)
    reclass_top: Mapped[float] = mapped_column(Float, default=0.0)
    missed_area: Mapped[float] = mapped_column(Float, default=0.0)
    # the least the nearest prediction overlapped any label paired with none (IoU; 1 when every label is paired):
    # 물체가 없을 수 있는 라벨 at any IoU threshold is one query
    missed_near: Mapped[float] = mapped_column(Float, default=1.0)
    labels: Mapped[list[list[float]]] = mapped_column(JSON, default=list)
    scores: Mapped[list[float]] = mapped_column(JSON, default=list)


class LabelSet(Base):
    """A labeling set: images of a dataset picked to be labelled by people, split into parts of ``part_size``.

    The labels themselves are saved like any fix: as revisions in the dataset's work in progress, which
    becomes a version when confirmed. The set keeps who does which part and how far each image got.
    """

    __tablename__ = "label_sets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    release_id: Mapped[str] = mapped_column(String(32))  # the version its images were picked from
    name: Mapped[str] = mapped_column(String(128))
    instructions: Mapped[str] = mapped_column(Text, default="")
    part_size: Mapped[int] = mapped_column(Integer, default=50)
    review: Mapped[bool] = mapped_column(Boolean, default=True)  # every image is checked by a reviewer
    # the last 오토라벨: op id, model, confidences, how many images it was asked for, what came of it
    autolabel: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class LabelSetPart(Base):
    """One part of a set: who labels it and who reviews it."""

    __tablename__ = "label_set_parts"

    set_id: Mapped[str] = mapped_column(ForeignKey("label_sets.id", ondelete="CASCADE"), primary_key=True)
    part: Mapped[int] = mapped_column(Integer, primary_key=True)
    assignee: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(32), nullable=True)


class LabelSetItem(Base):
    """An image of a set: its part and place, and how far it got (todo → labelled → approved, or rejected)."""

    __tablename__ = "label_set_items"
    __table_args__ = (
        Index("ix_label_set_items_part", "set_id", "part", "position"),
        Index("ix_label_set_items_item", "item_id"),
    )

    set_id: Mapped[str] = mapped_column(ForeignKey("label_sets.id", ondelete="CASCADE"), primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), primary_key=True)
    part: Mapped[int] = mapped_column(Integer)
    position: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(12), default="todo")  # todo | labelled | approved | rejected
    note: Mapped[str] = mapped_column(Text, default="")  # why a reviewer sent it back
    labelled_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    labelled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class QaRun(Base):
    """The label checks of a release were run on these labels (``fingerprint``: its items and revisions)."""

    __tablename__ = "qa_runs"

    release_id: Mapped[str] = mapped_column(ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    counts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class QaResult(Base):
    """An image a label check flagged: which of its labels (indices in the release's revision) and why."""

    __tablename__ = "qa_results"
    __table_args__ = (
        Index("ix_qa_results_check", "release_id", "check", "item_id"),
        Index("ix_qa_results_item", "item_id"),
    )

    release_id: Mapped[str] = mapped_column(ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True)
    check: Mapped[str] = mapped_column(String(24), primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), primary_key=True)
    boxes: Mapped[list[int]] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(String(255), default="")


class QaScore(Base):
    """A box check a threshold decides (``qa.DEFAULTS``), before any threshold: the image's value (the most two
    boxes overlap, the shortest side in pixels, the biggest share of the image) and each label's, from a loose bound
    (``qa.LOOSE``). 검수 counts at whatever threshold it asks for from these; ``QaResult`` keeps the defaults."""

    __tablename__ = "qa_scores"
    __table_args__ = (
        Index("ix_qa_scores_check", "release_id", "check", "score"),
        Index("ix_qa_scores_item", "item_id"),
    )

    release_id: Mapped[str] = mapped_column(ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True)
    check: Mapped[str] = mapped_column(String(24), primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    labels: Mapped[list[list[float]]] = mapped_column(JSON, default=list)  # [[label index, value, partner or -1]]


class QaDismissal(Base):
    """Someone looked at what a check flagged and found the labels right ("문제 없음").

    Kept for the labels they looked at (``revision_id``): a later version with the same labels for the image
    keeps the judgement; once the labels change, the check counts it again. ``check`` is a rule check's id or
    ``extra`` / ``missed`` (what a model's evaluation found).
    """

    __tablename__ = "qa_dismissals"
    __table_args__ = (Index("ix_qa_dismissals_dataset", "dataset_id", "check"),)

    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), primary_key=True)
    check: Mapped[str] = mapped_column(String(24), primary_key=True)
    revision_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    note: Mapped[str] = mapped_column(String(255), default="")
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
