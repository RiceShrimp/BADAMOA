from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# (table, column, DDL) for columns added after 0.1.0.
_ADDED_COLUMNS = [
    ("jobs", "gpu_indices", "JSON DEFAULT '[]'"),
    ("runs", "template", "JSON DEFAULT '{}'"),
    ("runs", "environment", "JSON DEFAULT '{}'"),
    ("runs", "stage", "VARCHAR(16) DEFAULT ''"),
    ("runs", "checkpoint", "JSON DEFAULT '{}'"),
    ("model_versions", "template", "JSON DEFAULT '{}'"),
    ("projects", "settings", "JSON DEFAULT '{}'"),
    ("runs", "note", "TEXT DEFAULT ''"),
    ("model_versions", "aliases", "JSON DEFAULT '[]'"),
    ("runs", "sweep_id", "VARCHAR(32)"),
    ("run_files", "key", "VARCHAR(64) DEFAULT ''"),
    ("run_files", "step", "INTEGER"),
    ("run_files", "caption", "VARCHAR(255) DEFAULT ''"),
    ("nodes", "reserved_gpus", "JSON DEFAULT '[]'"),
    ("users", "disabled_at", "DATETIME"),
    ("datasets", "task", "VARCHAR(32) DEFAULT ''"),
    ("nodes", "cache_dir", "VARCHAR(512) DEFAULT ''"),
    ("label_revisions", "parent_id", "INTEGER"),
    ("label_revisions", "created_by", "VARCHAR(32)"),
    ("releases", "digest", "VARCHAR(64) DEFAULT ''"),
    ("node_releases", "state", "VARCHAR(12) DEFAULT ''"),
    ("node_releases", "job_id", "VARCHAR(32)"),
    ("node_releases", "present_items", "INTEGER"),
    ("node_releases", "present_bytes", "INTEGER"),
    ("node_releases", "checked_at", "DATETIME"),
    ("nodes", "data_check", "JSON DEFAULT '{}'"),
    ("environments", "image", "VARCHAR(512) DEFAULT ''"),
    ("releases", "review", "JSON DEFAULT '{}'"),
    ("nodes", "storage", "JSON DEFAULT '{}'"),
]


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone support; we store UTC everywhere)."""
    return datetime.now(UTC).replace(tzinfo=None)


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat(timespec="milliseconds") + "Z"


def make_engine(url: str) -> Engine:
    is_sqlite = url.startswith("sqlite")
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30} if is_sqlite else {},
        pool_pre_ping=not is_sqlite,
    )
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


class Database:
    def __init__(self, url: str) -> None:
        self.engine = make_engine(url)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        from . import catalog_models, models, train_models  # noqa: F401  (register tables)

        Base.metadata.create_all(self.engine)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Minimal forward migration until Alembic lands: add columns newer versions introduced."""
        inspector = inspect(self.engine)
        with self.engine.begin() as conn:
            added = set()
            for table, column, ddl in _ADDED_COLUMNS:
                if column not in {c["name"] for c in inspector.get_columns(table)}:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                    added.add((table, column))
            if ("model_versions", "aliases") in added:  # the one alias a version could have had before
                conn.execute(text("""UPDATE model_versions SET aliases = '["' || alias || '"]' WHERE alias != ''"""))
            if ("datasets", "task") in added:  # 0.34.0: the task moved from the project to each dataset
                conn.execute(
                    text("UPDATE datasets SET task = (SELECT task FROM projects WHERE projects.id = datasets.project_id)")
                )
            # Before 0.38.0 "unlabelled" was an image without a label file, which is background since 0.34.0.
            # From 0.38.0 it means "nobody has labelled it yet": convert only a database older than that
            # (one without the columns 0.38.0 added, or older still: without the dataset task of 0.34.0).
            if ("label_revisions", "parent_id") in added or ("datasets", "task") in added:
                self._unlabelled_is_background(conn)

    @staticmethod
    def _unlabelled_is_background(conn) -> None:
        """0.34.0: an image without labels is background; "unlabelled" (no label file) is no more."""
        if not conn.execute(text("SELECT 1 FROM release_items WHERE state = 'unlabelled' LIMIT 1")).first():
            return
        conn.execute(text("UPDATE release_items SET state = 'background' WHERE state = 'unlabelled'"))
        conn.execute(text("UPDATE label_revisions SET state = 'background' WHERE state = 'unlabelled'"))
        for release_id, raw in conn.execute(text("SELECT id, stats FROM releases WHERE stats LIKE '%unlabelled%'")).all():
            stats = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            states = stats.get("states") or {}
            if "unlabelled" in states:
                states["background"] = int(states.get("background") or 0) + int(states.pop("unlabelled") or 0)
                conn.execute(
                    text("UPDATE releases SET stats = :stats WHERE id = :id"),
                    {"stats": json.dumps(stats, ensure_ascii=False), "id": release_id},
                )

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.Session() as s:
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise
