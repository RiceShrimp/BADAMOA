from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# (table, column, DDL) for columns added after 0.1.0.
_ADDED_COLUMNS = [
    ("api_tokens", "last_ip", "VARCHAR(64) DEFAULT ''"),
    ("items", "thumb_bytes", "INTEGER DEFAULT 0"),
    ("items", "thumb_v", "INTEGER DEFAULT 0"),
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
    ("users", "orders", "JSON DEFAULT '{}'"),
    ("datasets", "task", "VARCHAR(32) DEFAULT ''"),
    ("nodes", "cache_dir", "VARCHAR(512) DEFAULT ''"),
    ("nodes", "work_dir", "VARCHAR(512) DEFAULT ''"),
    ("label_revisions", "parent_id", "INTEGER"),
    ("label_revisions", "created_by", "VARCHAR(32)"),
    ("releases", "digest", "VARCHAR(64) DEFAULT ''"),
    ("releases", "charts", "JSON DEFAULT '{}'"),
    ("node_releases", "state", "VARCHAR(12) DEFAULT ''"),
    ("node_releases", "job_id", "VARCHAR(32)"),
    ("node_releases", "present_items", "INTEGER"),
    ("node_releases", "present_bytes", "INTEGER"),
    ("node_releases", "checked_at", "DATETIME"),
    ("nodes", "data_check", "JSON DEFAULT '{}'"),
    ("environments", "image", "VARCHAR(512) DEFAULT ''"),
    ("releases", "review", "JSON DEFAULT '{}'"),
    ("nodes", "storage", "JSON DEFAULT '{}'"),
    ("label_sets", "autolabel", "JSON DEFAULT '{}'"),
    ("templates", "trial", "JSON DEFAULT '{}'"),
    ("users", "template_upload", "BOOLEAN DEFAULT 0"),
    ("templates", "report", "JSON DEFAULT '{}'"),
    ("templates", "memo", "TEXT DEFAULT ''"),
    ("item_predictions", "reclass", "INTEGER DEFAULT 0"),
    ("item_predictions", "extra_top", "REAL DEFAULT 0"),
    ("item_predictions", "reclass_top", "REAL DEFAULT 0"),
    ("item_predictions", "missed_area", "REAL DEFAULT 0"),
    ("items", "mtime_ns", "INTEGER DEFAULT 0"),
]


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone support; we store UTC everywhere)."""
    return datetime.now(UTC).replace(tzinfo=None)


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat(timespec="milliseconds") + "Z"


def write_lock(db: Session) -> None:
    """Take SQLite's write lock now, held until the session commits or rolls back.

    SQLite starts a transaction only at a session's first write, so what it read before was read outside it: two
    requests can both find a row missing and both insert it. Called before such reads, the second request waits
    here and then reads what the first committed. A session that has written already holds the lock."""
    conn = db.connection()
    if conn.dialect.name == "sqlite" and not conn.connection.dbapi_connection.in_transaction:
        conn.exec_driver_sql("BEGIN IMMEDIATE")


def insert_new(db: Session, model: type, rows: list[dict[str, Any]], unique: list[str]) -> None:
    """Insert ``rows``, skipping any whose ``unique`` columns match a row already there (ON CONFLICT DO NOTHING)."""
    if not rows:
        return
    if db.get_bind().dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    db.execute(insert(model).on_conflict_do_nothing(index_elements=unique), rows)


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
        self._split_autolabel_role()
        self._sota_alias()

    def _sota_alias(self) -> None:
        """0.98.0: the alias "champion" is called "SOTA" (the best model so far), as the screens suggest it now.
        Once: the versions holding it and the pipelines naming it (a gate's alias, a model to evaluate) take the new
        name, the runs of pipelines too (one waiting at its gate goes on with the new name); a model whose family
        has a "SOTA" already keeps its "champion"."""
        import copy

        from .models import HubSetting
        from .train_models import ModelVersion, Pipeline, PipelineRun

        def renamed_steps(steps: list[Any]) -> list[Any] | None:
            out, changed = copy.deepcopy(steps or []), False
            for step in out:
                if not isinstance(step, dict):
                    continue
                for holder in (step.get("config"), (step.get("config") or {}).get("model"), step.get("refs")):
                    if isinstance(holder, dict) and holder.get("alias") == "champion":
                        holder["alias"], changed = "SOTA", True
            return out if changed else None

        key = "migration.sota_alias"
        with self.session() as s:
            if s.get(HubSetting, key) is not None:
                return
            versions = s.query(ModelVersion).all()
            taken = {(v.project_id, v.name) for v in versions if "SOTA" in (v.aliases or [])}
            renamed = 0
            for v in versions:
                if "champion" in (v.aliases or []) and (v.project_id, v.name) not in taken:
                    v.aliases = ["SOTA" if a == "champion" else a for a in v.aliases]
                    v.alias = v.aliases[0]
                    renamed += 1
            for row in [*s.query(Pipeline).all(), *s.query(PipelineRun).all()]:
                steps = renamed_steps(row.steps)
                if steps is not None:
                    row.steps = steps
            s.add(HubSetting(key=key, value={"done": utcnow().isoformat(), "versions": renamed}))

    def _split_autolabel_role(self) -> None:
        """0.54.0: the "학습·오토라벨" role became 학습용 (compute) and 오토라벨용 (autolabel). A server (or an
        unused join command) that had it keeps doing both. Once: an admin may switch 오토라벨용 off afterwards."""
        from .models import HubSetting, JoinToken, Node

        key = "migration.autolabel_role"
        with self.session() as s:
            if s.get(HubSetting, key) is not None:
                return
            for row in [*s.query(Node).all(), *s.query(JoinToken).all()]:
                roles = list(row.roles or [])
                if "compute" in roles and "autolabel" not in roles:
                    roles.insert(roles.index("compute") + 1, "autolabel")
                    row.roles = roles
            s.add(HubSetting(key=key, value={"done": utcnow().isoformat()}))

    def _add_missing_columns(self) -> None:
        """Minimal forward migration until Alembic lands: add columns newer versions introduced."""
        inspector = inspect(self.engine)
        with self.engine.begin() as conn:
            added = set()
            for table, column, ddl in _ADDED_COLUMNS:
                if column not in {c["name"] for c in inspector.get_columns(table)}:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                    added.add((table, column))
            # indexes newer versions added to a table that was there already (create_all makes only new tables)
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_items_project_path ON items (project_id, rel_path)"))
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
