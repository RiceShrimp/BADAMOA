from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# (table, column, DDL) for columns added after 0.1.0.
_ADDED_COLUMNS = [
    ("jobs", "gpu_indices", "JSON DEFAULT '[]'"),
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
        from . import models  # noqa: F401  (register tables)

        Base.metadata.create_all(self.engine)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Minimal forward migration until Alembic lands: add columns newer versions introduced."""
        inspector = inspect(self.engine)
        with self.engine.begin() as conn:
            for table, column, ddl in _ADDED_COLUMNS:
                if column not in {c["name"] for c in inspector.get_columns(table)}:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.Session() as s:
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise
