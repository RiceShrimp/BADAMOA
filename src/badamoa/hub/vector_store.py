"""The vector DB 이미지 탐색 searches: LanceDB, a folder on the Hub PC (no server to run).

By default ``<hub home>/vectors``; an admin can move it (another disk, say), the data copied along. One table a
model, a row an image of a project — ``project_id``, ``item_id``, ``sha256``, ``vector`` (float16, L2-normalised) —
searched by cosine within a project (or a version), with an IVF-PQ index once there are many (and the rows added
since searched too, until the index takes them in). The image a vector belongs to is known by its content, so a
project that gets a picture another project has a vector for takes that vector (``sync_project``).
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from .config import HubSettings
from .models import HubSetting
from .services import ServiceError

log = logging.getLogger("badamoa.hub.vectors")

KEY = "vectors.location"
INDEX_FROM = 50_000  # rows from which a table is searched through an index (fewer: every row, exact)
REINDEX_SHARE = 0.2  # rows not in the index yet, as a share of all, before the index is brought up to date
IN_LIST = 5_000  # a version this small is searched within; a bigger one by searching wider and keeping its own
# How wide an indexed search looks and how many it re-ranks exactly: on clustered vectors as image embeddings are,
# all of the 10 nearest found (100,000 vectors, about 25 ms)
NPROBES = 40
REFINE = 20

_lock = threading.RLock()
_conn: dict[str, Any] = {"path": None, "db": None}
_moving: dict[str, Any] = {"state": "", "from": "", "to": "", "copied": 0, "total": 0, "error": ""}
_indexing: set[str] = set()


def default_location(settings: HubSettings) -> Path:
    return settings.home / "vectors"


def location(db: Session, settings: HubSettings) -> Path:
    row = db.get(HubSetting, KEY)
    path = str((row.value or {}).get("path") or "") if row is not None else ""
    return Path(path) if path else default_location(settings)


def _connect(path: Path) -> Any:
    import lancedb  # noqa: PLC0415  (only when vectors are kept or searched)

    with _lock:
        if _conn["path"] != str(path):
            if _moving["state"] == "moving":
                raise ServiceError(409, "벡터 DB를 옮기는 중입니다. 끝난 뒤 다시 하세요.")
            path.mkdir(parents=True, exist_ok=True)
            _conn.update(path=str(path), db=lancedb.connect(str(path)))
        return _conn["db"]


def _tables(conn: Any) -> list[str]:
    return list(conn.list_tables().tables)


def _name(model: str) -> str:
    return "images_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def _schema(dim: int) -> Any:
    import pyarrow as pa  # noqa: PLC0415

    return pa.schema(
        [
            pa.field("project_id", pa.string()),
            pa.field("item_id", pa.int64()),
            pa.field("sha256", pa.string()),
            pa.field("vector", pa.list_(pa.float16(), dim)),
        ]
    )


def _table(db: Session, settings: HubSettings, model: str, dim: int, create: bool = True) -> Any:
    conn = _connect(location(db, settings))
    name = _name(model)
    with _lock:
        if name in _tables(conn):
            return conn.open_table(name)
        if not create:
            return None
        return conn.create_table(name, schema=_schema(dim))


def _rows(dim: int, rows: list[tuple[str, int, str]], vectors: Any) -> Any:
    import numpy as np  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    flat = pa.array(np.ascontiguousarray(vectors, dtype=np.float16).reshape(-1), type=pa.float16())
    return pa.Table.from_arrays(
        [
            pa.array([r[0] for r in rows], pa.string()),
            pa.array([r[1] for r in rows], pa.int64()),
            pa.array([r[2] for r in rows], pa.string()),
            pa.FixedSizeListArray.from_arrays(flat, dim),
        ],
        schema=_schema(dim),
    )


def add(db: Session, settings: HubSettings, model: str, dim: int, rows: list[tuple[str, int, str]], vectors: Any) -> int:
    """Keep vectors for these images (``rows``: project, item, sha256; ``vectors`` row for row); an image kept already
    stays as it was. How many were new."""
    if not rows:
        return 0
    table = _table(db, settings, model, dim)
    before = table.count_rows()
    table.merge_insert("item_id").when_not_matched_insert_all().execute(_rows(dim, rows, vectors))
    added = table.count_rows() - before
    _keep_indexed(table, dim)
    return added


def _keep_indexed(table: Any, dim: int) -> None:
    """A table this big gets an index, and one far behind is brought up to date, in the background."""
    name = table.name
    with _lock:
        if name in _indexing:
            return
        rows = table.count_rows()
        if rows < INDEX_FROM:
            return
        indexed = [i for i in table.list_indices() if "vector" in (getattr(i, "columns", None) or [])]
        behind = 0
        if indexed:
            try:
                behind = int(getattr(table.index_stats(indexed[0].name), "num_unindexed_rows", 0) or 0)
            except Exception:  # noqa: BLE001 - an index we cannot read is made again
                indexed = []
        if indexed and behind < rows * REINDEX_SHARE:
            return
        _indexing.add(name)

    def build() -> None:
        try:
            from lancedb.index import IvfPq  # noqa: PLC0415

            if indexed:
                table.optimize(cleanup_older_than=timedelta(hours=1))  # the rows added since into the index
            else:
                partitions = max(16, int(rows**0.5))
                config = IvfPq(distance_type="cosine", num_partitions=partitions, num_sub_vectors=max(1, dim // 16))
                table.create_index("vector", config=config)
                table.create_scalar_index("project_id", index_type="BITMAP")
                table.create_scalar_index("item_id")
                table.optimize(cleanup_older_than=timedelta(hours=1))  # earlier versions of the table let go
            log.info("벡터 색인을 만들었습니다: %s (%d개)", name, rows)
        except Exception as e:  # noqa: BLE001 - searching goes on without (every row, exact)
            log.warning("벡터 색인을 만들지 못했습니다 (%s): %s", name, e)
        finally:
            with _lock:
                _indexing.discard(name)

    threading.Thread(target=build, name=f"index-{name}", daemon=True).start()


def items_of(db: Session, settings: HubSettings, model: str, dim: int, project_id: str) -> set[int]:
    """The images of a project the DB has a vector for."""
    table = _table(db, settings, model, dim, create=False)
    if table is None:
        return set()
    found = table.search().where(f"project_id = '{_quoted(project_id)}'").select(["item_id"]).limit(None).to_arrow()
    return {int(i) for i in found.column("item_id").to_pylist()}


def vectors_of(db: Session, settings: HubSettings, model: str, dim: int, shas: list[str]) -> dict[str, Any]:
    """The vector kept for each of these images (by content), whatever project it was kept for."""
    import numpy as np  # noqa: PLC0415

    table = _table(db, settings, model, dim, create=False)
    out: dict[str, Any] = {}
    if table is None:
        return out
    for start in range(0, len(shas), 1000):
        chunk = shas[start : start + 1000]
        listed = ", ".join(f"'{_quoted(s)}'" for s in chunk)
        found = table.search().where(f"sha256 IN ({listed})").select(["sha256", "vector"]).limit(None).to_arrow()
        for sha, vector in zip(found.column("sha256").to_pylist(), found.column("vector").to_pylist(), strict=True):
            out.setdefault(sha, np.asarray(vector, dtype=np.float16))
    return out


def vector_of_item(db: Session, settings: HubSettings, model: str, dim: int, item_id: int) -> Any:
    import numpy as np  # noqa: PLC0415

    table = _table(db, settings, model, dim, create=False)
    if table is None:
        return None
    found = table.search().where(f"item_id = {int(item_id)}").select(["vector"]).limit(1).to_arrow()
    if found.num_rows == 0:
        return None
    return np.asarray(found.column("vector")[0].as_py(), dtype=np.float32)


def search(
    db: Session,
    settings: HubSettings,
    model: str,
    dim: int,
    project_id: str,
    query: Any,
    limit: int,
    within: list[int] | None = None,
) -> list[tuple[int, float]]:
    """The project's images nearest to ``query`` (cosine, the nearest first): (item id, similarity). ``within``: only
    these images (a version's)."""
    table = _table(db, settings, model, dim, create=False)
    if table is None:
        return []
    where = f"project_id = '{_quoted(project_id)}'"
    small = within is not None and len(within) <= IN_LIST
    if small:
        if not within:
            return []
        where += f" AND item_id IN ({', '.join(str(int(i)) for i in within)})"
    keep = set(within) if within is not None and not small else None
    wanted = limit if keep is None else limit * 10
    while True:
        found = (
            table.search(query.astype("float32"))
            .distance_type("cosine")
            .where(where, prefilter=True)
            .nprobes(NPROBES)
            .refine_factor(REFINE)  # the index's guesses re-ranked by the vectors themselves: exact similarities
            .select(["item_id"])
            .limit(wanted)
            .to_list()
        )
        hits = [(int(r["item_id"]), 1.0 - float(r["_distance"])) for r in found]
        if keep is not None:
            mine = [h for h in hits if h[0] in keep]
            if len(mine) >= limit or len(found) < wanted:
                return mine[:limit]
            wanted *= 4  # a version far smaller than its project: look wider
            continue
        return hits[:limit]


def _quoted(text: str) -> str:
    return str(text).replace("'", "''")


# ---- where it is, and moving it -------------------------------------------------------------------------------------


def _size(path: Path) -> int:
    total = 0
    if path.is_dir():
        for file in path.rglob("*"):
            try:
                if file.is_file():
                    total += file.stat().st_size
            except OSError:
                continue
    return total


def view(db: Session, settings: HubSettings) -> dict[str, Any]:
    path = location(db, settings)
    tables = []
    try:
        conn = _connect(path)
        for name in _tables(conn):
            table = conn.open_table(name)
            indexed = [i.name for i in table.list_indices() if "vector" in (getattr(i, "columns", None) or [])]
            tables.append({"name": name, "rows": table.count_rows(), "indexed": bool(indexed)})
    except ServiceError:
        pass
    return {
        "path": str(path),
        "default": str(default_location(settings)),
        "size": _size(path),
        "tables": tables,
        "moving": dict(_moving),
    }


def move(db: Session, settings: HubSettings, target: str) -> dict[str, Any]:
    """Move the vector DB to the folder ``target`` (absolute, on the Hub PC): the data copied there in the background,
    the Hub using it from there once whole, the old folder removed."""
    path = Path(target.strip()).expanduser() if target.strip() else None
    if path is None or not path.is_absolute():
        raise ServiceError(400, "벡터 DB를 둘 폴더를 Hub PC의 절대 경로로 적으세요.")
    old = location(db, settings)
    if path.resolve() == old.resolve():
        raise ServiceError(400, "지금 쓰는 곳과 같은 폴더입니다.")
    if path.exists() and any(path.iterdir()):
        raise ServiceError(400, "비어 있는 폴더(또는 아직 없는 폴더)를 고르세요.")
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".badamoa-probe"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as e:
        raise ServiceError(400, f"그 폴더에 쓸 수 없습니다: {e}") from e
    with _lock:
        if _moving["state"] == "moving":
            raise ServiceError(409, "벡터 DB를 이미 옮기는 중입니다.")
        _moving.update(state="moving", **{"from": str(old), "to": str(path)}, copied=0, total=_size(old), error="")
        _conn.update(path=None, db=None)
    database = db.get_bind()

    def run() -> None:
        try:
            if old.is_dir():
                for source in old.rglob("*"):
                    dest = path / source.relative_to(old)
                    if source.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    elif source.is_file():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, dest)
                        _moving["copied"] += source.stat().st_size
            from sqlalchemy.orm import Session as Opened  # noqa: PLC0415

            with Opened(database) as session:
                row = session.get(HubSetting, KEY)
                if row is None:
                    session.add(HubSetting(key=KEY, value={"path": str(path)}))
                else:
                    row.value = {"path": str(path)}
                session.commit()
            shutil.rmtree(old, ignore_errors=True)
            _moving.update(state="done")
            log.info("벡터 DB를 옮겼습니다: %s → %s", old, path)
        except Exception as e:  # noqa: BLE001 - the old folder stays in use
            _moving.update(state="failed", error=f"{type(e).__name__}: {e}")
            shutil.rmtree(path, ignore_errors=True)
            log.warning("벡터 DB를 옮기지 못했습니다: %s", e)

    threading.Thread(target=run, name="vectors-move", daemon=True).start()
    return view(db, settings)
