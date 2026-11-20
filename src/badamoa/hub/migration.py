"""Moving the Hub to another PC with one file: export here, import there, then hand the nodes over.

What the file holds: the database (a consistent snapshot), every file the Hub keeps (``storage/``: thumbnails,
run files and weights, model kinds), the node packages and in-house wheels (``wheels/``) and the key that signs
media links. What stays where it is: datasets' originals and received copies (on the nodes, which follow the
Hub), Docker images (in their registry), and this PC's own node with its environments and caches.

The states (hub setting ``migration.state``):

* ``exporting``: the snapshot is taken first, then the files are packed; from then on this Hub is read-only, as
  whatever changed here afterwards would not reach the new one (people still look; nodes get no new jobs);
* ``exported``: the file is ready; an admin moves it to the new PC and imports it there (``badamoa hub import``);
* ``moved``: an admin gave the new Hub's address and it answered as this Hub's import: every node's next
  heartbeat tells it the new address, the node checks the new Hub knows it and switches (``node.json``), and
  browsers are sent there. This PC's own node goes along as an ordinary node of the new Hub while this program
  runs.

Cancel (before or after) makes this Hub writable again; the file stays.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tarfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import services
from .config import HubSettings
from .db import iso, utcnow
from .models import HubSetting, Job, User
from .services import ServiceError

log = logging.getLogger("badamoa.hub")

FORMAT = 1
KEY = "migration.state"
ID_KEY = "hub.id"
IMPORTED_KEY = "migration.imported"
MANIFEST = "badamoa-hub.json"
PARTS = ("hub.db", "media.key", "wheels/", "storage/")  # what an export holds, under these names
DROPPED_SETTINGS = ("hub.address", "docker.registry", KEY)  # this PC's: the new Hub finds its own
READ_ONLY = (
    "이 Hub는 새 PC로 옮기는 중이라 읽기만 됩니다. 여기서 바꾼 것은 새 Hub에 가지 않습니다. "
    "옮기기를 취소하면(설정 → Hub 옮기기) 다시 쓸 수 있습니다."
)
GB = 1024**3

_states: dict[str, dict[str, Any]] = {}  # per Hub (its database): the setting's value, for every request's check
_progresses: dict[str, dict[str, Any]] = {}  # per Hub: the running export's files and bytes packed so far
_lock = threading.Lock()


# ---- state ---------------------------------------------------------------------------------------------


def _key(where: Any) -> str:
    """Which Hub: a session of its database, the Hub (``deps.Hub``) or its ``Database``."""
    if isinstance(where, Session):
        return str(where.get_bind().url)
    database = getattr(where, "db", where)
    engine = getattr(database, "engine", None)
    return str(engine.url) if engine is not None else str(where)


def _state(where: Any) -> dict[str, Any]:
    return _states.setdefault(_key(where), {})


def _progress(where: Any) -> dict[str, Any]:
    return _progresses.setdefault(_key(where), {})


def load(db: Session) -> None:
    """Read the state at start (a Hub restarted while exported or moved stays so)."""
    row = db.get(HubSetting, KEY)
    state = _state(db)
    state.clear()
    state.update(dict(row.value) if row and isinstance(row.value, dict) else {})
    if state.get("phase") == "exporting":  # the process ended while packing: that file is not whole
        state.update(phase="failed", error="Hub가 내보내는 중에 꺼졌습니다. 다시 내보내세요.")
        _save(db)


def phase(where: Any) -> str:
    return str(_state(where).get("phase") or "")


def read_only(where: Any) -> bool:
    return phase(where) in ("exporting", "exported", "moved")


def moved_to(where: Any) -> str:
    return str(_state(where).get("url") or "") if phase(where) == "moved" else ""


def exported_file(where: Any) -> Path | None:
    """The finished export's file, while there is one."""
    path = Path(str(_state(where).get("file") or ""))
    return path if phase(where) in ("exported", "moved") and path.is_file() else None


def _save(db: Session) -> None:
    row = db.get(HubSetting, KEY)
    value = dict(_state(db))
    if row is None:
        db.add(HubSetting(key=KEY, value=value))
    else:
        row.value = value


def hub_id(db: Session) -> str:
    """This Hub's identity: the same after an import, so the old Hub can tell its new place answers."""
    row = db.get(HubSetting, ID_KEY)
    if row is None or not (row.value or {}).get("id"):
        value = {"id": uuid.uuid4().hex}
        if row is None:
            db.add(HubSetting(key=ID_KEY, value=value))
        else:
            row.value = value
        db.flush()
        return value["id"]
    return str(row.value["id"])


def view(db: Session) -> dict[str, Any]:
    """For everyone: whether this Hub is read-only or moved (and where), and what an import brought."""
    imported = db.get(HubSetting, IMPORTED_KEY)
    return {
        "phase": phase(db),
        "read_only": read_only(db),
        "url": moved_to(db),
        "imported": imported.value if imported else None,
    }


# ---- what an export would take ------------------------------------------------------------------------


def _disks() -> list[dict[str, Any]]:
    found = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        if usage.total < 2 * GB or part.mountpoint.startswith(("/boot", "/snap")):
            continue
        found.append(
            {"mount": part.mountpoint, "free_gb": round(usage.free / GB, 1), "total_gb": round(usage.total / GB, 1)}
        )
    return sorted(found, key=lambda d: -d["free_gb"])


def _size(folder: Path) -> tuple[int, int]:
    total = files = 0
    for root, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
                files += 1
            except OSError:
                continue
    return total, files


_counts: dict[str, tuple[float, tuple[int, int]]] = {}


def _counted(folder: Path) -> tuple[int, int]:
    """The storage folder's size before the Hub's own count is in: walked at most every ten minutes (a Hub may
    keep a million thumbnails, and the moving screen asks every few seconds)."""
    at, found = _counts.get(str(folder), (0.0, (0, 0)))
    if time.monotonic() - at > 600:
        found = _size(folder)
        _counts[str(folder)] = (time.monotonic(), found)
    return found


def _db_path(settings: HubSettings) -> Path:
    url = settings.database_url or ""
    if not url.startswith("sqlite:///"):
        raise ServiceError(400, "SQLite를 쓰는 Hub만 파일로 옮길 수 있습니다.")
    return Path(url[len("sqlite:///") :])


def overview(db: Session, settings: HubSettings) -> dict[str, Any]:
    """For the admin before exporting: how big the file gets, where it could go, what blocks it now, and what
    stays behind on this PC."""
    from . import catalog, hub_storage
    from .catalog_models import Dataset, Release, StorageLocation
    from .models import Node

    counted = hub_storage.measured()
    storage = (int(counted.get("bytes") or 0), int(counted.get("files") or 0))
    if not counted.get("at"):
        storage = _counted(Path(settings.storage_dir))
    database = _db_path(settings).stat().st_size if _db_path(settings).is_file() else 0
    wheels = _size(Path(settings.wheels_dir or settings.home / "wheels"))[0]
    running = db.scalars(select(Job).where(Job.status == "running")).all()
    local = db.scalar(select(Node).where(Node.is_local.is_(True)))
    here = []  # datasets whose originals are on this PC's node: they stay here
    if local is not None:
        roots = db.scalars(select(StorageLocation).where(StorageLocation.node_id == local.id)).all()
        here = [r.root for r in roots]
    registry = (db.get(HubSetting, "docker.registry") or HubSetting(value={})).value or {}
    now = utcnow()
    nodes = [
        {
            "id": n.id,
            "name": n.name,
            "here": bool(n.last_seen_at and (now - n.last_seen_at).total_seconds() < 120),
            "version": n.agent_version or "",
        }
        for n in db.scalars(select(Node).order_by(Node.name))
    ]
    file = exported_file(db)
    state = _state(db)
    view_ = {
        **view(db),
        "nodes": nodes,
        "download": catalog.media_url(settings, "hub-export", file.name) if file else "",
        "file_bytes": file.stat().st_size if file else 0,
        "hub_id": hub_id(db),
        "bytes": database + storage[0] + wheels,
        "files": storage[1],
        "parts": {"database": database, "storage": storage[0], "wheels": wheels},
        "running": [{"id": j.id, "type": j.type} for j in running],
        "disks": _disks(),
        "stays": {
            "originals": here,
            "registry": registry.get("address") if registry.get("mode") in ("made", "adopted") else "",
            "local_node": local.name if local else "",
        },
        "file": state.get("file"),
        "error": state.get("error"),
        "progress": dict(_progress(db)) if phase(db) == "exporting" else None,
        "exported_at": state.get("exported_at"),
        "datasets": db.query(Dataset).count(),
        "releases": db.query(Release).count(),
    }
    return view_


# ---- export ---------------------------------------------------------------------------------------------


def start_export(db: Session, user: User, settings: HubSettings, folder: str, local_agent: Any = None) -> dict[str, Any]:
    """Make the Hub read-only and pack it into ``<folder>/badamoa-hub-<when>.tar`` in the background."""
    with _lock:
        if phase(db) == "exporting":
            raise ServiceError(409, "이미 내보내는 중입니다.")
        if phase(db) == "moved":
            raise ServiceError(409, "이미 새 Hub로 넘겼습니다. 다시 하려면 먼저 옮기기를 취소하세요.")
        running = db.scalars(select(Job).where(Job.status == "running")).all()
        if running:
            raise ServiceError(409, f"돌고 있는 작업 {len(running)}개가 끝난 뒤에 내보내세요(작업 화면에서 보거나 중단).")
        target = Path(folder.strip()).expanduser() if folder.strip() else None
        if target is None or not target.is_absolute():
            raise ServiceError(400, "내보낼 폴더를 이 Hub PC의 절대 경로로 적으세요.")
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / f".badamoa-probe-{uuid.uuid4().hex[:6]}"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as e:
            raise ServiceError(400, f"{target}에 쓸 수 없습니다: {e}") from e
        need = overview(db, settings)["bytes"]
        free = shutil.disk_usage(target).free
        if free < need * 1.05 + GB:
            raise ServiceError(
                400, f"{target}의 디스크에 {free / GB:.1f} GB 남아 약 {need / GB:.1f} GB인 파일을 둘 수 없습니다."
            )
        if local_agent is not None:  # this PC's node keeps its identity (it may follow the Hub as a node)
            try:
                local_agent.config.save(Path(local_agent.home))
            except OSError:
                pass
        name = f"badamoa-hub-{datetime.now().strftime('%Y%m%d-%H%M')}.tar"
        state = _state(db)
        state.clear()
        state.update(
            phase="exporting", folder=str(target), file=str(target / name), by=user.username, started_at=iso(utcnow())
        )
        _save(db)
        services.audit(db, user.id, "ui", "hub.export", "", folder=str(target))
        db.commit()
        progress = _progress(db)
        progress.clear()
        progress.update(files=0, bytes=0, total_files=0, total_bytes=need, step="db")
    threading.Thread(target=_export_in_background, args=(settings, target / name), name="hub-export", daemon=True).start()
    return {"file": str(target / name)}


def _export_in_background(settings: HubSettings, path: Path) -> None:
    from .db import Database

    database = Database(settings.database_url or "")
    try:
        result = export(settings, path, progress=_progress(database))
        with database.session() as db:
            _state(db).update(phase="exported", exported_at=iso(utcnow()), bytes=result["bytes"], error="")
            _save(db)
        log.info("hub export: %s (%.1f GB)", path, result["bytes"] / GB)
    except Exception as e:  # noqa: BLE001 - said on the screen; the Hub goes back to normal
        log.exception("hub export")
        with database.session() as db:
            _state(db).update(phase="failed", error=str(e))
            _save(db)
    finally:
        database.engine.dispose()


def _snapshot(source: Path, target: Path) -> None:
    """A consistent copy of the database while the Hub runs (SQLite's online backup), without the migration
    state: the new Hub starts writable."""
    src = sqlite3.connect(str(source))
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
        dst.execute(f"DELETE FROM hub_settings WHERE key IN ({','.join('?' * len(DROPPED_SETTINGS))})", DROPPED_SETTINGS)
        dst.commit()
    finally:
        dst.close()
        src.close()


def export(settings: HubSettings, path: Path, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pack this Hub into ``path`` (a tar: the manifest first, then the database, the key, wheels and storage)."""
    progress = progress if progress is not None else {}
    home = Path(settings.home)
    part = path.with_name(path.name + ".part")
    snapshot = path.with_name(path.name + ".db")
    try:
        progress.update(step="db")
        _snapshot(_db_path(settings), snapshot)
        with sqlite3.connect(str(snapshot)) as conn:
            row = conn.execute("SELECT value FROM hub_settings WHERE key = ?", (ID_KEY,)).fetchone()
            identity = json.loads(row[0]).get("id") if row else ""
            counts = {
                t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("users", "nodes", "projects", "runs")
            }
        conn.close()
        storage = Path(settings.storage_dir)
        wheels = Path(settings.wheels_dir or home / "wheels")
        manifest = {
            "format": FORMAT,
            "version": services.HUB_VERSION,
            "hub_id": identity,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "counts": counts,
        }
        progress.update(step="files")
        packed = 0
        with tarfile.open(part, "w", format=tarfile.PAX_FORMAT) as tar:
            data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
            info = tarfile.TarInfo(MANIFEST)
            info.size, info.mtime = len(data), int(time.time())
            import io

            tar.addfile(info, io.BytesIO(data))
            tar.add(snapshot, arcname="hub.db")
            if (home / "media.key").is_file():
                tar.add(home / "media.key", arcname="media.key")
            for prefix, folder in (("wheels", wheels), ("storage", storage)):
                for root, dirs, names in os.walk(folder):
                    dirs.sort()
                    for name in sorted(names):
                        if name.endswith(".part"):
                            continue  # a file being written now
                        full = Path(root) / name
                        try:
                            tar.add(full, arcname=f"{prefix}/{full.relative_to(folder).as_posix()}", recursive=False)
                        except OSError:
                            continue  # removed meanwhile by the cleanup: nothing names it any more
                        packed += 1
                        progress["files"] = packed
                        progress["bytes"] = progress.get("bytes", 0) + (tar.members[-1].size if tar.members else 0)
                        if len(tar.members) > 10000:
                            tar.members.clear()  # tarfile keeps every member: 1.3 M of them is a lot of memory
        os.replace(part, path)
        return {"file": str(path), "bytes": path.stat().st_size, "files": packed, "manifest": manifest}
    finally:
        snapshot.unlink(missing_ok=True)
        part.unlink(missing_ok=True)


def cancel(db: Session, user: User) -> dict[str, Any]:
    """This Hub writable again (nodes already sent to the new Hub stay there)."""
    if phase(db) == "exporting":
        raise ServiceError(409, "내보내는 중에는 취소하지 못합니다. 끝난 뒤에 하세요.")
    kept = _state(db).get("file")
    _state(db).clear()
    _save(db)
    services.audit(db, user.id, "ui", "hub.export.cancel", "", file=kept or "")
    return {"ok": True, "file": kept}


def hand_over(db: Session, user: User, url: str) -> dict[str, Any]:
    """Send nodes and browsers to the new Hub, once it answers as this Hub's import."""
    if phase(db) not in ("exported", "moved"):
        raise ServiceError(409, "먼저 내보내고 새 PC에서 가져오세요.")
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ServiceError(400, "새 Hub의 주소를 http://로 시작하게 적으세요 (예: http://192.168.0.10:8470).")
    try:
        found = httpx.get(f"{url}/api/health", timeout=10).json()
    except (httpx.HTTPError, ValueError) as e:
        raise ServiceError(400, f"{url}에서 Hub가 답하지 않습니다: {e}") from e
    if found.get("hub_id") != hub_id(db):
        raise ServiceError(
            400, f"{url}의 Hub는 이 Hub를 가져온 것이 아닙니다. 새 PC에서 내보낸 파일을 가져왔는지 확인하세요."
        )
    _state(db).update(phase="moved", url=url, moved_at=iso(utcnow()))
    _save(db)
    services.audit(db, user.id, "ui", "hub.hand_over", "", url=url)
    return view(db)


# ---- import (on the new PC, with its Hub stopped) ------------------------------------------------------


def _members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    keep = []
    for member in tar:
        name = member.name
        if name == MANIFEST:
            continue
        if name.startswith("/") or ".." in Path(name).parts or not name.startswith(PARTS):
            raise ServiceError(400, f"내보낸 파일에 있으면 안 되는 항목이 있습니다: {name}")
        if member.isfile() or member.isdir():
            keep.append(member)
    return keep


def read_manifest(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r") as tar:
        first = tar.next()
        if first is None or first.name != MANIFEST:
            raise ServiceError(400, f"{path.name}은(는) BADAMOA Hub를 내보낸 파일이 아닙니다.")
        manifest = json.loads(tar.extractfile(first).read().decode("utf-8"))  # type: ignore[union-attr]
    if manifest.get("format") != FORMAT:
        raise ServiceError(400, f"이 BADAMOA가 모르는 형식({manifest.get('format')})입니다. BADAMOA를 업데이트하세요.")
    return manifest


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in text.split(".")[:3] if n.isdigit())


def import_file(path: Path, home: Path, force: bool = False, log_line: Callable[[str], None] = print) -> dict[str, Any]:
    """Unpack an export into this PC's Hub folder (its Hub stopped). A folder already holding a Hub with people
    is refused unless ``force``, which moves what is there aside (``before-import-<when>``)."""
    manifest = read_manifest(path)
    if _version(str(manifest.get("version"))) > _version(services.HUB_VERSION):
        raise ServiceError(
            400,
            f"BADAMOA {manifest.get('version')}에서 내보낸 파일입니다. "
            f"이 PC의 BADAMOA({services.HUB_VERSION})를 먼저 업데이트하세요.",
        )
    home.mkdir(parents=True, exist_ok=True)
    present = [home / p.rstrip("/") for p in PARTS if (home / p.rstrip("/")).exists()]
    if (home / "hub.db").is_file() and _has_people(home / "hub.db") and not force:
        raise ServiceError(409, f"{home}에 이미 쓰던 Hub가 있습니다. 덮어쓰려면 --force (지금 것은 옆 폴더로 옮겨 둡니다).")
    if present:
        aside = home / f"before-import-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        aside.mkdir()
        try:
            for item in [*present, *home.glob("hub.db-*")]:
                shutil.move(str(item), str(aside / item.name))
        except OSError as e:
            raise ServiceError(409, f"{home}의 Hub가 켜져 있는 것 같습니다({e}). Hub를 끈 뒤에 하세요.") from e
        log_line(f"이 폴더에 있던 것은 {aside}로 옮겼습니다.")
    started = time.monotonic()
    count = 0
    with tarfile.open(path, "r") as tar:
        for member in _members(tar):
            tar.extract(member, home, filter="data")
            count += 1
            if count % 50000 == 0:
                log_line(f"{count:,}개 풀었습니다…")
    _fix_database(home / "hub.db", manifest, path)
    log_line(f"{count:,}개를 풀었습니다 ({time.monotonic() - started:.0f}초).")
    return {"manifest": manifest, "files": count}


def _has_people(db_file: Path) -> bool:
    try:
        with sqlite3.connect(str(db_file)) as conn:
            found = conn.execute("SELECT count(*) FROM users").fetchone()[0] > 0
        conn.close()
        return found
    except sqlite3.Error:
        return False


def _fix_database(db_file: Path, manifest: dict[str, Any], path: Path) -> None:
    """The old PC's own node becomes an ordinary node (this PC gets its own when the Hub starts), and what was
    that PC's (its address, its Docker registry) goes: the new Hub finds its own."""
    conn = sqlite3.connect(str(db_file))
    try:
        for node_id, name in conn.execute("SELECT id, name FROM nodes WHERE is_local = 1").fetchall():
            renamed = (name[: -len(" (hub)")] if name.endswith(" (hub)") else name) + " (예전 Hub)"
            conn.execute("UPDATE nodes SET is_local = 0, name = ? WHERE id = ?", (renamed, node_id))
        marks = ",".join("?" * len(DROPPED_SETTINGS))
        conn.execute(f"DELETE FROM hub_settings WHERE key IN ({marks})", DROPPED_SETTINGS)
        imported = {
            "hub_id": manifest.get("hub_id"),
            "version": manifest.get("version"),
            "exported_at": manifest.get("exported_at"),
            "file": path.name,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        conn.execute("DELETE FROM hub_settings WHERE key = ?", (IMPORTED_KEY,))
        conn.execute("INSERT INTO hub_settings (key, value) VALUES (?, ?)", (IMPORTED_KEY, json.dumps(imported)))
        conn.commit()
    finally:
        conn.close()
