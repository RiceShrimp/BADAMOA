"""How much room the Hub's own folder takes, measured in the background, and keeping it from filling up.

A dataset registered on a node leaves two things here for each image: its record in the database (fingerprint,
place, size, labels and their history, version membership: about 2 KB) and a 320 px thumbnail (about 11 KB).
Hundreds of thousands of images make that gigabytes, so:

- the storage folder is counted file by file in the background (every ``MEASURE_EVERY_S``), never while a
  screen waits: the storage card and the home screen read the last count
- each image's thumbnail size is kept with it (``Item.thumb_bytes``), so the datasets' share is a sum, per
  project too; images registered before 0.61.0 get theirs filled in by the same background pass
- an import says how much it will need before it reads a single image (``check_room``), and stops if the
  disk gets close to full while it runs
- the database file does not shrink when records go: ``compact`` gives the space back (VACUUM), on an admin's
  click or by itself after the trash was emptied, only while no job runs
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from .catalog_models import Item
from .config import HubSettings
from .models import Job

log = logging.getLogger("badamoa.hub.storage")

MEASURE_EVERY_S = 600
BACKFILL_BATCH = 5000
THUMB_GUESS = 12 * 1024  # an image's thumbnail before any is known
RECORD_GUESS = 2 * 1024  # an image's records in the database before any is known
GB = 1024**3
# Left free on the Hub's disk: an import that would need more is refused, one that gets this close stops.
KEEP_FREE_MIN = 2 * GB
KEEP_FREE_SHARE = 0.02
# The database is compacted by itself when this much of it is empty pages (and at least this many bytes).
COMPACT_SHARE = 0.25
COMPACT_MIN = 256 * 1024**2

# Thumbnails as nodes make them since 0.61.0; older ones (320 px JPEG at 75) the Hub makes so in the background,
# from its own copy, while no import runs. The old files go with the regular cleanup (nothing refers to them).
THUMB_SIDE = 256
THUMB_QUALITY = 60
SHRINK_BATCH = 500

_lock = threading.Lock()
_shrinking: dict[str, Any] = {"thread": None, "done": 0}
_measured: dict[str, Any] = {"bytes": 0, "files": 0, "at": 0.0}
_compacting: dict[str, Any] = {"running": False, "started_at": 0.0, "last": None}


# ---- counting the folder ------------------------------------------------------------------------


def _walk(folder: Path) -> tuple[int, int]:
    """Bytes and files under ``folder``: ``os.scandir`` gives each file's size with the listing on Windows,
    so a folder of a million thumbnails is counted in seconds, not minutes."""
    total = files = 0
    stack = [str(folder)]
    while stack:
        try:
            with os.scandir(stack.pop()) as listing:
                for entry in listing:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue  # removed meanwhile
        except OSError:
            continue
    return total, files


def measured() -> dict[str, Any]:
    """The last count of the storage folder: ``bytes``, ``files``, ``at`` (0 before the first)."""
    return dict(_measured)


def measure(database: Any, settings: HubSettings) -> dict[str, Any]:
    """Count the storage folder and fill in thumbnail sizes still unknown (the Hub's background loop)."""
    started = time.monotonic()
    total, files = _walk(Path(settings.storage_dir))
    _measured.update(bytes=total, files=files, at=time.time())
    filled = 0
    try:
        filled = fill_thumb_sizes(database, settings)
    except Exception:  # noqa: BLE001 - the count stands without it
        log.exception("thumbnail sizes")
    took = time.monotonic() - started
    if took > 5 or filled:
        log.info("storage: %d files, %.1f GB (%.0f s); thumbnail sizes for %d images", files, total / GB, took, filled)
    return measured()


def _blob_path(settings: HubSettings, key: str) -> Path:
    return Path(settings.storage_dir) / "blobs" / key[:2] / key[2:4] / key


def fill_thumb_sizes(database: Any, settings: HubSettings, limit: int = 200_000) -> int:
    """Thumbnail sizes of images registered before they were kept (0.61.0), a batch at a time."""
    done = 0
    while done < limit:
        with database.session() as db:
            rows = db.execute(
                select(Item.id, Item.thumb_key)
                .where(Item.thumb_key.is_not(None), Item.thumb_bytes == 0)
                .limit(BACKFILL_BATCH)
            ).all()
            if not rows:
                break
            sizes: dict[int, int] = {}
            for item_id, key in rows:
                try:
                    sizes[item_id] = _blob_path(settings, key).stat().st_size
                except OSError:
                    sizes[item_id] = -1  # gone: not looked at again
            for size in set(sizes.values()):
                ids = [i for i, s in sizes.items() if s == size]
                db.execute(update(Item).where(Item.id.in_(ids)).values(thumb_bytes=size))
            done += len(rows)
    return done


# ---- thumbnails ---------------------------------------------------------------------------------


def thumb_current(data: bytes) -> bool:
    """Whether a stored thumbnail is of today's kind: WebP, its longer side at most ``THUMB_SIDE``."""
    from PIL import Image

    if not (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        return False
    try:
        with Image.open(io.BytesIO(data)) as image:
            return max(image.size) <= THUMB_SIDE
    except Exception:  # noqa: BLE001 - not readable: left as it is
        return False


def shrink(data: bytes) -> bytes:
    """A stored thumbnail made again as today's kind (from the stored copy: the original is on a node)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        picture = image.convert("RGB")
    picture.thumbnail((THUMB_SIDE, THUMB_SIDE))
    out = io.BytesIO()
    picture.save(out, format="WEBP", quality=THUMB_QUALITY, method=4)
    return out.getvalue()


def importing(db: Session) -> bool:
    return bool(
        db.scalar(select(func.count(Job.id)).where(Job.type == "data.import", Job.status.in_(("queued", "running"))))
    )


def shrink_thumbs(database: Any, settings: HubSettings, limit: int | None = None) -> int:
    """Older thumbnails made again as today's kind, a batch at a time, stopping while an import runs (it needs the
    Hub's time more). Every image sharing a thumbnail gets the new one. Returns how many were made."""
    from .catalog import storage_for

    storage = storage_for(str(settings.storage_dir))
    made = 0
    while limit is None or made < limit:
        with database.session() as db:
            if importing(db):
                break
            keys = db.scalars(
                select(Item.thumb_key).where(Item.thumb_v == 0, Item.thumb_key.is_not(None)).distinct().limit(SHRINK_BATCH)
            ).all()
            if not keys:
                break
            for key in keys:
                try:
                    data = storage.get_blob(key)
                    small = None if thumb_current(data) else shrink(data)
                except Exception:  # noqa: BLE001 - gone or unreadable: left as it is, not tried again
                    small = None
                if small is None:
                    db.execute(update(Item).where(Item.thumb_key == key).values(thumb_v=1))
                    continue
                new = storage.put_blob(small)
                db.execute(
                    update(Item).where(Item.thumb_key == key).values(thumb_key=new, thumb_bytes=len(small), thumb_v=1)
                )
                made += 1
        _shrinking["done"] += len(keys)
        time.sleep(0.05)  # let the requests in between batches
    return made


def old_thumbs(db: Session) -> int:
    """Images whose thumbnail is still of the older, larger kind."""
    return db.scalar(select(func.count(Item.id)).where(Item.thumb_v == 0, Item.thumb_key.is_not(None))) or 0


def start_shrinking(database: Any, settings: HubSettings) -> bool:
    """Shrink older thumbnails in a thread of its own (it can take an hour for a million), unless one runs."""
    running = _shrinking["thread"]
    if running is not None and running.is_alive():
        return False

    def run() -> None:
        try:
            made = shrink_thumbs(database, settings)
            if made:
                log.info("thumbnails: %d made smaller (256 px WebP)", made)
        except Exception:  # noqa: BLE001 - tried again at the next count
            log.exception("shrinking thumbnails")

    _shrinking["thread"] = threading.Thread(target=run, name="shrink-thumbs", daemon=True)
    _shrinking["thread"].start()
    return True


def shrinking() -> bool:
    running = _shrinking["thread"]
    return running is not None and running.is_alive()


# ---- an import's room -----------------------------------------------------------------------------


def keep_free(settings: HubSettings) -> int:
    """Bytes always left free on the Hub's disk."""
    disk = shutil.disk_usage(settings.home)
    return max(KEEP_FREE_MIN, int(disk.total * KEEP_FREE_SHARE))


def per_image(db: Session, settings: HubSettings) -> int:
    """What one more image takes on the Hub: its thumbnail and its records, from what is here already."""
    count, thumbs = db.execute(select(func.count(Item.id), func.avg(Item.thumb_bytes)).where(Item.thumb_bytes > 0)).one()
    thumb = int(thumbs or 0) if (count or 0) >= 100 else THUMB_GUESS
    items = db.scalar(select(func.count(Item.id))) or 0
    records = RECORD_GUESS
    if items >= 10_000:
        size = sum(p.stat().st_size for p in Path(settings.home).glob("hub.db*") if p.is_file())
        records = max(512, min(16 * 1024, size // items))
    return thumb + records


def check_room(db: Session, settings: HubSettings, images: int) -> dict[str, int]:
    """Whether ``images`` more fit on the Hub's disk (with ``keep_free`` left): what they need and what is free.
    Raises ``ServiceError`` 507 when they do not."""
    from .services import ServiceError

    need = int(images * per_image(db, settings) * 1.1)
    free = shutil.disk_usage(settings.home).free
    keep = keep_free(settings)
    if need > free - keep:
        raise ServiceError(
            507,
            f"Hub의 디스크가 모자랍니다: 이미지 {images:,}장을 등록하면 약 {_gb(need)}가 필요한데 남은 공간은 "
            f"{_gb(free)}입니다({_gb(keep)}는 늘 남겨 둡니다). 설정 → 저장 공간에서 정리하거나 Hub 디스크를 늘리세요.",
        )
    return {"hub_bytes": need, "hub_free": free}


def check_running(settings: HubSettings) -> None:
    """An import under way stops when the disk gets this close to full."""
    from .services import ServiceError

    free = shutil.disk_usage(settings.home).free
    keep = keep_free(settings) // 2
    if free < keep:
        what = f"Hub 디스크의 남은 공간이 {_gb(free)}뿐이라 등록을 멈춥니다."
        raise ServiceError(507, f"{what} 설정 → 저장 공간에서 정리한 뒤 다시 가져오세요.")


def _gb(size: int) -> str:
    return f"{size / GB:,.1f} GB" if size >= GB else f"{size / 1024**2:,.0f} MB"


# ---- the database file ---------------------------------------------------------------------------


def database_usage(db: Session, settings: HubSettings) -> dict[str, Any]:
    """The database files' size, how much of it is empty pages (given back by compacting), and compacting now."""
    size = sum(p.stat().st_size for p in Path(settings.home).glob("hub.db*") if p.is_file())
    empty = 0
    if db.get_bind().dialect.name == "sqlite":
        page = db.execute(text("PRAGMA page_size")).scalar() or 0
        empty = (db.execute(text("PRAGMA freelist_count")).scalar() or 0) * page
    return {"bytes": size, "empty": empty, **compacting()}


def compacting() -> dict[str, Any]:
    return {"compacting": _compacting["running"], "last_compact": _compacting["last"]}


def busy(db: Session) -> bool:
    """Jobs that write to the database as they go (imports above all): compacting waits for them."""
    return bool(db.scalar(select(func.count(Job.id)).where(Job.status.in_(("queued", "running")))))


def worth_compacting(db: Session, settings: HubSettings) -> bool:
    usage = database_usage(db, settings)
    return usage["empty"] >= COMPACT_MIN and usage["empty"] >= usage["bytes"] * COMPACT_SHARE


def compact(database: Any, settings: HubSettings) -> dict[str, Any]:
    """VACUUM the database: the file shrinks by its empty pages. Needs about the file's size free on the disk
    for a moment, and holds every write off while it runs (a minute for a gigabyte), so only while no job runs."""
    from .services import ServiceError

    with _lock:
        if _compacting["running"]:
            raise ServiceError(409, "이미 DB를 압축하고 있습니다.")
        with database.session() as db:
            if db.get_bind().dialect.name != "sqlite":
                raise ServiceError(400, "SQLite DB만 여기서 압축합니다.")
            if busy(db):
                raise ServiceError(409, "작업이 돌고 있어 지금은 압축할 수 없습니다. 작업이 끝난 뒤에 하세요.")
            before = database_usage(db, settings)
        if shutil.disk_usage(settings.home).free < before["bytes"]:
            raise ServiceError(507, f"DB를 압축하려면 디스크에 {_gb(before['bytes'])} 정도가 잠시 필요합니다.")
        _compacting.update(running=True, started_at=time.time())
    started = time.monotonic()
    try:
        # outside a transaction: SQLite refuses VACUUM inside one
        with database.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql("VACUUM")
            conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        with database.session() as db:
            after = database_usage(db, settings)["bytes"]
        took = round(time.monotonic() - started, 1)
        result = {"before": before["bytes"], "after": after, "took_s": took, "at": time.time()}
        log.info("database compacted: %.0f MB → %.0f MB in %.0f s", before["bytes"] / 1024**2, after / 1024**2, took)
        _compacting["last"] = result
        return result
    finally:
        _compacting["running"] = False


def compact_in_background(database: Any, settings: HubSettings) -> None:
    """Start compacting (refused at once, with why, when it cannot start); it runs on without the request."""
    from .services import ServiceError

    with database.session() as db:
        if busy(db):
            raise ServiceError(409, "작업이 돌고 있어 지금은 압축할 수 없습니다. 작업이 끝난 뒤에 하세요.")
    if _compacting["running"]:
        raise ServiceError(409, "이미 DB를 압축하고 있습니다.")

    def run() -> None:
        try:
            compact(database, settings)
        except Exception:  # noqa: BLE001 - shown as "not compacted" on the card; the log says why
            log.exception("database compaction")

    threading.Thread(target=run, name="db-compact", daemon=True).start()
