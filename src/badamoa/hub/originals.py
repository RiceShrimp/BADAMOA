"""Original images that live on another server: the Hub asks that node for the file and keeps a copy.

The item view shows the original, not the 320 px thumbnail. When the Hub cannot read the file
itself (it sits on a remote node), it asks the node over the question channel; the node checks
the file is the one registered (same size) and uploads it. Copies are kept in ``cache/originals``
up to a size limit, the oldest used going first, so paging back and forth stays quick.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from sqlalchemy.orm import Session

from . import asks, services
from .catalog_models import Item, StorageLocation
from .config import HubSettings
from .models import Node
from .services import ServiceError

CACHE_BYTES = 4 * 1024**3  # copies kept from remote nodes
SINCE = (0, 32, 0)  # nodes from here on send originals
_KEY = re.compile(r"^item-\d+-[0-9a-f]{16}$")
_lock = threading.Lock()


def _folder(settings: HubSettings) -> Path:
    folder = settings.home / "cache" / "originals"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def key_of(item: Item) -> str:
    return f"item-{item.id}-{item.sha256[:16]}"


def cached(settings: HubSettings, item: Item) -> Path | None:
    path = _folder(settings) / key_of(item)
    if path.is_file() and path.stat().st_size == item.bytes:
        os.utime(path)  # recently used: kept longer
        return path
    return None


def store(settings: HubSettings, key: str, content: bytes) -> None:
    """What a node uploads for a question: written whole, then the cache trimmed."""
    if not _KEY.match(key):
        raise ServiceError(400, "알 수 없는 파일 이름입니다.")
    folder = _folder(settings)
    part = folder / f"{key}.part"
    part.write_bytes(content)
    part.replace(folder / key)
    with _lock:
        files = [p for p in folder.iterdir() if p.is_file() and not p.name.endswith(".part")]
        files.sort(key=lambda p: p.stat().st_mtime)  # the least recently used first
        total = sum(p.stat().st_size for p in files)
        for old in files:
            if total <= CACHE_BYTES:
                break
            total -= old.stat().st_size
            old.unlink(missing_ok=True)


def holder(db: Session, item: Item) -> tuple[Node, str] | None:
    """The node with the file and the file's path there."""
    location = db.get(StorageLocation, item.location_id) if item.location_id else None
    node = db.get(Node, location.node_id) if location else None
    if location is None or node is None:
        return None
    sep = "\\" if re.match(r"^[A-Za-z]:[\\/]", location.root) or "\\" in location.root else "/"
    return node, location.root.rstrip("/\\") + sep + item.rel_path.replace("/", sep)


def fetch(db: Session, settings: HubSettings, item: Item) -> Path:
    """The original from the node that has it (or from the cache). Raises with the reason when it cannot."""
    hit = cached(settings, item)
    if hit:
        return hit
    found = holder(db, item)
    if found is None:
        raise ServiceError(404, "이 이미지가 있는 노드를 알 수 없습니다.")
    node, path = found
    if services.node_status(node, settings) == "offline":
        raise ServiceError(409, f"원본이 있는 노드({node.name})가 꺼져 있습니다.")
    if not asks.answers_questions(node.agent_version or "", SINCE):
        raise ServiceError(409, f"노드({node.name})의 프로그램이 원본 보내기를 모릅니다. 노드를 업데이트하세요.")
    node_id = node.id
    db.close()  # nothing more from the database while the node sends the file
    asks.ask(node_id, "file", {"path": path, "bytes": item.bytes, "key": key_of(item)}, wait=60)
    hit = cached(settings, item)
    if hit is None:
        raise ServiceError(502, "노드가 원본을 보냈다고 했지만 받은 파일이 없습니다.")
    return hit
