"""Where a node keeps images it received from other nodes: one folder, by content.

    <cache>/images/ab/cd/<sha256><ext>

Files are named by their SHA-256, so an image shared by several versions or datasets is kept
once and a later version only brings what is new. The folder is the one an admin set for the
node; otherwise the node picks the disk with the most room once and remembers it
(``cache-dir.txt`` in the node's home), so it does not wander as disks fill up.

The Hub decides what stays: ``check`` counts what is there of each version the Hub says the node
keeps, and with ``clean`` removes the files none of them needs (a copy someone removed, a version
removed for good, an interrupted transfer nobody came back to).
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from badamoa.core import sysinfo

SAVED = "cache-dir.txt"
FOLDER = "badamoa-cache"
MIN_DISK_GB = 2.0


def path_of(root: Path, sha256: str, ext: str) -> Path:
    return root / "images" / sha256[:2] / sha256[2:4] / f"{sha256}{ext.lower()}"


def rel_of(sha256: str, ext: str) -> str:
    return f"images/{sha256[:2]}/{sha256[2:4]}/{sha256}{ext.lower()}"


def has(root: Path, sha256: str, ext: str, size: int) -> bool:
    try:
        return path_of(root, sha256, ext).stat().st_size == size
    except OSError:
        return False


def mount_of(path: Path, disks: list[dict[str, Any]]) -> str:
    """The mount point a path lies on (the longest one that contains it)."""
    text = str(path.resolve()).lower()
    best = ""
    for disk in disks:
        mount = str(disk.get("mount") or "")
        if text.startswith(mount.lower()) and len(mount) > len(best):
            best = mount
    return best


def choose(home: Path) -> Path:
    """The disk with the most room; on the disk of the node's home, a folder there."""
    disks = [d for d in sysinfo.MetricsSampler._disks() if (d.get("total_gb") or 0) >= MIN_DISK_GB]
    fallback = home / "cache"
    if not disks:
        return fallback
    best = max(disks, key=lambda d: d.get("free_gb") or 0)
    if best.get("mount") == mount_of(home, disks):
        return fallback
    return Path(str(best["mount"])) / FOLDER  # made when the first images arrive


def root(home: Path, setting: str = "") -> Path:
    """The folder in use: the admin's choice, else the one picked before, else a new pick (remembered)."""
    if setting.strip():
        return Path(setting.strip())
    saved = home / SAVED
    try:
        text = saved.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    except OSError:
        pass
    chosen = choose(home)
    try:
        saved.write_text(str(chosen), encoding="utf-8")
    except OSError:
        pass
    return chosen


def usage(folder: Path) -> tuple[int, int]:
    """Bytes and files kept (a walk over the folder: done now and then, not on every heartbeat)."""
    size = count = 0
    images = folder / "images"
    if not images.is_dir():
        return 0, 0
    for current, _, names in os.walk(images):
        for name in names:
            if name.endswith(".part") or ".part-" in name:
                continue
            try:
                size += os.path.getsize(os.path.join(current, name))
                count += 1
            except OSError:
                pass
    return size, count


def free_gb(folder: Path) -> float | None:
    """Room left on the disk the folder is (or will be) on."""
    probe = folder
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / 1024**3, 1)
    except OSError:
        return None


# ---- checking against the Hub ----------------------------------------------------------------


def tally(root: Path, items: Iterable[dict[str, Any]], needed: set[str]) -> dict[str, int]:
    """How much of a version is here (``items``: its images as the Hub lists them). Adds their hashes to ``needed``."""
    count = size = 0
    for item in items:
        needed.add(item["sha256"])
        if has(root, item["sha256"], item["ext"], int(item["bytes"])):
            count, size = count + 1, size + int(item["bytes"])
    return {"items": count, "bytes": size}


def sweep(root: Path, needed: set[str], clean: bool, since: float | None = None) -> dict[str, dict[str, int]]:
    """The files none of the images in ``needed`` is; removed with ``clean``.

    A file that arrived at ``since`` or later is left alone: it belongs to a transfer that began
    after the list of what to keep was made.
    """
    unknown = {"files": 0, "bytes": 0}
    removed = {"files": 0, "bytes": 0}
    images = root / "images"
    if not images.is_dir():
        return {"unknown": unknown, "removed": removed}
    since = time.time() if since is None else since
    for current, _, names in os.walk(images):
        for name in names:
            if name.endswith(".part") or ".part-" in name or name.split(".", 1)[0] in needed:
                continue
            path = os.path.join(current, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            unknown["files"] += 1
            unknown["bytes"] += stat.st_size
            # ctime: when it arrived here (a transfer may keep the original's modification time)
            if clean and max(stat.st_mtime, stat.st_ctime) < since:
                try:
                    os.remove(path)
                except OSError:
                    continue
                removed["files"] += 1
                removed["bytes"] += stat.st_size
    return {"unknown": unknown, "removed": removed}
