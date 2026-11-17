"""Where a node keeps images it received from other nodes: one folder, by content.

    <cache>/images/ab/cd/<sha256><ext>

Files are named by their SHA-256, so an image shared by several versions or datasets is kept
once and a later version only brings what is new. The folder is the one an admin set for the
node; otherwise the node picks the disk with the most room once and remembers it
(``cache-dir.txt`` in the node's home), so it does not wander as disks fill up.
"""

from __future__ import annotations

import os
import shutil
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


def _mount_of(path: Path, disks: list[dict[str, Any]]) -> str:
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
    if best.get("mount") == _mount_of(home, disks):
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
