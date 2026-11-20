"""Work folders jobs leave on a node, and removing the old ones.

Every job works in ``<work folder>/runs/<job id>`` (``places``; before 0.74.0 the node's home). A
finished run removes its folder (everything it produced is on the Hub by then); a failed or stopped
one leaves it, so "continue" can pick up the local checkpoint. The Hub keeps the latest checkpoint as
well (0.13.0), so a folder nobody came back to for ``days`` days only takes disk space: the node
removes it, never while a job runs.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from badamoa.node import places

log = logging.getLogger("badamoa.node.workdirs")

SWEEP_EVERY_S = 3600.0


def _newest(folder: Path) -> float:
    """When anything in the folder last changed: the folder and two levels below it (runs write into out/…)."""
    newest = folder.stat().st_mtime
    for top in os.scandir(folder):
        try:
            newest = max(newest, top.stat(follow_symlinks=False).st_mtime)
            if top.is_dir(follow_symlinks=False):
                for inner in os.scandir(top.path):
                    newest = max(newest, inner.stat(follow_symlinks=False).st_mtime)
        except OSError:
            continue
    return newest


def _size(folder: Path) -> int:
    total = 0
    for root, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def sweep(home: Path, days: int, running: set[str], now: float | None = None) -> tuple[int, int]:
    """Remove work folders untouched for ``days`` days that no running job owns. Returns (folders, bytes)."""
    roots = places.of(Path(home)).everywhere("runs")  # the work folder's, and one the node used before
    if days <= 0 or not roots:
        return 0, 0
    cutoff = (now if now is not None else time.time()) - days * 86400
    removed = freed = 0
    for folder in [f for root in roots for f in root.iterdir()]:
        if not folder.is_dir() or folder.name in running:
            continue
        try:
            if _newest(folder) >= cutoff:
                continue
            size = _size(folder)
        except OSError:
            continue
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            removed, freed = removed + 1, freed + size
    if removed:
        log.info("작업 폴더 %d개 정리 (%.1f MB, %d일 넘게 그대로였던 것)", removed, freed / 1024**2, days)
    return removed, freed
