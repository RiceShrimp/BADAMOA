"""Where a node keeps what grows: its work folder.

    <work>/envs/<id>               virtual environments built for templates
    <work>/runs/<job id>           work folders of jobs (a stopped training's checkpoints, for continuing)
    <work>/cache/models            model weights fetched for evaluations, exports, tests and continued training
    <work>/cache/templates         template code fetched from the Hub
    <work>/cache/template-data     what template code keeps between runs (pretrained weights)
    <work>/cache/wheels, ultralytics
    <work>/uv-cache                the node's own uv cache (``envs.uv_env``): it goes with the node

The node's config, logs and lock stay in its home. Like received images (``cache``), the work folder is
the one an admin set for the node; otherwise the disk with the most room, picked once and remembered
(``work-dir.txt`` in the home), so it does not wander as disks fill up. On the home's own disk it is the
home itself, as before 0.74.0.

What a node kept elsewhere before (in its home, or in a work folder an admin has since changed) stays
usable: every folder the node has used is a ``root`` that lookups go through. Virtual environments are
used where they are (one cannot be moved); work folders, weights, template code and template caches move
into the work folder once, at a moment no job runs (``move_old``).
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from badamoa.core import sysinfo
from badamoa.node import cache

SAVED = "work-dir.txt"  # the node's own pick
USED = "work-dirs.json"  # every work folder this node has used besides its home
FOLDER = "badamoa-work"
ENV = "BADAMOA_WORK_DIR"  # the work folder, for the jobs the node starts
MOVED = (("runs",), ("cache", "models"), ("cache", "templates"), ("cache", "template-data"))

_chosen: dict[str, Path] = {}  # node home → the work folder its agent resolved (the admin's or its pick)


def _key(path: Path) -> str:
    text = os.path.normpath(str(Path(path).absolute()))
    return text.lower() if os.name == "nt" else text


def same(a: Path, b: Path) -> bool:
    return _key(a) == _key(b)


def _existing(path: Path) -> Path:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def same_disk(a: Path, b: Path) -> bool:
    """Whether two folders (or where they would be made) are on one file system: hard links work between them."""
    try:
        return os.stat(_existing(a)).st_dev == os.stat(_existing(b)).st_dev
    except OSError:
        return False


def writable(folder: Path) -> bool:
    """Make the folder if needed and write a file in it."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f".badamoa-probe-{uuid.uuid4().hex[:8]}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def choose(home: Path) -> Path:
    """The disk with the most room that the node can write to; on the disk of the node's home, the home itself."""
    disks = [d for d in sysinfo.MetricsSampler._disks() if (d.get("total_gb") or 0) >= cache.MIN_DISK_GB]
    at_home = cache.mount_of(home, disks)
    for disk in sorted(disks, key=lambda d: -(d.get("free_gb") or 0)):
        if disk.get("mount") == at_home:
            return home
        folder = Path(str(disk["mount"])) / FOLDER
        if writable(folder):
            return folder
    return home


def _remember(home: Path, work: Path) -> None:
    if same(work, home):
        return
    used = used_before(home)
    if any(same(work, p) for p in used):
        return
    try:
        (home / USED).write_text(json.dumps([str(p) for p in [*used, work]], ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def used_before(home: Path) -> list[Path]:
    try:
        return [Path(p) for p in json.loads((home / USED).read_text(encoding="utf-8")) if p]
    except (OSError, ValueError, TypeError):
        return []


def _saved(home: Path) -> Path | None:
    try:
        text = (home / SAVED).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(text) if text else None


def root(home: Path, setting: str = "") -> tuple[Path, str]:
    """The work folder in use: the admin's choice, else the one picked before, else a new pick (remembered).

    Returns it and, when the admin's folder cannot be used, why (the node's own pick is used meanwhile).
    A pick whose folder has gone (a disk no longer mounted) is made again.
    """
    home = Path(home)
    error = ""
    if setting.strip():
        wanted = Path(setting.strip())
        if writable(wanted):
            _remember(home, wanted)
            return wanted, ""
        error = f"관리자가 정한 작업 폴더 {wanted}을(를) 만들거나 쓸 수 없어 노드가 고른 곳을 씁니다."
    picked = _saved(home)
    if picked is None or not (picked.is_dir() or same(picked, home)):
        picked = choose(home)
        try:
            (home / SAVED).write_text(str(picked), encoding="utf-8")
        except OSError:
            pass
    _remember(home, picked)
    return picked, error


def use(home: Path, work: Path | None) -> None:
    """The node agent's resolved work folder, for what it does itself (jobs get ``BADAMOA_WORK_DIR``)."""
    if work is None:
        _chosen.pop(_key(home), None)
    else:
        _chosen[_key(home)] = Path(work)


@dataclass(frozen=True)
class Places:
    home: Path
    work: Path

    @property
    def envs(self) -> Path:
        return self.work / "envs"

    @property
    def runs(self) -> Path:
        return self.work / "runs"

    @property
    def models(self) -> Path:
        return self.work / "cache" / "models"

    @property
    def templates(self) -> Path:
        return self.work / "cache" / "templates"

    @property
    def template_data(self) -> Path:
        return self.work / "cache" / "template-data"

    @property
    def wheels(self) -> Path:
        return self.work / "cache" / "wheels"

    @property
    def ultralytics(self) -> Path:
        return self.work / "cache" / "ultralytics"  # Ultralytics' settings, not the user's own

    def roots(self) -> list[Path]:
        """The work folder first, then every other folder things may still be in: the home, earlier work folders."""
        found: list[Path] = []
        for path in (self.work, self.home, *used_before(self.home)):
            if any(same(path, p) for p in found):
                continue
            if path is self.work or path.is_dir():
                found.append(path)
        return found

    def find(self, *parts: str) -> Path:
        """Where something is (the first root that has it), else where it goes: in the work folder."""
        for base in self.roots():
            candidate = base.joinpath(*parts)
            if candidate.exists():
                return candidate
        return self.work.joinpath(*parts)

    def everywhere(self, *parts: str) -> list[Path]:
        """That folder in every root that has it."""
        return [base.joinpath(*parts) for base in self.roots() if base.joinpath(*parts).exists()]


def of(home: Path | str) -> Places:
    """The node's places: its agent's work folder, else the one a job was given, else the one picked before,
    else the home."""
    home = Path(home)
    work = _chosen.get(_key(home))
    if work is None and os.environ.get(ENV):
        work = Path(os.environ[ENV])
    if work is None:
        picked = _saved(home)
        work = picked if picked is not None and picked.is_dir() else home
    return Places(home=home, work=work)


# ---- moving what was kept elsewhere -----------------------------------------------------------------------


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for current, _, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(current, name)).st_size
            except OSError:
                pass
    return total


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _move(source: Path, target: Path) -> None:
    """Rename on one disk; across disks copy beside the target, rename, then remove the source, so a move cut
    short leaves the source as it was."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(source, target)
        return
    except OSError:
        pass
    temporary = target.with_name(f".{target.name}.moving")
    if temporary.exists():
        _remove(temporary)
    if source.is_dir():
        shutil.copytree(source, temporary, symlinks=True)  # an export's links to images stay links
    else:
        shutil.copy2(source, temporary)
    os.rename(temporary, target)
    _remove(source)


def pending(where: Places) -> bool:
    """Whether anything is still to move into the work folder."""
    for base in where.roots()[1:]:
        for parts in MOVED:
            folder = base.joinpath(*parts)
            try:
                if folder.is_dir() and any(not p.name.endswith((".part", ".lock")) for p in folder.iterdir()):
                    return True
            except OSError:
                continue
    return False


def move_old(where: Places, log: Callable[[str], Any] = print) -> dict[str, int]:
    """Move work folders, weights, template code and template caches kept outside the work folder into it.

    Call only while no job runs. Something already in the work folder under the same name is the same
    thing (named by content or by job): the copy elsewhere goes. Environments stay where they are.
    """
    moved = removed = size = failed = 0
    for base in where.roots()[1:]:
        for parts in MOVED:
            folder = base.joinpath(*parts)
            if not folder.is_dir() or same(folder, where.work.joinpath(*parts)):
                continue
            for source in sorted(folder.iterdir()):
                if source.name.endswith((".part", ".lock")) or source.name.endswith(".moving"):
                    continue
                target = where.work.joinpath(*parts, source.name)
                try:
                    bytes_ = _size(source)
                    if target.exists():
                        _remove(source)
                        removed += 1
                    else:
                        _move(source, target)
                        moved += 1
                    size += bytes_
                except OSError as e:
                    failed += 1
                    log(f"옮기지 못했습니다 ({source}): {e}")
            try:
                if not any(folder.iterdir()):
                    folder.rmdir()
            except OSError:
                pass
    if moved or removed:
        mb = size / 1024**2
        log(f"작업 폴더 {where.work}(으)로 {moved:,}개를 옮겼습니다 ({mb:,.0f} MB, 같은 것이 있어 지운 것 {removed:,}개).")
    return {"moved": moved, "removed": removed, "bytes": size, "failed": failed}


# ---- the disks the node writes to -----------------------------------------------------------------------


def disk_of(path: Path, disks: list[dict[str, Any]]) -> dict[str, Any]:
    """The disk a folder is (or will be) on: its mount point, room left and size, in GB."""
    entry: dict[str, Any] = {"path": str(path), "mount": cache.mount_of(_existing(Path(path)), disks)}
    try:
        usage = shutil.disk_usage(_existing(Path(path)))
        entry.update(free_gb=round(usage.free / 1024**3, 1), total_gb=round(usage.total / 1024**3, 1))
    except OSError:
        entry.update(free_gb=None, total_gb=None)
    return entry
