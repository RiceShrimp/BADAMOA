"""This machine's folders for the Hub's "pick a folder" dialog: names and counts, never file contents.

An empty path gives the places to start from (drives on Windows; ``/``, the home folder and
mounted disks elsewhere). A folder gives its sub-folders, how many images sit directly in it,
and what dataset layout it looks like (YOLO, COCO, one folder per class) so the right one is easy to spot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import psutil

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_FOLDERS = 500
GUESS_CHILDREN = 150  # sub-folders that get a quick look of their own
SYSTEM_MOUNTS = ("/boot", "/snap", "/var/snap", "/run", "/proc", "/sys", "/dev")


def roots() -> list[dict[str, str]]:
    places: list[dict[str, str]] = []

    def add(path: str, name: str) -> None:
        if path and all(p["path"] != path for p in places) and Path(path).is_dir():
            places.append({"name": name, "path": path, "guess": ""})

    if sys.platform == "win32":
        for part in psutil.disk_partitions(all=False):
            add(part.mountpoint, part.mountpoint.rstrip("\\"))
        add(str(Path.home()), f"홈 ({Path.home().name})")
        return places
    add("/", "/ (루트)")
    add(str(Path.home()), f"홈 ({Path.home()})")
    for part in psutil.disk_partitions(all=False):
        point = part.mountpoint
        if point != "/" and not point.startswith(SYSTEM_MOUNTS) and part.fstype not in ("squashfs", "tmpfs"):
            add(point, point)
    for common in ("/data", "/mnt", "/media", "/srv"):
        add(common, common)
    return places


def _quick_guess(folder: Path) -> str:
    """A few file checks, no listing: cheap enough for every sub-folder shown."""
    try:
        if (folder / "data.yaml").is_file() or (folder / "dataset.yaml").is_file():
            return "yolo"
        if (folder / "images").is_dir() and (folder / "labels").is_dir():
            return "yolo"
        if (folder / "annotations").is_dir():
            return "coco"
    except OSError:
        return ""
    return ""


def _has_images(folder: Path, look: int = 30) -> bool:
    try:
        with os.scandir(folder) as entries:
            for n, entry in enumerate(entries):
                if n >= look:
                    break
                if entry.is_file() and Path(entry.name).suffix.lower() in IMAGES:
                    return True
    except OSError:
        return False
    return False


def _guess(folder: Path, subfolders: list[Path]) -> str:
    found = _quick_guess(folder)
    if found:
        return found
    if any(p.suffix.lower() == ".json" and "instances" in p.name for p in folder.glob("*.json")):
        return "coco"
    # One folder per class, directly or under train/val/test.
    splits = [p for p in subfolders if p.name.lower() in ("train", "val", "valid", "test")]
    classes = [p for p in (sorted(splits[0].iterdir()) if splits else subfolders) if p.is_dir()][:5]
    if len(classes) >= 2 and sum(_has_images(p) for p in classes[:3]) >= 2:
        return "imagefolder"
    return ""


def _hidden_on_windows(entry: os.DirEntry) -> bool:
    """Folders Windows itself hides ($Recycle.Bin, System Volume Information …)."""
    if sys.platform != "win32":
        return False
    try:
        return bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x6)  # HIDDEN | SYSTEM
    except (OSError, AttributeError):
        return False


def listing(path: str, hidden: bool = False) -> dict[str, Any]:
    if not path.strip():
        return {"path": "", "parent": None, "folders": roots(), "images": 0, "guess": "", "truncated": False}
    folder = Path(path.strip()).expanduser()
    if not folder.is_dir():
        raise ValueError(f"폴더가 아니거나 없습니다: {folder}")
    subfolders: list[Path] = []
    images = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if not hidden and (entry.name.startswith(".") or _hidden_on_windows(entry)):
                    continue
                try:
                    if entry.is_dir():
                        subfolders.append(Path(entry.path))
                    elif Path(entry.name).suffix.lower() in IMAGES:
                        images += 1
                except OSError:
                    continue
    except PermissionError as e:
        raise ValueError(f"이 폴더를 읽을 권한이 없습니다: {folder}") from e
    subfolders.sort(key=lambda p: p.name.lower())
    shown = subfolders[:MAX_FOLDERS]
    parent = folder.parent
    return {
        "path": str(folder),
        "parent": "" if parent == folder else str(parent),  # "" goes back to the starting places
        "folders": [
            {"name": p.name, "path": str(p), "guess": _quick_guess(p) if n < GUESS_CHILDREN else ""}
            for n, p in enumerate(shown)
        ],
        "images": images,
        "guess": _guess(folder, subfolders),
        "truncated": len(subfolders) > MAX_FOLDERS,
    }
