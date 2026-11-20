"""What BADAMOA keeps on a node, by kind, with sizes — measured now and then, never on every heartbeat.

    images         versions received from other nodes (``cache``; managed per version by ``node_data``)
    envs           virtual environments built for templates (``envs/<id>``)
    containers     Docker images BADAMOA pulled or uses
    models         model weights fetched for evaluations, exports, tests and continued training
    templates      template code fetched from the Hub (``cache/templates/<sha256>``)
    template_data  what template code keeps between runs, mostly pretrained weights (``cache/template-data/<name>``)
    runs           work folders of jobs (a finished run leaves nothing; a stopped or failed one stays for continuing)
    uv             uv's download cache, which builds the virtual environments (the server user's, shared)

The node sends the report with a heartbeat after measuring; the Hub adds what each item belongs to
and whether a job uses it now, and asks the node to remove items (``clean``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from badamoa.core import sysinfo
from badamoa.node import cache, containers, envs

KINDS = ("images", "envs", "containers", "models", "templates", "template_data", "runs", "uv")


def _iso(seconds: float | None) -> str | None:
    if not seconds:
        return None
    return datetime.fromtimestamp(seconds, UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def _size(folder: Path) -> tuple[int, int, int]:
    """Bytes, files, and bytes in files that have other names elsewhere too (hard links, e.g. with uv's cache)."""
    total = files = shared = 0
    for root, _, names in os.walk(folder):
        for name in names:
            try:
                stat = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += stat.st_size
            files += 1
            if getattr(stat, "st_nlink", 1) > 1:
                shared += stat.st_size
    return total, files, shared


def _newest(folder: Path) -> float:
    try:
        return max([folder.stat().st_mtime, *(p.stat().st_mtime for p in folder.iterdir())])
    except OSError:
        return 0.0


def _free(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def uv_cache_dir() -> Path | None:
    try:
        done = subprocess.run([envs.find_uv(), "cache", "dir"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None
    text = done.stdout.strip()
    return Path(text) if done.returncode == 0 and text else None


def _folders(parent: Path) -> list[Path]:
    try:
        return sorted(p for p in parent.iterdir() if p.is_dir() and not p.name.endswith((".part", ".lock")))
    except OSError:
        return []


def measure(home: Path, cache_root: Path) -> dict[str, Any]:
    """The report: per kind its items (with sizes) or its total, and the free space of the disks involved."""
    started = time.monotonic()
    kinds: dict[str, Any] = {}
    images_bytes, images_files = cache.usage(cache_root)
    kinds["images"] = {"path": str(cache_root), "bytes": images_bytes, "files": images_files}

    environments = []
    for folder in _folders(home / "envs"):
        marker = folder / "badamoa-env.json"
        if not marker.is_file():
            continue
        entry = next((e for e in envs.listing(home) if e["id"] == folder.name), None)
        size, files, shared = _size(folder)
        used = folder / envs.USED
        environments.append(
            {
                "id": folder.name,
                **(entry or {}),
                "bytes": size,
                "files": files,
                "shared": shared,
                "last_used": _iso(used.stat().st_mtime if used.is_file() else marker.stat().st_mtime),
            }
        )
    kinds["envs"] = {"path": str(home / "envs"), "items": environments}

    images = []
    for entry in containers.listing(home):
        images.append({**entry, **containers.details(home, entry["image"])})
    kinds["containers"] = {"items": images}
    docker_root = containers.storage_root() if images else None
    if docker_root:  # Docker's own folder: Docker, not BADAMOA, decides where images go
        kinds["containers"].update(path=docker_root["path"], desktop=docker_root["desktop"])

    models = []
    for path in sorted((home / "cache" / "models").glob("*")) if (home / "cache" / "models").is_dir() else []:
        if path.is_file() and not path.name.endswith(".part"):
            stat = path.stat()
            models.append({"id": path.name, "blob": path.stem, "bytes": stat.st_size, "last_used": _iso(stat.st_atime)})
    kinds["models"] = {"path": str(home / "cache" / "models"), "items": models}

    for kind, folder in (
        ("templates", home / "cache" / "templates"),
        ("template_data", home / "cache" / "template-data"),
        ("runs", home / "runs"),
    ):
        items = [{"id": f.name, "bytes": _size(f)[0], "last_used": _iso(_newest(f))} for f in _folders(folder)]
        kinds[kind] = {"path": str(folder), "items": items}
    uv = uv_cache_dir()
    if uv is not None and uv.is_dir():  # no uv here (yet): nothing to show
        size, files, shared = _size(uv)
        kinds["uv"] = {"path": str(uv), "bytes": size, "files": files, "shared": shared}

    # each folder's disk: the mount it is on and the room left there
    mounts = sysinfo.MetricsSampler._disks()
    disks: dict[str, Any] = {}
    places = [home, cache_root, uv, *(Path(k["path"]) for k in kinds.values() if k.get("path") and not k.get("desktop"))]
    for where in places:
        if where is None or str(where) in disks:
            continue
        free = _free(Path(where))
        anchor = Path(where).anchor or str(where)
        disks[str(where)] = {"free": free, "anchor": anchor, "mount": cache.mount_of(Path(where), mounts) or anchor}
    return {
        "measured_at": _iso(time.time()),
        "took_s": round(time.monotonic() - started, 1),
        "home": str(home),
        "disks": disks,
        "kinds": kinds,
    }


# ---- removing ----------------------------------------------------------------------------------


def _remove_tree(path: Path) -> int:
    size = _size(path)[0] if path.is_dir() else (path.stat().st_size if path.exists() else 0)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    return size


def clean(
    home: Path,
    kind: str,
    items: list[str],
    everything: bool = False,
    log: Any = print,
    running: set[str] | None = None,
) -> dict[str, Any]:
    """Remove items of one kind (or every one with ``everything``). The Hub has checked no job uses them;
    here an environment being built and the work folders of ``running`` jobs are skipped as well.

    Received images are not removed here: they go version by version (``node_data``).
    """
    running = running or set()
    freed = removed = 0
    if kind == "uv":
        # uv's cache is the server user's, shared with whatever else runs uv there: only what nothing
        # needs any more goes (``uv cache prune``), never the whole of it
        uv = envs.find_uv()
        if any((home / "envs").glob("*.lock")):
            raise RuntimeError("가상환경을 만드는 작업이 있어 uv 캐시를 정리하지 않았습니다. 끝난 뒤에 다시 하세요.")
        where = uv_cache_dir()
        before = _size(where)[0] if where and where.is_dir() else 0
        done = subprocess.run([uv, "cache", "prune"], capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"uv 캐시를 정리하지 못했습니다: {(done.stderr or done.stdout).strip()[-300:]}")
        after = _size(where)[0] if where and where.is_dir() else 0
        freed = max(0, before - after)
        log(f"uv 캐시에서 쓰지 않는 것을 정리했습니다 ({freed / 1024**2:,.0f} MB). 가상환경은 그대로입니다.")
        return {"removed": 1 if freed else 0, "bytes": freed}
    if kind == "containers":
        names = items or ([e["image"] for e in containers.listing(home)] if everything else [])
        for image in names:
            size = containers.details(home, image)["bytes"]
            result = containers.remove(home, image, log)
            removed += 1 if result.get("removed") or result.get("forgotten") else 0
            freed += size if result.get("removed") else 0
        return {"removed": removed, "bytes": freed}
    folders = {
        "envs": home / "envs",
        "models": home / "cache" / "models",
        "templates": home / "cache" / "templates",
        "template_data": home / "cache" / "template-data",
        "runs": home / "runs",
    }
    if kind not in folders:
        raise RuntimeError(f"지울 수 없는 종류입니다: {kind}")
    parent = folders[kind]
    if not parent.is_dir():
        return {"removed": 0, "bytes": 0}
    names = items or ([p.name for p in parent.iterdir() if not p.name.endswith(".lock")] if everything else [])
    for name in names:
        target = parent / Path(name).name  # a bare name: never a path out of the folder
        if not target.exists():
            continue
        if kind == "envs" and (parent / f"{target.name}.lock").exists():
            log(f"만드는 중이라 건너뜀: {target.name}")
            continue
        if kind == "runs" and target.name in running:
            log(f"실행 중인 작업의 폴더라 건너뜀: {target.name}")
            continue
        try:
            freed += _remove_tree(target)
            removed += 1
        except OSError as e:
            log(f"지우지 못했습니다 ({target.name}): {e}")
    log(f"{removed:,}개를 지웠습니다 ({freed / 1024**2:,.0f} MB).")
    return {"removed": removed, "bytes": freed}
