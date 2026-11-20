"""What BADAMOA keeps on a node, by kind, with sizes — measured now and then, never on every heartbeat.

    images         versions received from other nodes (``cache``; managed per version by ``node_data``)
    envs           virtual environments built for templates (``envs/<id>``)
    containers     Docker images BADAMOA pulled or uses
    models         model weights fetched for evaluations, exports, tests and continued training
    templates      template code fetched from the Hub (``cache/templates/<sha256>``)
    template_data  what template code keeps between runs, mostly pretrained weights (``cache/template-data/<name>``)
    runs           work folders of jobs (a finished run leaves nothing; a stopped or failed one stays for continuing)
    uv             uv's download cache, which builds the virtual environments (the server user's, shared;
                   the node's own in the work folder when uv's is on another disk)

Everything but images, Docker images and uv's own cache is in the node's work folder (``places``); what is
still in a folder the node used before (its home) is listed too, with its own ``path``.

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
from badamoa.node import cache, containers, envs, places

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


def uv_cache_dir(where: places.Places | None = None) -> Path | None:
    """The uv cache the node's environments are built from: the node's own (``envs.uv_cache``) or uv's."""
    own = envs.uv_cache(where) if where is not None else None
    if own is not None:
        return own
    try:
        done = subprocess.run(
            [envs.find_uv(), "cache", "dir"], capture_output=True, text=True, timeout=30, check=False, env=envs.code_env()
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None
    text = done.stdout.strip()
    return Path(text) if done.returncode == 0 and text else None


FOLDERS = {
    "envs": ("envs",),
    "models": ("cache", "models"),
    "templates": ("cache", "templates"),
    "template_data": ("cache", "template-data"),
    "runs": ("runs",),
}


def _in_every_root(where: places.Places, kind: str, files: bool = False) -> list[tuple[Path, bool]]:
    """A kind's items in every folder the node keeps them in (the work folder's first, each name once), and
    whether each is in the work folder."""
    found: dict[str, tuple[Path, bool]] = {}
    for base in where.roots():
        parent = base.joinpath(*FOLDERS[kind])
        try:
            entries = sorted(parent.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.endswith((".part", ".lock", ".moving")):
                continue
            if entry.is_file() if files else entry.is_dir():
                found.setdefault(entry.name, (entry, base is where.work))
    return [found[name] for name in sorted(found)]


def _placed(item: dict[str, Any], path: Path, in_work: bool) -> dict[str, Any]:
    return item if in_work else {**item, "path": str(path)}  # still where the node kept it before


def measure(home: Path, cache_root: Path) -> dict[str, Any]:
    """The report: per kind its items (with sizes) or its total, and the free space of the disks involved."""
    started = time.monotonic()
    where = places.of(home)
    kinds: dict[str, Any] = {}
    images_bytes, images_files = cache.usage(cache_root)
    kinds["images"] = {"path": str(cache_root), "bytes": images_bytes, "files": images_files}

    environments = []
    described = {e["id"]: e for e in envs.listing(home)}
    for folder, in_work in _in_every_root(where, "envs"):
        marker = folder / "badamoa-env.json"
        if not marker.is_file():
            continue
        size, files, shared = _size(folder)
        used = folder / envs.USED
        item = {
            "id": folder.name,
            **described.get(folder.name, {}),
            "bytes": size,
            "files": files,
            "shared": shared,
            "last_used": _iso(used.stat().st_mtime if used.is_file() else marker.stat().st_mtime),
        }
        environments.append(_placed(item, folder, in_work))
    kinds["envs"] = {"path": str(where.envs), "items": environments}

    images = []
    for entry in containers.listing(home):
        images.append({**entry, **containers.details(home, entry["image"])})
    kinds["containers"] = {"items": images}
    docker_root = containers.storage_root() if images else None
    if docker_root:  # Docker's own folder: Docker, not BADAMOA, decides where images go
        kinds["containers"].update(path=docker_root["path"], desktop=docker_root["desktop"])

    models = []
    for path, in_work in _in_every_root(where, "models", files=True):
        stat = path.stat()
        item = {"id": path.name, "blob": path.stem, "bytes": stat.st_size, "last_used": _iso(stat.st_atime)}
        models.append(_placed(item, path, in_work))
    kinds["models"] = {"path": str(where.models), "items": models}

    for kind, folder in (("templates", where.templates), ("template_data", where.template_data), ("runs", where.runs)):
        items = [
            _placed({"id": f.name, "bytes": _size(f)[0], "last_used": _iso(_newest(f))}, f, in_work)
            for f, in_work in _in_every_root(where, kind)
        ]
        kinds[kind] = {"path": str(folder), "items": items}
    uv = uv_cache_dir(where)
    if uv is not None and uv.is_dir():  # no uv here (yet): nothing to show
        size, files, shared = _size(uv)
        kinds["uv"] = {"path": str(uv), "bytes": size, "files": files, "shared": shared}
        if envs.uv_cache(where) is not None:  # the node's own, beside its environments
            kinds["uv"]["work"] = True

    # each folder's disk: the mount it is on and the room left there (and of every item kept elsewhere)
    mounts = sysinfo.MetricsSampler._disks()
    disks: dict[str, Any] = {}
    elsewhere = [Path(i["path"]) for k in kinds.values() for i in k.get("items") or [] if i.get("path")]
    kept = [Path(k["path"]) for k in kinds.values() if k.get("path") and not k.get("desktop")]
    folders = [home, where.work, cache_root, uv, *kept]
    for folder in [*folders, *elsewhere]:
        if folder is None or str(folder) in disks:
            continue
        free = _free(Path(folder))
        anchor = Path(folder).anchor or str(folder)
        disks[str(folder)] = {"free": free, "anchor": anchor, "mount": cache.mount_of(Path(folder), mounts) or anchor}
    return {
        "measured_at": _iso(time.time()),
        "took_s": round(time.monotonic() - started, 1),
        "home": str(home),
        "work": str(where.work),
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
    there = places.of(home)
    if kind == "uv":
        # uv's cache is the server user's, shared with whatever else runs uv there: only what nothing
        # needs any more goes (``uv cache prune``), never the whole of it
        uv = envs.find_uv()
        if any(any(base.joinpath("envs").glob("*.lock")) for base in there.roots()):
            raise RuntimeError("가상환경을 만드는 작업이 있어 uv 캐시를 정리하지 않았습니다. 끝난 뒤에 다시 하세요.")
        where = uv_cache_dir(there)
        before = _size(where)[0] if where and where.is_dir() else 0
        done = subprocess.run(
            [uv, "cache", "prune"], capture_output=True, text=True, check=False, env=envs.code_env(envs.uv_env(there))
        )
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
    if kind not in FOLDERS:
        raise RuntimeError(f"지울 수 없는 종류입니다: {kind}")
    parents = there.everywhere(*FOLDERS[kind])  # the work folder's, and one the node used before
    if not parents:
        return {"removed": 0, "bytes": 0}
    names = items or (
        sorted({p.name for parent in parents for p in parent.iterdir() if not p.name.endswith(".lock")})
        if everything
        else []
    )
    for name in names:
        name = Path(name).name  # a bare name: never a path out of the folder
        if kind == "envs" and any((parent / f"{name}.lock").exists() for parent in parents):
            log(f"만드는 중이라 건너뜀: {name}")
            continue
        if kind == "runs" and name in running:
            log(f"실행 중인 작업의 폴더라 건너뜀: {name}")
            continue
        targets = [parent / name for parent in parents if (parent / name).exists() and name]
        if not targets:
            continue
        try:
            for target in targets:
                freed += _remove_tree(target)
            removed += 1
        except OSError as e:
            log(f"지우지 못했습니다 ({name}): {e}")
    log(f"{removed:,}개를 지웠습니다 ({freed / 1024**2:,.0f} MB).")
    return {"removed": removed, "bytes": freed}
