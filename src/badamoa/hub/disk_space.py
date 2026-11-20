"""Disks a node writes to that are nearly full: its home, its work folder, received images, Docker's images.

The node sends where each of these is with every heartbeat (``metrics.places``: the folder, its disk and
the room left there); nodes before 0.74.0 send the home and the image folder only, read here from the
storage report and ``metrics.cache``. A disk is short of room below 10 GB, or below 5 % of its size when
that is less than 50 GB (a big disk with hundreds of GB left is fine); it is full below 3 GB, where jobs
start failing. A disk several of them share is said once, with all of them.

BADAMOA moves its own folders (the work folder and received images go to the disk with the most room);
Docker's images it never moves: for Docker the warning says how an admin moves Docker's data-root.
"""

from __future__ import annotations

from typing import Any

LOW_GB = 10.0
LOW_PCT = 5.0
LOW_CAP_GB = 50.0  # 5 % of a 7 TB disk is no shortage
FULL_GB = 3.0
WHAT = {"home": "BADAMOA 홈", "work": "작업 폴더", "images": "받은 이미지", "docker": "Docker 이미지"}
ORDER = ("home", "work", "images", "docker")


def level(free_gb: float | None, total_gb: float | None) -> str:
    """"full", "low", or "" for a disk with room (or one whose room is not known)."""
    if free_gb is None:
        return ""
    if free_gb < FULL_GB:
        return "full"
    limit = LOW_GB
    if total_gb:
        limit = max(LOW_GB, min(total_gb * LOW_PCT / 100, LOW_CAP_GB))
    return "low" if free_gb < limit else ""


def _mount_of(path: str, disks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The disk a node path is on, by the longest mount point that contains it (for nodes before 0.74.0)."""
    text = path.lower().replace("\\", "/")
    best: dict[str, Any] | None = None
    for disk in disks:
        mount = str(disk.get("mount") or "").lower().replace("\\", "/")
        if mount and text.startswith(mount) and len(mount) > len(str((best or {}).get("mount") or "")):
            best = disk
    return best


def places(metrics: dict[str, Any], storage: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Where BADAMOA writes on the node: each with its folder, disk and room (``kind``: home, work, images, docker)."""
    sent = metrics.get("places")
    if isinstance(sent, list):
        return [p for p in sent if isinstance(p, dict) and p.get("kind") in WHAT]
    disks = metrics.get("disks") or []
    found = []
    home = str((storage or {}).get("home") or "")
    images = str((metrics.get("cache") or {}).get("dir") or "")
    for kind, path in (("home", home), ("images", images)):
        disk = _mount_of(path, disks) if path else None
        if disk is not None:
            room = {"free_gb": disk.get("free_gb"), "total_gb": disk.get("total_gb")}
            found.append({"kind": kind, "path": path, "mount": disk["mount"], **room})
    return found


def alerts(metrics: dict[str, Any], storage: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The nearly full disks, fullest first: ``mount``, room, ``level``, what is on it (``uses``) and, when
    Docker's images are there, Docker's folder (``docker``) for the admin's how-to."""
    by_disk: dict[str, dict[str, Any]] = {}
    for place in places(metrics, storage):
        if place.get("desktop"):
            continue  # Docker Desktop's virtual disk: its room is not one of this machine's disks
        found = level(place.get("free_gb"), place.get("total_gb"))
        if not found:
            continue
        key = str(place.get("mount") or place.get("path") or "")
        entry = by_disk.setdefault(
            key,
            {
                "mount": key,
                "free_gb": place.get("free_gb"),
                "total_gb": place.get("total_gb"),
                "level": found,
                "uses": [],
                "paths": {},
            },
        )
        if place["kind"] not in entry["uses"]:
            entry["uses"].append(place["kind"])
            entry["paths"][place["kind"]] = place.get("path") or ""
        if place["kind"] == "docker":
            entry["docker"] = place.get("path") or ""
    found = list(by_disk.values())
    for entry in found:
        entry["uses"].sort(key=ORDER.index)
        entry["label"] = " · ".join(WHAT[u] for u in entry["uses"])
    return sorted(found, key=lambda e: (e["level"] != "full", e.get("free_gb") or 0))


DOCKER_MOVE_SINCE = (0, 75, 0)  # a node moves Docker's images itself (``node.docker_move``) from this BADAMOA on
ACTIVE = ("queued", "running")


def docker_move_state(node: Any, busy: int = 0) -> dict[str, Any]:
    """Whether BADAMOA can move this node's Docker images for an admin, and if not why; where they are and where
    they could go. ``busy``: BADAMOA jobs queued or running on the node."""
    metrics = node.metrics or {}
    docker = next((p for p in places(metrics) if p.get("kind") == "docker"), None)
    path = str((docker or {}).get("path") or "")
    linux_docker = bool(path) and not (docker or {}).get("desktop") and path.startswith("/")
    found = {"path": path, "suggest": docker_target(metrics, path) if linux_docker else "", "possible": False, "why": ""}
    version = tuple(int(n) for n in (node.agent_version or "0").split(".")[:3] if n.isdigit())
    account = (node.info or {}).get("account") or {}
    if docker is None:
        found["why"] = "이 서버의 Docker 이미지 위치를 아직 모릅니다(Docker가 없거나 노드가 알려 오지 않음)."
    elif docker.get("desktop"):
        found["why"] = "Docker Desktop은 그 앱의 설정(Resources → Disk image location)에서 옮깁니다."
    elif node.os != "linux" or not account.get("admin"):
        found["why"] = (
            "sudo로 설치한 Linux 노드만 Docker 설정을 바꿀 수 있습니다. 노드 추가의 sudo 명령으로 다시 설치하면 됩니다."
        )
    elif version < DOCKER_MOVE_SINCE:
        found["why"] = f"노드를 {'.'.join(str(n) for n in DOCKER_MOVE_SINCE)} 이상으로 업데이트해야 합니다."
    elif busy:
        found["why"] = f"이 서버에 BADAMOA 작업 {busy}개가 있습니다. 끝난 뒤에 옮기세요."
    else:
        found["possible"] = True
    return found


def docker_target(metrics: dict[str, Any], current: str) -> str:
    """A folder to suggest for Docker's data-root: ``docker`` on the disk with the most room (not the current one)."""
    disks = [d for d in metrics.get("disks") or [] if (d.get("total_gb") or 0) >= 2]
    here = _mount_of(current, disks)
    others = [d for d in disks if d is not here]
    if not others:
        return ""
    best = max(others, key=lambda d: d.get("free_gb") or 0)
    mount = str(best.get("mount") or "")
    if not mount or mount.startswith(("/boot", "/snap")):
        return ""
    return mount.rstrip("/") + "/docker"
