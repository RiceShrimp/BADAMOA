"""A node takes BADAMOA off its machine when the Hub deletes it: the service, the program, its settings.

Data folders (the images and labels datasets were registered from) are never touched; the node's work
folders (``places``: environments, work folders of jobs, fetched files, its uv cache) and the images it
received from other nodes (``cache``) go with its home when BADAMOA made them (``badamoa-work``, ``badamoa-cache``; a
folder an admin named otherwise stays). The work
runs in a helper outside the node's own service, because stopping the service stops everything in
it: ``systemd-run`` on Linux, a WMI-started script on Windows. A node not put in by the install
scripts (run from a source checkout) only forgets its registration.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from badamoa.node import cache, containers, places, updater

log = logging.getLogger("badamoa.node.remover")

UNIT = "badamoa-node"
# What an admin may leave on the server when deleting the node (the program, its settings and service always go)
KEEPABLE = ("images", "envs", "containers", "models", "templates", "template_data", "runs", "uv")
_PARTS = {
    "envs": ("envs",),
    "models": ("cache", "models"),
    "templates": ("cache", "templates"),
    "template_data": ("cache", "template-data"),
    "runs": ("runs",),
    "uv": ("uv-cache",),
}
TASK = "BADAMOA Node"


def _ours(path: Path, name: str) -> bool:
    """A folder BADAMOA made (…/badamoa, …/BADAMOA/node): the only kind this module removes."""
    return path.name.lower() == name and len(path.resolve().parts) >= 3


def remove(home: Path, images: Path | None = None, keep: Iterable[str] = ()) -> str:
    """Start removing BADAMOA from this machine; returns what will go. Raises when it cannot start.

    ``images``: the folder of received images in use (the admin's or the node's pick). ``keep``: kinds an
    admin chose to leave on the server (``KEEPABLE``); the rest BADAMOA made goes, the program always."""
    kept = {k for k in keep if k in KEEPABLE}
    where = updater.layout()
    home = Path(home).resolve()
    if where is None:
        (home / "node.json").unlink(missing_ok=True)
        return "등록만 지웠습니다 (설치 스크립트로 설치한 노드가 아님)"
    root = where["root"].resolve()
    if not _ours(root, "badamoa") or not _ours(home, "node"):
        raise RuntimeError(f"지울 폴더가 예상과 다릅니다: {root}, {home}")
    work = [p for p in [places.of(home).work, *places.used_before(home)] if _ours(p, places.FOLDER) and p.is_dir()]
    picked = cache.root(home) if (home / cache.SAVED).is_file() else None
    received = list(dict.fromkeys(Path(p).resolve() for p in (images, picked) if p is not None))
    targets = _targets(home, [w.resolve() for w in work], kept, received)
    if "images" not in kept:  # received images, where BADAMOA made the folder (elsewhere than the home)
        for folder in received:
            if not _ours(folder, cache.FOLDER) or not folder.is_dir() or home in folder.parents:
                continue
            if not any(places.same(folder, t) or t in folder.parents for t in targets):
                targets.append(folder)
    dropped = [] if "containers" in kept else _drop_docker_images(home)
    if sys.platform == "win32":
        said = _windows(root, home, targets)
    else:
        said = _linux(root, home, targets)
    return said + (f", Docker 이미지 {len(dropped)}개" if dropped else "")


def _targets(home: Path, work: list[Path], kept: set[str], received: list[Path]) -> list[Path]:
    """The node's home and work folders, less what an admin kept: whole folders when nothing in them is kept."""
    held = [base.joinpath(*_PARTS[k]) for base in (home, *work) for k in kept if k in _PARTS]
    if "images" in kept:
        held += received
    found: list[Path] = []
    for base in (home, *work):
        found += [p for p in _prune(base, [h for h in held if h.exists()]) if p not in found]
    return found


def _prune(path: Path, held: list[Path]) -> list[Path]:
    """What to remove under ``path`` so that every held folder stays."""
    inside = [h for h in held if h == path or path in h.parents]
    if not inside:
        return [path]
    if path in inside:
        return []
    return [p for child in sorted(path.iterdir()) for p in _prune(child, inside)]


def _drop_docker_images(home: Path) -> list[str]:
    """The node's Docker images: the ones it pulled go, the ones that were there before are only forgotten."""
    dropped: list[str] = []
    for entry in containers.listing(home):
        try:
            if containers.remove(home, entry["image"], log.info).get("removed"):
                dropped.append(entry["image"])
        except Exception as e:  # noqa: BLE001 - the rest of the removal goes on
            log.warning("Docker 이미지를 지우지 못했습니다 (%s): %s", entry["image"], e)
    return dropped


def _linux(root: Path, home: Path, work: list[Path] | None = None) -> str:
    link = Path("/usr/local/bin/badamoa")
    steps = [
        "sleep 2",  # the node answers the Hub first
        f"systemctl disable --now {UNIT}",
        f"rm -f /etc/systemd/system/{UNIT}.service",
        "systemctl daemon-reload",
        f"rm -rf {shlex.quote(str(root))}",
        *(f"rm -rf {shlex.quote(str(target))}" for target in work or []),
        f"[ -L {link} ] && rm -f {link}",
        f"rmdir {shlex.quote(str(home.parent))} 2>/dev/null",
        "true",
    ]
    script = "; ".join(steps)
    if os.environ.get("INVOCATION_ID") and shutil.which("systemd-run"):  # running as the systemd service
        subprocess.run(
            ["systemd-run", "--unit", f"{UNIT}-remove", "--collect", "--quiet", "/bin/sh", "-c", script], check=True
        )
    else:
        subprocess.Popen(["/bin/sh", "-c", script], start_new_session=True)
    return ", ".join([f"서비스 {UNIT}", str(root), *map(str, work or [])])


def _windows(root: Path, home: Path, work: list[Path] | None = None) -> str:
    folder = Path(tempfile.mkdtemp(prefix="badamoa-remove-"))
    lines = [
        "@echo off",
        *updater.wait_lines({os.getpid(), os.getppid()}),
        f'schtasks /End /TN "{TASK}" >nul 2>&1',
        f'schtasks /Delete /F /TN "{TASK}" >nul 2>&1',
        f'rmdir /s /q "{root}"',
        *(f'rmdir /s /q "{target}" 2>nul & del /f /q "{target}" 2>nul' for target in work or []),
        f'rmdir "{home.parent}" >nul 2>&1',
        "exit /b 0",
    ]
    script = folder / "remove.cmd"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    if not updater.start_outside(script):
        raise RuntimeError("정리 도우미를 시작하지 못했습니다.")
    return ", ".join([f'예약 작업 "{TASK}"', str(root), *map(str, work or [])])
