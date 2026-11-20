"""A node takes BADAMOA off its machine when the Hub deletes it: the service, the program, its settings.

Data folders (the images and labels datasets were registered from) are never touched; the node's work
folders (``places``: environments, work folders of jobs, fetched files) go with its home when BADAMOA
made them (``badamoa-work``; a folder an admin named otherwise stays). The work
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
from pathlib import Path

from badamoa.node import places, updater

log = logging.getLogger("badamoa.node.remover")

UNIT = "badamoa-node"
TASK = "BADAMOA Node"


def _ours(path: Path, name: str) -> bool:
    """A folder BADAMOA made (…/badamoa, …/BADAMOA/node): the only kind this module removes."""
    return path.name.lower() == name and len(path.resolve().parts) >= 3


def remove(home: Path) -> str:
    """Start removing BADAMOA from this machine; returns what will go. Raises when it cannot start."""
    where = updater.layout()
    home = Path(home).resolve()
    if where is None:
        (home / "node.json").unlink(missing_ok=True)
        return "등록만 지웠습니다 (설치 스크립트로 설치한 노드가 아님)"
    root = where["root"].resolve()
    if not _ours(root, "badamoa") or not _ours(home, "node"):
        raise RuntimeError(f"지울 폴더가 예상과 다릅니다: {root}, {home}")
    work = [p for p in [places.of(home).work, *places.used_before(home)] if _ours(p, places.FOLDER) and p.is_dir()]
    if sys.platform == "win32":
        return _windows(root, home, work)
    return _linux(root, home, work)


def _linux(root: Path, home: Path, work: list[Path] | None = None) -> str:
    link = Path("/usr/local/bin/badamoa")
    steps = [
        "sleep 2",  # the node answers the Hub first
        f"systemctl disable --now {UNIT}",
        f"rm -f /etc/systemd/system/{UNIT}.service",
        "systemctl daemon-reload",
        f"rm -rf {shlex.quote(str(root))} {shlex.quote(str(home))}",
        *(f"rm -rf {shlex.quote(str(folder))}" for folder in work or []),
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
    return ", ".join([f"서비스 {UNIT}", str(root), str(home), *map(str, work or [])])


def _windows(root: Path, home: Path, work: list[Path] | None = None) -> str:
    folder = Path(tempfile.mkdtemp(prefix="badamoa-remove-"))
    lines = [
        "@echo off",
        *updater.wait_lines({os.getpid(), os.getppid()}),
        f'schtasks /End /TN "{TASK}" >nul 2>&1',
        f'schtasks /Delete /F /TN "{TASK}" >nul 2>&1',
        f'rmdir /s /q "{root}"',
        f'rmdir /s /q "{home}"',
        *(f'rmdir /s /q "{folder}"' for folder in work or []),
        f'rmdir "{home.parent}" >nul 2>&1',
        "exit /b 0",
    ]
    script = folder / "remove.cmd"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    if not updater.start_outside(script):
        raise RuntimeError("정리 도우미를 시작하지 못했습니다.")
    return ", ".join([f'예약 작업 "{TASK}"', str(root), str(home), *map(str, work or [])])
