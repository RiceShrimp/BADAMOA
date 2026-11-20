"""Moving Docker's images (its ``data-root``) to a disk with room: on an admin's word (node 상세 → 디스크 공간
부족), or by a job about to pull an image its disk cannot take (``containers.make_room``).

Docker keeps every image of a server in one folder (``/var/lib/docker`` unless ``data-root`` in
``/etc/docker/daemon.json`` says otherwise); BADAMOA cannot place an image elsewhere. Moving that folder means
stopping Docker, so every container of the server stops for a while, BADAMOA's or anyone's: it happens only on
a Linux node running as root (installed with sudo), only while the node runs no other job, and only when no
container runs at all (an admin may choose to stop them; a job never does).

    1. Docker's folder, its images and its size; the new folder must be empty, on another disk, with room
    2. stop Docker (``docker.socket`` too, so a call does not wake it), copy everything (``rsync -aHAX``)
    3. ``data-root`` in daemon.json (the rest of it kept, its first state saved beside it), start Docker
    4. check Docker uses the new folder and has every image it had; only then the old folder goes
       (or stays, when asked), and the space is free

Any step failing after Docker stopped puts daemon.json back and starts Docker on the old folder, which was
never touched; a copy that got made stays in the new folder for an admin to look at and remove.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DAEMON_JSON = Path("/etc/docker/daemon.json")
START_WAIT_S = 90
HOLD = "docker-moving"  # in the node's home while a job moves Docker: the agent starts no new job meanwhile
RUNNING = "running.json"  # the agent's running jobs, for a job that has to know whether it is alone
SETTLE_S = 3.0
Run = Callable[..., subprocess.CompletedProcess]


def choices(current: str, disks: list[dict[str, Any]] | None = None) -> list[str]:
    """Folders Docker's images could move to: ``docker`` on each other local disk, the one with the most room first
    (a folder already holding something is skipped: the move needs an empty one)."""
    if disks is None:
        from badamoa.core import sysinfo

        disks = sysinfo.MetricsSampler._disks()
    disks = [d for d in disks if (d.get("total_gb") or 0) >= 2]
    real = os.path.realpath(current) if current and sys.platform.startswith("linux") else current
    inside = [
        str(d.get("mount") or "")
        for d in disks
        if real == d.get("mount") or real.startswith(str(d.get("mount") or "").rstrip("/") + "/")
    ]
    here = max(inside, key=len, default="")
    found = []
    for disk in sorted(disks, key=lambda d: -(d.get("free_gb") or 0)):
        mount = str(disk.get("mount") or "")
        if mount == here or not mount.startswith("/") or mount.startswith(("/boot", "/snap")):
            continue
        folder = mount.rstrip("/") + "/docker"
        if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
            continue
        found.append(folder)
    return found


def running_jobs(home: Path) -> list[str]:
    """The jobs the node's agent runs now (it writes them down as they start and end)."""
    try:
        return [str(j) for j in json.loads((home / RUNNING).read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError):
        return []


@contextmanager
def alone(home: Path, job_id: str, settle_s: float | None = None) -> Iterator[list[str]]:
    """Keep new jobs off this node while the block runs (the agent starts none while the hold file names a job it
    runs) and say which other jobs run: an empty list means this job is alone until the block ends."""
    hold = home / HOLD
    hold.write_text(job_id, encoding="utf-8")
    try:
        time.sleep(
            SETTLE_S if settle_s is None else settle_s
        )  # a job the agent was starting as the hold appeared has written itself down by now
        yield [j for j in running_jobs(home) if j != job_id]
    finally:
        hold.unlink(missing_ok=True)


def _run(command: list[str], runner: Run, check: bool = False) -> subprocess.CompletedProcess:
    done = runner(command, capture_output=True, text=True, check=False)
    if check and done.returncode != 0:
        why = (done.stderr or done.stdout or "").strip()[-300:]
        raise RuntimeError(f"{' '.join(command[:3])} 실패: {why or f'종료 코드 {done.returncode}'}")
    return done


def _existing(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def _size(folder: Path, runner: Run) -> int:
    done = _run(["du", "-sb", str(folder)], runner)
    if done.returncode == 0 and done.stdout.split():
        return int(done.stdout.split()[0])
    total = 0
    for root, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def _images(runner: Run) -> set[str]:
    done = _run(["docker", "image", "ls", "-aq", "--no-trunc"], runner, check=True)
    return {line.strip() for line in done.stdout.splitlines() if line.strip()}


def _root(runner: Run) -> str:
    done = _run(["docker", "info", "--format", "{{.DockerRootDir}}"], runner)
    return done.stdout.strip() if done.returncode == 0 else ""


def _set_root(daemon: Path, target: str | None, saved: str | None) -> None:
    """``data-root`` set to ``target`` (keeping the rest), or the file put back as it was (``saved``; None: none)."""
    if target is None:
        if saved is None:
            daemon.unlink(missing_ok=True)
        else:
            daemon.write_text(saved, encoding="utf-8")
        return
    config = json.loads(saved) if saved else {}
    if not isinstance(config, dict):
        raise RuntimeError(f"{daemon}이(가) JSON 객체가 아니라 고치지 않았습니다.")
    config["data-root"] = target
    daemon.parent.mkdir(parents=True, exist_ok=True)
    written = daemon.with_name(daemon.name + ".badamoa-new")
    written.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(written, daemon)


def _start(runner: Run, log: Callable[[str], None]) -> bool:
    _run(["systemctl", "start", "docker"], runner)
    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline:
        if _root(runner):
            return True
        time.sleep(2)
    log("Docker가 다시 켜지지 않았습니다.")
    return False


def check(
    target: str, running_jobs: list[str], stop_containers: bool, runner: Run = subprocess.run, extra: int = 0
) -> dict[str, Any]:
    """What moving would do, or why it cannot: nothing is changed here. ``extra``: room the new disk must still
    have after the copy (an image about to be pulled there)."""
    if not sys.platform.startswith("linux") or not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise RuntimeError("Docker 이미지 위치는 sudo로 설치한 Linux 노드에서만 옮길 수 있습니다.")
    if not shutil.which("docker") or not shutil.which("systemctl"):
        raise RuntimeError("이 서버에 Docker나 systemctl이 없습니다.")
    if running_jobs:
        busy = len(running_jobs)
        raise RuntimeError(f"이 서버에서 MOA 작업 {busy}개가 돌고 있어 옮기지 않았습니다. 끝난 뒤에 하세요.")
    old = _root(runner)
    if not old:
        raise RuntimeError("Docker가 응답하지 않습니다. Docker가 켜져 있을 때 하세요.")
    if "/.local/share/docker" in old:
        raise RuntimeError("rootless Docker는 그 계정의 설정이라 여기서 옮기지 않습니다.")
    source = Path(old)
    new = Path(os.path.normpath(target.strip())) if target.strip() else None
    if new is None or not new.is_absolute():
        raise RuntimeError("옮길 폴더를 절대 경로로 적으세요 (예: /home4/docker).")
    if new == source or source in new.parents or new in source.parents:
        raise RuntimeError(f"{new}은(는) 지금 Docker 폴더 {source}와 겹칩니다. 다른 디스크의 빈 폴더를 고르세요.")
    if new.exists() and (not new.is_dir() or any(new.iterdir())):
        raise RuntimeError(f"{new}이(가) 비어 있지 않습니다. 빈 폴더나 없는 폴더를 고르세요.")
    if os.stat(_existing(new)).st_dev == os.stat(source).st_dev:
        raise RuntimeError(f"{new}은(는) 지금 Docker 폴더와 같은 디스크라 옮겨도 공간이 비지 않습니다.")
    size = _size(source, runner)
    free = shutil.disk_usage(_existing(new)).free
    if free < size * 1.05 + 1024**3 + extra:
        room, need = free / 1024**3, (size + extra) / 1024**3
        what = f"Docker 폴더와 받을 이미지({need:.0f} GB)" if extra else f"Docker 폴더({need:.0f} GB)"
        raise RuntimeError(f"{new}의 디스크에 {room:.0f} GB 남아 {what}를 둘 수 없습니다.")
    containers = [n for n in _run(["docker", "ps", "--format", "{{.Names}}"], runner, check=True).stdout.split() if n]
    if containers and not stop_containers:
        listed = ", ".join(containers[:8]) + (f" 외 {len(containers) - 8}개" if len(containers) > 8 else "")
        raise RuntimeError(
            f"돌고 있는 컨테이너 {len(containers)}개가 있어 옮기지 않았습니다: {listed}. "
            "멈춰도 되면 '돌고 있는 컨테이너도 멈춤'을 켜고 다시 하세요."
        )
    return {"from": str(source), "to": str(new), "bytes": size, "free": free, "containers": containers}


def move(
    target: str,
    *,
    remove_old: bool = True,
    stop_containers: bool = False,
    running_jobs: list[str] | None = None,
    extra: int = 0,
    daemon: Path = DAEMON_JSON,
    runner: Run = subprocess.run,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Move Docker's data-root to ``target`` (see the module's notes). Returns what was done."""
    plan = check(target, running_jobs or [], stop_containers, runner, extra)
    source, new = Path(plan["from"]), Path(plan["to"])
    before = _images(runner)
    log(f"Docker 이미지 {len(before)}개 · {plan['bytes'] / 1024**3:.1f} GB를 {source}에서 {new}로 옮깁니다.")
    if plan["containers"]:
        log(f"돌고 있는 컨테이너 {len(plan['containers'])}개가 멈춥니다: {', '.join(plan['containers'][:8])}")
    saved = daemon.read_text(encoding="utf-8") if daemon.is_file() else None
    if saved is not None:
        backup = daemon.with_name(daemon.name + ".badamoa-backup")
        if not backup.exists():
            backup.write_text(saved, encoding="utf-8")
    log("Docker를 멈춥니다.")
    _run(["systemctl", "stop", "docker.socket"], runner)
    _run(["systemctl", "stop", "docker"], runner, check=True)
    try:
        new.mkdir(parents=True, exist_ok=True)
        new.chmod(0o710)  # as Docker keeps its own folder
        if shutil.which("rsync"):
            log("복사합니다 (rsync).")
            _run(["rsync", "-aHAX", "--numeric-ids", f"{source}/", f"{new}/"], runner, check=True)
        else:
            log("복사합니다 (cp -a).")
            _run(["cp", "-a", f"{source}/.", f"{new}/"], runner, check=True)
        _set_root(daemon, str(new), saved)
        log(f"{daemon}의 data-root를 {new}로 바꾸고 Docker를 다시 켭니다.")
        if not _start(runner, log):
            raise RuntimeError("새 위치로 Docker가 켜지지 않았습니다.")
        now = _root(runner)
        if os.path.normpath(now) != os.path.normpath(str(new)):
            raise RuntimeError(f"Docker가 아직 {now}을(를) 씁니다.")
        after = _images(runner)
        if not before <= after:
            raise RuntimeError(f"옮긴 뒤 이미지 {len(before - after)}개가 보이지 않습니다.")
    except Exception as e:
        log(f"옮기지 못했습니다: {e} 원래대로 되돌립니다.")
        _run(["systemctl", "stop", "docker.socket"], runner)
        _run(["systemctl", "stop", "docker"], runner)
        _set_root(daemon, None, saved)
        started = _start(runner, log)
        left = f" 복사한 것은 {new}에 남아 있으니 확인한 뒤 지우세요." if new.exists() and any(new.iterdir()) else ""
        state = "원래 폴더로 Docker를 다시 켰습니다." if started else "Docker가 다시 켜지지 않았습니다. 서버에서 확인하세요."
        raise RuntimeError(f"{e} {state}{left}") from e
    log(f"Docker가 새 위치 {new}에서 이미지 {len(after)}개로 켜졌습니다.")
    removed = False
    if remove_old:
        log(f"예전 폴더 {source}를 지워 공간을 비웁니다.")
        shutil.rmtree(source, ignore_errors=True)
        removed = not source.exists() or not any(source.iterdir())
        if not removed:
            log(f"예전 폴더를 다 지우지 못했습니다: {source}. 서버에서 확인하세요.")
    else:
        log(f"예전 폴더 {source}는 그대로 두었습니다. 확인한 뒤 지우면 공간이 빕니다.")
    return {"from": str(source), "to": str(new), "bytes": plan["bytes"], "images": len(after), "removed_old": removed}
