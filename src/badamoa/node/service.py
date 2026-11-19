"""Register BADAMOA components to start at boot.

Linux as root: a systemd unit. Windows: a Task Scheduler task that runs as SYSTEM at
startup (no extra dependency such as pywin32 or a service wrapper needed).

Linux without root (a server where the account has no sudo): the node runs as that account. This
account's crontab starts it at boot and every few minutes starts it again if it is not running; a lock
file in the node's home keeps it to one agent however often it is started.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import IO

_UNIT = """\
[Unit]
Description=BADAMOA {title}
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={exe} {args}
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""

CRON_MARK = "# badamoa-node"  # the crontab lines this module owns
LOCK = "agent.lock"
LOG = "node.log"
LOG_KEEP = 10 * 1024 * 1024  # a started agent's output: moved aside to node.log.1 past this


def _exe() -> str:
    found = shutil.which("badamoa")
    if found:
        return found
    candidate = Path(sys.executable).with_name("badamoa.exe" if sys.platform == "win32" else "badamoa")
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("badamoa 실행 파일을 찾을 수 없습니다.")


def user_mode() -> bool:
    """Linux without root: the node belongs to this account (crontab), not to the machine (systemd)."""
    return sys.platform.startswith("linux") and os.geteuid() != 0


# ---- one agent per node home ---------------------------------------------------------------------


def hold_lock(home: Path) -> IO[str] | None:
    """Take the node's agent lock: the open file holds it until closed (or the process ends).

    None when another agent holds it. Linux / macOS only; elsewhere the lock is always granted.
    """
    handle = open(home / LOCK, "a+", encoding="utf-8")  # noqa: SIM115  (held for the agent's life)
    if os.name == "nt":
        return handle
    import fcntl

    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def running(home: Path) -> bool:
    """An agent for this node home is running (Linux / macOS; False elsewhere)."""
    held = hold_lock(home)
    if held is None:
        return True
    held.close()
    return False


def start(home: Path) -> bool:
    """Start the agent in the background unless one runs already. True when one was started."""
    if running(home):
        return False
    log = home / LOG
    if log.is_file() and log.stat().st_size > LOG_KEEP:
        log.replace(home / f"{LOG}.1")
    with open(log, "ab") as out:
        subprocess.Popen(  # noqa: S603
            [_exe(), "node", "run", "--home", str(home)],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=home,
            start_new_session=True,  # not tied to the terminal or SSH login that started it
        )
    return True


def _agents() -> list:
    """This account's running ``badamoa node run`` processes."""
    import psutil

    found = []
    for proc in psutil.process_iter(["pid", "cmdline", "uids"]):
        if proc.info["pid"] == os.getpid() or (proc.info.get("uids") and proc.info["uids"].real != os.getuid()):
            continue
        cmd = proc.info.get("cmdline") or []
        if any(Path(arg).name == "badamoa" and cmd[i + 1 : i + 3] == ["node", "run"] for i, arg in enumerate(cmd)):
            found.append(proc)
    return found


def stop() -> int:
    """Stop this account's node agents. Returns how many there were."""
    import psutil

    agents = _agents()
    for proc in agents:
        try:
            proc.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(agents, timeout=15)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    return len(agents)


def _cron_lines(home: Path) -> list[str]:
    command = f"{shlex.quote(_exe())} node start --home {shlex.quote(str(home))} >/dev/null 2>&1".replace("%", r"\%")
    return [f"@reboot {command} {CRON_MARK}", f"*/5 * * * * {command} {CRON_MARK}"]


def _crontab(lines: list[str]) -> str:
    """Put ``lines`` in place of this module's lines in the account's crontab. "" or why it could not."""
    if not shutil.which("crontab"):
        return "crontab 명령이 없습니다"
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    kept = [line for line in current.stdout.splitlines() if CRON_MARK not in line] if current.returncode == 0 else []
    if not kept and not lines and current.returncode != 0:
        return ""  # nothing there, nothing to put
    text = "\n".join(kept + lines) + "\n" if kept or lines else ""
    done = subprocess.run(["crontab", "-"], input=text, capture_output=True, text=True, check=False)
    return "" if done.returncode == 0 else (done.stderr or done.stdout).strip() or "crontab이 거절했습니다"


def _cron_running() -> bool:
    import psutil

    names = {"cron", "crond", "cronie", "busybox-crond"}
    return any((p.info.get("name") or "") in names for p in psutil.process_iter(["name"]))


def _install_user(home: Path) -> str:
    why = _crontab(_cron_lines(home))
    stopped = stop()  # the installer run again: the new program from now on
    start(home)
    lines = [f"Node Agent를 시작했습니다{' (다시 시작)' if stopped else ''}. 로그: {home / LOG}"]
    if why:
        lines.append(
            f"부팅 시 자동 실행은 등록하지 못했습니다: {why}. 서버 관리자에게 cron을 부탁하거나, "
            "재부팅 뒤에는 `badamoa node start`를 직접 실행하세요."
        )
    elif not _cron_running():
        lines.append(
            "crontab에 등록했지만 cron이 돌고 있지 않습니다. 부팅 시 자동 실행되려면 서버 관리자에게 cron을 부탁하세요."
        )
    else:
        lines.append("부팅 시와 5분마다(꺼져 있으면) 자동으로 켜지도록 이 계정의 crontab에 등록했습니다.")
    return "\n".join(lines)


def install(component: str, args: str) -> str:
    """component: 'hub' | 'node'. Returns a human-readable description of what was installed."""
    title = "Hub" if component == "hub" else "Node"
    name = f"badamoa-{component}"
    exe = _exe()
    if component == "node" and user_mode():
        from badamoa.core.paths import component_home

        return _install_user(component_home("node"))
    if sys.platform.startswith("linux"):
        unit_path = Path(f"/etc/systemd/system/{name}.service")
        unit_path.write_text(_UNIT.format(title=title, exe=exe, args=args), encoding="utf-8")
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", name], check=True)
        subprocess.run(["systemctl", "restart", name], check=True)  # running already (installer run again): the new program
        return f"systemd 서비스 {name} 등록 및 시작 완료 (journalctl -u {name} -f 로 로그 확인)"
    if sys.platform == "win32":
        task = f"BADAMOA {title}"
        command = f'"{exe}" {args}'
        subprocess.run(
            ["schtasks", "/Create", "/F", "/TN", task, "/SC", "ONSTART", "/RU", "SYSTEM", "/RL", "HIGHEST", "/TR", command],
            check=True,
        )
        subprocess.run(["schtasks", "/End", "/TN", task], check=False, capture_output=True)  # running already
        subprocess.run(["schtasks", "/Run", "/TN", task], check=True)
        return f'예약 작업 "{task}" 등록 및 시작 완료 (부팅 시 자동 실행)'
    raise RuntimeError(f"이 OS에서는 서비스 등록을 지원하지 않습니다: {sys.platform}")


def uninstall(component: str) -> str:
    title = "Hub" if component == "hub" else "Node"
    name = f"badamoa-{component}"
    if component == "node" and user_mode():
        why = _crontab([])
        stopped = stop()
        done = f"Node Agent {stopped}개를 멈추고 crontab에서 뺐습니다"
        return done + (f" (crontab: {why})" if why else "")
    if sys.platform.startswith("linux"):
        subprocess.run(["systemctl", "disable", "--now", name], check=False)
        Path(f"/etc/systemd/system/{name}.service").unlink(missing_ok=True)
        subprocess.run(["systemctl", "daemon-reload"], check=False)
        return f"systemd 서비스 {name} 제거 완료"
    if sys.platform == "win32":
        task = f"BADAMOA {title}"
        subprocess.run(["schtasks", "/End", "/TN", task], check=False)
        subprocess.run(["schtasks", "/Delete", "/F", "/TN", task], check=False)
        return f'예약 작업 "{task}" 제거 완료'
    raise RuntimeError(f"이 OS에서는 서비스 등록을 지원하지 않습니다: {sys.platform}")
