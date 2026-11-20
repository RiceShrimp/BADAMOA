"""Register BADAMOA components to start at boot.

Linux as root: a systemd unit. Windows: a Task Scheduler task that runs as SYSTEM at
startup (no extra dependency such as pywin32 or a service wrapper needed). The task is registered from
XML so it has no run-time limit (``schtasks /Create`` alone stops a task after 72 hours), runs on
battery, comes back a minute after it fails, and runs at normal priority (the jobs it starts inherit it).

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
import tempfile
from pathlib import Path
from typing import IO
from xml.sax.saxutils import escape

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


# ---- Windows: Task Scheduler ---------------------------------------------------------------------

SYSTEM = "S-1-5-18"  # the LocalSystem account


def task_xml(command: str, arguments: str, workdir: str, *, boot: bool, user: str = SYSTEM) -> str:
    """A task that starts ``command arguments`` at boot (``boot``) or when ``user`` logs on.

    ``user``: SYSTEM, or DOMAIN\\name: at boot that account without a stored password (S4U), at log-on its
    own session. Neither runs elevated except SYSTEM.
    """
    if boot:
        trigger = "<BootTrigger><Enabled>true</Enabled></BootTrigger>"
    else:
        trigger = f"<LogonTrigger><Enabled>true</Enabled><UserId>{escape(user)}</UserId></LogonTrigger>"
    if user == SYSTEM:
        principal = f"<UserId>{SYSTEM}</UserId><RunLevel>HighestAvailable</RunLevel>"
    else:
        logon = "S4U" if boot else "InteractiveToken"
        principal = f"<UserId>{escape(user)}</UserId><LogonType>{logon}</LogonType><RunLevel>LeastPrivilege</RunLevel>"
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>BADAMOA</Description></RegistrationInfo>
  <Triggers>{trigger}</Triggers>
  <Principals><Principal id="Author">{principal}</Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>
    <Priority>5</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>{escape(command)}</Command><Arguments>{escape(arguments)}</Arguments><WorkingDirectory>{escape(workdir)}</WorkingDirectory></Exec>
  </Actions>
</Task>
"""


def register_task(name: str, xml: str) -> str:
    """Register (or replace) the task. "" when done, else what Windows said."""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-16") as f:
        f.write(xml)
    try:
        done = subprocess.run(["schtasks", "/Create", "/F", "/TN", name, "/XML", f.name], capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    return "" if done.returncode == 0 else (done.stderr or done.stdout).strip() or "schtasks가 거절했습니다"


def query_task(name: str) -> str | None:
    """The task's XML, or None when there is no such task (or it cannot be read)."""
    done = subprocess.run(["schtasks", "/Query", "/TN", name, "/XML"], capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else None


def delete_task(name: str) -> str:
    """"" when the task is gone (or was not there), else what Windows said."""
    if query_task(name) is None:
        return ""
    done = subprocess.run(["schtasks", "/Delete", "/F", "/TN", name], capture_output=True, text=True)
    return "" if done.returncode == 0 else (done.stderr or done.stdout).strip() or "schtasks가 거절했습니다"


def user_mode() -> bool:
    """Linux without root, or without systemd (a container, WSL): the node starts from the account's crontab."""
    if not sys.platform.startswith("linux"):
        return False
    systemd = bool(shutil.which("systemctl")) and Path("/run/systemd/system").is_dir()
    return os.geteuid() != 0 or not systemd


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


def _agents(uid: int | None = None) -> list:
    """The running ``badamoa node run`` processes of this account (or of the account ``uid``)."""
    import psutil

    uid = os.getuid() if uid is None else uid
    found = []
    for proc in psutil.process_iter(["pid", "cmdline", "uids"]):
        if proc.info["pid"] == os.getpid() or (proc.info.get("uids") and proc.info["uids"].real != uid):
            continue
        cmd = proc.info.get("cmdline") or []
        if any(Path(arg).name == "badamoa" and cmd[i + 1 : i + 3] == ["node", "run"] for i, arg in enumerate(cmd)):
            found.append(proc)
    return found


def stop(uid: int | None = None) -> int:
    """Stop this account's node agents (or those of the account ``uid``). Returns how many there were."""
    import psutil

    agents = _agents(uid)
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


def _crontab(lines: list[str], user: str = "") -> str:
    """Put ``lines`` in place of this module's lines in the account's crontab (root: ``user``'s). "" or why it could not."""
    if not shutil.which("crontab"):
        return "crontab 명령이 없습니다"
    who = ["-u", user] if user else []
    current = subprocess.run(["crontab", *who, "-l"], capture_output=True, text=True, check=False)
    kept = [line for line in current.stdout.splitlines() if CRON_MARK not in line] if current.returncode == 0 else []
    if not kept and not lines and current.returncode != 0:
        return ""  # nothing there, nothing to put
    text = "\n".join(kept + lines) + "\n" if kept or lines else ""
    done = subprocess.run(["crontab", *who, "-"], input=text, capture_output=True, text=True, check=False)
    return "" if done.returncode == 0 else (done.stderr or done.stdout).strip() or "crontab이 거절했습니다"


def _their_node_home(user: str, home_dir: Path) -> Path:
    """Where ``user``'s own node keeps its registration: as its crontab line says, else the usual place."""
    listed = subprocess.run(["crontab", "-u", user, "-l"], capture_output=True, text=True, check=False)
    for line in listed.stdout.splitlines() if listed.returncode == 0 else []:
        if CRON_MARK in line:
            words = shlex.split(line.split(CRON_MARK)[0].replace(r"\%", "%"))
            if "--home" in words[:-1]:
                return Path(words[words.index("--home") + 1])
    return home_dir / ".local" / "share" / "badamoa" / "node"


def take_over_user_node(home: Path, user: str | None = None) -> str:
    """Installing with sudo on a server whose node ran without it: the same node carries on as root.

    The account that ran sudo had a node of its own (installed without sudo, started from its crontab). Its
    registration moves to root's node home, its crontab lines go and its agent stops, so the machine stays one node
    with its history and datasets. Nothing moves while that agent runs a job. Returns what was done ("" if nothing)."""
    if not sys.platform.startswith("linux") or os.geteuid() != 0 or (home / "node.json").is_file():
        return ""
    user = user if user is not None else os.environ.get("SUDO_USER", "")
    if not user or user == "root":
        return ""
    import pwd

    import psutil

    try:
        entry = pwd.getpwnam(user)
    except KeyError:
        return ""
    theirs = _their_node_home(user, Path(entry.pw_dir))
    if not (theirs / "node.json").is_file():
        return ""
    agents = _agents(entry.pw_uid)
    working = []
    for proc in agents:
        try:
            working += [c for c in proc.children(recursive=True) if c.name() not in ("nvidia-smi",)]
        except psutil.Error:
            continue
    if working:
        raise RuntimeError(
            f"{user} 계정의 노드가 작업을 하고 있어 지금은 이어받지 않았습니다. "
            "작업이 끝난 뒤 이 설치 명령을 다시 실행하세요."
        )
    home.mkdir(parents=True, exist_ok=True)
    shutil.copy2(theirs / "node.json", home / "node.json")
    (home / "node.json").chmod(0o600)
    why = _crontab([], user)
    stopped = stop(entry.pw_uid)
    done = f"{user} 계정으로 돌던 노드를 root가 이어받습니다(같은 노드 · 같은 등록): 그 계정의 노드 {stopped}개를 멈추고"
    return done + (f" crontab에서 빼지 못했습니다({why})." if why else " crontab에서 뺐습니다.")


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
        why = register_task(task, task_xml(exe, args, str(Path(exe).parent), boot=True))
        if why:
            raise RuntimeError(f'예약 작업 "{task}"을(를) 등록하지 못했습니다: {why}')
        subprocess.run(["schtasks", "/End", "/TN", task], check=False, capture_output=True)  # running already
        subprocess.run(["schtasks", "/Run", "/TN", task], check=True)
        return f'예약 작업 "{task}" 등록 및 시작 완료 (부팅 시 자동 실행, 멈추면 1분 뒤 다시 실행)'
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
