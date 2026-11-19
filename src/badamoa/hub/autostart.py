"""Starting the Hub when its machine starts: an admin turns it on or off (설정 → Hub 실행).

The Hub records how it runs now (data folder, address, port, options, the ``BADAMOA_*`` settings and the
uv it found) in ``autostart.json`` in its data folder, and registers the machine to start it that way:

- Windows: a Task Scheduler task "BADAMOA Hub <port>". From a Hub with administrator rights it runs at
  boot without anyone logging on (as the same account, no password stored; as SYSTEM when the Hub runs
  as SYSTEM); without them, when the account that runs the Hub logs on. ``python -m badamoa.hub.autostart
  on --home <folder>`` in an administrator PowerShell turns the second into the first.
- Linux (Ubuntu …): a systemd unit ``badamoa-hub-<port>`` as root on a machine booted with systemd, else
  ``@reboot`` in the account's crontab (no sudo, or no systemd: a container, WSL).

The machine runs ``python -m badamoa.hub.autostart run --home <folder>``: no console window (pythonw on
Windows), output in ``hub.log`` there.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from badamoa.core.paths import is_elevated
from badamoa.node import service

LAUNCH = "autostart.json"
LOG = "hub.log"
LOG_KEEP = 10 * 1024 * 1024
UNIT_DIR = Path("/etc/systemd/system")
_UNIT = """\
[Unit]
Description=BADAMOA Hub {port}
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={command}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")


def task_name(port: int) -> str:
    return f"BADAMOA Hub {port}"


def unit_name(port: int) -> str:
    return f"badamoa-hub-{port}"


def _cron_mark(port: int) -> str:
    return f"# badamoa-hub-{port}"


def _systemd() -> bool:
    """Linux as root on a machine that booted with systemd: a unit, not a crontab line."""
    return LINUX and os.geteuid() == 0 and bool(shutil.which("systemctl")) and Path("/run/systemd/system").is_dir()


def _python() -> str:
    """The interpreter this Hub runs on; pythonw beside it on Windows (no console window)."""
    exe = Path(sys.executable)
    if WINDOWS:
        quiet = exe.with_name("pythonw.exe")
        if quiet.is_file():
            return str(quiet)
    return str(exe)


def _account() -> str:
    """Who registers the task: SYSTEM, or DOMAIN\\name."""
    if WINDOWS:
        import getpass

        name = getpass.getuser()
        if name.upper().endswith("$") or name.upper() == "SYSTEM":
            return service.SYSTEM
        return f"{os.environ.get('USERDOMAIN', '.')}\\{name}"
    return ""


# ---- how this Hub runs ---------------------------------------------------------------------------


def launch_args(settings: Any, launch_gpu: tuple[float, float] | None = None) -> list[str]:
    """``badamoa`` arguments that start this Hub again the way it runs now."""
    memory, util = launch_gpu or (settings.gpu_busy_memory_pct, settings.gpu_busy_util_pct)
    args = ["hub", "run", "--home", str(Path(settings.home).resolve()), "--host", settings.host]
    args += ["--port", str(settings.port), "--no-open", "--gpu-busy-memory-pct", f"{memory:g}"]
    args += ["--gpu-busy-util-pct", f"{util:g}"]
    if settings.public_url:
        args += ["--public-url", settings.public_url]
    if not settings.local_node:
        args.append("--no-local-node")
    return args


def launch_env() -> dict[str, str]:
    """What the started Hub needs from this environment: BADAMOA's settings and uv (not on a boot task's PATH)."""
    env = {k: v for k, v in os.environ.items() if k.startswith("BADAMOA_")}
    uv = env.get("BADAMOA_UV") or shutil.which("uv")
    if uv:
        env["BADAMOA_UV"] = uv
    for name in ("BADAMOA_GPU_BUSY_MEMORY_PCT", "BADAMOA_GPU_BUSY_UTIL_PCT", "BADAMOA_OPEN_BROWSER"):
        env.pop(name, None)  # in the arguments already
    return env


def save_launch(settings: Any, launch_gpu: tuple[float, float] | None = None) -> dict[str, Any]:
    home = Path(settings.home).resolve()
    launch = {
        "python": _python(),
        "args": launch_args(settings, launch_gpu),
        "env": launch_env(),
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (home / LAUNCH).write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
    return launch


def _load(home: Path) -> dict[str, Any]:
    path = home / LAUNCH
    if not path.is_file():
        raise RuntimeError(f"{path}가 없습니다. Hub 화면의 설정 → Hub 실행에서 먼저 켜세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def _port(launch: dict[str, Any]) -> int:
    args = launch["args"]
    return int(args[args.index("--port") + 1])


def _command(launch: dict[str, Any], home: Path) -> tuple[str, str]:
    """(program, arguments) the machine runs."""
    return launch["python"], f'-m badamoa.hub.autostart run --home "{home}"'


# ---- turning it on and off -----------------------------------------------------------------------


def status(home: Path, port: int) -> dict[str, Any]:
    """``on``; ``when``: boot (no log-on needed) · logon; ``can_boot``: this Hub can register boot start
    itself; ``admin_command``: what an administrator runs for boot start when it cannot."""
    home = home.resolve()
    view: dict[str, Any] = {
        "supported": WINDOWS or LINUX,
        "on": False,
        "when": "",
        "can_boot": LINUX or is_elevated(),
        "admin_command": "",
        "name": task_name(port) if WINDOWS else f"systemd {unit_name(port)}" if _systemd() else "crontab @reboot",
        "log": str(home / LOG),
        "launch": None,
    }
    if (home / LAUNCH).is_file():
        launch = _load(home)
        view["launch"] = " ".join(["badamoa", *launch["args"]])
    if WINDOWS:
        xml = service.query_task(task_name(port))
        if xml is not None:
            view["on"], view["when"] = True, "boot" if "<BootTrigger" in xml else "logon"
        if view["launch"] and not view["can_boot"] and view["when"] != "boot":
            python = Path(sys.executable)
            if python.name.lower() == "pythonw.exe":  # started by the task: the admin wants to see what it says
                python = python.with_name("python.exe")
            view["admin_command"] = f'& "{python}" -m badamoa.hub.autostart on --home "{home}"'
    elif LINUX:
        if _systemd():
            unit = UNIT_DIR / f"{unit_name(port)}.service"
            done = subprocess.run(["systemctl", "is-enabled", unit_name(port)], capture_output=True, text=True)
            view["on"] = unit.is_file() and done.stdout.strip() == "enabled"
        else:
            listed = subprocess.run(["crontab", "-l"], capture_output=True, text=True) if shutil.which("crontab") else None
            view["on"] = bool(listed and listed.returncode == 0 and _cron_mark(port) in listed.stdout)
        view["when"] = "boot" if view["on"] else ""
    return view


def turn_on(home: Path) -> str:
    """Register the machine to start the Hub as ``autostart.json`` says. "" or why it could not."""
    home = home.resolve()
    launch = _load(home)
    port = _port(launch)
    program, arguments = _command(launch, home)
    if WINDOWS:
        account = _account()
        boot = is_elevated() or account == service.SYSTEM
        xml = service.task_xml(program, arguments, str(home), boot=boot, user=account)
        return service.register_task(task_name(port), xml)
    if _systemd():
        unit = UNIT_DIR / f"{unit_name(port)}.service"
        command = f"{shlex.quote(program)} {arguments}"
        unit.write_text(_UNIT.format(port=port, command=command), encoding="utf-8")
        subprocess.run(["systemctl", "daemon-reload"], check=False)
        done = subprocess.run(["systemctl", "enable", unit_name(port)], capture_output=True, text=True)
        return "" if done.returncode == 0 else (done.stderr or done.stdout).strip()
    if LINUX:
        line = f"@reboot {shlex.quote(program)} {arguments} >/dev/null 2>&1 {_cron_mark(port)}".replace("%", r"\%")
        return _set_cron(port, [line])
    return f"이 OS에서는 자동 실행을 등록하지 못합니다: {sys.platform}"


def turn_off(home: Path, port: int) -> str:
    if WINDOWS:
        why = service.delete_task(task_name(port))
        if why:
            how = f'schtasks /Delete /F /TN "{task_name(port)}"'
            return f"{why} — 관리자 권한으로 등록된 작업입니다. 관리자 PowerShell에서: {how}"
        return ""
    if _systemd():
        subprocess.run(["systemctl", "disable", unit_name(port)], capture_output=True, check=False)
        (UNIT_DIR / f"{unit_name(port)}.service").unlink(missing_ok=True)
        subprocess.run(["systemctl", "daemon-reload"], check=False)
        return ""
    if LINUX:
        return _set_cron(port, [])
    return ""


def _set_cron(port: int, lines: list[str]) -> str:
    if not shutil.which("crontab"):
        return "crontab 명령이 없습니다. 서버 관리자에게 cron을 부탁하세요"
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    kept = [x for x in current.stdout.splitlines() if _cron_mark(port) not in x] if current.returncode == 0 else []
    if not kept and not lines and current.returncode != 0:
        return ""
    done = subprocess.run(["crontab", "-"], input="\n".join(kept + lines) + "\n", capture_output=True, text=True)
    return "" if done.returncode == 0 else (done.stderr or done.stdout).strip() or "crontab이 거절했습니다"


# ---- what the machine runs -----------------------------------------------------------------------


def run(home: Path) -> None:
    """Start the Hub as recorded, its output in ``hub.log`` (a boot task has no console)."""
    home = home.resolve()
    launch = _load(home)
    os.environ.update(launch.get("env") or {})
    os.environ["PYTHONUTF8"] = "1"
    uv = os.environ.get("BADAMOA_UV")
    if uv:  # the Hub builds the node package with it, and the local node makes environments with it
        os.environ["PATH"] = str(Path(uv).parent) + os.pathsep + os.environ.get("PATH", "")
    log = home / LOG
    if log.is_file() and log.stat().st_size > LOG_KEEP:
        log.replace(home / f"{LOG}.1")
    out = open(log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115  (the Hub's output for its whole life)
    sys.stdout = sys.stderr = out
    os.chdir(home)
    from badamoa.cli import app

    app(launch["args"], prog_name="badamoa")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m badamoa.hub.autostart")
    parser.add_argument("action", choices=["run", "on", "off", "status"])
    parser.add_argument("--home", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "run":
        run(args.home)
        return 0
    port = _port(_load(args.home))
    if args.action == "on":
        why = turn_on(args.home)
        print(why or json.dumps(status(args.home, port), ensure_ascii=False, indent=2))
        return 1 if why else 0
    if args.action == "off":
        why = turn_off(args.home, port)
        print(why or "자동 실행을 껐습니다.")
        return 1 if why else 0
    print(json.dumps(status(args.home, port), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
