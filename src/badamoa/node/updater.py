"""A node keeps up with its Hub: when the Hub is newer, install that version from the Hub and restart.

Only for nodes put in by the install scripts (a ``uv tool`` install under /opt/badamoa or
C:\\Program Files\\BADAMOA); a node run from a source checkout is left alone. Never while a job runs.

Linux replaces the files in place (a running process keeps the old ones) and re-executes itself.
Windows cannot replace files a running program holds, so a small helper waits for this process
to end, installs, and starts the node again.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("badamoa.node.updater")

WINDOWS = sys.platform == "win32"


def newer(remote: str, local: str) -> bool:
    """True when ``remote`` is a later x.y.z than ``local`` (anything else is not compared)."""
    try:
        return tuple(int(p) for p in remote.split(".")) > tuple(int(p) for p in local.split("."))
    except ValueError:
        return False


def layout() -> dict[str, Path] | None:
    """The folders of a script install (uv tool), or None when this node runs some other way."""
    prefix = Path(sys.prefix)
    if not (prefix / "uv-receipt.toml").is_file():
        return None
    tools = prefix.parent
    root = tools.parent
    bin_dir = root / "bin"
    uv = bin_dir / ("uv.exe" if WINDOWS else "uv")
    if not uv.is_file():
        found = shutil.which("uv")
        if found is None:
            return None
        uv = Path(found)
    return {"root": root, "tools": tools, "bin": bin_dir, "python": root / "python", "uv": uv}


def _env(where: dict[str, Path]) -> dict[str, str]:
    return {
        **os.environ,
        "UV_INSTALL_DIR": str(where["bin"]),
        "UV_TOOL_DIR": str(where["tools"]),
        "UV_TOOL_BIN_DIR": str(where["bin"]),
        "UV_PYTHON_INSTALL_DIR": str(where["python"]),
        "UV_NO_MODIFY_PATH": "1",
        # the program's own downloads in its folder, gone with it (unless the server set a cache itself)
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(where["root"] / "uv-cache"),
    }


def _install_args(where: dict[str, Path], hub_url: str, version: str) -> list[str]:
    index = f"{hub_url.rstrip('/')}/install/simple/"
    return [str(where["uv"]), "tool", "install", "--force", "--python", "3.12", "--index", index, f"badamoa=={version}"]


def _program(where: dict[str, Path]) -> Path:
    return where["bin"] / ("badamoa.exe" if WINDOWS else "badamoa")


def update(hub_url: str, version: str) -> bool:
    """Start moving to ``version``. Returns False when this node cannot update itself (or it failed).

    On success Linux does not return (the process becomes the new version); Windows hands over
    to a helper and the caller should stop the agent.
    """
    where = layout()
    if where is None:
        return False
    if WINDOWS:
        return _hand_over(where, hub_url, version)
    done = subprocess.run(_install_args(where, hub_url, version), env=_env(where), capture_output=True, text=True)
    if done.returncode != 0:
        log.warning("노드 업데이트 실패 (%s): %s", version, (done.stderr or done.stdout)[-400:])
        return False
    program = _program(where)
    check = subprocess.run([str(program), "version"], capture_output=True, text=True)
    if check.stdout.strip() != version:
        log.warning("노드 업데이트 뒤 버전이 맞지 않습니다: %r (기대 %s)", check.stdout.strip(), version)
        return False
    log.info("노드를 %s로 업데이트했습니다. 다시 시작합니다.", version)
    os.execv(str(program), [str(program), *sys.argv[1:]])  # the same process, now the new version
    return True  # pragma: no cover - execv does not return


def _hand_over(where: dict[str, Path], hub_url: str, version: str) -> bool:
    """Windows: a helper waits for this agent (and its launcher) to exit, installs, and starts it again.

    The helper is started through WMI so that it is nobody's child: Windows ends a job's processes
    together (the Task Scheduler task, a terminal), and the helper has to outlive this agent.
    """
    folder = Path(tempfile.mkdtemp(prefix="badamoa-update-"))
    install = subprocess.list2cmdline(_install_args(where, hub_url, version))
    program = _program(where)
    again = subprocess.list2cmdline([str(program), *sys.argv[1:]])
    waits = " ".join(str(p) for p in {os.getpid(), os.getppid()})
    settings = {k: v for k, v in _env(where).items() if k.startswith("UV_")}
    lines = [
        "@echo off",
        *(f'set "{k}={v}"' for k, v in settings.items()),  # a WMI process does not inherit this environment
        f"for %%P in ({waits}) do call :wait %%P",
        f'{install} > "{folder / "update.log"}" 2>&1',
        f'start "" {again}',
        "exit /b 0",
        ":wait",
        # ping as the pause: "timeout" needs a console, and the helper has none
        'tasklist /FI "PID eq %1" | find "%1" >nul && (ping -n 2 127.0.0.1 >nul & goto wait)',
        "exit /b 0",
    ]
    script = folder / "update.cmd"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    if not start_outside(script):
        log.warning("노드 업데이트 도우미를 시작하지 못했습니다.")
        return False
    log.info("노드를 %s로 업데이트합니다 (도우미: %s).", version, script)
    return True


def wait_lines(pids: set[int]) -> list[str]:
    """Batch lines that wait until these processes are gone (ping as the pause: "timeout" needs a console)."""
    return [
        f"for %%P in ({' '.join(str(p) for p in pids)}) do call :wait %%P",
        "goto :after",
        ":wait",
        'tasklist /FI "PID eq %1" | find "%1" >nul && (ping -n 2 127.0.0.1 >nul & goto wait)',
        "exit /b 0",
        ":after",
    ]


def start_outside(script: Path) -> bool:
    """Windows: run a .cmd that must outlive this process, started through WMI so it is nobody's child."""
    command = f'cmd /c "{script}"'.replace("'", "''")
    create = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{CommandLine='{command}'}}; exit $r.ReturnValue"
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", create], capture_output=True, text=True, timeout=60
        )
        if done.returncode == 0:
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    # No WMI: a detached child that leaves the job when it may.
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
    for extra in (breakaway, 0):
        try:
            subprocess.Popen(["cmd", "/c", str(script)], creationflags=flags | extra, close_fds=True)
            return True
        except OSError:
            continue
    return False
