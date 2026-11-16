"""Register BADAMOA components to start at boot.

Linux: a systemd unit. Windows: a Task Scheduler task that runs as SYSTEM at
startup (no extra dependency such as pywin32 or a service wrapper needed).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

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


def _exe() -> str:
    found = shutil.which("badamoa")
    if found:
        return found
    candidate = Path(sys.executable).with_name("badamoa.exe" if sys.platform == "win32" else "badamoa")
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("badamoa 실행 파일을 찾을 수 없습니다.")


def install(component: str, args: str) -> str:
    """component: 'hub' | 'node'. Returns a human-readable description of what was installed."""
    title = "Hub" if component == "hub" else "Node"
    name = f"badamoa-{component}"
    exe = _exe()
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
