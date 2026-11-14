"""Plugin environments on a node: one virtual environment per plugin, built on first use.

``uv`` creates the environment and installs the plugin's requirements. For
plugins that need PyTorch, uv picks the build matching the node's GPU driver
(``--torch-backend auto``; the CPU build when there is no GPU). Set
``BADAMOA_TORCH_BACKEND`` on the node to force one, e.g. ``cpu`` or ``cu128``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from badamoa.core.plugins import PluginSpec

LOCK_STALE_S = 3600.0
LOCK_WAIT_S = 3600.0


def find_uv() -> str:
    found = os.environ.get("BADAMOA_UV") or shutil.which("uv")
    if not found:
        raise RuntimeError(
            "uv를 찾을 수 없습니다. 플러그인 환경을 만들려면 이 노드에 uv가 필요합니다 "
            "(https://docs.astral.sh/uv/). 설치한 뒤 Node Agent를 다시 시작하세요."
        )
    return found


def torch_backend() -> str:
    return os.environ.get("BADAMOA_TORCH_BACKEND", "").strip() or "auto"


def env_dir(home: Path, plugin: PluginSpec) -> Path:
    """The directory changes whenever anything that defines the environment does."""
    identity = [plugin.version, plugin.python, plugin.requirements, torch_backend() if plugin.torch else ""]
    digest = hashlib.sha1(json.dumps(identity).encode()).hexdigest()[:10]
    return home / "envs" / f"{plugin.name}-{digest}"


def python_of(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(command: list[str], log: Callable[[str], None]) -> None:
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip():
            log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"환경 설치 실패 (종료 코드 {proc.returncode}): {' '.join(command[:3])} …")


def ensure(home: Path, plugin: PluginSpec, log: Callable[[str], None] = print) -> Path:
    """Return the plugin environment's Python, building the environment if it is not there yet."""
    platform = "windows" if os.name == "nt" else sys.platform
    if not any(platform.startswith(p) for p in plugin.platforms):
        raise RuntimeError(f"{plugin.title} 플러그인은 이 OS({platform})를 지원하지 않습니다.")
    env = env_dir(home, plugin)
    ready = env / "badamoa-env.json"
    if ready.is_file():
        return python_of(env)

    env.parent.mkdir(parents=True, exist_ok=True)
    lock = env.parent / f"{env.name}.lock"
    waited = 0.0
    while True:  # two jobs asking for the same new environment: the second waits for the first
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if waited == 0:
                log(f"다른 작업이 {plugin.title} 환경을 만드는 중입니다. 기다립니다…")
            if waited >= LOCK_WAIT_S:
                raise RuntimeError("플러그인 환경이 준비되기를 기다리다 시간이 초과되었습니다.") from None
            time.sleep(2)
            waited += 2
    try:
        if ready.is_file():
            return python_of(env)
        uv = find_uv()
        started = time.monotonic()
        log(f"{plugin.title} 환경을 처음 만듭니다 (한 번만 필요, 몇 분 걸릴 수 있습니다): {env}")
        shutil.rmtree(env, ignore_errors=True)  # leftovers of an interrupted build
        _run([uv, "venv", "--color", "never", "--python", plugin.python, str(env)], log)
        install = [uv, "pip", "install", "--color", "never", "--python", str(python_of(env))]
        if plugin.torch:
            install += ["--torch-backend", torch_backend()]
        _run(install + plugin.requirements, log)
        ready.write_text(
            json.dumps({"plugin": plugin.name, "version": plugin.version, "requirements": plugin.requirements}),
            encoding="utf-8",
        )
        log(f"환경 준비 완료 ({time.monotonic() - started:.0f}초)")
        return python_of(env)
    finally:
        shutil.rmtree(lock, ignore_errors=True)
