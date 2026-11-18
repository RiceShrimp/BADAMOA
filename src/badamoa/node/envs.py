"""Environments on a node: where a template's code runs.

- **auto**: a virtual environment built on first use from the template's
  requirements. ``uv`` creates it; for PyTorch it picks the build matching the
  node's GPU driver (``--torch-backend auto``; the CPU build when there is no
  GPU). Set ``BADAMOA_TORCH_BACKEND`` on the node to force one (``cpu``, ``cu128``).
  Environments are shared by content: two templates with the same requirements
  use the same one.
- **python**: an interpreter that already exists on the node (a conda or venv
  environment the team maintains). BADAMOA only runs it.
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
from typing import Any

LOCK_STALE_S = 3600.0
LOCK_WAIT_S = 3600.0
# What the node keeps to itself: code it runs without having written it (a template, a package's
# build script) never sees these. With the token such code could act on the Hub as the node.
SECRETS = ("BADAMOA_NODE_TOKEN",)


def code_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for code the node runs but did not write: the node's own, minus its credentials."""
    env = {key: value for key, value in os.environ.items() if key not in SECRETS}
    env.update(extra or {})
    return env

# Run by ``check`` inside the environment being examined: nothing but the standard library.
PROBE = """
import json, sys
found = {"python": sys.version.split()[0], "executable": sys.executable, "packages": {}}
for name in ("torch", "torchvision", "ultralytics", "numpy"):
    try:
        module = __import__(name)
        found["packages"][name] = getattr(module, "__version__", "?")
    except Exception:
        pass
try:
    import torch
    found["cuda"] = bool(torch.cuda.is_available())
    found["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if found["cuda"] else []
except Exception:
    pass
print(json.dumps(found))
"""


def find_uv() -> str:
    found = os.environ.get("BADAMOA_UV") or shutil.which("uv")
    if not found:
        raise RuntimeError(
            "uv를 찾을 수 없습니다. 환경을 자동으로 만들려면 이 노드에 uv가 필요합니다 "
            "(https://docs.astral.sh/uv/). 설치한 뒤 Node Agent를 다시 시작하세요."
        )
    return found


def torch_backend() -> str:
    return os.environ.get("BADAMOA_TORCH_BACKEND", "").strip() or "auto"


def env_dir(home: Path, requirements: list[str], python: str, torch: bool) -> Path:
    """The directory changes whenever anything that defines the environment does."""
    identity = [python, sorted(requirements), torch_backend() if torch else ""]
    return home / "envs" / hashlib.sha1(json.dumps(identity).encode()).hexdigest()[:12]


def python_of(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(command: list[str], log: Callable[[str], None]) -> None:
    # A package built from source runs its own code while installing: it gets no credentials either.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=code_env(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip():
            log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"환경 설치 실패 (종료 코드 {proc.returncode}): {' '.join(command[:3])} …")


def ensure(
    home: Path, requirements: list[str], python: str = "3.12", torch: bool = False, log: Callable[[str], None] = print
) -> Path:
    """Return the Python of the automatically installed environment, building it if it is not there yet.

    Code that needs nothing beyond the standard library runs with the node's own Python.
    """
    if not requirements:
        return Path(sys.executable)
    env = env_dir(home, requirements, python, torch)
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
                log("다른 작업이 같은 환경을 만드는 중입니다. 기다립니다…")
            if waited >= LOCK_WAIT_S:
                raise RuntimeError("환경이 준비되기를 기다리다 시간이 초과되었습니다.") from None
            time.sleep(2)
            waited += 2
    try:
        if ready.is_file():
            return python_of(env)
        uv = find_uv()
        started = time.monotonic()
        log(f"학습 환경을 처음 만듭니다 (이 서버에서 한 번만 필요, 몇 분 걸릴 수 있습니다): {env}")
        shutil.rmtree(env, ignore_errors=True)  # leftovers of an interrupted build
        _run([uv, "venv", "--color", "never", "--python", python, str(env)], log)
        install = [uv, "pip", "install", "--color", "never", "--python", str(python_of(env))]
        if torch:
            install += ["--torch-backend", torch_backend()]
        _run(install + requirements, log)
        ready.write_text(json.dumps({"requirements": requirements, "python": python, "torch": torch}), encoding="utf-8")
        log(f"환경 준비 완료 ({time.monotonic() - started:.0f}초)")
        return python_of(env)
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def listing(home: Path) -> list[dict[str, Any]]:
    """The automatically installed environments ready on this node (what defines each, not where it is)."""
    found = []
    for marker in sorted((home / "envs").glob("*/badamoa-env.json")):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        requirements = sorted(data.get("requirements") or [])
        found.append({"requirements": requirements, "python": data.get("python"), "torch": data.get("torch")})
    return found


def remove(home: Path, spec: dict[str, Any], log: Callable[[str], None] = print) -> dict[str, Any]:
    """Take an automatically installed environment off this node: its folder. Refused while it is being built."""
    env = env_dir(home, list(spec.get("requirements") or []), spec.get("python") or "3.12", bool(spec.get("torch")))
    if (env.parent / f"{env.name}.lock").exists():
        raise RuntimeError("이 환경을 만드는 작업이 있습니다. 끝난 뒤에 지우세요.")
    if not env.exists():
        log("이 서버에는 이 환경이 없습니다.")
        return {"removed": False}
    size = sum(p.stat().st_size for p in env.rglob("*") if p.is_file() and not p.is_symlink())
    marker = env / "badamoa-env.json"
    marker.unlink(missing_ok=True)  # no longer ready, even if a file below is in use and stays
    try:
        shutil.rmtree(env)
    except OSError as e:
        raise RuntimeError(f"환경 폴더를 다 지우지 못했습니다 (쓰는 프로그램이 있을 수 있음): {e}") from e
    log(f"가상환경을 지웠습니다: {env} ({size / 1024**2:,.0f} MB)")
    return {"removed": True, "bytes": size}


def existing(path: str) -> Path:
    """An interpreter the admin registered for this node."""
    python = Path(path).expanduser()
    if python.is_dir():  # the environment's folder was given: find its interpreter
        python = python_of(python)
    if not python.is_file():
        raise RuntimeError(f"등록된 환경의 Python을 찾을 수 없습니다: {path}. 설정 → 환경에서 이 서버의 경로를 확인하세요.")
    return python


def resolve(home: Path, spec: dict[str, Any], log: Callable[[str], None] = print) -> Path:
    """The Python a run should use, from the environment snapshot the Hub sent (not for Docker: ``containers``)."""
    if spec.get("kind") == "docker":
        raise RuntimeError("Docker 환경에는 Python 경로가 없습니다 (컨테이너로 실행합니다).")
    if spec.get("kind") == "python":
        return existing(str(spec.get("path", "")))
    return ensure(home, list(spec.get("requirements") or []), spec.get("python") or "3.12", bool(spec.get("torch")), log)


PACKAGES = (
    "import importlib.metadata as m, json;"
    "print(json.dumps(sorted({f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions() if d.metadata['Name']})))"
)


def packages(python: Path) -> list[str]:
    """What is installed in an environment (``name==version``), for the run's record. Empty if it cannot be read."""
    try:
        done = subprocess.run(
            [str(python), "-c", PACKAGES], capture_output=True, text=True, timeout=60, check=False, env=code_env()
        )
        return list(json.loads(done.stdout.strip().splitlines()[-1]))[:1000] if done.returncode == 0 else []
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return []


def check(path: str) -> dict[str, Any]:
    """Try an existing environment: does it start, and what is in it."""
    python = existing(path)
    try:
        done = subprocess.run(
            [str(python), "-c", PROBE], capture_output=True, text=True, timeout=120, check=False, env=code_env()
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"환경의 Python을 실행하지 못했습니다: {e}") from e
    if done.returncode != 0:
        raise RuntimeError(f"환경의 Python이 오류로 끝났습니다: {done.stderr.strip()[-400:]}")
    return {"ok": True, **json.loads(done.stdout.strip().splitlines()[-1])}
