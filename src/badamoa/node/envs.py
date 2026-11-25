"""Environments on a node: where a template's code runs.

- **auto**: a virtual environment built on first use from the template's
  requirements. ``uv`` creates it; for PyTorch it picks the build matching the
  node's GPU driver (``--torch-backend auto``; the CPU build when there is no
  GPU). Set ``BADAMOA_TORCH_BACKEND`` on the node to force one (``cpu``, ``cu128``).
  Environments are shared by content: two templates with the same requirements
  use the same one.
- **python**: an interpreter that already exists on the node (a conda or venv
  environment the team maintains). BADAMOA only runs it.

The same versions on every server: when a template version passed its trial, the Hub sends the
versions installed then (``pinned``) and the install keeps to them (PyTorch's version only: its
build still follows the server's driver). In-house wheels come from the Hub's package list
(``wheels``): the node fetches them into ``cache/wheels`` and uv looks there first.

Environments are built in the node's work folder (``places``); one built before in another folder (the
node's home) is used where it is. uv links packages from its cache into an environment, so the node's own cache
is kept in the work folder (``uv_env``), and goes with the node.
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

from badamoa.node import places

LOCK_STALE_S = 3600.0
LOCK_WAIT_S = 3600.0
USED = "badamoa-used"  # touched whenever a job uses the environment: when it was last used
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
    if not found:  # the uv the install script put beside the node (its folder is not on a service's PATH)
        from badamoa.node import updater

        where = updater.layout()
        found = str(where["uv"]) if where else None
    if not found:
        raise RuntimeError(
            "uv를 찾을 수 없습니다. 환경을 자동으로 만들려면 이 노드에 uv가 필요합니다 "
            "(https://docs.astral.sh/uv/). 설치한 뒤 Node Agent를 다시 시작하세요."
        )
    return found


def uv_cache(where: places.Places) -> Path | None:
    """The uv cache the node uses: its own, in its work folder (``uv-cache``). It is on the disk of the node's
    environments (uv hard-links packages from its cache into them) and goes with the node when it is removed;
    uv's cache of the account (``~/.cache/uv``) may hold other people's work, so the node leaves it alone.
    None: the server set ``UV_CACHE_DIR`` itself."""
    if os.environ.get("UV_CACHE_DIR"):
        return None
    return where.work / "uv-cache"


def uv_env(where: places.Places) -> dict[str, str]:
    """What the uv the node runs needs in its environment to use the node's uv cache."""
    folder = uv_cache(where)
    return {"UV_CACHE_DIR": str(folder)} if folder is not None else {}


def torch_backend() -> str:
    return os.environ.get("BADAMOA_TORCH_BACKEND", "").strip() or "auto"


def fetch_wheels(home: Path, wheels: list[dict[str, Any]], get: Callable[[str], bytes]) -> Path:
    """The Hub's in-house wheels in this node's ``cache/wheels``, each checked against its hash."""
    folder = places.of(home).wheels
    folder.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        name = Path(str(wheel.get("filename") or "")).name
        digest = str(wheel.get("sha256") or "")
        target = folder / name
        if not name.endswith(".whl") or not digest:
            continue
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        data = get(digest)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Hub에서 받은 패키지 {name}의 해시가 맞지 않습니다.")
        partial = target.with_name(f"{name}.{os.getpid()}.part")
        partial.write_bytes(data)
        partial.replace(target)
    return folder


def env_name(requirements: list[str], python: str, torch: bool) -> str:
    """The folder's name changes whenever anything that defines the environment does."""
    identity = [python, sorted(requirements), torch_backend() if torch else ""]
    return hashlib.sha1(json.dumps(identity).encode()).hexdigest()[:12]


def env_dir(home: Path, requirements: list[str], python: str, torch: bool) -> Path:
    """The environment's folder: where it is ready (the work folder, or one the node used before), else where
    it is built (the work folder)."""
    where = places.of(home)
    name = env_name(requirements, python, torch)
    for base in where.roots():
        if (base / "envs" / name / "badamoa-env.json").is_file():
            return base / "envs" / name
    return where.envs / name


def python_of(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(command: list[str], log: Callable[[str], None], extra: dict[str, str] | None = None) -> None:
    # A package built from source runs its own code while installing: it gets no credentials either.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=code_env(extra),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip():
            log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"환경 설치 실패 (종료 코드 {proc.returncode}): {' '.join(command[:3])} …")


def ensure(
    home: Path,
    requirements: list[str],
    python: str = "3.12",
    torch: bool = False,
    log: Callable[[str], None] = print,
    pinned: list[str] | None = None,
    find_links: Path | None = None,
) -> Path:
    """Return the Python of the automatically installed environment, building it if it is not there yet.

    Code that needs nothing beyond the standard library runs with the node's own Python. ``pinned``:
    the versions to keep to (if they cannot be installed here, the install goes on without them and
    says so). ``find_links``: a folder of wheels uv looks in besides the index.
    """
    if not requirements:
        return Path(sys.executable)
    env = env_dir(home, requirements, python, torch)
    ready = env / "badamoa-env.json"
    if ready.is_file():
        _touch(env)
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
            _touch(env)
            return python_of(env)
        uv = find_uv()
        started = time.monotonic()
        log(f"학습 환경을 처음 만듭니다 (이 서버에서 한 번만 필요, 몇 분 걸릴 수 있습니다): {env}")
        shutil.rmtree(env, ignore_errors=True)  # leftovers of an interrupted build
        cached = uv_env(places.of(home))
        if cached:
            log(f"uv 캐시: {cached['UV_CACHE_DIR']} (이 노드의 것: 노드를 지우면 함께 지워짐)")
        _run([uv, "venv", "--color", "never", "--python", python, str(env)], log, cached)
        install = [uv, "pip", "install", "--color", "never", "--python", str(python_of(env))]
        if torch:
            install += ["--torch-backend", torch_backend()]
        if find_links is not None:
            install += ["--find-links", str(find_links)]
        kept = 0
        if pinned:
            constraints = env / "badamoa-pinned.txt"
            constraints.write_text("\n".join(pinned) + "\n", encoding="utf-8")
            try:
                _run(install + ["-c", str(constraints)] + requirements, log, cached)
                kept = len(pinned)
                log(f"시험 실행 때의 버전으로 설치했습니다 (고정 {kept}개).")
            except RuntimeError as e:
                log(f"시험 실행 때의 버전으로는 이 서버에 설치하지 못해 고정 없이 설치합니다: {str(e)[-200:]}")
                _run(install + requirements, log, cached)
        else:
            _run(install + requirements, log, cached)
        marker = {"requirements": requirements, "python": python, "torch": torch, "pinned": kept}
        ready.write_text(json.dumps(marker), encoding="utf-8")
        log(f"환경 준비 완료 ({time.monotonic() - started:.0f}초)")
        return python_of(env)
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def _touch(env: Path) -> None:
    try:
        (env / USED).write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def markers(home: Path) -> list[Path]:
    """The ready environments' markers in every folder the node keeps environments in, each name once (the work
    folder's first)."""
    found: dict[str, Path] = {}
    for base in places.of(home).roots():
        for marker in sorted((base / "envs").glob("*/badamoa-env.json")):
            found.setdefault(marker.parent.name, marker)
    return [found[name] for name in sorted(found)]


def listing(home: Path) -> list[dict[str, Any]]:
    """The automatically installed environments ready on this node: what defines each, and its folder's name."""
    found = []
    for marker in markers(home):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        requirements = sorted(data.get("requirements") or [])
        found.append(
            {
                "id": marker.parent.name,
                "requirements": requirements,
                "python": data.get("python"),
                "torch": data.get("torch"),
            }
        )
    return found


def remove(home: Path, spec: dict[str, Any], log: Callable[[str], None] = print) -> dict[str, Any]:
    """Take an automatically installed environment off this node: its folder (in every folder the node keeps
    environments in). Refused while it is being built."""
    name = env_name(list(spec.get("requirements") or []), spec.get("python") or "3.12", bool(spec.get("torch")))
    where = places.of(home)
    if (where.envs / f"{name}.lock").exists():
        raise RuntimeError("이 환경을 만드는 작업이 있습니다. 끝난 뒤에 지우세요.")
    folders = where.everywhere("envs", name)
    if not folders:
        log("이 서버에는 이 환경이 없습니다.")
        return {"removed": False}
    size = 0
    for env in folders:
        here = sum(p.stat().st_size for p in env.rglob("*") if p.is_file() and not p.is_symlink())
        marker = env / "badamoa-env.json"
        marker.unlink(missing_ok=True)  # no longer ready, even if a file below is in use and stays
        try:
            shutil.rmtree(env)
        except OSError as e:
            raise RuntimeError(f"환경 폴더를 다 지우지 못했습니다 (쓰는 프로그램이 있을 수 있음): {e}") from e
        log(f"가상환경을 지웠습니다: {env} ({here / 1024**2:,.0f} MB)")
        size += here
    return {"removed": True, "bytes": size}


def existing(path: str) -> Path:
    """An interpreter the admin registered for this node."""
    python = Path(path).expanduser()
    if python.is_dir():  # the environment's folder was given: find its interpreter
        python = python_of(python)
    if not python.is_file():
        raise RuntimeError(f"등록된 환경의 Python을 찾을 수 없습니다: {path}. 설정 → 환경에서 이 서버의 경로를 확인하세요.")
    return python


def resolve(
    home: Path, spec: dict[str, Any], log: Callable[[str], None] = print, get: Callable[[str], bytes] | None = None
) -> Path:
    """The Python a run should use, from the environment snapshot the Hub sent (not for Docker: ``containers``).

    ``get``: fetches an in-house wheel from the Hub by its hash (needed when the spec names ``wheels``).
    """
    if spec.get("kind") == "docker":
        raise RuntimeError("Docker 환경에는 Python 경로가 없습니다 (컨테이너로 실행합니다).")
    if spec.get("kind") == "python":
        return existing(str(spec.get("path", "")))
    requirements = list(spec.get("requirements") or [])
    links = fetch_wheels(home, spec["wheels"], get) if spec.get("wheels") and get and requirements else None
    return ensure(
        home,
        requirements,
        spec.get("python") or "3.12",
        bool(spec.get("torch")),
        log,
        pinned=list(spec.get("pinned") or []),
        find_links=links,
    )


PACKAGES = (
    "import importlib.metadata as m, json;"
    "print(json.dumps(sorted({f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions() if d.metadata['Name']})))"
)
# Where each package came from, as it wrote down itself when it was installed from somewhere other
# than an index (PEP 610 ``direct_url.json``): a Git commit, a URL, a folder on the server.
SOURCES = """
import importlib.metadata as m, json
found = {}
for d in m.distributions():
    name = d.metadata["Name"]
    try:
        url = json.loads(d.read_text("direct_url.json") or "null")
    except Exception:
        url = None
    if name and isinstance(url, dict):
        found[name] = url
print(json.dumps(found))
"""


def describe(direct: dict[str, Any]) -> dict[str, Any]:
    """One package's ``direct_url.json`` as a source: git (with the commit), local (a folder or file on the
    server), editable, or url."""
    url = str(direct.get("url") or "")
    if direct.get("vcs_info"):
        info = direct["vcs_info"]
        return {"kind": str(info.get("vcs") or "git"), "url": url, "commit": str(info.get("commit_id") or "")}
    if (direct.get("dir_info") or {}).get("editable"):
        return {"kind": "editable", "url": url}
    if url.startswith("file:"):
        return {"kind": "local", "url": url}
    return {"kind": "url", "url": url}


def sources(python: Path) -> dict[str, dict[str, Any]]:
    """The packages of an environment that did not come from an index, and where they came from."""
    try:
        done = subprocess.run(
            [str(python), "-c", SOURCES], capture_output=True, text=True, timeout=60, check=False, env=code_env()
        )
        found = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else {}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {}
    return {name: describe(url) for name, url in found.items() if isinstance(url, dict)}


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
