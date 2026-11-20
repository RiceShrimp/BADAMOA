"""Running a template's code in a Docker container instead of a virtual environment.

An environment is a declaration; the node makes it real either as a virtual environment (``envs``)
or, when the environment is a Docker image, as a container. The image holds the environment only:
the template's code, the exported data, the output folder and the caches stay on the node and are
mounted in, so a new template version does not need a new image.

Paths: on Linux every folder is mounted at the same path it has on the node, so the config, the
symbolic links of the exported data and the paths the code reports back need no translation. On
Windows (Docker Desktop runs Linux containers) a folder ``C:\\x\\y`` is mounted at ``/host/c/x/y``;
the config and ``data.yaml`` are rewritten for the container, paths in the code's reports are
mapped back, and the export uses hard links or copies (a symbolic link to ``C:\\…`` means nothing
inside the container).

What the container gets: the GPUs the Hub reserved for the job (``--gpus device=…``, numbered from
0 inside like with ``CUDA_VISIBLE_DEVICES``), the node's user on Linux, ``--ipc=host`` for PyTorch's
data loader workers, and none of the node's credentials. It is named after the job, so a stopped
job's container can be removed (``kill``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PULLED = "docker-images.json"  # the images this node pulled or found for BADAMOA (``listing``)
PROBE_TIMEOUT_S = 20


def docker() -> str | None:
    """The docker command, if there is one."""
    return os.environ.get("BADAMOA_DOCKER") or shutil.which("docker")


def capability() -> dict[str, Any] | None:
    """Whether this node can run containers: the server's version and OS, and whether GPUs can be passed in.

    None when there is no Docker or its server does not answer (sent once with the node's static facts).
    """
    command = docker()
    if not command:
        return None
    try:
        done = subprocess.run(
            [command, "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
        info = json.loads(done.stdout.strip() or "{}") if done.returncode == 0 else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if not info.get("ServerVersion"):
        return None
    runtimes = info.get("Runtimes") or {}
    return {
        "version": info.get("ServerVersion"),
        "os": (info.get("OSType") or "").lower(),  # linux: Linux containers (also Docker Desktop on Windows)
        "gpu": "nvidia" in runtimes or "nvidia" in str(info.get("DefaultRuntime") or ""),
    }


# ---- paths ------------------------------------------------------------------------------------


def inside(path: str | Path) -> str:
    """Where a node folder or file appears inside the container."""
    text = str(path)
    if os.name != "nt":
        return text
    windows = PureWindowsPath(text)
    drive = windows.drive.rstrip(":").lower()
    parts = [p for p in windows.parts[1:]]
    return str(PurePosixPath("/host", drive, *parts)) if drive else text.replace("\\", "/")


def outside(path: str) -> str:
    """The node path of a path the code reported from inside the container."""
    if os.name != "nt" or not path.startswith("/host/"):
        return path
    parts = PurePosixPath(path).parts  # ("/", "host", "c", …)
    if len(parts) < 3:
        return path
    return str(PureWindowsPath(f"{parts[2].upper()}:\\", *parts[3:]))


def translate(value: Any, folders: list[Path]) -> Any:
    """A config with every node path under ``folders`` turned into the path inside the container."""
    if isinstance(value, dict):
        return {k: translate(v, folders) for k, v in value.items()}
    if isinstance(value, list):
        return [translate(v, folders) for v in value]
    if isinstance(value, str) and os.name == "nt":
        for folder in folders:
            try:
                Path(value).relative_to(folder)
            except (ValueError, OSError):
                continue
            return inside(value)
    return value


def rewrite_data_yaml(path: Path) -> None:
    """``path:`` in a YOLO data.yaml is the export folder on the node: point it at the container's."""
    if os.name != "nt" or not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.startswith("path:"):
            node_path = json.loads(line[len("path:") :].strip())
            line = f"path: {json.dumps(inside(node_path), ensure_ascii=False)}"
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---- images ----------------------------------------------------------------------------------


def _known(home: Path) -> dict[str, dict[str, Any]]:
    try:
        known = json.loads((home / PULLED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in known.items() if isinstance(v, dict)}


def _remember(home: Path, image: str, image_id: str, pulled: bool) -> None:
    known = _known(home)
    # an image that was here before BADAMOA first used it stays the node's own: removing it only forgets it
    known[image] = {
        "id": image_id,
        "pulled": pulled or bool((known.get(image) or {}).get("pulled")),
        "last_used": time.time(),
    }
    (home / PULLED).write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")


def details(home: Path, image: str) -> dict[str, Any]:
    """An image's size on disk, whether BADAMOA pulled it, and when a job last used it."""
    entry = _known(home).get(image) or {}
    size = 0
    command = docker()
    if command:
        try:
            done = subprocess.run(
                [command, "image", "inspect", "--format", "{{.Size}}", image],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_S,
                check=False,
            )
            text = done.stdout.strip()
            size = int(text) if done.returncode == 0 and text.isdigit() else 0
        except (OSError, subprocess.TimeoutExpired):
            size = 0
    used = entry.get("last_used")
    return {
        "bytes": size,
        "pulled": bool(entry.get("pulled")),
        "last_used": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(used)) if used else None,
    }


def listing(home: Path) -> list[dict[str, Any]]:
    """The images this node has for BADAMOA, as environments (what the Hub shows as ready)."""
    return [{"kind": "docker", "image": image, "id": entry.get("id")} for image, entry in sorted(_known(home).items())]


def remove(home: Path, image: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Take an image off this node if BADAMOA pulled it; one that was here before is only forgotten."""
    known = _known(home)
    entry = known.pop(image, None)
    (home / PULLED).write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
    if entry is None:
        log(f"이 서버에서 BADAMOA가 쓰던 이미지가 아닙니다: {image}")
        return {"removed": False}
    if not entry.get("pulled"):
        log(f"BADAMOA가 받기 전부터 있던 이미지라 지우지 않고 목록에서만 뺐습니다: {image}")
        return {"removed": False, "forgotten": True}
    command = docker()
    done = subprocess.run([command or "docker", "rmi", image], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"Docker 이미지를 지우지 못했습니다: {(done.stderr or done.stdout).strip()[-300:]}")
    log(f"Docker 이미지를 지웠습니다: {image}")
    return {"removed": True}


def run_python(image: str, code: str, gpus: bool = False, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    """Run a little Python in the image (no mounts, no credentials)."""
    run = [docker() or "docker", "run", "--rm"]
    if gpus:
        run += ["--gpus", "all"]
    return subprocess.run(
        [*run, image, "python", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def packages(image: str, probe: str) -> list[str]:
    """What is installed in the image (``name==version``), for the run's record. Empty if it cannot be read."""
    try:
        done = run_python(image, probe, timeout=120)
        return list(json.loads(done.stdout.strip().splitlines()[-1]))[:1000] if done.returncode == 0 else []
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return []


def image_id(image: str) -> str | None:
    """The local image's id (its content digest), or None when the node does not have it."""
    command = docker()
    if not command:
        return None
    done = subprocess.run(
        [command, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, check=False
    )
    return done.stdout.strip() or None if done.returncode == 0 else None


def pull_failure(image: str, output: str) -> str:
    """Why ``docker pull`` failed, in words a person can act on."""
    head = image.split("/", 1)[0] if "/" in image else ""
    registry = head if any(c in head for c in ".:") else ""
    failed = f"Docker 이미지를 받지 못했습니다: {image}."
    if "HTTP response to HTTPS client" in output and registry:
        # A registry without TLS (``docker run registry``): each Docker that pulls from it has to be told it may.
        return (
            f"{failed} 레지스트리 {registry}는 HTTPS 없이(HTTP로) 열려 있는데, "
            "이 서버의 Docker는 HTTP 레지스트리를 허용하지 않습니다. "
            f'이 서버의 /etc/docker/daemon.json에 "insecure-registries": ["{registry}"]를 넣고 '
            "sudo systemctl reload docker를 하세요(돌고 있는 컨테이너는 멈추지 않음). "
            "Windows · Mac의 Docker Desktop은 Settings → Docker Engine에서 같은 값을 넣습니다."
        )
    if "unauthorized" in output or "authentication required" in output or "pull access denied" in output:
        return f"{failed} 레지스트리에 로그인이 필요합니다(이 서버에서 docker login {registry})."
    if "manifest unknown" in output or "repository does not exist" in output:
        return f"{failed} 레지스트리에 그 이름 · 태그의 이미지가 없습니다."
    return f"{failed} 이미지 이름과 레지스트리 로그인을 확인하세요."


def ensure(home: Path, image: str, log: Callable[[str], None] = print) -> str:
    """Have the image on this node (pulled from its registry the first time). Returns its id."""
    command = docker()
    if not command:
        raise RuntimeError("이 서버에 Docker가 없습니다. Docker 환경으로 학습하려면 Docker를 설치하세요.")
    found = image_id(image)
    pulled = found is None
    if found is None:
        log(f"Docker 이미지를 받습니다 (이 서버에서 처음 한 번): {image}")
        proc = subprocess.Popen(
            [command, "pull", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        last = ""
        said: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            if line and line != last:  # layer progress repeats itself
                log(line)
                last = line
                said = [*said[-20:], line]
        if proc.wait() != 0:
            raise RuntimeError(pull_failure(image, "\n".join(said)))
        found = image_id(image)
        if found is None:
            raise RuntimeError(f"받은 Docker 이미지를 찾을 수 없습니다: {image}")
    _remember(home, image, found, pulled)
    return found


# ---- running ---------------------------------------------------------------------------------


def name_of(job_id: str) -> str:
    return f"badamoa-{job_id}"


def command(
    image: str,
    job_id: str,
    args: list[str],
    *,
    mounts: list[tuple[Path, bool]],
    workdir: Path,
    env: dict[str, str],
    gpus: list[str],
) -> list[str]:
    """The ``docker run`` command for one entry point: ``mounts`` are (node folder, read-only)."""
    run = [docker() or "docker", "run", "--rm", "--name", name_of(job_id), "--ipc=host"]
    if gpus:
        run += ["--gpus", f'"device={",".join(gpus)}"']
    if os.name != "nt" and hasattr(os, "getuid"):
        run += ["--user", f"{os.getuid()}:{os.getgid()}"]  # files it writes belong to the node's user
        run += ["-e", "HOME=/tmp"]  # a user the image does not know still needs somewhere to write
    seen: set[str] = set()
    for folder, read_only in mounts:
        target = inside(folder)
        if target in seen or not Path(folder).exists():
            continue
        seen.add(target)
        run += ["-v", f"{folder}:{target}{':ro' if read_only else ''}"]
    for key, value in env.items():
        run += ["-e", f"{key}={value}"]
    return [*run, "-w", inside(workdir), image, *args]


def kill(job_id: str) -> None:
    """Remove a job's container if it is still there (the job was stopped, or its process died)."""
    command_ = docker()
    if command_:
        subprocess.run([command_, "rm", "-f", name_of(job_id)], capture_output=True, check=False, timeout=60)


def gpu_list() -> list[str]:
    """The GPUs the Hub reserved for this job, as the node numbers them."""
    return [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
