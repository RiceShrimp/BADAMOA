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
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import registries

PULLED = "docker-images.json"  # the images this node pulled or found for BADAMOA (``listing``)
PROBE_TIMEOUT_S = 20
GB = 1024**3
UNPACKED = 2  # an image unpacked is about twice its layers as the registry keeps them (compressed)
ROOM_TIMES = 3  # a pull needs at least three times the image's size free on Docker's disk
MANIFESTS = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


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


def storage_root() -> dict[str, Any] | None:
    """Where Docker keeps its images: its data folder, and whether that is inside Docker Desktop's virtual disk
    (a folder of the Linux VM, not one of this machine's)."""
    command = docker()
    if not command:
        return None
    try:
        done = subprocess.run(
            [command, "info", "--format", "{{.DockerRootDir}}|{{.OperatingSystem}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    root, _, system = done.stdout.strip().partition("|")
    if done.returncode != 0 or not root:
        return None
    return {"path": root, "desktop": "docker desktop" in system.lower()}


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
        log(f"이 서버에서 MOA가 쓰던 이미지가 아닙니다: {image}")
        return {"removed": False}
    if not entry.get("pulled"):
        log(f"MOA가 받기 전부터 있던 이미지라 지우지 않고 목록에서만 뺐습니다: {image}")
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


def pull_failure(image: str, output: str, hub_url: str = "") -> str:
    """Why ``docker pull`` failed, in words a person can act on (``hub_url``: where the one-line fix is served)."""
    registry = registries.registry_of(image)
    failed = f"Docker 이미지를 받지 못했습니다: {image}."
    if registries.refused(output) and registry:
        # A registry without TLS (``docker run registry``): each Docker that pulls from it has to be told it may.
        fix = (
            f"이 서버에서 한 번 실행하세요: {registries.command(hub_url)} "
            if hub_url
            else f'이 서버의 /etc/docker/daemon.json에 "insecure-registries": ["{registry}"]를 넣고 '
            "sudo systemctl reload docker를 하세요 "
        )
        return (
            f"{failed} 레지스트리 {registry}는 HTTPS 없이(또는 믿을 수 없는 인증서로) 열려 있는데, "
            "이 서버의 Docker는 그런 레지스트리를 허용하지 않고, 노드가 root가 아니라 직접 허용하지 못했습니다. "
            + fix
            + "(Docker의 insecure-registries에 넣고 설정을 다시 읽게 함, 돌고 있는 컨테이너는 그대로). "
            f'Windows · Mac의 Docker Desktop은 Settings → Docker Engine에 "insecure-registries": ["{registry}"]를 넣습니다.'
        )
    if "unauthorized" in output or "authentication required" in output or "pull access denied" in output:
        return f"{failed} 레지스트리에 로그인이 필요합니다(이 서버에서 docker login {registry})."
    if "manifest unknown" in output or "repository does not exist" in output:
        return f"{failed} 레지스트리에 그 이름 · 태그의 이미지가 없습니다."
    return f"{failed} 이미지 이름과 레지스트리 로그인을 확인하세요."


def _arch() -> str:
    machine = platform.machine().lower()
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)


def _registry_layers(image: str) -> int | None:
    """The layers' size read from the image's registry over HTTP(S), without a login (a registry of the team)."""
    import httpx

    registry = registries.registry_of(image)
    if not registry:
        return None  # Docker Hub wants a token: ``docker manifest inspect`` answers for it
    rest = image[len(registry) + 1 :]
    if "@" in rest:
        repo, reference = rest.split("@", 1)
    elif ":" in rest.rsplit("/", 1)[-1]:
        repo, reference = rest.rsplit(":", 1)
    else:
        repo, reference = rest, "latest"
    for scheme in ("http", "https"):
        base = f"{scheme}://{registry}/v2/{repo}/manifests/"
        try:
            found = httpx.get(base + reference, headers={"Accept": MANIFESTS}, timeout=15, verify=False)
            if found.status_code != 200:
                continue
            body = found.json()
            if body.get("manifests"):  # an index: this platform's manifest
                mine = [
                    m
                    for m in body["manifests"]
                    if (m.get("platform") or {}).get("os") == "linux"
                    and (m.get("platform") or {}).get("architecture") == _arch()
                ]
                if not mine:
                    return None
                body = httpx.get(base + mine[0]["digest"], headers={"Accept": MANIFESTS}, timeout=15, verify=False).json()
            layers = body.get("layers") or []
            return sum(int(layer.get("size") or 0) for layer in layers) or None
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return None


def pull_size(command: str, image: str) -> int | None:
    """The image's layers as its registry keeps them (compressed) for this machine; None when it cannot be told."""
    try:
        done = subprocess.run(
            [command, "manifest", "inspect", "-v", "--insecure", image],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        data = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        data = None
    arch = _arch()
    found = None
    for entry in data if isinstance(data, list) else [data] if isinstance(data, dict) else []:
        manifest = entry.get("SchemaV2Manifest") or entry.get("OCIManifest") or {}
        layers = manifest.get("layers") or []
        where = (entry.get("Descriptor") or {}).get("platform") or {}
        if not layers or where.get("os", "linux") != "linux" or where.get("architecture", arch) != arch:
            continue  # another platform, or an attestation
        found = max(found or 0, sum(int(layer.get("size") or 0) for layer in layers))
    return found or _registry_layers(image)


def low_line(total: int) -> int:
    """Below this much free a disk is short of room, as the Hub's disk warnings say (10 GB, or 5 % up to 50 GB)."""
    return int(max(10 * GB, min(total * 0.05, 50 * GB)))


def _gb(value: float) -> str:
    return f"{value / GB:.1f} GB"


def make_room(home: Path, command: str, image: str, log: Callable[[str], None]) -> None:
    """Before pulling: Docker puts the image in its own folder (``data-root``), wherever that is, and a pull goes
    ahead only with three times the image's size free there (the image is about twice its compressed layers in its
    registry). Short of that, Docker's folder moves first to a disk with room, if this node may (Linux, root,
    nothing else running: ``docker_root``); else the pull does not happen and the job says how to make room. An
    image whose size cannot be told is pulled unless the disk is already short of room (the Hub's disk warning)."""
    where = storage_root()
    path = str((where or {}).get("path") or "")
    if not path or (where or {}).get("desktop"):
        return  # Docker Desktop's images are in its own virtual disk, not one of this machine's
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return
    layers = pull_size(command, image)
    size = layers * UNPACKED if layers else 0
    need = size * ROOM_TIMES if size else low_line(usage.total)
    if usage.free >= need:
        return
    if size:
        why = f"이미지(약 {_gb(size)})의 {ROOM_TIMES}배인 {_gb(need)}가 남아 있어야 받는데 {_gb(usage.free)} 남았습니다"
    else:
        why = f"이미지 크기를 알 수 없고 그 디스크에 {_gb(usage.free)}밖에 남지 않았습니다"
    log(f"Docker는 이미지를 {path}에 둡니다. {why}.")
    if _move_docker(home, path, need, log):
        return
    raise RuntimeError(
        f"Docker 이미지를 받지 않았습니다: {path}의 디스크에 {why}. 노드 화면 → 저장 공간의 Docker 이미지 "
        "'다른 디스크로 옮기기'로 여유 있는 디스크로 옮기거나 쓰지 않는 Docker 이미지를 지운 뒤 다시 실행하세요."
    )


def _move_docker(home: Path, current: str, need: int, log: Callable[[str], None]) -> bool:
    """Docker's folder moved to the disk with the most room (enough for it and then ``need`` more), when this node
    may and nothing else runs."""
    from . import docker_root

    job = os.environ.get("BADAMOA_JOB_ID", "")
    if not sys.platform.startswith("linux") or not hasattr(os, "geteuid") or os.geteuid() != 0 or not job:
        log("노드가 root가 아니라(sudo 없이 설치) Docker 이미지 위치를 자동으로 옮기지 못합니다.")
        return False
    targets = docker_root.choices(current)
    if not targets:
        log("Docker 이미지를 옮길 만한 다른 디스크가 없습니다.")
        return False
    log(f"이 서버에서 돌고 있는 것이 없으면 Docker 이미지 위치를 {targets[0]}(으)로 옮긴 뒤 받습니다.")
    with docker_root.alone(home, job) as others:
        if others:
            log(f"이 서버에서 다른 MOA 작업 {len(others)}개가 돌고 있어 옮기지 않았습니다.")
            return False
        try:
            docker_root.move(targets[0], extra=need, log=log)
        except RuntimeError as e:
            log(f"Docker 이미지 위치를 옮기지 못했습니다: {e}")
            return False
    return True


def _hub_url(home: Path) -> str:
    try:
        return str(json.loads((home / "node.json").read_text(encoding="utf-8")).get("hub_url") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def _pull(command: str, image: str, log: Callable[[str], None]) -> tuple[bool, str]:
    """``docker pull``, its progress logged; whether it worked and its last lines."""
    return stream([command, "pull", image], log)


def stream(args: list[str], log: Callable[[str], None]) -> tuple[bool, str]:
    """A docker command whose progress goes to the log; whether it worked and its last lines."""
    proc = subprocess.Popen(
        args,
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
    return proc.wait() == 0, "\n".join(said)


def ensure(home: Path, image: str, log: Callable[[str], None] = print) -> str:
    """Have the image on this node (pulled from its registry the first time). Returns its id.

    A registry without HTTPS refuses the pull until Docker is told it may use it: a node running as root (installed
    with sudo) does that itself and pulls again; one without root says the line to run with sudo."""
    command = docker()
    if not command:
        raise RuntimeError("이 서버에 Docker가 없습니다. Docker 환경으로 학습하려면 Docker를 설치하세요.")
    found = image_id(image)
    pulled = found is None
    if found is None:
        make_room(home, command, image, log)
        log(f"Docker 이미지를 받습니다 (이 서버에서 처음 한 번): {image}")
        ok, said = _pull(command, image, log)
        registry = registries.registry_of(image)
        nearby = registries.local_name(image) if not ok and registries.refused(said) else ""
        if nearby:  # the registry runs on this machine: through localhost, which Docker always allows
            log(f"이 PC의 레지스트리라 {nearby}로 받습니다.")
            ok, said = _pull(command, nearby, log)
            if ok:
                subprocess.run([command, "tag", nearby, image], capture_output=True, check=False)
                subprocess.run([command, "rmi", nearby], capture_output=True, check=False)  # only that name
        if not ok and registry and registries.refused(said) and registries.can_write():
            registries.trust([registry], log)
            time.sleep(2)  # Docker reads its settings again
            log(f"Docker 이미지를 다시 받습니다: {image}")
            ok, said = _pull(command, image, log)
        if not ok:
            raise RuntimeError(pull_failure(image, said, _hub_url(home)))
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
