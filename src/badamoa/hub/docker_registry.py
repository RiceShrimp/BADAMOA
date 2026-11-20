"""The Hub's Docker registry: where the images of model kinds and environments are kept for every server.

Where the Hub runs it sets one up by itself (``ensure``: at start, then every ten minutes):

    adopted   a registry container already on this machine (``registry:2``, e.g. one a team ran before BADAMOA):
              used as it is, never replaced, restarted or removed
    managed   none there: ``badamoa-registry`` (``registry:2``), its images in the Hub's home (``<home>/registry``),
              on port 5000 or the next free one; started again if it stopped
    external  an address an admin gave (a registry elsewhere)
    off       an admin turned it off
    none      not set up: no Docker here, or no right to use it (a Linux Hub that is neither root nor in the docker
              group; Docker Desktop not started yet when the Hub starts at boot). Said why, and tried again

The registry speaks plain HTTP (``registry:2`` as it comes). The servers' Docker are told they may use it
(``node/registries.py``); this machine's own Docker reaches it as ``localhost:<port>``, which Docker always allows on
Linux and in Docker Desktop alike, so images are pushed from here that way.

Images are checked against their registry (``image_status``): a version that passed its trial with an image only
this machine had would fail on every other server with "manifest unknown".
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from badamoa.node.registries import registry_of

from . import address, services
from .config import HubSettings
from .db import iso, utcnow
from .models import HubSetting, Node, User
from .services import ServiceError

log = logging.getLogger("badamoa.hub.docker_registry")

KEY = "docker.registry"
CONTAINER = "badamoa-registry"
IMAGE = "registry:2"
PORTS = range(5000, 5011)
INSIDE_PORT = 5000  # where registry:2 listens in its container
EVERY_S = 600
TIMEOUT_S = 10.0
# Manifests come as OCI indexes (Docker's containerd image store) or Docker's own kinds: ask for all of them,
# else the registry answers 404 for a tag it has.
ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
_last = {"at": 0.0}


# ---- state ------------------------------------------------------------------------------------------


def state(db: Session) -> dict[str, Any]:
    row = db.get(HubSetting, KEY)
    return dict(row.value or {}) if row is not None else {}


def _save(db: Session, value: dict[str, Any]) -> dict[str, Any]:
    value = {**value, "checked_at": iso(utcnow())}
    row = db.get(HubSetting, KEY)
    if row is None:
        db.add(HubSetting(key=KEY, value=value))
    else:
        row.value = value
    return value


def address_of(db: Session) -> str:
    """The registry servers pull from ("" when there is none)."""
    current = state(db)
    return str(current.get("address") or "") if current.get("mode") in ("managed", "adopted", "external") else ""


def auto() -> bool:
    """Setting a registry up by itself; ``BADAMOA_DOCKER_REGISTRY=off`` keeps a Hub from it (tests, a lab Hub)."""
    return os.environ.get("BADAMOA_DOCKER_REGISTRY", "auto").strip().lower() not in ("off", "0", "false", "no")


# ---- Docker on this machine -------------------------------------------------------------------------


def docker() -> str | None:
    return shutil.which("docker")


def _run(args: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False
    )


def _why(done: subprocess.CompletedProcess) -> str:
    """Docker's own complaint in words a person can act on."""
    text = (done.stderr or done.stdout or "").strip()
    low = text.lower()
    if "permission denied" in low:
        return "이 계정은 Docker를 쓸 권한이 없습니다(root로 돌리거나 docker 그룹에 넣으세요)."
    if "cannot connect" in low or "is the docker daemon running" in low or "pipe" in low:
        return "Docker가 켜져 있지 않습니다(Docker Desktop이면 켜지면 다시 해 봅니다)."
    return text[-300:] or f"docker가 {done.returncode}로 끝났습니다."


def containers() -> list[dict[str, Any]]:
    """Every container on this machine (``docker ps -a``): name, image, state, the host ports it publishes."""
    command = docker()
    if not command:
        raise RuntimeError("이 PC에 Docker가 없습니다.")
    done = _run([command, "ps", "-a", "--no-trunc", "--format", "{{json .}}"])
    if done.returncode != 0:
        raise RuntimeError(_why(done))
    found = []
    for line in done.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        found.append(
            {
                "name": str(row.get("Names") or "").split(",")[0],
                "image": str(row.get("Image") or ""),
                "state": str(row.get("State") or "").lower(),
                "ports": published(str(row.get("Ports") or "")),
            }
        )
    return found


def published(ports: str) -> dict[int, int]:
    """``0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp`` → {inside: host}."""
    found: dict[int, int] = {}
    for host, inside in re.findall(r":(\d+)->(\d+)/tcp", ports):
        found.setdefault(int(inside), int(host))
    return found


def is_registry(image: str) -> bool:
    """``registry:2``, ``docker.io/library/registry:2.8``, ``distribution/distribution`` …"""
    name = image.split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    return last.split(":", 1)[0] in ("registry", "distribution")


def data_of(name: str) -> str:
    """Where a registry container keeps its images on this machine ("" when it is a volume or unknown)."""
    command = docker()
    if not command:
        return ""
    done = _run([command, "inspect", name, "--format", "{{json .Mounts}}"])
    try:
        mounts = json.loads(done.stdout.strip() or "[]") if done.returncode == 0 else []
    except ValueError:
        return ""
    for mount in mounts or []:
        if mount.get("Destination") == "/var/lib/registry":
            return str(mount.get("Source") or mount.get("Name") or "")
    return ""


def has_local(image: str) -> bool:
    """This machine's Docker has the image (built or pulled here)."""
    command = docker()
    return bool(command) and _run([command, "image", "inspect", "--format", "{{.Id}}", image]).returncode == 0


def _free_port(ports: range, taken: set[int]) -> int | None:
    for port in ports:
        if port in taken:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
        return port
    return None


# ---- reaching it --------------------------------------------------------------------------------------


def reachable(registry: str, timeout: float = TIMEOUT_S) -> bool:
    """The registry answers (``/v2/``: 200, or 401 when it wants a login)."""
    for scheme in ("http", "https"):
        try:
            resp = httpx.get(f"{scheme}://{registry}/v2/", timeout=timeout, verify=False)
        except httpx.HTTPError:
            continue
        if resp.status_code in (200, 401):
            return True
    return False


def catalog(registry: str) -> int | None:
    """How many repositories the registry holds (None when it does not say)."""
    for scheme in ("http", "https"):
        try:
            resp = httpx.get(f"{scheme}://{registry}/v2/_catalog?n=1000", timeout=TIMEOUT_S, verify=False)
        except httpx.HTTPError:
            continue
        if resp.status_code == 200:
            try:
                return len(resp.json().get("repositories") or [])
            except ValueError:
                return None
    return None


def split(image: str) -> tuple[str, str, str]:
    """``reg:5000/team/x:1.2`` → (``reg:5000``, ``team/x``, ``1.2``); a digest (``@sha256:…``) is the reference."""
    registry = registry_of(image)
    rest = image[len(registry) + 1 :] if registry else image
    if "@" in rest:
        repo, reference = rest.split("@", 1)
    elif ":" in rest.rsplit("/", 1)[-1]:
        repo, reference = rest.rsplit(":", 1)
    else:
        repo, reference = rest, "latest"
    if not registry and "/" not in repo:
        repo = f"library/{repo}"
    return registry, repo, reference


def present(image: str) -> bool | None:
    """Whether the image's registry has it: True, False (no such tag), None (could not tell)."""
    registry, repo, reference = split(image)
    if not registry:
        return None  # Docker Hub: public images, not checked here
    for scheme in ("http", "https"):
        try:
            resp = httpx.head(
                f"{scheme}://{registry}/v2/{repo}/manifests/{reference}",
                headers={"Accept": ACCEPT},
                timeout=TIMEOUT_S,
                verify=False,
            )
        except httpx.HTTPError:
            continue
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        return None  # 401 and the like: it wants a login
    return None


# ---- setting it up ------------------------------------------------------------------------------------


def _host(settings: HubSettings) -> str:
    """The name other servers reach this machine by: the Hub's saved or launch address, else its network address."""
    for url in (settings.extra.get(address.EXTRA), settings.public_url):
        host = urlsplit(str(url or "")).hostname
        if host and not address.is_loopback(host):
            return host
    found = address.lan_addresses()
    return found[0] if found else "localhost"


def setup(db: Session, settings: HubSettings) -> dict[str, Any]:
    """Find, adopt or make the registry; keep what was found (``state``)."""
    current = state(db)
    mode = current.get("mode")
    if mode == "off":
        return current
    if mode == "external":
        ok = reachable(current["address"])
        return _save(db, {**current, "ok": ok, "error": "" if ok else f"{current['address']}에 닿지 않습니다."})
    host = _host(settings)
    if mode in ("managed", "adopted") and current.get("address"):
        if reachable(current["address"]):
            return _save(db, {**current, "ok": True, "error": "", "images": catalog(current["address"])})
        command = docker()
        if mode == "managed" and command:  # ours: started again (its data is kept in the Hub's home)
            _run([command, "start", current.get("container") or CONTAINER])
            for _ in range(10):
                if reachable(current["address"], timeout=2):
                    return _save(db, {**current, "ok": True, "error": "", "images": catalog(current["address"])})
                time.sleep(1)
        why = "MOA가 만든 레지스트리" if mode == "managed" else f"이 PC의 레지스트리({current.get('container')})"
        return _save(db, {**current, "ok": False, "error": f"{why}가 응답하지 않습니다: {current['address']}"})

    try:
        found = containers()
    except RuntimeError as e:
        return _save(db, {"mode": "none", "ok": False, "error": str(e)})
    registries = [c for c in found if is_registry(c["image"])]
    running = [c for c in registries if c["state"] == "running" and INSIDE_PORT in c["ports"]]
    ours = next((c for c in registries if c["name"] == CONTAINER), None)
    if ours is None and running:  # a registry already here (a team's, from before): used as it is
        chosen = sorted(running, key=lambda c: c["ports"][INSIDE_PORT])[0]
        where = f"{host}:{chosen['ports'][INSIDE_PORT]}"
        ok = reachable(where)
        return _save(
            db,
            {
                "mode": "adopted",
                "address": where,
                "container": chosen["name"],
                "data": data_of(chosen["name"]),
                "ok": ok,
                "error": "" if ok else f"{where}에 닿지 않습니다.",
                "images": catalog(where) if ok else None,
            },
        )
    command = docker()
    assert command is not None
    if ours is not None:  # ours, from before (the Hub's record was lost): started again
        _run([command, "start", CONTAINER])
        port = ours["ports"].get(INSIDE_PORT) or 5000
    else:
        taken = {p for c in registries for p in c["ports"].values()}  # a stopped team registry keeps its port
        port = _free_port(PORTS, taken)
        if port is None:
            return _save(db, {"mode": "none", "ok": False, "error": "5000~5010번 포트가 모두 쓰이고 있습니다."})
        data = Path(settings.home).resolve() / "registry"
        data.mkdir(parents=True, exist_ok=True)
        made = _run(
            [
                command,
                "run",
                "-d",
                "--name",
                CONTAINER,
                "--restart",
                "unless-stopped",
                "-p",
                f"{port}:{INSIDE_PORT}",
                "-v",
                f"{data}:/var/lib/registry",
                IMAGE,
            ],
            timeout=600,  # the first time Docker fetches registry:2 (about 10 MB)
        )
        if made.returncode != 0:
            return _save(db, {"mode": "none", "ok": False, "error": f"레지스트리를 만들지 못했습니다: {_why(made)}"})
        log.info("Docker 레지스트리를 만들었습니다: %s:%d (%s)", host, port, data)
    where = f"{host}:{port}"
    ok = False
    for _ in range(15):
        if reachable(f"localhost:{port}", timeout=2):
            ok = True
            break
        time.sleep(1)
    lan = ok and reachable(where)
    error = (
        ""
        if lan
        else (
            f"이 PC에서는 되지만 {where}로는 닿지 않습니다: 방화벽이 {port}/tcp를 막으면 다른 서버가 받지 못합니다."
            if ok
            else "레지스트리가 응답하지 않습니다."
        )
    )
    return _save(
        db,
        {
            "mode": "managed",
            "address": where,
            "container": CONTAINER,
            "data": data_of(CONTAINER) or str(Path(settings.home).resolve() / "registry"),
            "ok": lan,
            "error": error,
            "images": catalog(where) if lan else None,
        },
    )


def ensure(database: Any, settings: HubSettings, force: bool = False) -> dict[str, Any] | None:
    """The Hub's background loop: set up or check the registry now and then (``force``: now)."""
    if not force and (not auto() or time.monotonic() - _last["at"] < EVERY_S):
        return None
    _last["at"] = time.monotonic()
    with database.session() as db:
        return setup(db, settings)


# ---- what admins do ------------------------------------------------------------------------------------


def configure(db: Session, user: User, settings: HubSettings, address_: str | None, off: bool) -> dict[str, Any]:
    """Another registry's address (external), off, or back to finding one by itself (neither)."""
    services.require_admin(user)
    if off:
        value = _save(db, {"mode": "off", "ok": False, "error": ""})
    elif address_:
        where = re.sub(r"^https?://", "", address_.strip()).rstrip("/")
        if not where or "/" in where:
            raise ServiceError(400, "주소는 192.168.0.10:5000 처럼 적으세요.")
        ok = reachable(where)
        value = _save(
            db, {"mode": "external", "address": where, "ok": ok, "error": "" if ok else f"{where}에 닿지 않습니다."}
        )
    else:
        db.query(HubSetting).filter(HubSetting.key == KEY).delete()
        db.flush()
        value = setup(db, settings)
    services.audit(db, user.id, "ui", "docker.registry", "", mode=value.get("mode"), address=value.get("address"))
    return value


def view(db: Session) -> dict[str, Any]:
    current = state(db)
    return {
        "mode": current.get("mode") or "none",
        "address": current.get("address") or "",
        "container": current.get("container") or "",
        "data": current.get("data") or "",
        "ok": bool(current.get("ok")),
        "error": current.get("error") or ("아직 확인하지 않았습니다." if not current else ""),
        "images": current.get("images"),
        "checked_at": current.get("checked_at"),
        "auto": auto(),
        "docker": bool(docker()),
    }


# ---- images: in the registry? -------------------------------------------------------------------------

_seen: dict[str, tuple[float, bool | None]] = {}
SEEN_S = 60.0


def _present(image: str) -> bool | None:
    hit = _seen.get(image)
    if hit and time.monotonic() - hit[0] < SEEN_S:
        return hit[1]
    found = present(image)
    _seen[image] = (time.monotonic(), found)
    return found


def used_images(db: Session) -> dict[str, list[str]]:
    """Every Docker image a model kind (its newest version) or an environment names, with who names it."""
    from .train_models import Environment, Template

    names: dict[str, list[str]] = {}
    for t in db.scalars(select(Template).where(Template.deleted_at.is_(None))):
        image = str(((t.manifest or {}).get("environment") or {}).get("image") or "")
        if image:
            names.setdefault(image, []).append(f"{t.title or t.name} {t.version}")
    for env in db.scalars(select(Environment)):
        if env.image:
            names.setdefault(env.image, []).append(f"환경 {env.name}")
    return names


def images(db: Session, settings: HubSettings) -> list[dict[str, Any]]:
    """The images in use and whether their registry has them; one this machine has can be pushed from here."""
    ours = address_of(db)
    rows = []
    for image, used_by in sorted(used_images(db).items()):
        registry = registry_of(image)
        found = _present(image) if registry else None
        local = has_local(image) if found is False else False
        rows.append(
            {
                "image": image,
                "used_by": used_by,
                "registry": registry,
                "present": found,
                "hub_has": local,  # this machine has it: it can be pushed from here
                "ours": bool(ours) and registry == ours,
                "push": f"docker push {image}",
            }
        )
    return rows


def _local_name(db: Session, settings: HubSettings, image: str) -> str:
    """How this machine's Docker pushes the image: through ``localhost:<port>`` when it is this machine's registry."""
    registry, repo, reference = split(image)
    current = state(db)
    if current.get("mode") in ("managed", "adopted") and registry == current.get("address"):
        port = registry.rsplit(":", 1)[-1]
        joiner = "@" if reference.startswith("sha256:") else ":"
        return f"localhost:{port}/{repo}{joiner}{reference}"
    return image


def push(db: Session, user: User, settings: HubSettings, image: str) -> Any:
    """Push an image this machine has to its registry: a job on the Hub's own node, its log on the jobs screen."""
    services.require_admin(user)
    if image not in used_images(db):
        raise ServiceError(400, "모델 종류나 환경이 쓰는 이미지만 올립니다.")
    node = db.scalars(select(Node).where(Node.is_local.is_(True))).first()
    if node is None or not node.enabled:
        raise ServiceError(400, "Hub의 노드가 없어 올릴 수 없습니다.")
    params = {"image": image, "via": _local_name(db, settings, image)}
    _seen.pop(image, None)
    job = services.create_job(db, user, "env.push", params, node.id, via="ui", gpus=0)
    services.audit(db, user.id, "ui", "docker.push", job.id, image=image)
    return job
