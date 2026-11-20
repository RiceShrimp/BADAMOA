"""In-house Docker registries without HTTPS, which each server's Docker has to be told it may pull from.

Docker pulls only over HTTPS with a certificate it trusts, unless the registry is listed in ``insecure-registries`` of
``/etc/docker/daemon.json``. An in-house registry (``docker run registry:2`` at ``192.168.0.10:5000``) usually has
neither, so every server that pulls from it needs that line. A node installed as root adds it itself: at install and
update, for the registries the Hub's model kinds and environments use (``badamoa node trust-registries``), and when
a pull is refused for a registry added later. A node without root cannot write that file: the failed pull names the
one line to run with sudo (``<hub>/install/docker-registries.sh``).

Docker reads ``insecure-registries`` again on SIGHUP (``systemctl reload docker``): running containers go on.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx

DAEMON_JSON = Path("/etc/docker/daemon.json")


def registry_of(image: str) -> str:
    """The registry an image comes from ("" for Docker Hub): ``192.168.0.10:5000/team/x:1`` → ``192.168.0.10:5000``."""
    head, sep, _ = image.strip().partition("/")
    if not sep:
        return ""
    return head if ("." in head or ":" in head or head == "localhost") else ""


def refused(output: str) -> bool:
    """A pull refused because the registry speaks plain HTTP or has a certificate Docker does not trust."""
    return any(
        sign in output for sign in ("HTTP response to HTTPS client", "x509:", "certificate signed by unknown authority")
    )


def plain(registry: str, timeout: float = 5.0) -> bool:
    """The registry answers, but not over HTTPS with a certificate this server trusts: Docker needs it listed."""
    try:
        httpx.get(f"https://{registry}/v2/", timeout=timeout)
        return False  # HTTPS that this server trusts: Docker pulls as it is
    except httpx.HTTPError:
        pass
    for url, verify in ((f"http://{registry}/v2/", True), (f"https://{registry}/v2/", False)):
        try:
            httpx.get(url, timeout=timeout, verify=verify)
            return True
        except httpx.HTTPError:
            continue
    return False  # not reachable from here: nothing to say about it


def listed(daemon: Path = DAEMON_JSON) -> list[str]:
    """The registries this server's Docker already takes without HTTPS."""
    try:
        config = json.loads(daemon.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(config.get("insecure-registries") or []) if isinstance(config, dict) else []


def can_write() -> bool:
    """Only root on Linux can change the Docker daemon's settings (Docker Desktop keeps its own)."""
    return sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0


def trust(
    registries: Iterable[str], log: Callable[[str], None] = print, daemon: Path = DAEMON_JSON, reload: bool = True
) -> list[str]:
    """Add ``registries`` to Docker's ``insecure-registries`` and have Docker read it again. Returns the ones added.

    The rest of daemon.json stays as it was; its first state is kept beside it (``daemon.json.badamoa-backup``)."""
    try:
        config = json.loads(daemon.read_text(encoding="utf-8")) if daemon.is_file() else {}
    except ValueError as e:
        raise RuntimeError(f"{daemon}이(가) JSON이 아니라 고치지 않았습니다: {e}") from e
    if not isinstance(config, dict):
        raise RuntimeError(f"{daemon}이(가) JSON 객체가 아니라 고치지 않았습니다.")
    have = list(config.get("insecure-registries") or [])
    added = [r for r in dict.fromkeys(registries) if r and r not in have]
    if not added:
        return []
    if daemon.is_file():
        backup = daemon.with_name(daemon.name + ".badamoa-backup")
        if not backup.exists():
            shutil.copy2(daemon, backup)
    config["insecure-registries"] = have + added
    daemon.parent.mkdir(parents=True, exist_ok=True)
    written = daemon.with_name(daemon.name + ".badamoa-new")
    written.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(written, daemon)
    log(
        f"이 서버의 Docker가 HTTPS 없는 레지스트리 {', '.join(added)}에서 받도록 "
        f"{daemon}의 insecure-registries에 넣었습니다."
    )
    if reload:
        _reload(log)
    return added


def _reload(log: Callable[[str], None]) -> None:
    """Docker reads daemon.json again without stopping its containers."""
    if shutil.which("systemctl"):
        done = subprocess.run(["systemctl", "reload", "docker"], capture_output=True, text=True, check=False)
        if done.returncode == 0:
            return
    found = (
        subprocess.run(["pidof", "dockerd"], capture_output=True, text=True, check=False) if shutil.which("pidof") else None
    )
    pids = found.stdout.split() if found and found.returncode == 0 else []
    if pids:
        subprocess.run(["kill", "-HUP", *pids], check=False)
        return
    log("Docker가 설정을 다시 읽게 하지 못했습니다: Docker를 다시 시작하면 적용됩니다.")


def command(hub_url: str) -> str:
    """The line an admin runs once on a server whose node has no root."""
    return f"curl -fsSL {hub_url.rstrip('/')}/install/docker-registries.sh | sudo bash"
