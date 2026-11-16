"""The address other servers use to reach this Hub: in node install commands and node scripts.

The browser's own address is no good when it is ``localhost`` (another server would reach itself).
In order: an address an admin set on the screen, the ``--public-url`` launch option, the address
the browser used when it is a real one, and otherwise this machine's network address.
"""

from __future__ import annotations

import socket
from typing import Any
from urllib.parse import urlsplit

import psutil
from fastapi import Request
from sqlalchemy.orm import Session

from . import services
from .config import HubSettings
from .models import HubSetting, User
from .services import ServiceError

KEY = "hub.address"
EXTRA = "hub_address"  # the saved address, kept in the settings so every request can use it
LOCAL_BIND = {"127.0.0.1", "localhost", "::1"}  # a Hub bound here takes no connections from outside


def is_loopback(host: str | None) -> bool:
    host = (host or "").strip("[]").lower()
    return host in ("", "localhost", "::1", "0.0.0.0") or host.startswith("127.") or host.endswith(".localhost")


def lan_addresses() -> list[str]:
    """This machine's IPv4 addresses on its networks, the one with the default route first."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent: this only picks the outgoing interface
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for addresses in psutil.net_if_addrs().values():
            for a in addresses:
                if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                    found.append(a.address)
    except (OSError, RuntimeError):
        pass
    return [a for a in dict.fromkeys(found) if not is_loopback(a)]


def load(db: Session, settings: HubSettings) -> None:
    row = db.get(HubSetting, KEY)
    if row and (row.value or {}).get("url"):
        settings.extra[EXTRA] = row.value["url"]


def resolve(request: Request, settings: HubSettings) -> tuple[str, str]:
    """(address, where it came from): saved | launch | browser | detected | local-only."""
    if settings.extra.get(EXTRA):
        return settings.extra[EXTRA], "saved"
    if settings.public_url:
        return settings.public_url.rstrip("/"), "launch"
    base = str(request.base_url).rstrip("/")
    if not is_loopback(request.url.hostname):
        return base, "browser"
    if settings.host in LOCAL_BIND:
        return base, "local-only"  # nothing outside this machine can connect anyway
    found = lan_addresses()
    if not found:
        return base, "local-only"
    return f"{request.url.scheme}://{found[0]}:{request.url.port or settings.port}", "detected"


def view(request: Request, settings: HubSettings) -> dict[str, Any]:
    url, source = resolve(request, settings)
    port = request.url.port or settings.port
    scheme = request.url.scheme
    candidates = [f"{scheme}://{ip}:{port}" for ip in lan_addresses()]
    name = socket.gethostname()
    if name and not is_loopback(name):
        candidates.append(f"{scheme}://{name}:{port}")
    return {
        "url": url,
        "source": source,
        "candidates": list(dict.fromkeys([url, *candidates])) if source != "local-only" else candidates,
        "launch": settings.public_url,
        "listens_locally_only": settings.host in LOCAL_BIND,
    }


def normalize(url: str) -> str:
    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.path:
        raise ServiceError(400, "주소는 http://서버이름:8470 처럼 적으세요 (경로 없이).")
    if is_loopback(parts.hostname):
        raise ServiceError(400, "localhost는 다른 서버에서 이 Hub가 아닙니다. 네트워크 주소나 서버 이름을 적으세요.")
    return url


def save(db: Session, user: User, settings: HubSettings, url: str | None) -> None:
    """Keep the address nodes are told to use; None goes back to finding it by itself."""
    services.require_admin(user)
    row = db.get(HubSetting, KEY)
    if url is None:
        if row is not None:
            db.delete(row)
        settings.extra.pop(EXTRA, None)
        services.audit(db, user.id, "ui", "hub.address", "", reset=True)
        return
    url = normalize(url)
    if row is None:
        db.add(HubSetting(key=KEY, value={"url": url}))
    else:
        row.value = {"url": url}
    settings.extra[EXTRA] = url
    services.audit(db, user.id, "ui", "hub.address", "", url=url)
