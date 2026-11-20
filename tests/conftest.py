from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn

from badamoa.hub.app import create_app
from badamoa.hub.config import HubSettings

ADMIN = {"username": "admin", "password": "badamoa-test-pw"}


@pytest.fixture(autouse=True)
def _quiet_nodes(monkeypatch):
    """Node agents in tests: their image port on loopback, any free port (no firewall prompt, no clashes),
    and images received kept in the node's own folder, never on a real disk the machine has."""
    from badamoa.node import cache

    monkeypatch.setenv("BADAMOA_NODE_DATA_PORT", "0")
    monkeypatch.setenv("BADAMOA_NODE_DATA_HOST", "127.0.0.1")
    monkeypatch.setenv("BADAMOA_DOCKER_REGISTRY", "off")  # a test Hub never makes or adopts a registry container
    monkeypatch.setattr(cache, "choose", lambda home: home / "cache")


@pytest.fixture
def settings(tmp_path) -> HubSettings:
    return HubSettings(home=tmp_path / "hub", web_dir=tmp_path / "no-web")


@pytest.fixture
def client(settings) -> Iterator:
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def admin(client) -> dict[str, str]:
    """Authorization headers for a freshly created admin."""
    token = client.post("/api/setup", json=ADMIN).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_hub(settings) -> Iterator[str]:
    """A real Hub on a local port (for the Node Agent and MCP-over-HTTP tests)."""
    settings.port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=settings.port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{settings.port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/api/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        raise RuntimeError("hub did not start")
    yield url
    server.should_exit = True
    # A node's long-poll keeps the server open for up to 20 s; the thread is a daemon on its own port, so move on.
    thread.join(timeout=1)
