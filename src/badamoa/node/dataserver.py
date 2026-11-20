"""The node's own little server: other nodes take registered images from here, never through the Hub.

    GET  /badamoa/ping   → {"node_id": …}   (can the other node reach this one)
    POST /badamoa/files  {"grant": …, "ids": [...]}  → a tar stream, one member per image,
                         named by its item id, then ``errors.json`` (files gone or changed)

Nothing is served without a grant: for each request the node asks the Hub which of the asked-for
images the grant covers and where they lie (``/api/agent/transfers/resolve``), and the Hub says
yes only to the node that holds them, for the node the grant was made for, while that job runs.
Files are sent as they are on disk (images are compressed already), each checked against its
registered size first; the receiving node checks the content against the registered SHA-256.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import psutil

log = logging.getLogger("badamoa.node.data")

DEFAULT_PORT = 8475
MAX_IDS = 5000
MAX_STREAMS = 8  # requests served at once; more get "busy" and retry


def local_addresses() -> list[str]:
    """This machine's IPv4 addresses other nodes might reach (not loopback, not link-local)."""
    found: list[str] = []
    try:
        interfaces = psutil.net_if_addrs()
    except (OSError, RuntimeError):
        return found
    for addrs in interfaces.values():
        for addr in addrs:
            if addr.family != socket.AF_INET:
                continue
            ip = addr.address
            if ip.startswith(("127.", "169.254.")) or ip in found:
                continue
            found.append(ip)
    return found


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.0"  # the stream ends when the connection closes

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - quiet unless debugging
        log.debug("%s %s", self.address_string(), fmt % args)

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/badamoa/ping":
            self._json(200, {"node_id": self.server.node_id})
        else:
            self._json(404, {"detail": "없는 경로입니다."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/badamoa/files":
            self._json(404, {"detail": "없는 경로입니다."})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(length, 1024 * 1024)) or b"{}")
            grant, ids = str(body.get("grant") or ""), [int(i) for i in (body.get("ids") or [])][:MAX_IDS]
        except (ValueError, TypeError):
            self._json(400, {"detail": "요청 형식이 맞지 않습니다."})
            return
        if not self.server.slots.acquire(blocking=False):
            self._json(503, {"detail": "보내는 중인 요청이 많습니다. 잠시 뒤 다시 시도하세요."})
            return
        try:
            self._send(grant, ids)
        finally:
            self.server.slots.release()

    def _send(self, grant: str, ids: list[int]) -> None:
        try:
            resp = self.server.hub.post("/api/agent/transfers/resolve", json={"grant": grant, "ids": ids}, timeout=60)
        except httpx.HTTPError as e:
            self._json(502, {"detail": f"Hub에 확인하지 못했습니다: {e}"})
            return
        if resp.status_code != 200:
            is_json = resp.headers.get("content-type", "").startswith("application/json")
            detail = resp.json().get("detail") if is_json else ""
            status = 403 if resp.status_code in (401, 403, 404) else 502
            self._json(status, {"detail": detail or "허가되지 않은 전송입니다."})
            return
        files = resp.json().get("files") or []
        allowed = {int(f["id"]) for f in files}
        errors = [{"id": i, "reason": "이 노드에 등록된 이 버전의 이미지가 아닙니다."} for i in ids if i not in allowed]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-tar")
        self.end_headers()
        sent = 0
        try:
            with tarfile.open(fileobj=self.wfile, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for f in files:
                    root = Path(f["root"]).expanduser()
                    path = root / f["rel_path"]
                    try:
                        inside = path.resolve().is_relative_to(root.resolve())
                        size = path.stat().st_size if inside and path.is_file() else -1
                    except OSError:
                        size = -1
                    if size < 0:
                        errors.append({"id": f["id"], "reason": "파일이 없습니다(옮겨졌거나 지워짐)."})
                        continue
                    if size != int(f["bytes"]):
                        errors.append({"id": f["id"], "reason": "등록한 뒤 파일이 바뀌었습니다(크기가 다름)."})
                        continue
                    info = tarfile.TarInfo(str(f["id"]))
                    info.size = size
                    try:
                        with path.open("rb") as fh:
                            tar.addfile(info, fh)
                        sent += 1
                    except OSError as e:  # changed while being read: the stream is broken, the receiver retries
                        log.warning("보내다 실패 %s: %s", path, e)
                        raise
                report = json.dumps({"errors": errors}, ensure_ascii=False).encode("utf-8")
                info = tarfile.TarInfo("errors.json")
                info.size = len(report)
                tar.addfile(info, BytesIO(report))
        except (OSError, ConnectionError) as e:
            log.info("전송이 끊겼습니다 (%d장 보냄): %s", sent, e)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, hub: httpx.Client, node_id: str) -> None:
        # Other nodes on the network connect here (BADAMOA_NODE_DATA_HOST narrows it, e.g. for tests).
        host = os.environ.get("BADAMOA_NODE_DATA_HOST", "0.0.0.0")  # noqa: S104
        super().__init__((host, port), _Handler)
        self.hub = hub
        self.node_id = node_id
        self.slots = threading.BoundedSemaphore(MAX_STREAMS)


class DataServer:
    def __init__(self, hub_url: str, node_token: str, node_id: str, port: int = DEFAULT_PORT) -> None:
        self.port = port
        self._hub = httpx.Client(base_url=hub_url, headers={"Authorization": f"Bearer {node_token}"}, timeout=60)
        self._token = node_token
        self._node_id = node_id
        self._server: _Server | None = None
        self.error = ""

    def follow(self, hub_url: str) -> None:
        """The Hub moved: ask the new one who may take images from here."""
        self._hub = httpx.Client(base_url=hub_url, headers={"Authorization": f"Bearer {self._token}"}, timeout=60)
        if self._server is not None:
            self._server.hub = self._hub

    def start(self) -> int:
        """Listen; returns the port (0 when it could not, with the reason in ``error``)."""
        try:
            self._server = _Server(self.port, self._hub, self._node_id)
        except OSError as e:
            self.error = f"포트 {self.port}를 열지 못했습니다: {e}"
            log.warning("이미지 전송용 %s. 다른 노드가 이 노드의 이미지를 받을 수 없습니다.", self.error)
            return 0
        threading.Thread(target=self._server.serve_forever, name="data-server", daemon=True).start()
        self.port = self._server.server_address[1]
        log.info("이미지 전송용 포트 %d", self.port)
        return self.port

    def info(self) -> dict[str, Any]:
        """What the heartbeat tells the Hub: where other nodes can reach this one."""
        if self._server is None:
            return {"port": 0, "error": self.error}
        return {"port": self.port, "addresses": local_addresses()}

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._hub.close()
