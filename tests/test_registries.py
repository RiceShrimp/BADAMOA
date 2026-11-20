"""In-house Docker registries without HTTPS: found from the Hub's images, let in on a node's Docker."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from badamoa.node import containers, registries


def test_the_registry_of_an_image():
    assert registries.registry_of("192.168.106.2:5000/wrapper/yolo:v4") == "192.168.106.2:5000"
    assert registries.registry_of("registry.local/team/trainer:1.2") == "registry.local"
    assert registries.registry_of("localhost/x") == "localhost"
    assert registries.registry_of("ultralytics/ultralytics:latest") == ""  # Docker Hub
    assert registries.registry_of("python:3.10-slim") == ""


def test_a_refused_pull_is_told_apart():
    assert registries.refused('Get "https://r:5000/v2/": http: server gave HTTP response to HTTPS client')
    assert registries.refused("x509: certificate signed by unknown authority")
    assert not registries.refused("manifest unknown")


def test_trust_adds_to_daemon_json_and_keeps_the_rest(tmp_path):
    daemon = tmp_path / "docker" / "daemon.json"
    said: list[str] = []
    # no file yet: made with only the registry
    assert registries.trust(["192.168.106.2:5000"], said.append, daemon=daemon, reload=False) == ["192.168.106.2:5000"]
    assert json.loads(daemon.read_text()) == {"insecure-registries": ["192.168.106.2:5000"]}
    assert "insecure-registries" in said[0]

    daemon.write_text(json.dumps({"default-runtime": "nvidia", "insecure-registries": ["a:5000"]}))
    added = registries.trust(["a:5000", "b:5000", "b:5000"], said.append, daemon=daemon, reload=False)
    assert added == ["b:5000"]
    assert json.loads(daemon.read_text()) == {"default-runtime": "nvidia", "insecure-registries": ["a:5000", "b:5000"]}
    backup = daemon.with_name("daemon.json.badamoa-backup")
    assert json.loads(backup.read_text())["insecure-registries"] == ["a:5000"]  # the state before BADAMOA touched it
    assert registries.trust(["b:5000"], said.append, daemon=daemon, reload=False) == []  # already there
    assert registries.listed(daemon) == ["a:5000", "b:5000"]

    daemon.write_text("{not json")
    with pytest.raises(RuntimeError, match="JSON"):
        registries.trust(["c:5000"], said.append, daemon=daemon, reload=False)
    assert daemon.read_text() == "{not json"  # left as it was


def test_a_plain_http_registry_is_found(tmp_path):
    class V2(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), V2)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert registries.plain(f"127.0.0.1:{server.server_port}", timeout=3)
    finally:
        server.shutdown()
    assert not registries.plain("127.0.0.1:9", timeout=1)  # nothing answers: nothing to say


def test_a_refused_pull_names_the_one_line_fix():
    image = "192.168.106.2:5000/wrapper/mlops_yolo_curriculum:v4-cu128"
    said = containers.pull_failure(image, "server gave HTTP response to HTTPS client", "http://192.168.106.2:8470")
    assert "curl -fsSL http://192.168.106.2:8470/install/docker-registries.sh | sudo bash" in said


def test_the_hub_lists_its_registries_and_serves_the_script(client, admin):
    image = "192.168.0.10:5000/team/trainer:1.2"
    made = client.post("/api/environments", json={"name": "사내", "kind": "docker", "image": image}, headers=admin)
    assert made.status_code == 200, made.text
    client.post("/api/environments", json={"name": "hub", "kind": "docker", "image": "python:3.10-slim"}, headers=admin)
    assert client.get("/install/registries").json() == {"registries": ["192.168.0.10:5000"]}
    script = client.get("/install/docker-registries.sh").text
    assert "{{HUB_URL}}" not in script and "node trust-registries --hub" in script and "\r" not in script
