"""The Hub's Docker registry: adopted when one is there, made when none is, images checked against it."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from badamoa.hub import docker_registry as d
from badamoa.hub.config import HubSettings
from badamoa.hub.db import Database
from badamoa.node import registries


def test_reading_docker_and_image_names():
    assert d.published("0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp") == {5000: 5000}
    assert d.published("127.0.0.1:5099->5000/tcp") == {5000: 5099}
    assert d.published("") == {}
    assert d.is_registry("registry:2") and d.is_registry("docker.io/library/registry:2.8")
    assert d.is_registry("distribution/distribution:3") and not d.is_registry("nginx:1")
    assert d.split("192.168.106.2:5000/wrapper/yolo:v4-cu128") == ("192.168.106.2:5000", "wrapper/yolo", "v4-cu128")
    assert d.split("reg.local/team/x") == ("reg.local", "team/x", "latest")
    assert d.split("python:3.10-slim") == ("", "library/python", "3.10-slim")
    assert d.split("reg:5000/a@sha256:ab") == ("reg:5000", "a", "sha256:ab")


@pytest.fixture
def hub_db(tmp_path):
    settings = HubSettings(home=tmp_path / "hub", public_url="http://10.0.0.5:8470")
    database = Database(f"sqlite:///{(tmp_path / 'hub.db').as_posix()}")
    database.create_all()
    return settings, database


def test_a_registry_already_here_is_adopted_as_it_is(hub_db, monkeypatch):
    settings, database = hub_db
    ran: list[list[str]] = []
    monkeypatch.setattr(d, "docker", lambda: "docker")
    monkeypatch.setattr(d, "_run", lambda args, timeout=60: ran.append(args))
    monkeypatch.setattr(
        d,
        "containers",
        lambda: [
            {"name": "web", "image": "nginx:1", "state": "running", "ports": {80: 8080}},
            {"name": "team-registry", "image": "registry:2", "state": "running", "ports": {5000: 5000}},
        ],
    )
    monkeypatch.setattr(d, "reachable", lambda registry, timeout=10: True)
    monkeypatch.setattr(d, "data_of", lambda name: "/srv/registry")
    monkeypatch.setattr(d, "catalog", lambda registry: 4)
    with database.session() as db:
        found = d.setup(db, settings)
    assert (found["mode"], found["address"], found["container"], found["data"], found["images"]) == (
        "adopted",
        "10.0.0.5:5000",  # the address other servers reach this machine by
        "team-registry",
        "/srv/registry",
        4,
    )
    assert ran == []  # nothing made, started or changed
    with database.session() as db:
        assert d.address_of(db) == "10.0.0.5:5000"


@pytest.mark.parametrize("home", ["linux", "windows"])
def test_none_there_one_is_made_in_the_hubs_home(hub_db, monkeypatch, tmp_path, home):
    settings, database = hub_db
    ran: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = stderr = ""

    def run(args, timeout=60):
        ran.append(args)
        return Done()

    monkeypatch.setattr(d, "docker", lambda: "/usr/bin/docker" if home == "linux" else r"C:\Program Files\Docker\docker.exe")
    monkeypatch.setattr(d, "_run", run)
    monkeypatch.setattr(
        d, "containers", lambda: [{"name": "old", "image": "registry:2", "state": "exited", "ports": {5000: 5000}}]
    )
    monkeypatch.setattr(d, "_free_port", lambda ports, taken: next(p for p in ports if p not in taken))
    monkeypatch.setattr(d, "reachable", lambda registry, timeout=10: True)
    monkeypatch.setattr(d, "catalog", lambda registry: 0)
    monkeypatch.setattr(d, "data_of", lambda name: "")
    with database.session() as db:
        made = d.setup(db, settings)
    assert made["mode"] == "managed" and made["ok"] and made["container"] == d.CONTAINER
    assert made["address"] == "10.0.0.5:5001"  # a stopped team registry keeps its port
    [argv] = ran
    data = Path(settings.home).resolve() / "registry"
    assert argv[1:] == [
        "run",
        "-d",
        "--name",
        d.CONTAINER,
        "--restart",
        "unless-stopped",
        "-p",
        "5001:5000",
        "-v",
        f"{data}:/var/lib/registry",
        "registry:2",
    ]
    assert data.is_dir() and made["data"] == str(data)


def test_no_docker_or_no_right_says_why_and_is_tried_again(hub_db, monkeypatch):
    settings, database = hub_db
    monkeypatch.setattr(d, "docker", lambda: None)
    with database.session() as db:
        none = d.setup(db, settings)
    assert none["mode"] == "none" and "Docker가 없습니다" in none["error"]

    class Refused:
        returncode = 1
        stdout = ""
        stderr = "permission denied while trying to connect to the Docker daemon socket"

    monkeypatch.setattr(d, "docker", lambda: "docker")
    monkeypatch.setattr(d, "_run", lambda args, timeout=60: Refused())
    with database.session() as db:  # "none" is looked at again, not kept
        refused = d.setup(db, settings)
    assert refused["mode"] == "none" and "docker 그룹" in refused["error"]


def _registry(tags: dict[str, str]):
    """A fake registry: ``/v2/`` answers, a manifest is there only for ``tags`` and only when asked for its kind."""
    seen: list[str] = []

    class V2(BaseHTTPRequestHandler):
        def _answer(self, body: bool):
            seen.append(self.headers.get("Accept", ""))
            if self.path == "/v2/":
                self.send_response(200)
            elif self.path == "/v2/_catalog?n=1000":
                self.send_response(200)
                self.end_headers()
                if body:
                    self.wfile.write(b'{"repositories": ["team/x"]}')
                return
            else:
                tag = self.path.rsplit("/", 1)[-1]
                kind = tags.get(tag)
                self.send_response(200 if kind and kind in (self.headers.get("Accept") or "") else 404)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            self._answer(True)

        def do_HEAD(self):  # noqa: N802
            self._answer(False)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), V2)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_an_image_is_checked_in_its_registry_whatever_kind_its_manifest_is():
    server, seen = _registry({"oci": "application/vnd.oci.image.index.v1+json", "v2": "manifest.v2+json"})
    where = f"127.0.0.1:{server.server_port}"
    try:
        assert d.reachable(where) and d.catalog(where) == 1
        assert d.present(f"{where}/team/x:oci") is True  # an OCI index (Docker's containerd store)
        assert d.present(f"{where}/team/x:v2") is True
        assert d.present(f"{where}/team/x:gone") is False
        assert d.present("python:3.10-slim") is None  # Docker Hub: not checked
    finally:
        server.shutdown()
    assert d.present("127.0.0.1:9/team/x:1") is None  # nothing answers: could not tell


def test_this_machines_registry_is_pulled_through_localhost(monkeypatch):
    monkeypatch.setattr(registries, "own_hosts", lambda: {"localhost", "127.0.0.1", "10.0.0.5", "hubpc"})
    assert registries.local_name("10.0.0.5:5000/wrapper/x:v4") == "localhost:5000/wrapper/x:v4"
    assert registries.local_name("hubpc:5001/x@sha256:ab") == "localhost:5001/x@sha256:ab"
    assert registries.local_name("10.0.0.9:5000/x:1") == ""  # another machine's
    assert registries.local_name("localhost:5000/x:1") == ""  # already
    assert registries.local_name("python:3.10") == ""


def test_admins_see_set_and_push(client, admin):
    view = client.get("/api/docker-registry", headers=admin).json()
    assert view["auto"] is False  # tests never make containers (BADAMOA_DOCKER_REGISTRY=off)
    server, _ = _registry({"1": "manifest"})
    where = f"127.0.0.1:{server.server_port}"
    try:
        out = client.put("/api/docker-registry", json={"address": f"http://{where}/"}, headers=admin).json()
        assert (out["mode"], out["address"], out["ok"]) == ("external", where, True)
        assert where in client.get("/install/registries").json()["registries"]  # nodes let their Docker use it
        body = {"name": "사내", "kind": "docker", "image": f"{where}/team/x:2"}
        assert client.post("/api/environments", json=body, headers=admin).status_code == 200
        [row] = client.get("/api/docker-registry/images", headers=admin).json()
        assert (row["image"], row["present"], row["used_by"]) == (f"{where}/team/x:2", False, ["환경 사내"])
        assert row["push"] == f"docker push {where}/team/x:2"
        refused = client.post("/api/docker-registry/push", json={"image": "elsewhere/x:1"}, headers=admin)
        assert refused.status_code == 400
    finally:
        server.shutdown()
    off = client.put("/api/docker-registry", json={"off": True}, headers=admin).json()
    assert off["mode"] == "off" and off["address"] == ""
    # the environment still names that registry: nodes still let their Docker use it
    assert where in client.get("/install/registries").json()["registries"]
    member = client.post("/api/users", json={"username": "kim", "password": "kim-registry-pw"}, headers=admin)
    assert member.status_code == 200
    token = client.post("/api/auth/login", json={"username": "kim", "password": "kim-registry-pw"}).json()["token"]
    assert client.get("/api/docker-registry", headers={"Authorization": f"Bearer {token}"}).status_code == 403
