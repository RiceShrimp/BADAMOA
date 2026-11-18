"""Environments as virtual environments or Docker images: where they may go (the node's role), installing and
removing them ahead of time, and a run in a container (when this machine has Docker and a small Python image)."""

from __future__ import annotations

import io
import os
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_yolo

from badamoa.core import templates as fmt
from badamoa.node import agent as node_agent
from badamoa.node import containers, envs, trainer

IMAGE = "python:3.10-slim"  # already on a machine that runs these: no test pulls an image


def _have_image() -> bool:
    if not containers.docker() or containers.capability() is None:
        return False
    return containers.image_id(IMAGE) is not None


# ---- without Docker ----------------------------------------------------------------------------


def test_paths_go_into_the_container_and_back(tmp_path):
    for path in (tmp_path / "runs" / "job" / "out" / "best.pt", tmp_path):
        assert Path(containers.outside(containers.inside(path))) == path
    if os.name == "nt":
        assert containers.inside(r"C:\data\x.jpg") == "/host/c/data/x.jpg"
    else:
        assert containers.inside("/data/x.jpg") == "/data/x.jpg"


def test_the_docker_command_mounts_what_the_code_needs_and_only_the_reserved_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(containers, "docker", lambda: "docker")
    work, code = tmp_path / "work", tmp_path / "code"
    work.mkdir()
    code.mkdir()
    command = containers.command(
        "team/train:1",
        "job1",
        ["python", "train.py"],
        mounts=[(work, False), (code, True), (tmp_path / "missing", True)],
        workdir=code,
        env={"PYTHONPATH": "/sdk"},
        gpus=["1", "3"],
    )
    text = " ".join(command)
    assert command[:6] == ["docker", "run", "--rm", "--name", "badamoa-job1", "--ipc=host"]
    assert '--gpus "device=1,3"' in text
    assert f"{work}:{containers.inside(work)}" in text and f"{code}:{containers.inside(code)}:ro" in text
    assert "missing" not in text  # a folder that is not there is not mounted
    assert command[-3:] == ["team/train:1", "python", "train.py"] and "PYTHONPATH=/sdk" in command
    assert trainer.container_args("python train.py --config {config}", "/c/config.json") == [
        "python",
        "train.py",
        "--config",
        "/c/config.json",
    ]


def test_a_template_may_declare_an_image():
    head = 'name = "boxed"\nversion = "1"\n[entrypoints]\ntrain = "python t.py --config {config}"\n'
    text = head + '[environment]\nimage = "IMG"\n'
    assert fmt.parse_manifest(text.replace("IMG", "registry.local:5000/team/train:1.2")).environment.image
    with pytest.raises(fmt.TemplateError, match="Docker 이미지 이름"):
        fmt.parse_manifest(text.replace("IMG", "not an image"))


def test_a_virtual_environment_is_removed_unless_it_is_being_built(tmp_path):
    spec = {"kind": "auto", "requirements": ["six"], "python": "3.12", "torch": False}
    env = envs.env_dir(tmp_path, ["six"], "3.12", False)
    (env / "lib").mkdir(parents=True)
    (env / "badamoa-env.json").write_text("{}", encoding="utf-8")
    (env / "lib" / "x.py").write_text("x = 1\n", encoding="utf-8")
    assert envs.listing(tmp_path)
    lock = env.parent / f"{env.name}.lock"
    lock.mkdir()
    with pytest.raises(RuntimeError, match="만드는 작업"):
        envs.remove(tmp_path, spec)
    lock.rmdir()
    assert envs.remove(tmp_path, spec)["removed"] and not env.exists() and envs.listing(tmp_path) == []


# ---- with a Hub and a node --------------------------------------------------------------------


@pytest.fixture
def hub(live_hub, tmp_path, monkeypatch):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        join = client.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="gpu-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        client.node_id, client.home = runner.config.node_id, home
        deadline = time.monotonic() + 20  # the first heartbeat: what the node runs (Docker or not) is known then
        while time.monotonic() < deadline and not client.get(f"/api/nodes/{client.node_id}").json().get("metrics"):
            time.sleep(0.2)
        yield client
        runner.stop()


def _wait(hub, job_id, timeout=300):
    deadline = time.monotonic() + timeout
    job = hub.get(f"/api/jobs/{job_id}").json()
    while time.monotonic() < deadline and job["status"] in ("queued", "running"):
        time.sleep(0.3)
        job = hub.get(f"/api/jobs/{job_id}").json()
    return job


def _logs(hub, job_id):
    return "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job_id}/logs").json())


def _import(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "환경"}).json()
    body = {"node_id": hub.node_id, "path": str(make_yolo(tmp_path / "ds")), "dataset": "street", "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    return job["result"]["release_id"]


def _upload(hub, declaration: str, **files: str) -> dict:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("badamoa.toml", declaration)
        for name, text in {"train.py": "print('train')\n", **files}.items():
            archive.writestr(name, text)
    resp = hub.post("/api/templates", content=buffer.getvalue(), headers={"Content-Type": "application/zip"})
    assert resp.status_code == 200, resp.text
    return resp.json()


VENV = (
    'name = "venv-kind"\nversion = "1"\n[entrypoints]\ntrain = "python train.py --config {config}"\n'
    '[environment]\nrequirements = ["six"]\n'
)


def test_environments_go_only_to_servers_that_train(hub, tmp_path):
    template = _upload(hub, VENV)
    servers = template["environment"]["servers"]
    assert [(s["id"], s["why"], s["state"]) for s in servers] == [(hub.node_id, "", "")]

    # the node no longer trains: nothing is installed there, and runs are refused before they start
    assert hub.patch(f"/api/nodes/{hub.node_id}", json={"roles": ["storage"]}).status_code == 200
    template = hub.get(f"/api/templates/{template['id']}").json()
    assert template["environment"]["servers"][0]["why"] == "학습 역할 아님" and template["environment"]["nodes"] == 0
    picked = hub.post(f"/api/templates/{template['id']}/prepare", json={"node_ids": [hub.node_id]})
    assert picked.status_code == 400 and "학습 역할이 아니라" in picked.json()["detail"]
    nobody = hub.post(f"/api/templates/{template['id']}/prepare", json={"node_ids": []})
    assert nobody.status_code == 400 and "학습 역할이 있는" in nobody.json()["detail"]
    release = _import(hub, tmp_path)
    check = hub.post("/api/runs/check", json={"release_id": release, "template": "dummy", "gpus": 0}).json()
    assert check["problems"] and "학습 역할이 아닙니다" in check["problems"][0]

    # a Docker environment needs Docker on the server
    hub.patch(f"/api/nodes/{hub.node_id}", json={"roles": ["compute", "storage"]})
    boxed = hub.post("/api/environments", json={"name": "py310", "kind": "docker", "image": IMAGE}).json()
    assert boxed["kind"] == "docker" and boxed["image"] == IMAGE
    check = hub.post(
        "/api/runs/check", json={"release_id": release, "template": "dummy", "environment": "py310", "gpus": 0}
    ).json()
    has_docker = bool(hub.get(f"/api/nodes/{hub.node_id}").json()["info"].get("docker"))
    assert bool(check["problems"]) is not has_docker
    if not has_docker:
        assert "Docker가 없어" in check["problems"][0]


@pytest.mark.skipif(not _have_image(), reason=f"needs Docker and the image {IMAGE} on this machine")
def test_a_run_and_an_evaluation_in_a_container(hub, tmp_path):
    release = _import(hub, tmp_path)
    hub.post("/api/environments", json={"name": "py310", "kind": "docker", "image": IMAGE})
    run = hub.post(
        "/api/runs",
        json={"release_id": release, "template": "dummy", "environment": "py310", "params": {"epochs": 2}, "gpus": 0},
    ).json()
    assert run["environment"]["kind"] == "docker" and run["environment"]["image"] == IMAGE
    job = _wait(hub, run["job_id"])
    logs = _logs(hub, run["job_id"])
    assert job["status"] == "succeeded", logs
    assert f"Docker {IMAGE}" in logs
    view = hub.get(f"/api/runs/{run['id']}").json()
    assert view["info"]["image"] == IMAGE and view["info"]["image_id"].startswith("sha256:")
    assert view["model"] is not None and view["metrics"]["map50_95"] > 0

    # the evaluation of that model runs in the same image (its code imports badamoa_sdk, mounted in)
    model = view["model"]["id"]
    op = hub.post(f"/api/models/{model}/eval", json={"release_id": release, "split": "val", "gpus": 0}).json()
    done = _wait(hub, op["job_id"])
    assert done["status"] == "succeeded", _logs(hub, op["job_id"])
    assert f"badamoa-{run['job_id']}" not in subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, check=False
    ).stdout  # --rm: nothing left behind


@pytest.mark.skipif(not _have_image(), reason=f"needs Docker and the image {IMAGE} on this machine")
def test_an_image_prepared_ahead_and_removed_keeps_an_image_that_was_there_before(hub):
    declaration = 'name = "boxed"\nversion = "1"\n[entrypoints]\ntrain = "python train.py --config {config}"\n'
    template = _upload(hub, declaration + f'[environment]\nimage = "{IMAGE}"\n')
    assert template["environment"]["kind"] == "docker" and template["environment"]["image"] == IMAGE
    prepared = hub.post(f"/api/templates/{template['id']}/prepare", json={"node_ids": [hub.node_id]})
    assert prepared.status_code == 200, prepared.text
    jobs = prepared.json()["jobs"]
    assert _wait(hub, jobs[0])["status"] == "succeeded", _logs(hub, jobs[0])

    def state():
        return hub.get(f"/api/templates/{template['id']}").json()["environment"]["servers"][0]["state"]

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and state() != "ready":
        time.sleep(0.3)
    assert state() == "ready"
    removed = hub.delete(f"/api/templates/{template['id']}/environment/{hub.node_id}").json()["job"]
    assert _wait(hub, removed)["status"] == "succeeded", _logs(hub, removed)
    assert "지우지 않고 목록에서만" in _logs(hub, removed)  # it was on this machine before: only forgotten
    assert containers.image_id(IMAGE) is not None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and state() == "ready":
        time.sleep(0.3)
    assert state() == ""
