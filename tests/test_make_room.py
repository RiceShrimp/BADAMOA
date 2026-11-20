"""Before a job pulls a Docker image: room on Docker's disk, moving Docker's folder when the node may and is alone,
and not pulling an image that would fill the disk. Also the folders the move dialog lists, and per-image
predictions read a few at a time (a whole split given at once ran the GPU out of memory)."""

from __future__ import annotations

import json
import os
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from badamoa.hub import disk_space
from badamoa.node import containers, docker_root

GB = 1024**3
Usage = namedtuple("Usage", "total used free")


def _verbose(*entries):
    return SimpleNamespace(returncode=0, stdout=json.dumps(list(entries)), stderr="")


def test_the_pull_size_is_this_platforms_layers(monkeypatch):
    amd = {
        "Descriptor": {"platform": {"os": "linux", "architecture": "amd64"}},
        "OCIManifest": {"layers": [{"size": 2 * GB}, {"size": GB}]},
    }
    arm = {
        "Descriptor": {"platform": {"os": "linux", "architecture": "arm64"}},
        "OCIManifest": {"layers": [{"size": 9 * GB}]},
    }
    proof = {
        "Descriptor": {"platform": {"os": "unknown", "architecture": "unknown"}},
        "OCIManifest": {"layers": [{"size": 5}]},
    }
    monkeypatch.setattr(containers.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(containers.subprocess, "run", lambda *a, **k: _verbose(amd, arm, proof))
    assert containers.pull_size("docker", "reg:5000/x:1") == 3 * GB
    single = {"Descriptor": {}, "SchemaV2Manifest": {"layers": [{"size": GB}]}}  # one platform, not a list
    monkeypatch.setattr(
        containers.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(single))
    )
    assert containers.pull_size("docker", "reg:5000/x:1") == GB
    monkeypatch.setattr(containers.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    assert containers.pull_size("docker", "reg:5000/x:1") is None


@pytest.fixture
def docker_disk(monkeypatch, tmp_path):
    """Docker's folder on a 400 GB disk; an image of 1.5 GB in its registry (about 3 GB unpacked, 9 GB to pull)."""
    state = {"free": 300 * GB, "size": 1.5 * GB, "moved": [], "root": True}
    monkeypatch.setattr(containers, "storage_root", lambda: {"path": "/var/lib/docker", "desktop": False})
    monkeypatch.setattr(
        containers.shutil, "disk_usage", lambda path: Usage(400 * GB, 400 * GB - state["free"], state["free"])
    )
    monkeypatch.setattr(containers, "pull_size", lambda command, image: state["size"])
    monkeypatch.setattr(containers.sys, "platform", "linux")
    monkeypatch.setattr(os, "geteuid", lambda: 0 if state["root"] else 1000, raising=False)
    monkeypatch.setenv("BADAMOA_JOB_ID", "job-1")
    monkeypatch.setattr(docker_root, "SETTLE_S", 0)
    monkeypatch.setattr(docker_root, "choices", lambda current: ["/home4/docker"])
    monkeypatch.setattr(docker_root, "move", lambda target, **k: state["moved"].append((target, k["extra"])))
    return state


def test_three_times_the_image_free_pulls_where_docker_is(docker_disk, tmp_path):
    said = []
    docker_disk["free"] = 9 * GB
    containers.make_room(tmp_path, "docker", "reg:5000/x:1", said.append)
    assert said == [] and docker_disk["moved"] == []


def test_less_moves_dockers_folder_first_when_the_node_is_alone(docker_disk, tmp_path):
    docker_disk["free"] = 8 * GB
    (tmp_path / docker_root.RUNNING).write_text(json.dumps(["job-1"]), encoding="utf-8")
    said = []
    containers.make_room(tmp_path, "docker", "reg:5000/x:1", said.append)
    assert docker_disk["moved"] == [("/home4/docker", 9 * GB)]  # the new disk must hold the image three times too
    assert "이미지(약 3.0 GB)의 3배인 9.0 GB가 남아 있어야 받는데 8.0 GB 남았습니다" in said[0]
    assert not (tmp_path / docker_root.HOLD).exists()  # new jobs may start again


def test_with_another_job_running_it_neither_moves_nor_pulls(docker_disk, tmp_path):
    docker_disk["free"] = 8 * GB
    (tmp_path / docker_root.RUNNING).write_text(json.dumps(["job-1", "job-2"]), encoding="utf-8")
    said = []
    with pytest.raises(RuntimeError, match="받지 않았습니다.*3배.*다른 디스크로 옮기기"):
        containers.make_room(tmp_path, "docker", "reg:5000/x:1", said.append)
    assert docker_disk["moved"] == [] and any("다른 MOA 작업 1개" in line for line in said)


def test_without_root_it_does_not_pull_either(docker_disk, tmp_path):
    docker_disk["root"] = False
    docker_disk["free"] = 8 * GB
    said = []
    with pytest.raises(RuntimeError, match="받지 않았습니다"):
        containers.make_room(tmp_path, "docker", "reg:5000/x:1", said.append)
    assert docker_disk["moved"] == [] and any("root가 아니라" in line for line in said)


def test_an_image_of_unknown_size_is_pulled_unless_the_disk_is_already_short(docker_disk, tmp_path):
    docker_disk["size"] = None
    docker_disk["free"] = 30 * GB  # short below 20 GB on a 400 GB disk (5 %)
    containers.make_room(tmp_path, "docker", "reg:5000/x:1", print)
    assert docker_disk["moved"] == []
    docker_disk["free"] = 15 * GB
    (tmp_path / docker_root.RUNNING).write_text(json.dumps(["job-1"]), encoding="utf-8")
    said = []
    containers.make_room(tmp_path, "docker", "reg:5000/x:1", said.append)
    assert docker_disk["moved"] == [("/home4/docker", 20 * GB)] and "크기를 알 수 없고" in said[0]


def test_the_agent_starts_nothing_while_a_job_holds_the_node(tmp_path):
    from badamoa.node.agent import NodeAgent, NodeConfig

    agent = NodeAgent(NodeConfig(hub_url="http://hub", node_id="n", node_token="t", name="box"), tmp_path, max_jobs=4)
    agent._running = {"job-1": None}
    agent._write_running()
    assert docker_root.running_jobs(tmp_path) == ["job-1"]
    (tmp_path / docker_root.HOLD).write_text("job-1", encoding="utf-8")
    assert agent._slots_free() == 0
    agent._running = {}  # the job ended without taking its hold away
    assert agent._slots_free() == 4 and not (tmp_path / docker_root.HOLD).exists()


def test_the_dialog_lists_every_other_disk_roomiest_first():
    disks = [
        {"mount": "/", "free_gb": 2.0, "total_gb": 440.0},
        {"mount": "/home2", "free_gb": 420.0, "total_gb": 1700.0},
        {"mount": "/home4", "free_gb": 2100.0, "total_gb": 7000.0},
        {"mount": "/boot/efi", "free_gb": 1.0, "total_gb": 2.0},
    ]
    found = disk_space.docker_choices({"disks": disks}, "/var/lib/docker")
    assert [c["folder"] for c in found] == ["/home4/docker", "/home2/docker"]
    assert docker_root.choices("/var/lib/docker", disks)[0] in ("/home4/docker", "/home2/docker")


def test_per_image_predictions_go_a_few_at_a_time(tmp_path, monkeypatch):
    template = Path(__file__).resolve().parent.parent / "src" / "badamoa" / "templates" / "yolo"
    monkeypatch.syspath_prepend(str(template))
    monkeypatch.setitem(sys.modules, "badamoa_sdk", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(**{n: None for n in (
        "IMAGE_SUFFIXES", "MAX_OUTLINE", "SAMPLE_CONF", "SAMPLE_SIDE", "class_folder_index", "clean", "match",
        "task_of", "truth_of")}))  # fmt: skip
    sys.modules.pop("ops", None)
    import ops

    images = [tmp_path / f"{n:03d}.jpg" for n in range(20)]
    for image in images:
        image.write_bytes(b"")
    asked = {}

    class Model:
        def predict(self, source, **settings):
            asked.update(source=source, **settings)
            listed = Path(source).read_text(encoding="utf-8").splitlines()
            for line in reversed(listed):  # in any order: matched by path
                yield SimpleNamespace(path=line)

    got = list(ops.predict_each(Model(), images, conf=0.25, device="0"))
    assert sorted(image for image, _ in got) == images
    assert asked["batch"] == ops.PREDICT_BATCH and asked["stream"] is True and asked["source"].endswith(".txt")
    sys.modules.pop("ops", None)
