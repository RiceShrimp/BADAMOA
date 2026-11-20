"""Moving a server's Docker images: when the Hub offers it, and what the node refuses before touching anything.

The move itself (stop, copy, daemon.json, start, check, roll back) runs on Linux as root; it was checked in a
python:3.12-slim container with stand-in docker and systemctl and a second filesystem (see the 0.75.0 notes)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from badamoa.hub import disk_space
from badamoa.node import docker_root

DOCKER = {"kind": "docker", "path": "/var/lib/docker", "mount": "/", "free_gb": 2.0, "total_gb": 100.0}
DISKS = [{"mount": "/", "free_gb": 2.0, "total_gb": 100.0}, {"mount": "/home4", "free_gb": 2000.0, "total_gb": 4000.0}]


def _node(**over):
    base = {
        "metrics": {"places": [DOCKER], "disks": DISKS},
        "agent_version": "0.75.0",
        "os": "linux",
        "info": {"account": {"name": "root", "admin": True}},
    }
    return SimpleNamespace(**{**base, **over})


def test_the_hub_offers_the_move_only_where_it_can_be_done():
    state = disk_space.docker_move_state(_node())
    assert state == {
        "path": "/var/lib/docker",
        "suggest": "/home4/docker",
        "choices": [{"folder": "/home4/docker", "mount": "/home4", "free_gb": 2000.0, "total_gb": 4000.0}],
        "here": {"mount": "/", "free_gb": 2.0, "total_gb": 100.0},
        "possible": True,
        "why": "",
    }
    assert "끝난 뒤" in disk_space.docker_move_state(_node(), busy=1)["why"]
    user_install = _node(info={"account": {"name": "jhan", "admin": False}})
    assert "sudo로 설치한 Linux" in disk_space.docker_move_state(user_install)["why"]
    assert "업데이트" in disk_space.docker_move_state(_node(agent_version="0.74.0"))["why"]
    desktop = {"metrics": {"places": [{"kind": "docker", "path": "/var/lib/docker", "desktop": True}], "disks": []}}
    assert "Docker Desktop" in disk_space.docker_move_state(_node(**desktop))["why"]
    assert "모릅니다" in disk_space.docker_move_state(_node(metrics={"places": [], "disks": DISKS}))["why"]


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="the refusal for a server that is not Linux")
def test_a_node_that_is_not_linux_refuses_before_anything():
    with pytest.raises(RuntimeError, match="Linux"):
        docker_root.check("/home4/docker", [], False)


def test_the_admin_api_refuses_a_node_that_cannot(client, admin):
    join = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()
    info = {"hostname": "box", "os": "windows", "gpus": []}
    node = client.post("/api/agent/register", json={"join_token": join["token"], "agent_version": "0.75.0", "info": info})
    node_id = node.json()["node_id"]
    state = client.get(f"/api/nodes/{node_id}/docker-move", headers=admin).json()
    assert state["possible"] is False and state["why"]
    refused = client.post(f"/api/nodes/{node_id}/docker-move", json={"target": "/home4/docker"}, headers=admin)
    assert refused.status_code == 400 and refused.json()["detail"] == state["why"]
