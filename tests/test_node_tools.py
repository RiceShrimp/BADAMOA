"""What a node does for the Hub besides jobs: list its folders, join only once, keep up with the Hub's version."""

from __future__ import annotations

import threading

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_imagefolder, make_yolo

from badamoa import __version__
from badamoa.node import agent as node_agent
from badamoa.node import updater


@pytest.fixture
def hub(live_hub, tmp_path, monkeypatch):
    """A Hub with one real node running in this process."""
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        join = client.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="data-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        client.node_id = runner.config.node_id
        client.node_home = home
        client.url = live_hub
        yield client
        runner.stop()


def test_browse_a_nodes_folders(hub, tmp_path):
    data = tmp_path / "data"
    make_yolo(data / "street")
    make_imagefolder(data / "pets")
    (data / ".hidden").mkdir()

    def browse(path="", **extra):
        return hub.get(f"/api/nodes/{hub.node_id}/browse", params={"path": str(path), **extra}, timeout=30)

    first = browse("")
    assert first.status_code == 200, first.text
    start = first.json()
    assert start["path"] == "" and start["folders"]  # where to start: drives, home, mounts
    listing = browse(data).json()
    assert [(f["name"], f["guess"]) for f in listing["folders"]] == [("pets", ""), ("street", "yolo")]
    assert listing["parent"] == str(data.parent)
    assert browse(data / "street").json()["guess"] == "yolo"
    pets = browse(data / "pets").json()
    assert pets["guess"] == "imagefolder" and {f["name"] for f in pets["folders"]} == {"train", "val"}
    assert ".hidden" in {f["name"] for f in browse(data, hidden=True).json()["folders"]}
    missing = browse(data / "nope")
    assert missing.status_code == 400 and "nope" in missing.json()["detail"]

    # a node whose program cannot answer yet, and a node that is off
    old = hub.post("/api/nodes/join-tokens", json={}).json()
    info = {"hostname": "old", "os": "Linux", "cpu_count": 1, "memory_total_mb": 1, "gpus": []}
    body = {"join_token": old["token"], "agent_version": "0.30.0", "info": info}
    reg = httpx.post(f"{hub.url}/api/agent/register", json=body)
    answer = hub.get(f"/api/nodes/{reg.json()['node_id']}/browse")
    assert answer.status_code == 409 and "0.30.0" in answer.json()["detail"]


def test_join_again_keeps_the_registration(hub):
    again = node_agent.already_joined(hub.url, hub.node_home)
    assert again is not None and again.node_id == hub.node_id
    # the same Hub reached at another address: the node remembers the new one
    other = hub.url.replace("127.0.0.1", "localhost")
    assert node_agent.already_joined(other, hub.node_home).hub_url == other
    # deleted on the Hub: a new registration is needed
    hub.delete(f"/api/nodes/{hub.node_id}")
    assert node_agent.already_joined(hub.url, hub.node_home) is None


def test_update_helpers(client, admin):
    assert updater.newer("0.31.0", "0.30.0") and updater.newer("1.0.0", "0.99.9")
    assert not updater.newer("0.30.0", "0.30.0") and not updater.newer("0.30.0", "0.31.0")
    assert not updater.newer("dev", "0.1.0")
    assert updater.layout() is None  # the tests run from a checkout, not from an installed node
    board = client.get("/api/gpu-board", headers=admin).json()
    assert board["hub_version"] == __version__
    commands = client.get("/api/node-update-commands", headers=admin).json()["commands"]
    assert commands["linux"].endswith("/install/node-update.sh | sudo bash") and "node-update.ps1" in commands["windows"]
    script = client.get("/install/node-update.sh").text
    assert f'VERSION="{__version__}"' in script and "systemctl restart badamoa-node" in script
    assert "uv tool install --force" in client.get("/install/node-update.ps1").text
