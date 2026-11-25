"""What a node does for the Hub besides jobs: list its folders, join only once, keep up with the Hub's version."""

from __future__ import annotations

import threading
import time

import httpx
import pytest
from conftest import ADMIN
from sample_data import make_imagefolder, make_yolo

from badamoa import __version__
from badamoa.node import agent as node_agent
from badamoa.node import service, updater


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
        client.runner = runner
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


def test_a_node_without_root_starts_from_its_own_crontab(tmp_path):
    home = tmp_path / "my node"
    home.mkdir()
    boot, every = service._cron_lines(home)
    assert boot.startswith("@reboot ") and every.startswith("*/5 * * * * ")
    for line in (boot, every):
        assert line.endswith(service.CRON_MARK) and "node start --home" in line and ">/dev/null 2>&1" in line


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="the agent lock is Linux / macOS only")
def test_one_agent_per_node_home(tmp_path):
    assert not service.running(tmp_path)
    held = service.hold_lock(tmp_path)
    assert held is not None and service.running(tmp_path) and service.hold_lock(tmp_path) is None
    held.close()
    assert not service.running(tmp_path)


def test_delete_takes_the_node_off_its_machine(hub):
    # an old node cannot clean up: the commands to do it by hand come back
    old = hub.post("/api/nodes/join-tokens", json={}).json()
    info = {"hostname": "old", "os": "Linux", "cpu_count": 1, "memory_total_mb": 1, "gpus": []}
    body = {"join_token": old["token"], "agent_version": "0.31.0", "info": info}
    old_id = httpx.post(f"{hub.url}/api/agent/register", json=body).json()["node_id"]
    gone = hub.delete(f"/api/nodes/{old_id}").json()
    assert gone["removed_on_server"] is False and "0.31.0" in gone["reason"]
    assert gone["commands"]["linux"].endswith("/install/node-remove.sh | sudo bash")
    script = hub.get("/install/node-remove.sh").text
    assert "systemctl disable --now badamoa-node" in script and "rm -rf /opt/badamoa /var/lib/badamoa/node" in script
    assert "schtasks /Delete" in hub.get("/install/node-remove.ps1").text

    # a running node: told first, it forgets its registration (a checkout install: no files removed) and stops
    config = hub.node_home / "node.json"
    assert config.is_file()
    done = hub.delete(f"/api/nodes/{hub.node_id}").json()
    assert (done["removed_on_server"], done["reason"]) == (True, "")
    assert not config.exists() and hub.runner.stop_event.is_set()
    assert all(n["id"] != hub.node_id for n in hub.get("/api/nodes").json())


def test_delete_tells_the_node_what_to_keep(hub, monkeypatch):
    from badamoa.node import remover

    assert hub.delete(f"/api/nodes/{hub.node_id}", params={"keep": ["envs", "bogus"]}).status_code == 400
    asked: list = []
    monkeypatch.setattr(remover, "remove", lambda home, images=None, keep=(): asked.append(list(keep)) or "kept")
    done = hub.delete(f"/api/nodes/{hub.node_id}", params={"keep": ["envs", "images"]}).json()
    assert (done["removed_on_server"], done["reason"]) == (True, "") and asked == [["envs", "images"]]


def test_an_old_node_is_not_cleaned_when_something_should_stay(hub):
    old = hub.post("/api/nodes/join-tokens", json={}).json()
    info = {"hostname": "old", "os": "Linux", "cpu_count": 1, "memory_total_mb": 1, "gpus": []}
    body = {"join_token": old["token"], "agent_version": "0.100.0", "info": info}
    old_id = httpx.post(f"{hub.url}/api/agent/register", json=body).json()["node_id"]
    gone = hub.delete(f"/api/nodes/{old_id}", params={"keep": ["envs"]}).json()  # it would remove everything
    assert gone["removed_on_server"] is False and "골라 지우지" in gone["reason"]


def test_delete_without_cleanup_leaves_the_server_alone(hub):
    done = hub.delete(f"/api/nodes/{hub.node_id}", params={"cleanup": False}).json()
    assert (done["removed_on_server"], done["reason"]) == (False, "")
    assert (hub.node_home / "node.json").is_file() and not hub.runner.stop_event.is_set()


def test_imports_show_progress_and_originals_come_from_the_node(hub, tmp_path, monkeypatch):
    from badamoa.hub import catalog

    project = hub.post("/api/projects", json={"name": "진행", "task": "detect"}).json()
    body = {"node_id": hub.node_id, "path": str(make_yolo(tmp_path / "a")), "dataset": "street", "version": "v1"}
    job = hub.post(f"/api/projects/{project['id']}/imports", json=body).json()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and hub.get(f"/api/jobs/{job['id']}").json()["status"] in ("queued", "running"):
        time.sleep(0.3)

    # ---- the import is listed for every screen, with how far it got
    (entry,) = [i for i in hub.get("/api/imports").json() if i["job_id"] == job["id"]]
    assert (entry["status"], entry["done"], entry["total"], entry["dataset"]) == ("done", 8, 8, "street")
    assert entry["project"]["name"] == "진행" and entry["release_id"]
    logs = " ".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
    assert "동시에 읽습니다" in logs and "초당" in logs

    # ---- an original the Hub cannot read itself: the node that has it sends it, and it is kept
    item = hub.get(f"/api/releases/{entry['release_id']}/items").json()["items"][0]
    detail = hub.get(f"/api/releases/{entry['release_id']}/items/{item['id']}").json()
    assert detail["image_url"] and detail["image_local"] is True
    monkeypatch.setattr(catalog, "item_file", lambda db, it: None)  # as if the images were on another server
    original = hub.get(detail["image_url"], timeout=90)
    assert original.status_code == 200 and len(original.content) == detail["bytes"]
    assert hub.get(detail["image_url"]).content == original.content  # from the cache this time
