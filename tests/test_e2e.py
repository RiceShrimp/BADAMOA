"""End to end over real HTTP: Hub + a real Node Agent + an MCP client speaking JSON-RPC."""

from __future__ import annotations

import threading
import time

import httpx
import pytest
from conftest import ADMIN

from badamoa.node import agent as node_agent


@pytest.fixture
def hub(live_hub):
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=30) as c:
        yield c


@pytest.fixture
def node(hub, tmp_path, monkeypatch):
    """Join and run a real agent in this process."""
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 0.5)
    join = hub.post("/api/nodes/join-tokens", json={}).json()
    home = tmp_path / "node"
    home.mkdir()
    config = node_agent.join(str(hub.base_url), join["token"], home, name="test-node")
    runner = node_agent.NodeAgent(config, home)
    threading.Thread(target=runner.run_forever, daemon=True).start()
    yield config.node_id
    runner.stop()


def _wait(hub, job_id, timeout=60):
    deadline = time.monotonic() + timeout
    job = None
    while time.monotonic() < deadline:
        job = hub.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "canceled"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job did not finish: {job}")


def test_agent_runs_system_info_job(hub, node):
    job = hub.post("/api/jobs", json={"type": "system.info", "node_id": node}).json()
    done = _wait(hub, job["id"])
    assert done["status"] == "succeeded", done
    assert done["node_name"] == "test-node"
    assert done["result"]["info"]["hostname"]
    logs = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
    assert "host" in logs and "memory" in logs
    assert hub.get("/api/nodes").json()[0]["status"] == "online"


def test_agent_reports_failure_with_logs(hub, node):
    job = hub.post("/api/jobs", json={"type": "debug.sleep", "params": {"seconds": 1, "fail": True}}).json()
    done = _wait(hub, job["id"])
    assert done["status"] == "failed" and "종료 코드" in done["error"]
    logs = "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job['id']}/logs").json())
    assert "tick 1/1" in logs and "의도적 실패" in logs


def test_agent_cancels_running_job(hub, node):
    job = hub.post("/api/jobs", json={"type": "debug.sleep", "params": {"seconds": 60}}).json()
    for _ in range(100):
        if hub.get(f"/api/jobs/{job['id']}").json()["status"] == "running":
            break
        time.sleep(0.1)
    hub.post(f"/api/jobs/{job['id']}/cancel")
    done = _wait(hub, job["id"], timeout=30)
    assert done["status"] == "canceled"
    assert done["duration_s"] < 30


# ---- MCP over Streamable HTTP -------------------------------------------------

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
INIT = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}}


def _rpc(hub, method, params=None, token=None):
    headers = dict(MCP_HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    return httpx.post(str(hub.base_url).rstrip("/") + "/mcp", json=body, headers=headers, timeout=60)


def test_mcp_rejects_anonymous_and_bogus_tokens(hub):
    assert _rpc(hub, "initialize", INIT).status_code == 401
    assert _rpc(hub, "initialize", INIT, token="bma_not-a-real-token").status_code == 401


def test_mcp_tools_drive_a_job_end_to_end(hub, node):
    token = hub.post("/api/tokens", json={"name": "claude-code"}).json()["token"]

    init = _rpc(hub, "initialize", INIT, token)
    assert init.status_code == 200, init.text
    assert init.json()["result"]["serverInfo"]["name"] == "badamoa"

    tools = {t["name"]: t for t in _rpc(hub, "tools/list", token=token).json()["result"]["tools"]}
    assert {"list_nodes", "run_job", "wait_for_job", "get_job_logs", "cancel_job"} <= set(tools)
    assert tools["list_nodes"]["annotations"]["readOnlyHint"] is True
    assert tools["cancel_job"]["annotations"]["destructiveHint"] is True
    assert "ctx" not in tools["run_job"]["inputSchema"]["properties"]

    def call(name, arguments=None):
        res = _rpc(hub, "tools/call", {"name": name, "arguments": arguments or {}}, token).json()["result"]
        assert not res.get("isError"), res
        return res

    nodes = call("list_nodes")["structuredContent"]["result"]
    assert nodes[0]["name"] == "test-node" and nodes[0]["status"] == "online"
    board = call("get_gpu_board")["structuredContent"]
    assert board["nodes"][0]["name"] == "test-node" and "waiting" in board
    assert tools["get_gpu_board"]["annotations"]["readOnlyHint"] is True

    job = call("run_job", {"type": "system.info"})["structuredContent"]
    assert job["created_via"] == "mcp"
    done = call("wait_for_job", {"job_id": job["id"], "timeout_seconds": 60})["structuredContent"]
    assert done["status"] == "succeeded"
    assert "host" in call("get_job_logs", {"job_id": job["id"]})["content"][0]["text"]

    bad = _rpc(hub, "tools/call", {"name": "run_job", "arguments": {"type": "nope"}}, token).json()["result"]
    assert bad["isError"] is True and "unknown job type" in bad["content"][0]["text"]

    res = _rpc(hub, "resources/read", {"uri": f"badamoa://jobs/{job['id']}"}, token).json()["result"]
    assert '"succeeded"' in res["contents"][0]["text"]
    prompts = [p["name"] for p in _rpc(hub, "prompts/list", token=token).json()["result"]["prompts"]]
    assert "diagnose-failed-job" in prompts
