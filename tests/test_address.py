"""The address in node install commands: never localhost when the machine has a network address."""

from __future__ import annotations

from badamoa.hub import address


def test_join_commands_never_point_at_localhost(client, admin):
    # Opened by a real name: that name is used.
    body = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()
    assert (body["hub_url"], body["address"]["source"]) == ("http://testserver", "browser")

    # Opened as localhost: this machine's network address instead (or a warning when it has none).
    local = {**admin, "host": "localhost:8470"}
    body = client.post("/api/nodes/join-tokens", json={}, headers=local).json()
    found = address.lan_addresses()
    if found:
        assert body["address"]["source"] == "detected" and body["hub_url"] == f"http://{found[0]}:8470"
        assert "localhost" not in body["commands"]["linux"] and body["hub_url"] in body["address"]["candidates"]
    else:
        assert body["address"]["source"] == "local-only"

    # Set on the screen: used from then on, in the commands and in the node scripts.
    saved = client.put("/api/settings/hub-address", json={"url": "http://gpu-hub.lan:8470/"}, headers=local).json()
    assert (saved["url"], saved["source"]) == ("http://gpu-hub.lan:8470", "saved")
    body = client.post("/api/nodes/join-tokens", json={}, headers=local).json()
    assert body["commands"]["linux"].startswith("curl -fsSL http://gpu-hub.lan:8470/install/node.sh")
    assert "http://gpu-hub.lan:8470" in client.get("/install/node.sh").text
    for wrong in ("gpu-hub.lan", "http://localhost:8470", "ftp://x", "http://x:1/path"):
        assert client.put("/api/settings/hub-address", json={"url": wrong}, headers=admin).status_code == 400, wrong

    # Back to finding it by itself.
    assert client.put("/api/settings/hub-address", json={}, headers=admin).json()["source"] == "browser"


def test_loopback_names():
    assert all(address.is_loopback(h) for h in ("localhost", "127.0.0.1", "127.1.2.3", "::1", "[::1]", "app.localhost"))
    assert not any(address.is_loopback(h) for h in ("10.0.0.5", "gpu-01", "hub.example.com"))


def test_hub_start_opens_the_web_ui_at_the_network_address(live_hub, monkeypatch):
    from badamoa import cli

    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    port = int(live_hub.rsplit(":", 1)[1])
    cli._open_when_up(port, f"http://192.0.2.10:{port}")  # waits for the Hub, then opens that address
    assert opened == [f"http://192.0.2.10:{port}"]


def test_node_package_of_this_version(client, admin, settings):
    from badamoa.hub import install, services

    body = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()
    assert body["node_package"] == {"version": services.HUB_VERSION, "ready": False}  # the install would stop half-way
    (settings.wheels_dir / f"badamoa-{services.HUB_VERSION}-py3-none-any.whl").write_bytes(b"x")
    assert client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["node_package"]["ready"] is True
    assert install.ensure_node_wheel(settings) == "ready"


def test_hub_builds_the_node_package_from_its_checkout(tmp_path):
    import shutil

    import pytest

    from badamoa.hub import install, services
    from badamoa.hub.config import HubSettings

    if shutil.which("uv") is None or install._source_root() is None:
        pytest.skip("needs uv and a source checkout")
    settings = HubSettings(home=tmp_path / "hub")
    assert install.ensure_node_wheel(settings) == "built"
    assert install.node_wheel(settings).name == f"badamoa-{services.HUB_VERSION}-py3-none-any.whl"
