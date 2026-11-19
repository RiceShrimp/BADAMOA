"""Starting the Hub when its machine starts (설정 → Hub 실행), and the Windows tasks behind it."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import pytest

from badamoa.hub import autostart
from badamoa.hub.config import HubSettings
from badamoa.node import service

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def _parse(xml: str) -> ET.Element:
    return ET.fromstring(xml.split("?>", 1)[1])  # the declaration names UTF-16; the text is already decoded


def test_windows_tasks_never_time_out_and_come_back():
    boot = _parse(service.task_xml("C:\\py\\pythonw.exe", '-m x run --home "C:\\h"', "C:\\h", boot=True))
    assert boot.find("t:Triggers/t:BootTrigger", NS) is not None
    assert boot.find("t:Principals/t:Principal/t:UserId", NS).text == service.SYSTEM
    settings = boot.find("t:Settings", NS)
    assert settings.find("t:ExecutionTimeLimit", NS).text == "PT0S"  # schtasks alone: stopped after 72 hours
    assert settings.find("t:DisallowStartIfOnBatteries", NS).text == "false"
    assert settings.find("t:StopIfGoingOnBatteries", NS).text == "false"
    assert settings.find("t:RestartOnFailure/t:Interval", NS).text == "PT1M"
    assert settings.find("t:Priority", NS).text == "5"  # the jobs it starts inherit it
    assert boot.find("t:Actions/t:Exec/t:Arguments", NS).text == '-m x run --home "C:\\h"'

    logon = _parse(service.task_xml("p", "a", "w", boot=False, user="PC\\kim & lee"))
    assert logon.find("t:Triggers/t:LogonTrigger/t:UserId", NS).text == "PC\\kim & lee"
    principal = logon.find("t:Principals/t:Principal", NS)
    assert principal.find("t:LogonType", NS).text == "InteractiveToken"
    assert principal.find("t:RunLevel", NS).text == "LeastPrivilege"
    s4u = _parse(service.task_xml("p", "a", "w", boot=True, user="PC\\kim"))
    assert s4u.find("t:Principals/t:Principal/t:LogonType", NS).text == "S4U"


def test_the_hub_is_started_again_the_way_it_runs_now(tmp_path, monkeypatch):
    monkeypatch.setenv("BADAMOA_TORCH_BACKEND", "cu128")
    monkeypatch.setenv("BADAMOA_GPU_BUSY_MEMORY_PCT", "101")
    settings = HubSettings(home=tmp_path / "hub", port=8479, public_url="http://gpu-hub.lan:8479", local_node=False)
    launch = autostart.save_launch(settings, (101.0, 101.0))
    args = launch["args"]
    assert args[:4] == ["hub", "run", "--home", str((tmp_path / "hub").resolve())]
    assert args[args.index("--port") + 1] == "8479" and "--no-open" in args and "--no-local-node" in args
    assert args[args.index("--public-url") + 1] == "http://gpu-hub.lan:8479"
    assert args[args.index("--gpu-busy-memory-pct") + 1] == "101"
    assert launch["env"]["BADAMOA_TORCH_BACKEND"] == "cu128" and "BADAMOA_GPU_BUSY_MEMORY_PCT" not in launch["env"]

    # what the machine runs: the recorded command, its output in hub.log
    ran = {}
    import badamoa.cli

    monkeypatch.setattr(badamoa.cli, "app", lambda argv, prog_name: ran.update(argv=argv) or print("hub up"))
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.chdir(tmp_path)
    autostart.run(tmp_path / "hub")
    sys.stdout.flush()
    assert ran["argv"] == args and "hub up" in (tmp_path / "hub" / "hub.log").read_text(encoding="utf-8")


@pytest.fixture
def windows(monkeypatch):
    """The Windows path, with the Task Scheduler faked: what was registered, by name."""
    tasks: dict[str, str] = {}
    monkeypatch.setattr(autostart, "WINDOWS", True)
    monkeypatch.setattr(autostart, "LINUX", False)
    monkeypatch.setattr(autostart, "is_elevated", lambda: False)
    monkeypatch.setattr(service, "register_task", lambda name, xml: tasks.__setitem__(name, xml) or "")
    monkeypatch.setattr(service, "query_task", lambda name: tasks.get(name))
    monkeypatch.setattr(service, "delete_task", lambda name: tasks.pop(name, None) and "")
    return tasks


def test_an_admin_turns_it_on_and_off_on_the_screen(client, admin, settings, windows):
    view = client.get("/api/settings/hub-autostart", headers=admin).json()
    assert view["on"] is False and view["name"] == "BADAMOA Hub 8470" and view["admin_command"] == ""

    view = client.put("/api/settings/hub-autostart", json={"on": True}, headers=admin).json()
    # no administrator rights: when the account logs on, and the one command that makes it start at boot
    assert (view["on"], view["when"]) == (True, "logon")
    command = view["admin_command"]
    assert "-m badamoa.hub.autostart on --home" in command and str(settings.home.resolve()) in command
    xml = windows["BADAMOA Hub 8470"]
    assert "<LogonTrigger>" in xml and "-m badamoa.hub.autostart run --home" in xml
    assert (settings.home / "autostart.json").is_file()

    view = client.put("/api/settings/hub-autostart", json={"on": False}, headers=admin).json()
    assert view["on"] is False and not windows
    feed = client.get("/api/activity", headers=admin).json()
    assert [e["action"] for e in feed if e["action"] == "hub.autostart"] == ["hub.autostart", "hub.autostart"]


def test_ubuntu_as_root_gets_a_systemd_unit(tmp_path, monkeypatch):
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "enabled\n", ""

    monkeypatch.setattr(autostart, "WINDOWS", False)
    monkeypatch.setattr(autostart, "LINUX", True)
    monkeypatch.setattr(autostart, "_systemd", lambda: True)
    monkeypatch.setattr(autostart, "UNIT_DIR", tmp_path)
    monkeypatch.setattr(autostart.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Done())
    settings = HubSettings(home=tmp_path / "hub", port=8470)
    autostart.save_launch(settings)
    assert autostart.turn_on(settings.home) == ""
    unit = (tmp_path / "badamoa-hub-8470.service").read_text(encoding="utf-8")
    assert "-m badamoa.hub.autostart run --home" in unit and "Restart=always" in unit and "network-online" in unit
    assert ["systemctl", "enable", "badamoa-hub-8470"] in calls
    view = autostart.status(settings.home, 8470)
    assert (view["on"], view["when"], view["name"]) == (True, "boot", "systemd badamoa-hub-8470")
    assert autostart.turn_off(settings.home, 8470) == "" and not (tmp_path / "badamoa-hub-8470.service").exists()


def test_members_cannot_see_or_change_it(client, admin, windows):
    client.post("/api/users", json={"username": "alice", "password": "alice-password"}, headers=admin)
    token = client.post("/api/auth/login", json={"username": "alice", "password": "alice-password"}).json()["token"]
    member = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/settings/hub-autostart", headers=member).status_code == 403
    assert client.put("/api/settings/hub-autostart", json={"on": True}, headers=member).status_code == 403
