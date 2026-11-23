"""A Windows process without a console (pythonw: the Hub's task, a node's service) starts console programs with no
window, so nvidia-smi on every GPU reading no longer flashes a terminal up and down."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from badamoa.core import winproc


def test_child_programs_get_no_window_unless_the_caller_chose_one():
    seen = {}

    def init(self, *args, **kwargs):
        seen.update(kwargs)

    quiet = winproc.quiet(init)
    quiet(None, ["nvidia-smi"])
    assert seen["creationflags"] == winproc.CREATE_NO_WINDOW
    quiet(None, ["x"], creationflags=0x200)  # a new process group: kept, and no window
    assert seen["creationflags"] == 0x200 | winproc.CREATE_NO_WINDOW
    quiet(None, ["cmd"], creationflags=winproc.DETACHED_PROCESS)  # the caller's choice stands
    assert seen["creationflags"] == winproc.DETACHED_PROCESS


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' console windows")
def test_pythonw_turns_it_on(tmp_path):
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        pytest.skip("no pythonw beside this Python")
    out = tmp_path / "quiet.txt"
    flag = "str(getattr(subprocess.Popen, '_badamoa_quiet', False))"
    code = f"import badamoa, subprocess, pathlib; pathlib.Path(r'{out}').write_text({flag})"
    subprocess.run([str(pythonw), "-c", code], timeout=60, check=True)
    assert out.read_text() == "True"


def test_the_task_serves_the_hub_from_python_exe_with_no_window(tmp_path, monkeypatch):
    import json

    from badamoa.hub import autostart

    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "pythonw.exe").write_bytes(b"")
    (scripts / "python.exe").write_bytes(b"")
    monkeypatch.setattr(autostart, "WINDOWS", True)
    monkeypatch.setattr(autostart.sys, "executable", str(scripts / "pythonw.exe"))
    monkeypatch.setenv("PYTHONUTF8", "1")
    assert autostart._console_python() == str(scripts / "python.exe")

    home = tmp_path / "hub"
    home.mkdir()
    launch = {"python": str(scripts / "pythonw.exe"), "args": ["hub", "run", "--port", "8470"], "env": {}}
    (home / "autostart.json").write_text(json.dumps(launch), encoding="utf-8")
    started = {}

    class Child:
        def wait(self):
            return 0

    def popen(command, **options):
        started.update(command=command, **options)
        return Child()

    monkeypatch.setattr(autostart.subprocess, "Popen", popen)
    monkeypatch.setattr(autostart, "_end_with_me", lambda: None)
    with pytest.raises(SystemExit) as done:
        autostart.run(home)
    assert done.value.code == 0
    assert started["command"] == [str(scripts / "python.exe"), "-m", "badamoa.cli", "hub", "run", "--port", "8470"]
    assert started["creationflags"] & winproc.CREATE_NO_WINDOW and started["cwd"] == home.resolve()

    # started some other way (python.exe, Linux): the Hub runs in this very process as before
    monkeypatch.setattr(autostart.sys, "executable", str(scripts / "python.exe"))
    assert autostart._console_python() is None
