"""Where BADAMOA keeps its data on disk.

Resolution order for a component's home directory:
1. explicit path (CLI flag)
2. ``BADAMOA_HOME`` environment variable (``<BADAMOA_HOME>/<component>``)
3. machine-wide dir when running as root/Administrator (services run this way)
4. per-user dir otherwise (development)
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from platformdirs import site_data_dir, user_data_dir

APP_NAME = "badamoa"


def is_elevated() -> bool:
    if sys.platform == "win32":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except Exception:
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


def machine_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "BADAMOA"
    if sys.platform.startswith("linux"):
        return Path("/var/lib/badamoa")
    return Path(site_data_dir(APP_NAME))


def component_home(component: str, explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:  # absolute, so a relative one (--home .moa/hub) means the same to every process it is handed to
        path = Path(explicit).expanduser().absolute()
    elif os.environ.get("BADAMOA_HOME"):
        path = Path(os.environ["BADAMOA_HOME"]).expanduser().absolute() / component
    elif is_elevated():
        path = machine_root() / component
    else:
        path = Path(user_data_dir(APP_NAME, appauthor=False)) / component
    path.mkdir(parents=True, exist_ok=True)
    return path
