"""No flashing console windows on Windows.

A Windows process without a console of its own — ``pythonw``: the Hub started by its scheduled task, a node's
service, the jobs they start — gives every console program it runs (``nvidia-smi`` on each GPU reading,
``docker``, ``uv``, ``git``, ``schtasks`` …) a new console window, which flashes up and closes. Such a process
starts them with no window instead (``CREATE_NO_WINDOW``), unless the caller asked for a console or a detached
process itself. A process that has a console (a terminal) is left as it is: its children share that console.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from typing import Any

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_CONSOLE = 0x00000010


def quiet(init: Callable[..., None]) -> Callable[..., None]:
    """``Popen.__init__`` that adds ``CREATE_NO_WINDOW`` when the caller chose no console of its own."""

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        flags = kwargs.get("creationflags") or 0
        if not flags & (DETACHED_PROCESS | CREATE_NEW_CONSOLE):
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        init(self, *args, **kwargs)

    return __init__


def has_console() -> bool:
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return True  # cannot tell: leave things as they are


def hide_child_consoles() -> bool:
    """On Windows without a console, start child programs with no window. Returns whether it did."""
    if sys.platform != "win32" or getattr(subprocess.Popen, "_badamoa_quiet", False) or has_console():
        return False
    subprocess.Popen.__init__ = quiet(subprocess.Popen.__init__)  # type: ignore[method-assign]
    subprocess.Popen._badamoa_quiet = True  # type: ignore[attr-defined]
    return True
