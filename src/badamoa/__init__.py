"""BADAMOA — Build · Annotate · Develop · Automate / MLOps Orchestration & Agents."""

__version__ = "0.116.0"

import sys as _sys

if _sys.platform == "win32":  # pythonw (the Hub's task, a node's service): no console window flashing per program
    from badamoa.core.winproc import hide_child_consoles as _hide

    _hide()
