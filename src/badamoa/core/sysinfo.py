"""Hardware / OS inspection used by nodes to describe themselves to the Hub."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any

import psutil

_GPU_FIELDS = [
    "index",
    "name",
    "uuid",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "driver_version",
]


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if sys.platform == "win32":
        candidate = r"C:\Windows\System32\nvidia-smi.exe"
        if os.path.exists(candidate):
            return candidate
    return None


def gpus() -> list[dict[str, Any]]:
    """Query NVIDIA GPUs via nvidia-smi. Returns [] when unavailable."""
    exe = _nvidia_smi()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    result = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_FIELDS):
            continue
        idx, name, uuid, mem_total, mem_used, util, temp, driver = parts
        result.append(
            {
                "index": _to_int(idx),
                "name": name,
                "uuid": uuid,
                "memory_total_mb": _to_int(mem_total),
                "memory_used_mb": _to_int(mem_used),
                "utilization_pct": _to_int(util),
                "temperature_c": _to_int(temp),
                "driver_version": driver,
            }
        )
    return result


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def static_info() -> dict[str, Any]:
    """Slow-changing facts about the machine (sent at registration)."""
    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": platform.system().lower(),  # "windows" | "linux" | "darwin"
        "os_release": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_total_mb": vm.total // (1024 * 1024),
        "gpus": gpus(),
    }


def metrics(path_for_disk: str | None = None) -> dict[str, Any]:
    """Fast-changing utilization numbers (sent with each heartbeat)."""
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(path_for_disk or ("C:\\" if sys.platform == "win32" else "/"))
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "memory_used_mb": vm.used // (1024 * 1024),
        "memory_total_mb": vm.total // (1024 * 1024),
        "disk_free_gb": round(disk.free / 1024**3, 1),
        "disk_total_gb": round(disk.total / 1024**3, 1),
        "gpus": [
            {
                "index": g["index"],
                "memory_used_mb": g["memory_used_mb"],
                "memory_total_mb": g["memory_total_mb"],
                "utilization_pct": g["utilization_pct"],
                "temperature_c": g["temperature_c"],
            }
            for g in gpus()
        ],
    }
