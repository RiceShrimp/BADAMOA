"""Hardware / OS inspection used by nodes to describe themselves to the Hub.

``static_info()`` is sent once at registration. ``MetricsSampler.sample()`` is sent
with every heartbeat: CPU, memory, per-GPU load and occupants, disks, per-user
usage and login sessions. Everything is best effort - a value the platform
cannot provide is ``None`` rather than an error.
"""

from __future__ import annotations

import functools
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
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
    "power.draw",
    "power.limit",
    "driver_version",
]
_SKIP_FSTYPES = {"squashfs", "tmpfs", "devtmpfs", "overlay", "iso9660", "cdfs", "udf", "efivarfs", ""}
_SYSTEM_USERS = {"SYSTEM", "LOCAL SERVICE", "NETWORK SERVICE"}
# A user counts as "working" above this much CPU, or when holding GPU memory.
_ACTIVE_CPU_CORES = 0.5


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if sys.platform == "win32":
        candidate = r"C:\Windows\System32\nvidia-smi.exe"
        if os.path.exists(candidate):
            return candidate
    return None


def _run_smi(args: list[str]) -> list[list[str]]:
    exe = _nvidia_smi()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return [[part.strip() for part in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def _num(value: str, cast=float) -> Any:
    try:
        return cast(float(value))
    except (ValueError, TypeError):
        return None


def _short_user(name: str | None) -> str | None:
    """`DOMAIN\\user` → `user`."""
    return name.rsplit("\\", 1)[-1] if name else None


def gpus() -> list[dict[str, Any]]:
    """Query NVIDIA GPUs via nvidia-smi. Returns [] when unavailable."""
    result = []
    for parts in _run_smi([f"--query-gpu={','.join(_GPU_FIELDS)}"]):
        if len(parts) != len(_GPU_FIELDS):
            continue
        idx, name, uuid, mem_total, mem_used, util, temp, power, power_limit, driver = parts
        result.append(
            {
                "index": _num(idx, int),
                "name": name,
                "uuid": uuid,
                "memory_total_mb": _num(mem_total, int),
                "memory_used_mb": _num(mem_used, int),
                "utilization_pct": _num(util, int),
                "temperature_c": _num(temp, int),
                "power_w": _num(power, round),
                "power_limit_w": _num(power_limit, round),
                "driver_version": driver,
            }
        )
    return result


def gpu_processes() -> dict[str, list[dict[str, Any]]]:
    """Compute processes per GPU uuid: pid, owning user, process name, memory."""
    by_gpu: dict[str, list[dict[str, Any]]] = {}
    for parts in _run_smi(["--query-compute-apps=gpu_uuid,pid,used_memory"]):
        if len(parts) != 3:
            continue
        uuid, pid, mem = parts[0], _num(parts[1], int), _num(parts[2], int)
        if pid is None:
            continue
        user = name = None
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            user = _short_user(proc.username())
        except (psutil.Error, OSError):
            pass
        by_gpu.setdefault(uuid, []).append({"pid": pid, "user": user, "name": name, "memory_mb": mem})
    return by_gpu


def static_info() -> dict[str, Any]:
    """Slow-changing facts about the machine (sent at registration)."""
    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": platform.system().lower(),  # "windows" | "linux" | "darwin"
        "os_release": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_total_mb": vm.total // (1024 * 1024),
        "gpus": [
            {k: g[k] for k in ("index", "name", "uuid", "memory_total_mb", "power_limit_w", "driver_version")}
            for g in gpus()
        ],
    }


def _cpu_model() -> str | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def _cpu_temperature() -> int | None:
    sensors = getattr(psutil, "sensors_temperatures", None)
    if sensors is None:
        return None
    try:
        readings = sensors()
    except (OSError, psutil.Error):
        return None
    for name in ("k10temp", "coretemp", "zenpower", "cpu_thermal"):
        if readings.get(name):
            return round(max(r.current for r in readings[name]))
    return None


def _rapl_energy_uj() -> int | None:
    """Total CPU package energy counter (Linux powercap), or None when unreadable."""
    total, found = 0, False
    for path in Path("/sys/class/powercap").glob("intel-rapl:[0-9]*/energy_uj"):
        try:
            total += int(path.read_text())
            found = True
        except (OSError, ValueError):
            return None
    return total if found else None


def _is_whole_disk(name: str) -> bool:
    """True for sda / nvme0n1 / PhysicalDrive0, False for partitions and virtual devices."""
    low = name.lower()
    if low.startswith("physicaldrive"):
        return True
    if low.startswith(("loop", "ram", "zram", "dm-", "sr", "md")):
        return False
    if low.startswith("nvme"):
        return re.fullmatch(r"nvme\d+n\d+", low) is not None
    return not low[-1:].isdigit()


def _disk_kind(name: str) -> tuple[str | None, str | None]:
    """(model, kind) of a Linux block device; kind is NVMe | SSD | HDD."""
    base = Path("/sys/block") / name
    if not base.exists():
        return None, None
    model = None
    try:
        model = (base / "device" / "model").read_text().strip() or None
    except OSError:
        pass
    try:
        rotational = (base / "queue" / "rotational").read_text().strip() == "1"
    except OSError:
        return model, None
    return model, "NVMe" if name.startswith("nvme") else ("HDD" if rotational else "SSD")


# IOCTL_DISK_PERFORMANCE = CTL_CODE(IOCTL_DISK_BASE, 0x0008, METHOD_BUFFERED, FILE_ANY_ACCESS)
_IOCTL_DISK_PERFORMANCE = 0x00070020
# DISK_PERFORMANCE up to QueryTime: BytesRead, BytesWritten, ReadTime, WriteTime, IdleTime (LARGE_INTEGER),
# ReadCount, WriteCount, QueueDepth, SplitCount (DWORD), QueryTime (LARGE_INTEGER).
_DISK_PERFORMANCE = struct.Struct("<5q4Iq")


def _parse_disk_performance(raw: bytes) -> tuple[int, int] | None:
    """(IdleTime, QueryTime) from a DISK_PERFORMANCE buffer; None when short or the disk keeps no idle time."""
    if len(raw) < _DISK_PERFORMANCE.size:
        return None
    fields = _DISK_PERFORMANCE.unpack_from(raw)
    idle, query = fields[4], fields[9]
    return (idle, query) if idle > 0 and query > 0 else None


@functools.cache
def _kernel32() -> Any:
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        wintypes.HANDLE,
    ]  # fmt: skip
    k.CreateFileW.restype = wintypes.HANDLE
    k.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
    ]  # fmt: skip
    k.DeviceIoControl.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    return k


def _windows_disk_times(name: str) -> tuple[int, int] | None:
    """(IdleTime, QueryTime) of a Windows ``PhysicalDriveN``, the counters Task Manager's "active time" is made of.

    Opens ``\\\\.\\PhysicalDriveN`` with no access rights, as psutil does (no admin needed), and asks for
    IOCTL_DISK_PERFORMANCE. None off Windows or when the disk does not answer.
    """
    match = re.fullmatch(r"physicaldrive(\d+)", name.lower())
    if sys.platform != "win32" or not match:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k = _kernel32()
        handle = k.CreateFileW(rf"\\.\PhysicalDrive{match.group(1)}", 0, 0x1 | 0x2, None, 3, 0, None)
        if handle is None or handle == ctypes.c_void_p(-1).value:  # INVALID_HANDLE_VALUE
            return None
        try:
            buf = ctypes.create_string_buffer(256)  # larger than DISK_PERFORMANCE, as some drivers want
            returned = wintypes.DWORD()
            ok = k.DeviceIoControl(handle, _IOCTL_DISK_PERFORMANCE, None, 0, buf, len(buf), ctypes.byref(returned), None)
        finally:
            k.CloseHandle(handle)
        return _parse_disk_performance(buf.raw[: returned.value]) if ok else None
    except (OSError, AttributeError, ValueError):
        return None


def _busy_from_idle(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    """Percent of the time between two (IdleTime, QueryTime) readings the disk was not idle: 100 − % idle time."""
    if previous is None or current is None:
        return None
    idle, span = current[0] - previous[0], current[1] - previous[1]
    if span <= 0 or idle < 0:
        return None  # counters restarted
    return round(max(0.0, 100.0 * (1 - idle / span)), 1)


class MetricsSampler:
    """Collects one metrics snapshot per call; keeps state to turn counters into rates."""

    def __init__(self, disk_path: str | None = None) -> None:
        self.disk_path = disk_path or ("C:\\" if sys.platform == "win32" else "/")
        self._last_time: float | None = None
        self._last_io: dict[str, Any] = {}
        self._last_idle: dict[str, tuple[int, int] | None] = {}
        self._last_energy: int | None = None
        psutil.cpu_percent(interval=None)  # prime the counter

    def sample(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = (now - self._last_time) if self._last_time else None
        vm = psutil.virtual_memory()
        gpu_list = self._gpus()
        users = self._users(gpu_list)
        disks = self._disks()
        home = next((d for d in disks if self.disk_path.startswith(d["mount"])), disks[0] if disks else None)
        snapshot: dict[str, Any] = {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "load": self._load(),
            "cpu_temp_c": _cpu_temperature(),
            "cpu_power_w": self._cpu_power(elapsed),
            "memory_used_mb": (vm.total - vm.available) // (1024 * 1024),
            "memory_total_mb": vm.total // (1024 * 1024),
            "uptime_s": int(time.time() - psutil.boot_time()),
            "gpus": gpu_list,
            "disks": disks,
            "physical_disks": self._physical_disks(elapsed),
            "users": users,
            "sessions": self._sessions(),
            # kept for older Hubs / simple views
            "disk_free_gb": home["free_gb"] if home else None,
            "disk_total_gb": home["total_gb"] if home else None,
        }
        self._last_time = now
        return snapshot

    # ---- pieces ---------------------------------------------------------------

    @staticmethod
    def _load() -> list[float] | None:
        try:
            return [round(x, 2) for x in psutil.getloadavg()]
        except (OSError, AttributeError):
            return None

    def _cpu_power(self, elapsed: float | None) -> int | None:
        energy = _rapl_energy_uj() if sys.platform.startswith("linux") else None
        previous, self._last_energy = self._last_energy, energy
        if energy is None or previous is None or not elapsed or energy < previous:
            return None
        return round((energy - previous) / 1e6 / elapsed)

    @staticmethod
    def _gpus() -> list[dict[str, Any]]:
        processes = gpu_processes()
        result = []
        for gpu in gpus():
            procs = processes.get(gpu["uuid"], [])
            result.append(
                {
                    "index": gpu["index"],
                    "memory_used_mb": gpu["memory_used_mb"],
                    "memory_total_mb": gpu["memory_total_mb"],
                    "utilization_pct": gpu["utilization_pct"],
                    "temperature_c": gpu["temperature_c"],
                    "power_w": gpu["power_w"],
                    "power_limit_w": gpu["power_limit_w"],
                    "processes": procs,
                    "users": sorted({p["user"] for p in procs if p["user"] and p["user"] not in _SYSTEM_USERS}),
                }
            )
        return result

    @staticmethod
    def _disks() -> list[dict[str, Any]]:
        result, seen = [], set()
        for part in psutil.disk_partitions(all=False):
            if part.fstype.lower() in _SKIP_FSTYPES or part.device in seen or part.device.startswith("/dev/loop"):
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            if usage.total == 0:
                continue
            seen.add(part.device)
            result.append(
                {
                    "mount": part.mountpoint,
                    "device": part.device,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / 1024**3, 1),
                    "used_gb": round(usage.used / 1024**3, 1),
                    "free_gb": round(usage.free / 1024**3, 1),
                    "pct": round(usage.percent, 1),
                }
            )
        return result

    def _physical_disks(self, elapsed: float | None) -> list[dict[str, Any]]:
        try:
            counters = psutil.disk_io_counters(perdisk=True) or {}
        except (OSError, RuntimeError):
            return []
        result = []
        for name, io in sorted(counters.items()):
            if not _is_whole_disk(name):
                continue  # partitions and virtual devices
            previous = self._last_io.get(name)
            read = write = busy = None
            if previous and elapsed:
                read = round((io.read_bytes - previous.read_bytes) / elapsed / 1024**2, 2)
                write = round((io.write_bytes - previous.write_bytes) / elapsed / 1024**2, 2)
                if hasattr(io, "busy_time"):
                    busy = round(min(100.0, (io.busy_time - previous.busy_time) / (elapsed * 1000) * 100), 1)
            if not hasattr(io, "busy_time"):  # Windows: Task Manager's "active time" from the disk's idle counter
                times = _windows_disk_times(name)
                busy = _busy_from_idle(self._last_idle.get(name), times)
                self._last_idle[name] = times
            model, kind = _disk_kind(name)
            result.append(
                {"name": name, "model": model, "kind": kind, "read_mbps": read, "write_mbps": write, "busy_pct": busy}
            )
            self._last_io[name] = io
        return result

    @staticmethod
    def _users(gpu_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        usage: dict[str, dict[str, Any]] = {}
        for proc in psutil.process_iter(["username", "cpu_percent", "memory_info"]):
            info = proc.info
            user = _short_user(info.get("username"))
            if not user or user in _SYSTEM_USERS:
                continue
            entry = usage.setdefault(
                user, {"user": user, "cpu_cores": 0.0, "memory_mb": 0, "processes": 0, "gpus": [], "gpu_memory_mb": 0}
            )
            entry["cpu_cores"] += (info.get("cpu_percent") or 0.0) / 100.0
            entry["memory_mb"] += (info["memory_info"].rss // (1024 * 1024)) if info.get("memory_info") else 0
            entry["processes"] += 1
        for gpu in gpu_list:
            for p in gpu["processes"]:
                entry = usage.get(p["user"] or "")
                if entry is None:
                    continue
                if gpu["index"] not in entry["gpus"]:
                    entry["gpus"].append(gpu["index"])
                entry["gpu_memory_mb"] += p["memory_mb"] or 0
        result = []
        for entry in usage.values():
            entry["cpu_cores"] = round(entry["cpu_cores"], 1)
            entry["active"] = bool(entry["gpus"]) or entry["cpu_cores"] >= _ACTIVE_CPU_CORES
            result.append(entry)
        return sorted(result, key=lambda e: (not e["active"], -e["cpu_cores"], e["user"]))

    @staticmethod
    def _sessions() -> list[dict[str, Any]]:
        try:
            return [
                {"user": _short_user(u.name), "terminal": u.terminal, "host": u.host, "started": int(u.started)}
                for u in psutil.users()
            ]
        except (OSError, psutil.Error):
            return []


def metrics(path_for_disk: str | None = None) -> dict[str, Any]:
    """One-off snapshot (rates are None because there is no previous sample)."""
    return MetricsSampler(path_for_disk).sample()
