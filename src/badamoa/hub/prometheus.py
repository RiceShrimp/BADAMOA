"""``/metrics`` in the Prometheus text format, for teams that already run Prometheus and Grafana.

Nothing is collected or kept for it: each scrape is written from what the Hub already holds
(the newest snapshot of every node, the GPU states, the job queue). With no scraper nothing
runs. A scraper signs in like any client, with an API token (``Authorization: Bearer …``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa import __version__

from . import resources, services
from .config import HubSettings
from .models import Job, Node

MB = 1024 * 1024
GB = 1024**3
GPU_STATES = ("free", "job", "busy", "reserved", "offline")


def _label(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.declared: set[str] = set()

    def metric(self, name: str, kind: str, help_text: str, samples: Iterable[tuple[dict[str, Any], float | None]]) -> None:
        rows = [(labels, value) for labels, value in samples if value is not None]
        if name not in self.declared:
            self.declared.add(name)
            self.lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
        for labels, value in rows:
            body = ",".join(f'{k}="{_label(v)}"' for k, v in labels.items())
            self.lines.append(f"{name}{{{body}}} {float(value):g}" if body else f"{name} {float(value):g}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def render(db: Session, settings: HubSettings) -> str:
    out = _Writer()
    out.metric("badamoa_info", "gauge", "The Hub's version.", [({"version": __version__}, 1)])
    nodes = list(db.scalars(select(Node).order_by(Node.name)))
    assigned = resources.assignments(db)
    up, cpu, mem_used, mem_total, uptime = [], [], [], [], []
    gpu_util, gpu_mem, gpu_total, gpu_temp, gpu_power, gpu_state = [], [], [], [], [], []
    disk_used, disk_total = [], []
    for node in nodes:
        status = services.node_status(node, settings)
        base = {"node": node.name}
        up.append((base, 1 if status == "online" else 0))
        if status != "online":
            for gpu in (node.info or {}).get("gpus") or []:
                gpu_state.append(({**base, "gpu": gpu.get("index"), "state": "offline"}, 1))
            continue
        m = node.metrics or {}
        cpu.append((base, m.get("cpu_pct")))
        mem_used.append((base, (m.get("memory_used_mb") or 0) * MB if m.get("memory_used_mb") is not None else None))
        mem_total.append((base, (m.get("memory_total_mb") or 0) * MB if m.get("memory_total_mb") is not None else None))
        uptime.append((base, m.get("uptime_s")))
        names = {g.get("index"): g.get("name") for g in (node.info or {}).get("gpus") or []}
        for gpu in m.get("gpus") or []:
            labels = {**base, "gpu": gpu.get("index"), "name": names.get(gpu.get("index"), "")}
            gpu_util.append((labels, gpu.get("utilization_pct")))
            used, total = gpu.get("memory_used_mb"), gpu.get("memory_total_mb")
            gpu_mem.append((labels, used * MB if used is not None else None))
            gpu_total.append((labels, total * MB if total is not None else None))
            gpu_temp.append((labels, gpu.get("temperature_c")))
            gpu_power.append((labels, gpu.get("power_w")))
        for state in resources.gpu_states(node, settings, assigned.get(node.id)):
            gpu_state.append(({**base, "gpu": state["index"], "state": state["state"]}, 1))
        for disk in m.get("disks") or []:
            labels = {**base, "mount": disk.get("mount")}
            used, total = disk.get("used_gb"), disk.get("total_gb")
            disk_used.append((labels, used * GB if used is not None else None))
            disk_total.append((labels, total * GB if total is not None else None))

    out.metric("badamoa_node_up", "gauge", "1 when the node answered recently.", up)
    out.metric("badamoa_node_cpu_percent", "gauge", "CPU use of the node.", cpu)
    out.metric("badamoa_node_memory_used_bytes", "gauge", "Memory in use.", mem_used)
    out.metric("badamoa_node_memory_total_bytes", "gauge", "Memory installed.", mem_total)
    out.metric("badamoa_node_uptime_seconds", "gauge", "Time since the node started.", uptime)
    out.metric("badamoa_gpu_utilization_percent", "gauge", "GPU utilization.", gpu_util)
    out.metric("badamoa_gpu_memory_used_bytes", "gauge", "GPU memory in use.", gpu_mem)
    out.metric("badamoa_gpu_memory_total_bytes", "gauge", "GPU memory installed.", gpu_total)
    out.metric("badamoa_gpu_temperature_celsius", "gauge", "GPU temperature.", gpu_temp)
    out.metric("badamoa_gpu_power_watts", "gauge", "GPU power draw.", gpu_power)
    out.metric(
        "badamoa_gpu_state",
        "gauge",
        "1 for what holds the GPU now: free, job (a BADAMOA job), busy (someone else), reserved (kept for people), offline.",
        gpu_state,
    )
    out.metric("badamoa_disk_used_bytes", "gauge", "Disk space in use per mount.", disk_used)
    out.metric("badamoa_disk_total_bytes", "gauge", "Disk size per mount.", disk_total)

    counts = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
    out.metric(
        "badamoa_jobs",
        "gauge",
        "Jobs by status (queued and running are the ones to watch).",
        [({"status": s}, counts.get(s, 0)) for s in ("queued", "running", "succeeded", "failed", "canceled")],
    )
    return out.text()


__all__ = ["GPU_STATES", "render"]
