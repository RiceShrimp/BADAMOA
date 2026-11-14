"""Node resources: metric history, per-GPU occupancy, and disk forecasts.

Servers are shared with people who log in and run work by hand, so the Hub never
assumes it owns a GPU: a GPU is free only if no running BADAMOA job reserved it
*and* its current load says nobody else is using it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, NodeMetric

RESOLUTIONS = {"1m": 60, "10m": 600, "1h": 3600}
RETENTION = {"1m": timedelta(hours=24), "10m": timedelta(days=7), "1h": timedelta(days=30)}
# range → (resolution, span)
RANGES: dict[str, tuple[str, timedelta]] = {
    "1h": ("1m", timedelta(hours=1)),
    "6h": ("1m", timedelta(hours=6)),
    "24h": ("10m", timedelta(hours=24)),
    "7d": ("1h", timedelta(days=7)),
    "30d": ("1h", timedelta(days=30)),
}


# ---- metric history -----------------------------------------------------------


def flatten(metrics: dict[str, Any]) -> dict[str, float]:
    """Turn one heartbeat snapshot into scalar series (see docs/SCREEN_SPECS.md 3.2)."""
    series: dict[str, float] = {}

    def put(key: str, value: Any) -> None:
        if isinstance(value, int | float) and not isinstance(value, bool):
            series[key] = float(value)

    def pct(used: Any, total: Any) -> float | None:
        return used / total * 100 if isinstance(used, int | float) and total else None

    put("cpu", metrics.get("cpu_pct"))
    put("mem", pct(metrics.get("memory_used_mb"), metrics.get("memory_total_mb")))
    put("cpu_power", metrics.get("cpu_power_w"))

    gpus = metrics.get("gpus") or []
    gpu_power = [g["power_w"] for g in gpus if isinstance(g.get("power_w"), int | float)]
    for gpu in gpus:
        i = gpu.get("index")
        put(f"gpu{i}_util", gpu.get("utilization_pct"))
        put(f"gpu{i}_mem", pct(gpu.get("memory_used_mb"), gpu.get("memory_total_mb")))
        put(f"gpu{i}_temp", gpu.get("temperature_c"))
        put(f"gpu{i}_power", gpu.get("power_w"))
    utils = [g["utilization_pct"] for g in gpus if isinstance(g.get("utilization_pct"), int | float)]
    if utils:
        put("gpu_util", sum(utils) / len(utils))
    if gpu_power:
        put("gpu_power", sum(gpu_power))
    if gpu_power or "cpu_power" in series:
        put("power", sum(gpu_power) + series.get("cpu_power", 0.0))

    for disk in metrics.get("disks") or []:
        put(f"disk_pct:{disk.get('mount')}", disk.get("pct"))
        put(f"disk_used:{disk.get('mount')}", disk.get("used_gb"))
    for disk in metrics.get("physical_disks") or []:
        put(f"io_busy:{disk.get('name')}", disk.get("busy_pct"))
        put(f"io_read:{disk.get('name')}", disk.get("read_mbps"))
        put(f"io_write:{disk.get('name')}", disk.get("write_mbps"))
    return series


def _bucket(now: datetime, seconds: int) -> datetime:
    epoch = int((now - datetime(1970, 1, 1)).total_seconds())
    return now - timedelta(seconds=epoch % seconds, microseconds=now.microsecond)


def record(db: Session, node_id: str, metrics: dict[str, Any], now: datetime | None = None) -> None:
    """Fold one snapshot into the current bucket of every resolution."""
    values = flatten(metrics)
    if not values:
        return
    now = now or utcnow()
    for res, seconds in RESOLUTIONS.items():
        ts = _bucket(now, seconds)
        row = db.scalar(
            select(NodeMetric).where(NodeMetric.node_id == node_id, NodeMetric.res == res, NodeMetric.ts == ts)
        )
        if row is None:
            row = NodeMetric(node_id=node_id, res=res, ts=ts, data={})
            db.add(row)
        data = dict(row.data or {})
        for key, value in values.items():
            total, count, peak = data.get(key, (0.0, 0, value))
            data[key] = [total + value, count + 1, max(peak, value)]
        row.data = data


def history(db: Session, node_id: str, range_name: str, now: datetime | None = None) -> dict[str, Any]:
    if range_name not in RANGES:
        raise ValueError(f"range는 {list(RANGES)} 중 하나여야 합니다.")
    res, span = RANGES[range_name]
    now = now or utcnow()
    rows = db.scalars(
        select(NodeMetric)
        .where(NodeMetric.node_id == node_id, NodeMetric.res == res, NodeMetric.ts >= now - span)
        .order_by(NodeMetric.ts)
    ).all()
    keys = sorted({key for row in rows for key in row.data})
    series: dict[str, dict[str, list[float | None]]] = {key: {"avg": [], "max": []} for key in keys}
    for row in rows:
        for key in keys:
            cell = row.data.get(key)
            series[key]["avg"].append(round(cell[0] / cell[1], 2) if cell and cell[1] else None)
            series[key]["max"].append(round(cell[2], 2) if cell else None)
    return {
        "range": range_name,
        "resolution": res,
        "step_s": RESOLUTIONS[res],
        "from": iso(now - span),
        "to": iso(now),
        "timestamps": [iso(row.ts) for row in rows],
        "series": series,
    }


def summarize(db: Session, node_id: str, range_name: str) -> dict[str, Any]:
    """Average and peak per series over a range (compact form for agents)."""
    data = history(db, node_id, range_name)
    summary = {}
    for key, values in data["series"].items():
        avg = [v for v in values["avg"] if v is not None]
        peak = [v for v in values["max"] if v is not None]
        if avg:
            summary[key] = {"avg": round(sum(avg) / len(avg), 1), "max": round(max(peak), 1), "last": avg[-1]}
    return {"range": range_name, "samples": len(data["timestamps"]), "series": summary}


def prune(db: Session, now: datetime | None = None) -> int:
    now = now or utcnow()
    removed = 0
    for res, keep in RETENTION.items():
        result = db.execute(delete(NodeMetric).where(NodeMetric.res == res, NodeMetric.ts < now - keep))
        removed += result.rowcount or 0
    return removed


def disk_forecast(db: Session, node: Node, now: datetime | None = None) -> list[dict[str, Any]]:
    """Per mount: growth over the last 7 days and when it would fill at that rate."""
    now = now or utcnow()
    rows = db.scalars(
        select(NodeMetric)
        .where(NodeMetric.node_id == node.id, NodeMetric.res == "1h", NodeMetric.ts >= now - timedelta(days=7))
        .order_by(NodeMetric.ts)
    ).all()
    result = []
    for disk in (node.metrics or {}).get("disks") or []:
        key = f"disk_used:{disk.get('mount')}"
        oldest = next((row for row in rows if row.data.get(key)), None)
        entry = {"mount": disk.get("mount"), "history_days": 0.0, "delta_7d_gb": None, "days_until_full": None}
        if oldest is not None:
            days = (now - oldest.ts).total_seconds() / 86400
            entry["history_days"] = round(days, 1)
            if days >= 0.5:  # less history than this says nothing about a trend
                cell = oldest.data[key]
                per_day = (disk.get("used_gb", 0.0) - cell[0] / cell[1]) / days
                entry["delta_7d_gb"] = round(per_day * 7, 1)
                if per_day > 0.01 and disk.get("free_gb") is not None:
                    entry["days_until_full"] = round(disk["free_gb"] / per_day)
        result.append(entry)
    return result


# ---- GPU occupancy --------------------------------------------------------------


def assignments(db: Session) -> dict[str, dict[int, str]]:
    """node_id → {gpu index: job id} for running jobs."""
    result: dict[str, dict[int, str]] = {}
    for job in db.scalars(select(Job).where(Job.status == "running")):
        if job.node_id:
            for index in job.gpu_indices or []:
                result.setdefault(job.node_id, {})[index] = job.id
    return result


def gpu_states(node: Node, settings: HubSettings, assigned: dict[int, str] | None = None) -> list[dict[str, Any]]:
    """State of every GPU on a node: ``job`` (reserved by BADAMOA), ``busy`` (someone else), ``free``."""
    assigned = assigned or {}
    live = {g.get("index"): g for g in (node.metrics or {}).get("gpus") or []}
    states = []
    for gpu in (node.info or {}).get("gpus") or []:
        index = gpu.get("index")
        now = live.get(index, {})
        used, total = now.get("memory_used_mb"), now.get("memory_total_mb") or gpu.get("memory_total_mb")
        memory_pct = used / total * 100 if isinstance(used, int | float) and total else 0.0
        util = now.get("utilization_pct") or 0
        if index in assigned:
            state = "job"
        elif memory_pct >= settings.gpu_busy_memory_pct or util >= settings.gpu_busy_util_pct:
            state = "busy"
        else:
            state = "free"
        states.append(
            {
                "index": index,
                "state": state,
                "job_id": assigned.get(index),
                "users": now.get("users") or [],
                "memory_total_mb": total,
            }
        )
    return states


def free_gpus(
    node: Node, settings: HubSettings, assigned: dict[int, str] | None = None, min_vram_mb: int = 0
) -> list[int]:
    return [
        g["index"]
        for g in gpu_states(node, settings, assigned)
        if g["state"] == "free" and (g["memory_total_mb"] or 0) >= min_vram_mb
    ]
