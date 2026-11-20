"""What the node screens show: every GPU on one board, and one server's GPUs over time.

The board answers "who holds which GPU now, and where can a job go": each GPU with what holds
it (a BADAMOA run with its progress, a person with their programs, nobody, or kept for people),
and the jobs waiting with the reason. The timeline answers "who used these GPUs, and when":
per GPU, its utilization over the range with the BADAMOA jobs (from the jobs table) and the
people (from ``GpuSpan``) that held it as labelled spans.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from badamoa.core.jobs import BUILTIN_JOB_TYPES

from . import resources, services
from .config import HubSettings
from .db import iso, utcnow
from .models import GpuSpan, Job, Node
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run

TIMELINE_RANGES = ("6h", "24h", "7d")
ACTION_LABEL = {"eval": "평가", "export": "내보내기", "predict": "테스트"}
QUEUED_SHOWN = 10


def _job_label(db: Session, job: Job, now: datetime) -> dict[str, Any]:
    """A job as people know it: the run or model it is for, where it leads, and how far along a training is."""
    if job.type == "train.run":
        run = db.scalar(select(Run).where(Run.job_id == job.id))
        if run is not None:
            entry: dict[str, Any] = {
                "label": run.name,
                "href": f"#/training/{run.id}",
                "epoch": run.epoch,
                "epochs": run.epochs,
            }
            if job.status == "running" and job.started_at and run.epoch and run.epochs and run.epochs > run.epoch:
                per_epoch = (now - job.started_at).total_seconds() / run.epoch
                entry["eta_s"] = round(per_epoch * (run.epochs - run.epoch))
            return entry
    if job.type == "model.run":
        op = db.scalar(select(ModelJob).where(ModelJob.job_id == job.id))
        model = db.get(ModelVersion, op.model_id) if op else None
        if op is not None and model is not None:
            return {
                "label": f"{ACTION_LABEL.get(op.kind, op.kind)} · {model.name} v{model.version}",
                "href": f"#/models/{model.id}",
            }
    spec = BUILTIN_JOB_TYPES.get(job.type)
    return {"label": spec.title if spec else job.type, "href": "#/jobs"}


MIN_DISK_GB = 2  # smaller partitions (EFI, boot) are not worth a meter


def board(db: Session, settings: HubSettings) -> dict[str, Any]:
    now = utcnow()
    nodes = db.scalars(select(Node).order_by(Node.is_local.desc(), Node.created_at)).all()
    assigned = resources.assignments(db)
    running = {j.id: j for j in db.scalars(select(Job).where(Job.status == "running"))}
    labels = {job_id: _job_label(db, job, now) for job_id, job in running.items()}
    counts = {"free": 0, "job": 0, "busy": 0, "reserved": 0, "offline": 0}
    shown = []
    for node in nodes:
        status = services.node_status(node, settings)
        metrics = node.metrics or {}
        live = {g.get("index"): g for g in metrics.get("gpus") or []}
        states = {s["index"]: s for s in resources.gpu_states(node, settings, assigned.get(node.id))}
        gpus = []
        for gpu in (node.info or {}).get("gpus") or []:
            index = gpu.get("index")
            current = live.get(index, {}) if status == "online" else {}
            state = states.get(index, {}).get("state", "free") if status == "online" else "offline"
            counts[state] = counts.get(state, 0) + 1
            processes = current.get("processes") or []
            occupant: dict[str, Any] | None = None
            if state == "job":
                job_id = states[index]["job_id"]
                occupant = {"kind": "job", "job_id": job_id, **labels.get(job_id, {})}
            elif processes or state == "busy":
                occupant = {
                    "kind": "person",
                    "who": ", ".join(current.get("users") or []),
                    "programs": ", ".join(sorted({str(p["name"]) for p in processes if p.get("name")})),
                    # light use beside which a job can still run (below the busy thresholds)
                    "light": state != "busy",
                }
            gpus.append(
                {
                    "index": index,
                    "name": gpu.get("name"),
                    "memory_total_mb": current.get("memory_total_mb") or gpu.get("memory_total_mb"),
                    "memory_used_mb": current.get("memory_used_mb"),
                    "utilization_pct": current.get("utilization_pct"),
                    "temperature_c": current.get("temperature_c"),
                    "power_w": current.get("power_w"),
                    "state": state,
                    "reserved": index in (node.reserved_gpus or []),
                    "occupant": occupant,
                }
            )
        disks = metrics.get("disks") or []
        fullest = max(disks, key=lambda d: d.get("pct") or 0, default=None)
        shown.append(
            {
                "id": node.id,
                "name": node.name,
                "status": status,
                "enabled": node.enabled,
                "is_local": node.is_local,
                "agent_version": node.agent_version or "",
                "roles": node.roles,
                "gpu_name": gpus[0]["name"] if gpus else None,
                "cpu_pct": metrics.get("cpu_pct"),
                "memory_pct": round(metrics["memory_used_mb"] / metrics["memory_total_mb"] * 100)
                if metrics.get("memory_used_mb") is not None and metrics.get("memory_total_mb")
                else None,
                "disk": {"mount": fullest.get("mount"), "pct": fullest.get("pct"), "free_gb": fullest.get("free_gb")}
                if fullest
                else None,
                # What the server has besides GPUs, shown beside them: processor, memory, every disk.
                "cpu": {
                    "pct": metrics.get("cpu_pct"),
                    "cores": (node.info or {}).get("cpu_count"),
                    "model": (node.info or {}).get("cpu_model"),
                },
                "memory": {
                    "used_gb": round(metrics["memory_used_mb"] / 1024, 1),
                    "total_gb": round(metrics["memory_total_mb"] / 1024, 1),
                }
                if metrics.get("memory_used_mb") is not None and metrics.get("memory_total_mb")
                else None,
                "disks": [
                    {k: d.get(k) for k in ("mount", "total_gb", "free_gb", "pct")}
                    for d in sorted(disks, key=lambda d: str(d.get("mount")))
                    if (d.get("total_gb") or 0) >= MIN_DISK_GB
                ],
                # Images taken from other nodes: where they are kept (as set, and in use), and this node's data port.
                "cache_dir": node.cache_dir or "",
                "cache": metrics.get("cache") or None,
                "serve": metrics.get("serve") or None,
                # the account the node runs as: one without root (installed without sudo) is marked on the board
                "os": node.os,
                "account": (node.info or {}).get("account") or None,
                "last_seen_at": iso(node.last_seen_at),
                "gpus": gpus,
            }
        )
    queued = db.scalars(select(Job).where(Job.status == "queued").order_by(Job.created_at)).all()
    waiting = [
        {
            "job_id": job.id,
            **_job_label(db, job, now),
            "gpus": (job.requirements or {}).get("gpus", 0),
            "min_vram_mb": (job.requirements or {}).get("min_vram_mb", 0),
            "reason": services.explain_wait(db, settings, job),
            "created_at": iso(job.created_at),
        }
        for job in queued[:QUEUED_SHOWN]
    ]
    # the BADAMOA training that should free its GPUs first, for "when could a waiting job start"
    names = {n.id: n.name for n in nodes}
    ends = [
        {
            "node": names.get(running[job_id].node_id or "", ""),
            "label": label["label"],
            "eta_s": label["eta_s"],
            "gpus": running[job_id].gpu_indices or [],
        }
        for job_id, label in labels.items()
        if label.get("eta_s") is not None and running[job_id].gpu_indices
    ]
    return {
        "hub_version": services.HUB_VERSION,
        "nodes": shown,
        "counts": {**counts, "waiting": len(queued)},
        "waiting": waiting,
        "next_free": min(ends, key=lambda e: e["eta_s"]) if ends else None,
    }


def timeline(db: Session, settings: HubSettings, node: Node, range_name: str = "24h") -> dict[str, Any]:
    if range_name not in TIMELINE_RANGES:
        raise ServiceError(400, f"range는 {', '.join(TIMELINE_RANGES)} 중 하나여야 합니다.")
    now = utcnow()
    history = resources.history(db, node.id, range_name, now)
    start = now - resources.RANGES[range_name][1]
    jobs = db.scalars(
        select(Job).where(
            Job.node_id == node.id,
            Job.started_at.is_not(None),
            Job.started_at < now,
            or_(Job.finished_at.is_(None), Job.finished_at > start),
        )
    ).all()
    people = db.scalars(
        select(GpuSpan).where(GpuSpan.node_id == node.id, GpuSpan.ended_at >= start, GpuSpan.started_at <= now)
    ).all()
    gpus = []
    for gpu in (node.info or {}).get("gpus") or []:
        index = gpu.get("index")
        spans = [
            {
                "kind": "job",
                "job_id": job.id,
                "status": job.status,
                "start": iso(max(job.started_at, start)),
                "end": iso(min(job.finished_at or now, now)),
                **_job_label(db, job, now),
            }
            for job in jobs
            if index in (job.gpu_indices or [])
        ]
        spans += [
            {
                "kind": "person",
                "who": span.who,
                "programs": span.programs,
                "memory_mb": span.memory_mb,
                "start": iso(max(span.started_at, start)),
                "end": iso(span.ended_at),
            }
            for span in people
            if span.gpu_index == index
        ]
        gpus.append(
            {
                "index": index,
                "name": gpu.get("name"),
                "memory_total_mb": gpu.get("memory_total_mb"),
                "reserved": index in (node.reserved_gpus or []),
                "util": history["series"].get(f"gpu{index}_util", {}).get("avg", []),
                "spans": sorted(spans, key=lambda s: s["start"]),
            }
        )
    return {
        "range": range_name,
        "from": iso(start),
        "to": iso(now),
        "step_s": history["step_s"],
        "timestamps": history["timestamps"],
        "gpus": gpus,
    }
