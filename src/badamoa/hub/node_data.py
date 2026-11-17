"""Copies of versions on the nodes that received them: what is where, keeping or removing, and checking.

A node that trains on a version registered on another node first receives its images into its
cache, by content (``transfers``); the Hub remembers that copy (``NodeRelease``). Here:

* **seeing** them: per version, the servers that hold it (where it was registered, where copies
  were received) with their size, how often and when they last served a run, and what uses them
  now; per node, every copy it keeps;
* **removing** a copy: a run can ask for the copy its server received to go once it succeeded,
  people remove copies from the version's or the node's screen, and a version removed for good
  takes its copies along. Never while a job on that server uses the version;
* **checking**: a ``data.check`` job has the node compare its cache with what the Hub thinks it
  keeps. It reports what is there of each copy, and with ``clean`` removes the files no kept copy
  needs. Removing a copy is just that: forget it here, then clean — so an image several versions
  share stays as long as one of them does.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import services
from .catalog_models import Dataset, Item, Project, Release, ReleaseItem
from .db import iso, utcnow
from .models import Job, Node, NodeRelease, User
from .services import ServiceError
from .train_models import ModelJob, Run, Sweep

CHECK_SINCE = (0, 42, 0)  # nodes from here on check and clean their received copies
ACTIVE = ("queued", "running")
USING = ("data.sync", "train.run", "model.run")  # jobs that read a version's images on a node
PAGE = 5000
CHOICES = ("", "keep", "delete")  # what a run does with the copy its server received, once it is over


# ---- who uses a copy ---------------------------------------------------------------------------


def _knows_check(node: Node) -> bool:
    try:
        return tuple(int(p) for p in (node.agent_version or "").split(".")[:3]) >= CHECK_SINCE
    except ValueError:
        return False


def _require_check(node: Node) -> None:
    if not _knows_check(node):
        need = ".".join(str(n) for n in CHECK_SINCE)
        version = node.agent_version or "버전 모름"
        why = f"받아 둔 이미지를 정리할 수 없습니다. {need} 이상으로 업데이트하세요"
        raise ServiceError(400, f"{node.name}({version})는 {why}.")
    if not node.enabled:
        raise ServiceError(400, f"{node.name}은 작업을 받지 않도록 꺼 두었습니다.")


def _on_node(job: Job, node_id: str) -> bool:
    """The job runs on that node, or may still be placed there."""
    if job.node_id:
        return job.node_id == node_id
    return job.target_node_id == node_id or node_id in ((job.requirements or {}).get("nodes") or [])


def users(db: Session, node_id: str, release_id: str) -> list[str]:
    """What uses a version's images on a node now or is about to, e.g. ["학습 street-v1-yolo11n", "탐색 lr"]."""
    found: list[str] = []
    jobs = db.scalars(select(Job).where(Job.type.in_(USING), Job.status.in_(ACTIVE))).all()
    for job in jobs:
        if (job.params or {}).get("release_id") != release_id or not _on_node(job, node_id):
            continue
        if job.type == "train.run":
            run = db.scalar(select(Run).where(Run.job_id == job.id))
            found.append(f"학습 {run.name}" if run else "학습")
        else:
            found.append("평가" if job.type == "model.run" else "이미지 받기")
    for sweep in db.scalars(select(Sweep).where(Sweep.release_id == release_id, Sweep.status == "running")):
        if node_id in ((sweep.settings or {}).get("nodes") or []):
            found.append(f"탐색 {sweep.name}")
    return list(dict.fromkeys(found))


def _usage(db: Session, node_id: str, release_ids: list[str]) -> dict[str, tuple[int, Any]]:
    """Per version: how many runs and evaluations it served on the node, and when the last one did."""
    if not release_ids:
        return {}
    when = func.max(func.coalesce(Job.finished_at, Job.started_at))
    out: dict[str, tuple[int, Any]] = {}
    queries = (
        select(Run.release_id, func.count(Run.id), when)
        .join(Job, Job.id == Run.job_id)
        .where(Job.node_id == node_id, Run.release_id.in_(release_ids))
        .group_by(Run.release_id),
        select(ModelJob.release_id, func.count(ModelJob.id), when)
        .join(Job, Job.id == ModelJob.job_id)
        .where(Job.node_id == node_id, ModelJob.release_id.in_(release_ids))
        .group_by(ModelJob.release_id),
    )
    for query in queries:
        for release_id, count, last in db.execute(query):
            runs, seen = out.get(release_id, (0, None))
            latest = [x for x in (seen, last) if x is not None]
            out[release_id] = (runs + int(count or 0), max(latest) if latest else None)
    return out


# ---- what the screens show ---------------------------------------------------------------------


def _copy(db: Session, settings: Any, row: NodeRelease, node: Node | None, usage: tuple[int, Any]) -> dict[str, Any]:
    using = users(db, row.node_id, row.release_id)
    why = ""
    if row.state == "removing":
        why = "지우는 중"
    elif using:
        why = f"{', '.join(using)}이(가) 쓰는 중"
    elif node is not None and not _knows_check(node):
        why = f"{'.'.join(str(n) for n in CHECK_SINCE)} 이상으로 업데이트해야 지울 수 있음"
    return {
        "node_id": row.node_id,
        "node_name": node.name if node else "삭제된 노드",
        "node_status": services.node_status(node, settings) if node else "offline",
        "items": row.items,
        "bytes": row.bytes,
        "received_at": iso(row.synced_at),
        "state": row.state or "kept",
        "present_items": row.present_items,
        "present_bytes": row.present_bytes,
        "checked_at": iso(row.checked_at),
        "runs": usage[0],
        "last_used_at": iso(usage[1]),
        "in_use": using,
        "removable": not why,
        "why_not": why,
    }


def release_copies(db: Session, settings: Any, release: Release) -> dict[str, Any]:
    """Where a version's images are: the server they were registered on, and the copies other servers received."""
    from . import training  # training imports this module

    try:
        node = training.data_node(db, release.id)
        origin = {"node_id": node.id, "node_name": node.name, "node_status": services.node_status(node, settings)}
    except ServiceError:
        origin = None
    rows = db.scalars(select(NodeRelease).where(NodeRelease.release_id == release.id)).all()
    copies = []
    for row in rows:
        usage = _usage(db, row.node_id, [release.id]).get(release.id, (0, None))
        copies.append(_copy(db, settings, row, db.get(Node, row.node_id), usage))
    copies.sort(key=lambda c: c["node_name"])
    return {"release_id": release.id, "digest": release.digest, "origin": origin, "copies": copies}


def node_copies(db: Session, settings: Any, node: Node) -> dict[str, Any]:
    """Every copy a node keeps, with the version and dataset it belongs to, and its last check."""
    rows = db.scalars(select(NodeRelease).where(NodeRelease.node_id == node.id)).all()
    usage = _usage(db, node.id, [r.release_id for r in rows])
    copies = []
    for row in rows:
        release = db.get(Release, row.release_id)
        dataset = db.get(Dataset, release.dataset_id) if release else None
        project = db.get(Project, dataset.project_id) if dataset else None
        copy = _copy(db, settings, row, node, usage.get(row.release_id, (0, None)))
        copy["release"] = {
            "id": row.release_id,
            "version": release.version if release else "(지워진 버전)",
            "digest": release.digest if release else "",
            "dataset": dataset.name if dataset else "",
            "project_id": dataset.project_id if dataset else "",
            "project": project.name if project else "",
            "deleted": bool(release is None or release.deleted_at is not None),
        }
        copies.append(copy)
    copies.sort(key=lambda c: (c["release"]["project"], c["release"]["dataset"], c["release"]["version"]))
    check = dict(node.data_check or {})
    active = db.scalar(
        select(Job.id).where(Job.type == "data.check", Job.status.in_(ACTIVE), Job.target_node_id == node.id).limit(1)
    )
    return {
        "node_id": node.id,
        "copies": copies,
        "bytes": sum(c["bytes"] for c in copies),
        "check": {**check, "running": bool(active)},
        "can_check": _knows_check(node) and node.enabled,
    }


# ---- removing and checking -----------------------------------------------------------------------


def _check_job(db: Session, node: Node, clean: bool, user_id: str | None, via: str) -> Job:
    """A ``data.check`` job for the node (made here directly: the Hub starts some without a person)."""
    job = Job(
        type="data.check",
        params={"clean": clean},
        target_node_id=node.id,
        requirements={},
        created_by=user_id,
        created_via=via,
    )
    db.add(job)
    db.flush()
    services.audit(db, user_id, via, "job.create", job.id, type="data.check", node_id=node.id, clean=clean)
    return job


def remove_copy(db: Session, user: User, release_id: str, node_id: str, via: str = "api") -> NodeRelease:
    """Remove the copy of a version a node received. The images the node's other copies share stay."""
    row = db.get(NodeRelease, (node_id, release_id))
    if row is None:
        raise ServiceError(404, "이 서버에 받아 둔 이 버전의 이미지가 없습니다.")
    node = services.get_node(db, node_id)
    if row.state == "removing":
        return row
    _require_check(node)
    using = users(db, node_id, release_id)
    if using:
        raise ServiceError(409, f"{', '.join(using)}이(가) 이 서버에서 이 버전을 쓰는 중입니다. 끝난 뒤에 지우세요.")
    job = _check_job(db, node, True, user.id, via)
    row.state, row.job_id = "removing", job.id
    services.audit(db, user.id, via, "copy.remove", release_id, node_id=node_id, bytes=row.bytes)
    return row


def check_node(db: Session, user: User, node_id: str, clean: bool = False, via: str = "api") -> Job:
    """Have a node compare its received copies with the Hub's records (and with ``clean`` drop what none needs)."""
    node = services.get_node(db, node_id)
    _require_check(node)
    for job in db.scalars(select(Job).where(Job.type == "data.check", Job.status == "queued")):
        if job.target_node_id == node.id and bool((job.params or {}).get("clean")) == clean:
            return job  # one waiting already does the same
    return _check_job(db, node, clean, user.id, via)


def settle(db: Session, node_id: str, release_id: str, user_id: str | None = None) -> None:
    """A copy marked to go: remove it once nothing on that node uses the version."""
    row = db.get(NodeRelease, (node_id, release_id))
    if row is None or row.state != "remove" or users(db, node_id, release_id):
        return
    node = db.get(Node, node_id)
    if node is None or not node.enabled or not _knows_check(node):
        return  # stays marked; it goes when someone removes it after the node is updated or switched on
    job = _check_job(db, node, True, user_id, "hub")
    row.state, row.job_id = "removing", job.id


def forget(db: Session, release_ids: list[str], user_id: str | None = None, via: str = "hub") -> int:
    """Versions removed for good: their copies go from every node that kept one."""
    rows = db.scalars(select(NodeRelease).where(NodeRelease.release_id.in_(release_ids))).all() if release_ids else []
    nodes = {row.node_id for row in rows}
    for row in rows:
        db.delete(row)
    db.flush()
    for node_id in nodes:
        node = db.get(Node, node_id)
        if node is not None and node.enabled and _knows_check(node):
            _check_job(db, node, True, user_id, via)
    return len(rows)


# ---- when jobs end -------------------------------------------------------------------------------


def _after_run(db: Session, job: Job) -> None:
    """A training ended: what it asked for the copy its server received."""
    run = db.scalar(select(Run).where(Run.job_id == job.id))
    if run is None or run.sweep_id or not job.node_id:
        return  # a search's runs leave it to the search (``after_sweep``)
    choice = (run.config or {}).get("data_after", "")
    row = db.get(NodeRelease, (job.node_id, run.release_id))
    if row is None or not choice:
        return  # registered there, or never received there
    if choice == "keep":
        if row.state in ("remove", "held"):
            row.state = ""
        return
    if job.status != "succeeded":
        if row.state == "":
            row.state = "held"  # kept so the run can continue; removed from the screens when no longer wanted
        return
    for other in db.scalars(select(Job).where(Job.type == "train.run", Job.status.in_(ACTIVE), Job.id != job.id)):
        later = db.scalar(select(Run).where(Run.job_id == other.id))
        if (
            later is not None
            and later.release_id == run.release_id
            and (later.config or {}).get("data_after") == "keep"
            and _on_node(other, job.node_id)
        ):
            return  # a run still to come there wants it kept
    if row.state != "removing":
        row.state = "remove"


def after_sweep(db: Session, sweep: Sweep) -> None:
    """A search finished: if it asked to, the copies its runs' servers received go now."""
    if (sweep.settings or {}).get("data_after") != "delete":
        return
    nodes = db.scalars(
        select(Job.node_id)
        .join(Run, Run.job_id == Job.id)
        .where(Run.sweep_id == sweep.id, Job.node_id.is_not(None))
        .distinct()
    ).all()
    for node_id in nodes:
        row = db.get(NodeRelease, (node_id, sweep.release_id))
        if row is not None and row.state != "removing":
            row.state = "remove"
            settle(db, node_id, sweep.release_id, sweep.created_by)


def _checked(db: Session, job: Job) -> None:
    """What a node's check found: the copies it was removing go, the others get their counts."""
    node = db.get(Node, job.node_id or job.target_node_id or "")
    if node is None:
        return
    now = utcnow()
    result = job.result or {}
    removing = db.scalars(
        select(NodeRelease).where(
            NodeRelease.node_id == node.id, NodeRelease.job_id == job.id, NodeRelease.state == "removing"
        )
    ).all()
    if job.status != "succeeded":
        for row in removing:
            row.state, row.job_id = "remove", None  # tried again when a job there ends or someone removes it
        node.data_check = {
            **(node.data_check or {}),
            "job_id": job.id,
            "error": job.error or "확인 작업이 끝나지 못했습니다.",
            "failed_at": iso(now),
        }
        return
    for row in removing:
        db.delete(row)
    for release_id, found in (result.get("releases") or {}).items():
        row = db.get(NodeRelease, (node.id, release_id))
        if row is None or row.state == "removing":
            continue
        items, size = int(found.get("items") or 0), int(found.get("bytes") or 0)
        if items == 0 and row.items:
            db.delete(row)  # nothing of it is there any more (the cache moved, or someone emptied it)
            continue
        row.present_items, row.present_bytes, row.checked_at = items, size, now
    node.data_check = {
        "job_id": job.id,
        "checked_at": iso(now),
        "clean": bool((job.params or {}).get("clean")),
        "unknown": result.get("unknown") or {"files": 0, "bytes": 0},
        "removed": result.get("removed") or {"files": 0, "bytes": 0},
        "error": None,
    }


def on_job_finished(db: Session, job: Job) -> None:
    if job.type == "data.check":
        _checked(db, job)
        return
    if job.type == "train.run":
        _after_run(db, job)
    release_id = str((job.params or {}).get("release_id") or "")
    if job.type in USING and job.node_id and release_id:
        settle(db, job.node_id, release_id, job.created_by)


# ---- the checking node ---------------------------------------------------------------------------


def expected(db: Session, node: Node, job_id: str, release_id: str = "", after: int = 0) -> dict[str, Any]:
    """For a node's check: which versions it should hold, then page by page their images.

    The first call lists the copies to check (``check``) and the versions jobs there use now
    (``keep``: their files are not counted, only spared). Then one version at a time.
    """
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.status != "running" or job.type != "data.check":
        raise ServiceError(404, "이 노드에서 실행 중인 확인 작업이 아닙니다.")
    if not release_id:
        kept = db.scalars(
            select(NodeRelease).where(NodeRelease.node_id == node.id, NodeRelease.state != "removing")
        ).all()
        check = [row.release_id for row in kept]
        busy = {
            str((j.params or {}).get("release_id"))
            for j in db.scalars(select(Job).where(Job.type.in_(USING), Job.status.in_(ACTIVE)))
            if (j.params or {}).get("release_id") and _on_node(j, node.id)
        }
        return {"check": check, "keep": sorted(busy - set(check))}
    limit = PAGE
    rows = db.execute(
        select(Item.id, Item.sha256, Item.bytes, Item.rel_path)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release_id, Item.id > after)
        .order_by(Item.id)
        .limit(limit)
    ).all()
    return {
        "items": [{"sha256": sha, "bytes": size, "ext": PurePosixPath(rel).suffix.lower()} for _, sha, size, rel in rows],
        "next": rows[-1][0] if len(rows) == limit else None,
    }
