"""Moving a version's images from the node that holds them straight to the node that trains on them.

Images are registered where they lie and can add up to hundreds of GB, so they never pass
through the Hub. The Hub only arranges the move:

* a run may name the nodes it is allowed on. A node that already has the images (the one they
  were registered on, or one that received them before) takes it as it is; otherwise one of the
  named nodes first receives them in a ``data.sync`` job, which holds no GPU, and the training
  waits for it (``requirements.after``);
* the receiving node asks for the list of images and a grant (``plan``) and fetches them from
  the holding node's own little server, in tar batches;
* the holding node serves only what a grant covers: it asks the Hub which files a grant lets it
  send (``resolve``), and only while the job the grant was made for still runs.

The receiving node checks every file against its registered size and SHA-256 and keeps it in a
cache by content, so a later version that shares images fetches only the new ones.
"""

from __future__ import annotations

import hashlib
import ipaddress
import secrets
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import services
from .catalog_models import Item, LabelSetItem, Release, ReleaseItem, StorageLocation
from .db import utcnow
from .models import Job, Node, NodeRelease, TransferGrant, User
from .services import ServiceError

ACTIVE = ("queued", "running")
PLAN_MAX = 5000  # images per page of a plan
RESOLVE_MAX = 5000  # images a holding node may ask about at once
DEFAULT_PORT = 8475  # the nodes' data port (BADAMOA_NODE_DATA_PORT changes it)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def holders(db: Session, release_id: str) -> set[str]:
    """Nodes that have a version's images: where they were registered, and where they were received in full."""
    from . import training  # training imports this module

    have = set(
        db.scalars(
            select(NodeRelease.node_id).where(NodeRelease.release_id == release_id, NodeRelease.state != "removing")
        )
    )
    try:
        have.add(training.data_node(db, release_id).id)
    except ServiceError:
        pass
    return have


def release_bytes(db: Session, release_id: str, only: Any = None) -> tuple[int, int]:
    """How many images a version has (of ``only``, a query of item ids, when given) and how many bytes they take."""
    query = (
        select(func.count(Item.id), func.coalesce(func.sum(Item.bytes), 0))
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release_id)
    )
    if only is not None:
        query = query.where(Item.id.in_(only))
    count, size = db.execute(query).one()
    return int(count or 0), int(size or 0)


def addresses(node: Node) -> list[str]:
    """Where other nodes can reach a node: what it reports, then what the Hub saw it connect from."""
    metrics = node.metrics or {}
    serve = metrics.get("serve") or {}
    found: list[str] = []
    for address in [*(serve.get("addresses") or []), metrics.get("seen_from") or ""]:
        try:
            ip = ipaddress.ip_address(str(address))
        except ValueError:
            continue
        if ip.is_loopback or ip.is_unspecified or str(ip) in found:
            continue
        found.append(str(ip))
    return found


def data_port(node: Node) -> int:
    return int(((node.metrics or {}).get("serve") or {}).get("port") or 0)


# ---- choosing where a run trains -------------------------------------------------------------


def allowed_nodes(db: Session, node_ids: list[str]) -> list[Node]:
    """The nodes a person allowed, in their order; each must exist and take jobs."""
    picked: list[Node] = []
    for node_id in dict.fromkeys(node_ids):
        node = db.get(Node, node_id)
        if node is None:
            raise ServiceError(404, "고른 노드를 찾을 수 없습니다. 노드 목록을 새로 고친 뒤 다시 고르세요.")
        if not node.enabled:
            raise ServiceError(400, f"{node.name}은 작업을 받지 않도록 꺼 두었습니다.")
        picked.append(node)
    return picked


def pick_receiver(nodes: list[Node]) -> Node:
    """Of the allowed nodes none of which has the images, the one to receive them: seen lately, most GPUs."""
    now = utcnow()

    def score(node: Node) -> tuple[int, int]:
        recent = node.last_seen_at is not None and (now - node.last_seen_at).total_seconds() < 120
        return (1 if recent else 0, len((node.info or {}).get("gpus") or []))

    return max(nodes, key=score)  # max keeps the first of equals: the person's order breaks ties


def check_route(source: Node, receiver: Node) -> None:
    """Both ends must speak the transfer: a receiver new enough, a holding node new enough with its port open."""
    need = ".".join(str(n) for n in SYNC_SINCE)
    if not _knows_sync(receiver.agent_version or ""):
        version = receiver.agent_version or "버전 모름"
        why = f"다른 노드의 이미지를 받을 수 없습니다. {need} 이상이어야 합니다"
        raise ServiceError(400, f"{receiver.name}({version})는 {why}.")
    serve = (source.metrics or {}).get("serve") or {}
    if not _knows_sync(source.agent_version or "") or not serve.get("port"):
        why = serve.get("error") or f"{need} 이상이어야 하고 이미지 전송용 포트가 열려 있어야 합니다"
        raise ServiceError(400, f"이미지가 있는 {source.name}가 다른 노드로 이미지를 보낼 수 없습니다: {why}.")


def sync_job(db: Session, user: User, release_id: str, node: Node, via: str = "api", set_id: str = "") -> Job:
    """The job that brings a version's images to a node, one at a time per version and node.

    ``set_id``: only a labeling set's images not labelled yet (what a 오토라벨 needs); a transfer of the whole
    version on its way to the node serves too."""
    for job in db.scalars(select(Job).where(Job.type == "data.sync", Job.status.in_(ACTIVE))):
        params = job.params or {}
        if job.target_node_id == node.id and params.get("release_id") == release_id:
            if not params.get("set_id") or params.get("set_id") == set_id:
                return job
    params = {"release_id": release_id, **({"set_id": set_id} if set_id else {})}
    return services.create_job(db, user, "data.sync", params, node.id, via=via, gpus=0)


def _only(job: Job) -> Any:
    """The images a transfer is limited to (a labeling set's not labelled yet), or None for the whole version."""
    set_id = str((job.params or {}).get("set_id") or "")
    if not set_id:
        return None
    return select(LabelSetItem.item_id).where(LabelSetItem.set_id == set_id, LabelSetItem.state == "todo")


# ---- the receiving node --------------------------------------------------------------------


def _running_job(db: Session, node: Node, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 작업이 아닙니다.")
    if job.type not in ("data.sync", "train.run"):
        raise ServiceError(400, "이미지를 받는 작업이 아닙니다.")
    return job


def plan(db: Session, node: Node, job_id: str, after: int = 0, limit: int = PLAN_MAX) -> dict[str, Any]:
    """A page of the images a job needs, and on the first page where to get them with which grant."""
    from . import training

    job = _running_job(db, node, job_id)
    release_id = str((job.params or {}).get("release_id") or "")
    release = db.get(Release, release_id)
    if release is None:
        raise ServiceError(404, "버전을 찾을 수 없습니다.")
    source = training.data_node(db, release.id)
    limit = max(1, min(limit, PLAN_MAX))
    only = _only(job)
    query = (
        select(Item.id, Item.sha256, Item.bytes, Item.rel_path)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release.id, Item.id > after)
    )
    if only is not None:
        query = query.where(Item.id.in_(only))
    rows = db.execute(query.order_by(Item.id).limit(limit)).all()
    page: dict[str, Any] = {
        "items": [
            {"id": i, "sha256": sha, "bytes": size, "ext": PurePosixPath(rel).suffix.lower()} for i, sha, size, rel in rows
        ],
        "next": rows[-1][0] if len(rows) == limit else None,
    }
    if after == 0:
        count, size = release_bytes(db, release.id, only)
        page["total"] = {"items": count, "bytes": size}
        page["local"] = source.id == node.id  # the images are registered here: nothing to fetch
        if not page["local"]:
            token = secrets.token_urlsafe(24)
            db.execute(delete(TransferGrant).where(TransferGrant.job_id == job.id))
            db.add(
                TransferGrant(
                    token_hash=_hash(token),
                    job_id=job.id,
                    release_id=release.id,
                    source_node_id=source.id,
                    dest_node_id=node.id,
                )
            )
            page["grant"] = token
            page["source"] = {
                "id": source.id,
                "name": source.name,
                "addresses": addresses(source),
                "port": data_port(source),
            }
    return page


def report(db: Session, node: Node, job_id: str, progress: dict[str, Any]) -> None:
    """How far a transfer got, for the run's screen (the final numbers come with the job's result)."""
    job = _running_job(db, node, job_id)
    keep = ("items", "bytes", "done_items", "done_bytes", "cached_items", "rate", "eta_s")
    job.result = {"progress": {k: progress[k] for k in keep if isinstance(progress.get(k), int | float)}}


# ---- the holding node ----------------------------------------------------------------------


def resolve(db: Session, node: Node, token: str, ids: list[int]) -> dict[str, Any]:
    """Which of ``ids`` a grant lets this node send, with the registered files to send."""
    grant = db.get(TransferGrant, _hash(token or ""))
    if grant is None or grant.source_node_id != node.id:
        raise ServiceError(403, "허가되지 않은 전송입니다.")
    job = db.get(Job, grant.job_id)
    if job is None or job.status != "running" or job.node_id != grant.dest_node_id:
        raise ServiceError(403, "전송을 허가한 작업이 끝났습니다.")
    ids = [int(i) for i in ids[:RESOLVE_MAX]]
    roots = {loc.id: loc.root for loc in db.scalars(select(StorageLocation).where(StorageLocation.node_id == node.id))}
    rows = db.execute(
        select(Item.id, Item.location_id, Item.rel_path, Item.bytes, Item.sha256)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == grant.release_id, Item.id.in_(ids))
    ).all()
    files = [
        {"id": i, "root": roots[loc], "rel_path": rel, "bytes": size, "sha256": sha}
        for i, loc, rel, size, sha in rows
        if loc in roots
    ]
    return {"files": files, "dest_node_id": grant.dest_node_id}


# ---- when a transfer ends --------------------------------------------------------------------


def on_job_finished(db: Session, job: Job) -> None:
    """A finished transfer: remember the node has the version, or let the training waiting for it go."""
    if job.type not in ("data.sync", "train.run"):
        return
    db.execute(delete(TransferGrant).where(TransferGrant.job_id == job.id))
    if job.type != "data.sync":
        return
    release_id = str((job.params or {}).get("release_id") or "")
    if job.status == "succeeded" and job.node_id and (job.params or {}).get("set_id"):
        return  # some of the version's images (a 오토라벨's): the node does not hold the version
    if job.status == "succeeded" and job.node_id:
        result = job.result or {}
        row = db.get(NodeRelease, (job.node_id, release_id))
        if row is None:
            row = NodeRelease(node_id=job.node_id, release_id=release_id)
            db.add(row)
        row.items, row.bytes, row.synced_at = int(result.get("items") or 0), int(result.get("bytes") or 0), utcnow()
        # the transfer checked every image on arrival: all of it is there now
        row.present_items, row.present_bytes, row.checked_at = row.items, row.bytes, row.synced_at
        if row.state == "removing":
            row.state, row.job_id = "", None
        return
    if job.status == "succeeded":
        return
    # It failed or was stopped: the trainings (evaluations, 오토라벨) waiting for it cannot start.
    reason = job.error or ("취소되었습니다." if job.status == "canceled" else "실패했습니다.")
    for waiting in db.scalars(select(Job).where(Job.type.in_(("train.run", "model.run")), Job.status == "queued")):
        if (waiting.requirements or {}).get("after") == job.id:
            action = (waiting.params or {}).get("action")
            what = "학습을" if waiting.type == "train.run" else "오토라벨을" if action == "autolabel" else "작업을"
            waiting.status, waiting.finished_at = "failed", utcnow()
            waiting.error = f"이미지를 받지 못해 {what} 시작하지 못했습니다: {reason}"


def waiting_for(db: Session, job: Job) -> Job | None:
    """The transfer a queued training waits for, if it still has to finish."""
    after = (job.requirements or {}).get("after")
    if not after:
        return None
    first = db.get(Job, after)
    return first if first is not None and first.status != "succeeded" else None


def progress_text(job: Job) -> str:
    """For a running transfer, e.g. 이미지 받는 중 37% · 120 / 320 GB · 110 MB/s."""
    p = (job.result or {}).get("progress") or {}
    total, done = int(p.get("bytes") or 0), int(p.get("done_bytes") or 0)
    if job.status == "queued" or not total:
        return "이미지를 받을 준비를 하고 있습니다."
    gb = 1024**3
    parts = [f"{done / total:.0%}", f"{done / gb:,.1f} / {total / gb:,.1f} GB"]
    if p.get("rate"):
        parts.append(f"{float(p['rate']) / 1024**2:,.0f} MB/s")
    return "이미지 받는 중 " + " · ".join(parts)


# ---- what the new-run dialog offers ------------------------------------------------------------


def node_choices(db: Session, settings: Any, release_id: str) -> dict[str, Any]:
    """Every node with what it means to train a version there: free GPUs, and whether the images must come first."""
    from . import resources, training

    count, size = release_bytes(db, release_id)
    try:
        source = training.data_node(db, release_id).id
    except ServiceError:
        source = ""
    received = set(
        db.scalars(
            select(NodeRelease.node_id).where(NodeRelease.release_id == release_id, NodeRelease.state != "removing")
        )
    )
    assigned = resources.assignments(db)
    nodes = db.scalars(select(Node).order_by(Node.is_local.desc(), Node.created_at)).all()
    shown = []
    for node in nodes:
        status = services.node_status(node, settings)
        gpus = (node.info or {}).get("gpus") or []
        free = resources.free_gpus(node, settings, assigned.get(node.id), 0) if status == "online" else []
        cache = (node.metrics or {}).get("cache") or {}
        shown.append(
            {
                "id": node.id,
                "name": node.name,
                "status": status,
                "enabled": node.enabled,
                "roles": list(node.roles or []),  # 학습용 (compute) ones train
                "gpus": len(gpus),
                "gpus_free": len(free),
                "gpu_list": gpu_list(node, settings, status, assigned.get(node.id)),
                # registered: the images were imported here · received: copied here before · "": must be copied
                "images": "registered" if node.id == source else "received" if node.id in received else "",
                "cache_free_gb": cache.get("free_gb"),
                "cache_dir": cache.get("dir") or "",  # where a copy goes there (its 받은 이미지 보관 위치)
                "cache_auto": bool(cache.get("auto")),  # picked by the node: the disk with the most room then
                "can_receive": bool(node.agent_version) and _knows_sync(node.agent_version),
            }
        )
    return {"items": count, "bytes": size, "source": source, "nodes": shown}


def gpu_list(node: Node, settings: Any, status: str, assigned: Any = None) -> list[dict[str, Any]]:
    """Each GPU of a node, to pick by number: free · busy (someone uses it) · job (a BADAMOA job) · reserved (people's)."""
    from . import resources  # noqa: PLC0415

    states = {g["index"]: g for g in resources.gpu_states(node, settings, assigned)}
    return [
        {
            "index": g.get("index"),
            "name": g.get("name") or "",
            "memory_total_mb": g.get("memory_total_mb"),
            "state": (states.get(g.get("index")) or {}).get("state", "free") if status == "online" else "offline",
            "users": (states.get(g.get("index")) or {}).get("users", []),
            "reserved": g.get("index") in (node.reserved_gpus or []),
        }
        for g in (node.info or {}).get("gpus") or []
    ]


def place_info(node: Node, settings: Any, status: str, assigned: Any, holder_id: str, have: Any) -> dict[str, Any]:
    """What a dialog shows of a server a job over a version's images may run on: its GPUs by number, whether it
    holds the images (registered there · received before · "": sent there first), and where a copy goes there (its
    받은 이미지 보관 위치: the admin's, else the disk with the most room when the first copy came) with the room left."""
    cache = (node.metrics or {}).get("cache") or {}
    return {
        "gpu_list": gpu_list(node, settings, status, assigned),
        "images": "registered" if node.id == holder_id else "received" if node.id in have else "",
        "cache": {"dir": cache.get("dir") or "", "free_gb": cache.get("free_gb"), "auto": bool(cache.get("auto"))},
    }


def picked_gpus(node: Node, gpu_indices: list[int] | None, gpus: int | None, one: bool = True) -> list[int]:
    """GPUs picked by number on the server a job runs on: there, not people's, and (``one``) only one."""
    picked = sorted({int(i) for i in gpu_indices or []})
    if not picked:
        return []
    if gpus == 0:
        raise ServiceError(400, "CPU로 돌릴 때는 GPU를 고르지 않습니다.")
    if one and len(picked) > 1:
        raise ServiceError(400, "이 작업은 GPU 하나로 돕니다. 하나만 고르세요.")
    listed = {g.get("index") for g in (node.info or {}).get("gpus") or []}
    for index in picked:
        if index not in listed:
            raise ServiceError(400, f"{node.name}에는 GPU {index}이(가) 없습니다.")
        if index in (node.reserved_gpus or []):
            raise ServiceError(400, f"{node.name}의 GPU {index}은(는) 사람 전용으로 빼 둔 GPU입니다.")
    return picked


DATA_AFTER = ("", "keep", "delete")  # what a job does with the copy its server received, once it is over


def data_after_of(choice: str, node_id: str, holder_id: str) -> str:
    """keep or delete the copy the server received (``node_data.on_job_finished``); none where it was registered."""
    if choice not in DATA_AFTER:
        raise ServiceError(400, "data_after는 keep(남겨 두기) 또는 delete(지우기)입니다.")
    return "" if node_id == holder_id else choice


SYNC_SINCE = (0, 37, 0)  # nodes from here on take images from other nodes


def _knows_sync(version: str) -> bool:
    try:
        parts = tuple(int(p) for p in version.split(".")[:3])
    except ValueError:
        return False
    return parts >= SYNC_SINCE
