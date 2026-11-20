"""Business logic shared by the REST API and the MCP server.

Every function takes an open SQLAlchemy session; callers own the transaction.
Errors are raised as ``ServiceError`` with an HTTP-style status code so both
front doors (REST, MCP) can translate them.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from badamoa import __version__ as core_version
from badamoa.core.jobs import BUILTIN_JOB_TYPES, validate_params
from badamoa.core.protocol import (
    ALL_ROLES,
    ROLE_LABELS,
    TERMINAL_STATUSES,
    CompleteRequest,
    HeartbeatRequest,
    LogLine,
    RegisterRequest,
)

from . import disk_space, resources
from .config import HubSettings
from .db import iso, utcnow
from .models import ApiToken, AuditLog, Job, JobLog, JoinToken, Node, User
from .security import (
    API_PREFIX,
    JOIN_PREFIX,
    NODE_PREFIX,
    SESSION_PREFIX,
    hash_password,
    hash_token,
    new_token,
    verify_password,
)

HUB_VERSION = core_version
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")


class ServiceError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def audit(db: Session, user_id: str | None, via: str, action: str, target: str = "", **detail: Any) -> None:
    db.add(AuditLog(user_id=user_id, via=via, action=action, target=target, detail=detail))


# ----------------------------------------------------------------------------
# Users & tokens
# ----------------------------------------------------------------------------


def needs_setup(db: Session) -> bool:
    return (db.scalar(select(func.count()).select_from(User)) or 0) == 0


def create_user(db: Session, username: str, password: str, role: str = "member") -> User:
    if not _USERNAME_RE.match(username):
        raise ServiceError(400, "아이디는 영문/숫자/._- 3~64자여야 합니다.")
    if len(password) < 8:
        raise ServiceError(400, "비밀번호는 8자 이상이어야 합니다.")
    if role not in ("admin", "member"):
        raise ServiceError(400, "role은 admin 또는 member입니다.")
    if db.scalar(select(User).where(User.username == username)):
        raise ServiceError(409, "이미 존재하는 아이디입니다.")
    user = User(username=username, password_hash=hash_password(password), role=role)
    db.add(user)
    db.flush()
    return user


def issue_token(
    db: Session, user: User, kind: str, name: str = "", ttl: timedelta | None = None
) -> tuple[str, ApiToken]:
    token = new_token(SESSION_PREFIX if kind == "session" else API_PREFIX)
    row = ApiToken(
        user_id=user.id,
        kind=kind,
        name=name,
        token_hash=hash_token(token),
        prefix=token[:10],
        expires_at=(utcnow() + ttl) if ttl else None,
    )
    db.add(row)
    db.flush()
    return token, row


def login(db: Session, username: str, password: str, settings: HubSettings) -> tuple[str, User]:
    user = db.scalar(select(User).where(User.username == username))
    if user is None or not verify_password(password, user.password_hash):
        raise ServiceError(401, "아이디 또는 비밀번호가 올바르지 않습니다.")
    if user.disabled_at is not None:
        raise ServiceError(403, "사용이 중지된 계정입니다. 관리자에게 문의하세요.")
    token, _ = issue_token(db, user, "session", "web", timedelta(hours=settings.session_ttl_hours))
    return token, user


def authenticate(db: Session, token: str, ip: str = "") -> User | None:
    """The token's person; its last use (and where from) is written down at most once a minute."""
    if not token or not token.startswith((SESSION_PREFIX + "_", API_PREFIX + "_")):
        return None
    row = db.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(token)))
    if row is None:
        return None
    now = utcnow()
    if row.expires_at and row.expires_at < now:
        return None
    if row.user.disabled_at is not None:
        return None
    if row.last_used_at is None or (now - row.last_used_at).total_seconds() > 60 or (ip and ip != row.last_ip):
        row.last_used_at = now
        if ip:
            row.last_ip = ip[:64]
    return row.user


def revoke_token(db: Session, user: User, token_id: str) -> None:
    row = db.get(ApiToken, token_id)
    if row is None or (row.user_id != user.id and user.role != "admin"):
        raise ServiceError(404, "토큰을 찾을 수 없습니다.")
    db.delete(row)


def list_api_tokens(db: Session, user: User) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(ApiToken)
        .where(ApiToken.user_id == user.id, ApiToken.kind == "api")
        .order_by(ApiToken.created_at.desc())
    )
    return [
        {
            "id": r.id,
            "name": r.name,
            "prefix": r.prefix,
            "created_at": iso(r.created_at),
            "last_used_at": iso(r.last_used_at),
            "expires_at": iso(r.expires_at),
        }
        for r in rows
    ]


def serialize_user(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "created_at": iso(user.created_at),
        "disabled": user.disabled_at is not None,
        "template_upload": bool(user.template_upload),
        "can_upload_templates": can_upload_templates(user),
    }


def can_upload_templates(user: User) -> bool:
    """Admins, and members an admin lets upload model kinds (their uploads are drafts)."""
    return user.role == "admin" or bool(user.template_upload)


def require_admin(user: User) -> None:
    if user.role != "admin":
        raise ServiceError(403, "관리자 권한이 필요합니다.")


# ----------------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------------


def _clean_roles(roles: list[str] | None) -> list[str]:
    roles = list(dict.fromkeys(roles or ALL_ROLES))
    bad = [r for r in roles if r not in ALL_ROLES]
    if bad:
        known = ", ".join(f"{r}({ROLE_LABELS[r]})" for r in ALL_ROLES)
        raise ServiceError(400, f"알 수 없는 역할: {bad}. 가능한 값: {known}")
    return roles


def create_join_token(
    db: Session, user: User, roles: list[str] | None, ttl_minutes: int = 60
) -> tuple[str, JoinToken]:
    require_admin(user)
    if not 1 <= ttl_minutes <= 60 * 24 * 7:
        raise ServiceError(400, "만료 시간은 1분 ~ 7일 사이여야 합니다.")
    token = new_token(JOIN_PREFIX)
    row = JoinToken(
        token_hash=hash_token(token),
        roles=_clean_roles(roles),
        created_by=user.id,
        expires_at=utcnow() + timedelta(minutes=ttl_minutes),
    )
    db.add(row)
    db.flush()
    return token, row


def register_node(db: Session, req: RegisterRequest) -> tuple[Node, str]:
    jt = db.scalar(select(JoinToken).where(JoinToken.token_hash == hash_token(req.join_token)))
    if jt is None or jt.used_at is not None or jt.expires_at < utcnow():
        raise ServiceError(401, "join token이 없거나 만료되었거나 이미 사용되었습니다.")
    token = new_token(NODE_PREFIX)
    info = dict(req.info)
    node = Node(
        name=(req.name or info.get("hostname") or "node")[:128],
        token_hash=hash_token(token),
        roles=list(jt.roles),
        os=str(info.get("os", ""))[:16],
        info=info,
        agent_version=req.agent_version,
        last_seen_at=utcnow(),
    )
    db.add(node)
    db.flush()
    jt.used_at = utcnow()
    jt.used_by_node_id = node.id
    audit(db, jt.created_by, "agent", "node.register", node.id, name=node.name, roles=node.roles)
    return node, token


def ensure_local_node(db: Session, info: dict[str, Any], agent_version: str) -> tuple[Node, str | None]:
    """All-in-one install: the Hub machine is also a node. Returns a new token only on creation."""
    node = db.scalar(select(Node).where(Node.is_local.is_(True)))
    if node is not None:
        return node, None
    token = new_token(NODE_PREFIX)
    node = Node(
        name=f"{info.get('hostname', 'hub')} (hub)",
        token_hash=hash_token(token),
        roles=list(ALL_ROLES),
        os=str(info.get("os", "")),
        info=info,
        agent_version=agent_version,
        is_local=True,
    )
    db.add(node)
    db.flush()
    return node, token


def rotate_node_token(db: Session, node: Node) -> str:
    token = new_token(NODE_PREFIX)
    node.token_hash = hash_token(token)
    return token


def authenticate_node(db: Session, token: str) -> Node | None:
    if not token or not token.startswith(NODE_PREFIX + "_"):
        return None
    return db.scalar(select(Node).where(Node.token_hash == hash_token(token)))


def node_status(node: Node, settings: HubSettings, now: datetime | None = None) -> str:
    if not node.enabled:
        return "disabled"
    now = now or utcnow()
    if node.last_seen_at and (now - node.last_seen_at).total_seconds() <= settings.node_offline_after_s:
        return "online"
    return "offline"


def heartbeat(db: Session, node: Node, req: HeartbeatRequest, seen_from: str = "") -> list[str]:
    now = utcnow()
    node.last_seen_at = now
    node.metrics = {**req.metrics, "seen_from": seen_from} if seen_from else req.metrics
    node.agent_version = req.agent_version
    resources.record(db, node.id, req.metrics, now)
    resources.record_people(db, node, resources.assignments(db).get(node.id), now)
    if req.info:
        node.info = req.info
        node.os = str(req.info.get("os", node.os))
    running = db.scalars(select(Job).where(Job.node_id == node.id, Job.status == "running")).all()
    reported = set(req.running_job_ids)
    cancel_ids = []
    for job in running:
        if job.id in reported:
            if job.cancel_requested:
                cancel_ids.append(job.id)
        elif job.started_at and (now - job.started_at).total_seconds() > 30:
            # The node no longer runs it (e.g. agent restarted mid-job).
            job.status = "failed"
            job.error = "노드가 더 이상 이 작업을 실행하고 있지 않습니다 (에이전트 재시작 등)."
            job.finished_at = now
    return cancel_ids


def serialize_node(
    node: Node, settings: HubSettings, assigned: dict[int, str] | None = None
) -> dict[str, Any]:
    info = node.info or {}
    status = node_status(node, settings)
    return {
        "id": node.id,
        "name": node.name,
        "status": status,
        "os": node.os,
        "roles": node.roles,
        "labels": node.labels,
        "is_local": node.is_local,
        "enabled": node.enabled,
        "agent_version": node.agent_version,
        "created_at": iso(node.created_at),
        "last_seen_at": iso(node.last_seen_at),
        "info": {
            "hostname": info.get("hostname"),
            "os_release": info.get("os_release"),
            "arch": info.get("arch"),
            "cpu_model": info.get("cpu_model"),
            "cpu_count": info.get("cpu_count"),
            "memory_total_mb": info.get("memory_total_mb"),
            "gpus": [
                {k: g.get(k) for k in ("index", "name", "memory_total_mb", "driver_version")}
                for g in info.get("gpus", [])
            ],
            "docker": info.get("docker"),  # {"version", "os", "gpu"} when the node can run containers
            "account": info.get("account"),  # {"name", "admin"}: the account the agent runs as
        },
        "metrics": node.metrics or {},
        "reserved_gpus": node.reserved_gpus or [],
        "cache_dir": node.cache_dir or "",  # as an admin set it; metrics.cache has the folder in use
        "work_dir": node.work_dir or "",  # as an admin set it; metrics.work has the folder in use
        # disks BADAMOA writes to there that are nearly full (none while the node is away: its numbers are old)
        "disk_alerts": disk_space.alerts(node.metrics or {}, node.storage) if status == "online" else [],
        "docker_target": _docker_target(node),
        "gpu_states": resources.gpu_states(node, settings, assigned),
    }


def node_busy(db: Session, node: Node) -> int:
    """BADAMOA jobs queued for or running on a node."""
    return int(
        db.scalar(
            select(func.count(Job.id)).where(
                Job.status.in_(("queued", "running")), or_(Job.node_id == node.id, Job.target_node_id == node.id)
            )
        )
        or 0
    )


def start_docker_move(
    db: Session, user: User, node_id: str, target: str, remove_old: bool = True, stop_containers: bool = False,
    via: str = "api",
) -> Job:
    """An admin moves a node's Docker images to another disk (``node.docker_move``): the node stops Docker, copies,
    points Docker at the new folder, checks, and puts everything back if any of it fails."""
    require_admin(user)
    node = get_node(db, node_id)
    state = disk_space.docker_move_state(node, node_busy(db, node))
    if not state["possible"]:
        raise ServiceError(400, state["why"])
    target = (target or "").strip()
    if not target.startswith("/"):
        raise ServiceError(400, "옮길 폴더를 절대 경로로 적으세요 (예: /home4/docker).")
    job = create_job(
        db,
        user,
        "node.docker_move",
        {"target": target, "remove_old": bool(remove_old), "stop_containers": bool(stop_containers)},
        node.id,
        via=via,
        gpus=0,
    )
    audit(db, user.id, via, "node.docker_move", node.id, target=target, current=state["path"], remove_old=remove_old)
    return job


def _docker_target(node: Node) -> str:
    """Where to suggest moving Docker's data-root, when Docker's disk is nearly full."""
    metrics = node.metrics or {}
    docker = next((p for p in disk_space.places(metrics) if p.get("kind") == "docker"), None)
    if docker is None or not disk_space.level(docker.get("free_gb"), docker.get("total_gb")):
        return ""
    return disk_space.docker_target(metrics, str(docker.get("path") or ""))


def list_nodes(db: Session, settings: HubSettings) -> list[dict[str, Any]]:
    nodes = db.scalars(select(Node).order_by(Node.is_local.desc(), Node.created_at)).all()
    assigned = resources.assignments(db)
    return [serialize_node(n, settings, assigned.get(n.id)) for n in nodes]


def node_view(db: Session, settings: HubSettings, node: Node) -> dict[str, Any]:
    return serialize_node(node, settings, resources.assignments(db).get(node.id))


def get_node(db: Session, node_id: str) -> Node:
    node = db.get(Node, node_id)
    if node is None:
        raise ServiceError(404, "노드를 찾을 수 없습니다.")
    return node


def update_node(
    db: Session,
    user: User,
    node_id: str,
    *,
    name: str | None = None,
    roles: list[str] | None = None,
    enabled: bool | None = None,
    labels: dict[str, Any] | None = None,
    reserved_gpus: list[int] | None = None,
    cache_dir: str | None = None,
    work_dir: str | None = None,
) -> Node:
    require_admin(user)
    node = get_node(db, node_id)
    if cache_dir is not None:
        node.cache_dir = _folder(cache_dir, "받은 이미지 보관 위치", "badamoa-cache")
    if work_dir is not None:
        node.work_dir = _folder(work_dir, "작업 폴더 위치", "badamoa-work")
    if reserved_gpus is not None:
        known = {g.get("index") for g in (node.info or {}).get("gpus") or []}
        node.reserved_gpus = sorted({int(i) for i in reserved_gpus if int(i) in known})
    if name is not None:
        node.name = name.strip()[:128] or node.name
    if roles is not None:
        node.roles = _clean_roles(roles)
    if enabled is not None:
        node.enabled = enabled
    if labels is not None:
        node.labels = labels
    audit(
        db, user.id, "api", "node.update", node.id,
        name=name, roles=roles, enabled=enabled, reserved=reserved_gpus, cache_dir=cache_dir, work_dir=work_dir,
    )  # fmt: skip
    return node


def _folder(text: str, what: str, example: str) -> str:
    """A folder on the node as an admin typed it: absolute, or "" for the node's own pick."""
    text = text.strip()
    absolute = text.startswith(("/", "\\")) or (len(text) > 2 and text[1:3] in (":\\", ":/"))
    if text and not absolute:
        raise ServiceError(400, f"{what}는 절대 경로로 적으세요 (예: /data/{example}, D:\\{example}).")
    return text[:512]


def delete_node(db: Session, user: User, node_id: str) -> None:
    require_admin(user)
    node = get_node(db, node_id)
    if node.is_local:
        raise ServiceError(400, "Hub 로컬 노드는 삭제할 수 없습니다. 비활성화하세요.")
    for job in db.scalars(select(Job).where(Job.node_id == node.id, Job.status == "running")):
        job.status, job.error, job.finished_at = "failed", "노드가 삭제되었습니다.", utcnow()
    audit(db, user.id, "api", "node.delete", node.id, name=node.name)
    db.delete(node)


# ----------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------


def list_job_types() -> list[dict[str, Any]]:
    return [spec.model_dump() for spec in BUILTIN_JOB_TYPES.values() if not spec.hidden]


def create_job(
    db: Session,
    user: User,
    job_type: str,
    params: dict[str, Any] | None = None,
    node_id: str | None = None,
    via: str = "api",
    gpus: int | None = None,
    min_vram_mb: int = 0,
    nodes: list[str] | None = None,
    after: str | None = None,
    role: str = "compute",
    gpu_indices: list[int] | None = None,
) -> Job:
    """``nodes``: the job may run on any of these (``node_id`` pins one). ``after``: it starts once that job succeeded.
    ``role``: the role a server needs to take it when it takes GPUs (autolabel: ``autolabel``).
    ``gpu_indices``: these GPUs of the pinned node and no others (the job waits until all of them are free)."""
    try:
        clean = validate_params(job_type, params or {})
    except ValueError as e:
        raise ServiceError(400, str(e)) from e
    spec = BUILTIN_JOB_TYPES[job_type]
    picked = sorted({int(i) for i in gpu_indices or []})
    gpu_count = len(picked) if picked else spec.gpus if gpus is None else gpus
    if not 0 <= gpu_count <= 16 or min_vram_mb < 0:
        raise ServiceError(400, "GPU 수는 0~16, 최소 VRAM은 0 이상이어야 합니다.")
    requirements: dict[str, Any] = {}
    if gpu_count:
        requirements = {"roles": [role], "gpus": gpu_count, "min_vram_mb": min_vram_mb}
    if picked:
        if not node_id:
            raise ServiceError(400, "GPU를 고르려면 작업이 갈 노드를 하나로 정해야 합니다.")
        requirements["gpu_indices"] = picked
    if node_id:
        node = get_node(db, node_id)
        if not node.enabled:
            raise ServiceError(400, "비활성화된 노드에는 작업을 보낼 수 없습니다.")
    if nodes:
        requirements["nodes"] = list(dict.fromkeys(nodes))
    if after:
        requirements["after"] = after
    job = Job(
        type=job_type,
        params=clean,
        target_node_id=node_id,
        requirements=requirements,
        created_by=user.id,
        created_via=via,
    )
    db.add(job)
    db.flush()
    audit(db, user.id, via, "job.create", job.id, type=job_type, node_id=node_id)
    return job


def _node_matches(job: Job, node: Node) -> bool:
    if job.target_node_id and job.target_node_id != node.id:
        return False
    req = job.requirements or {}
    if req.get("nodes") and node.id not in req["nodes"]:
        return False
    if any(role not in node.roles for role in req.get("roles", [])):
        return False
    if req.get("os") and req["os"] != node.os:
        return False
    return True


def try_claim(db: Session, node: Node, settings: HubSettings) -> Job | None:
    if not node.enabled:
        return None
    assigned = resources.assignments(db).get(node.id, {})
    candidates = db.scalars(
        select(Job).where(Job.status == "queued").order_by(Job.created_at).limit(50)
    ).all()
    for job in candidates:
        if not _node_matches(job, node):
            continue
        req = job.requirements or {}
        if req.get("after"):
            first = db.get(Job, req["after"])
            if first is not None and first.status != "succeeded":
                continue  # still waiting for what it needs (the images it trains on)
        gpu_indices: list[int] = []
        if req.get("gpu_indices"):  # these GPUs and no others: wait until every one of them is free
            free = resources.free_gpus(node, settings, assigned, req.get("min_vram_mb", 0))
            if not set(req["gpu_indices"]) <= set(free):
                continue
            gpu_indices = list(req["gpu_indices"])
        elif req.get("gpus"):
            free = resources.free_gpus(node, settings, assigned, req.get("min_vram_mb", 0))
            if len(free) < req["gpus"]:
                if not req.get("gpu_if_free"):
                    continue  # not enough idle GPUs here right now
                free = []  # it takes a GPU only if one is idle (a template's trial): the CPU now
            gpu_indices = free[: req["gpus"]]
        res = db.execute(
            update(Job)
            .where(Job.id == job.id, Job.status == "queued")
            .values(status="running", node_id=node.id, started_at=utcnow(), gpu_indices=gpu_indices)
        )
        if res.rowcount == 1:
            db.refresh(job)
            return job
    return None


def _job_for_node(db: Session, node: Node, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id:
        raise ServiceError(404, "이 노드에 할당된 작업이 아닙니다.")
    return job


def append_logs(db: Session, node: Node, job_id: str, lines: list[LogLine]) -> bool:
    job = _job_for_node(db, node, job_id)
    seq = job.log_count
    for line in lines:
        seq += 1
        ts = line.ts.replace(tzinfo=None) if line.ts.tzinfo else line.ts
        db.add(JobLog(job_id=job.id, seq=seq, ts=ts, stream=line.stream, text=line.text[:10_000]))
    job.log_count = seq
    return job.cancel_requested


def complete_job(db: Session, node: Node, job_id: str, req: CompleteRequest) -> Job:
    job = _job_for_node(db, node, job_id)
    if job.status != "running":
        return job  # already finalized (e.g. canceled or swept); ignore late completion
    job.status = req.status
    job.result = req.result
    job.error = req.error
    job.finished_at = utcnow()
    return job


def get_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise ServiceError(404, "작업을 찾을 수 없습니다.")
    return job


def cancel_job(db: Session, user: User, job_id: str, via: str = "api") -> Job:
    job = get_job(db, job_id)
    if job.status in TERMINAL_STATUSES:
        raise ServiceError(409, f"이미 종료된 작업입니다 ({job.status}).")
    if job.status == "queued":
        job.status, job.finished_at = "canceled", utcnow()
    else:
        job.cancel_requested = True
    audit(db, user.id, via, "job.cancel", job.id)
    return job


def get_logs(db: Session, job_id: str, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
    get_job(db, job_id)
    rows = db.scalars(
        select(JobLog)
        .where(JobLog.job_id == job_id, JobLog.seq > after)
        .order_by(JobLog.seq)
        .limit(max(1, min(limit, 5000)))
    )
    return [{"seq": r.seq, "ts": iso(r.ts), "stream": r.stream, "text": r.text} for r in rows]


def list_jobs(
    db: Session, status: str | None = None, node_id: str | None = None, limit: int = 100
) -> list[Job]:
    q = select(Job).order_by(Job.created_at.desc()).limit(max(1, min(limit, 500)))
    if status:
        q = q.where(Job.status == status)
    if node_id:
        q = q.where(Job.node_id == node_id)
    return list(db.scalars(q))


def serialize_job(
    job: Job, node_names: dict[str, str] | None = None, wait_reason: str | None = None
) -> dict[str, Any]:
    spec = BUILTIN_JOB_TYPES.get(job.type)
    end = job.finished_at or (utcnow() if job.status == "running" else None)
    duration = (end - job.started_at).total_seconds() if (job.started_at and end) else None
    return {
        "id": job.id,
        "type": job.type,
        "title": spec.title if spec else job.type,
        "params": job.params,
        "status": job.status,
        "node_id": job.node_id,
        "node_name": (node_names or {}).get(job.node_id or ""),
        "target_node_id": job.target_node_id,
        "created_by": job.created_by,
        "created_via": job.created_via,
        "created_at": iso(job.created_at),
        "started_at": iso(job.started_at),
        "finished_at": iso(job.finished_at),
        "duration_s": round(duration, 1) if duration is not None else None,
        "cancel_requested": job.cancel_requested,
        "result": job.result,
        "error": job.error,
        "log_count": job.log_count,
        "gpus_requested": (job.requirements or {}).get("gpus", 0),
        "gpu_indices": job.gpu_indices or [],
        "wait_reason": wait_reason,
    }


def job_view(db: Session, settings: HubSettings, job: Job, node_names: dict[str, str] | None = None) -> dict[str, Any]:
    reason = explain_wait(db, settings, job) if job.status == "queued" else None
    return serialize_job(job, node_names if node_names is not None else node_name_map(db), reason)


def explain_wait(db: Session, settings: HubSettings, job: Job) -> str:
    """Why a queued job has not started yet, in words the user can act on."""
    from . import transfers

    first = transfers.waiting_for(db, job)
    if first is not None:
        return transfers.progress_text(first)
    nodes = list(db.scalars(select(Node)))
    allowed = (job.requirements or {}).get("nodes")
    if allowed:
        nodes = [n for n in nodes if n.id in allowed]
    if job.target_node_id:
        nodes = [n for n in nodes if n.id == job.target_node_id]
        if not nodes:
            return "지정한 노드가 더 이상 없습니다."
    nodes = [n for n in nodes if n.enabled]
    if not nodes:
        return "작업을 받을 수 있는 노드가 없습니다."
    online = [n for n in nodes if node_status(n, settings) == "online"]
    if not online:
        return "지정한 노드가 오프라인입니다." if job.target_node_id else "온라인 노드가 없습니다."
    req = job.requirements or {}
    eligible = [n for n in online if all(role in n.roles for role in req.get("roles", []))]
    if not eligible:
        wanted = ", ".join(ROLE_LABELS.get(r, r) for r in req.get("roles", []))
        return f"{wanted} 역할을 가진 온라인 노드가 없습니다."
    need = req.get("gpus", 0)
    if req.get("gpu_indices"):
        assigned = resources.assignments(db)
        node = eligible[0]
        states = {g["index"]: g for g in resources.gpu_states(node, settings, assigned.get(node.id))}
        waiting = []
        for index in req["gpu_indices"]:
            g = states.get(index)
            if g is None:
                waiting.append(f"GPU {index} 없음")
            elif g["state"] == "job":
                waiting.append(f"GPU {index} MOA 작업 중")
            elif g["state"] == "busy":
                waiting.append(f"GPU {index} 사용 중" + (f" ({', '.join(g['users'])})" if g["users"] else ""))
            elif g["state"] == "reserved":
                waiting.append(f"GPU {index} 사람 전용")
        if not waiting:
            return "고른 GPU가 모두 비어 곧 배정됩니다."
        return f"{node.name}의 고른 GPU를 기다리는 중입니다: " + " · ".join(waiting)
    if need:
        assigned = resources.assignments(db)
        details = []
        for node in eligible:
            states = resources.gpu_states(node, settings, assigned.get(node.id))
            free = resources.free_gpus(node, settings, assigned.get(node.id), req.get("min_vram_mb", 0))
            if len(free) >= need:
                return "빈 GPU가 있어 곧 배정됩니다."
            users = sorted({u for g in states if g["state"] == "busy" for u in g["users"]})
            by_jobs = sum(1 for g in states if g["state"] == "job")
            kept = sum(1 for g in states if g["state"] == "reserved")
            who = ", ".join(users) if users else ("다른 프로세스" if any(g["state"] == "busy" for g in states) else "")
            parts = [f"빈 GPU {len(free)}/{len(states)}개"]
            if who:
                parts.append(f"사용 중: {who}")
            if by_jobs:
                parts.append(f"MOA 작업 {by_jobs}개")
            if kept:
                parts.append(f"사람 전용 {kept}개")
            details.append(f"{node.name} — {' · '.join(parts)}")
        return f"빈 GPU가 부족합니다 (필요 {need}개). " + " / ".join(details)
    return "노드가 작업을 가져가기를 기다리는 중입니다."


def node_name_map(db: Session) -> dict[str, str]:
    return {nid: name for nid, name in db.execute(select(Node.id, Node.name))}


def list_activity(db: Session, limit: int = 20) -> list[dict[str, Any]]:
    """Recent audit entries, newest first, for the activity feed."""
    rows = db.execute(
        select(AuditLog, User.username)
        .outerjoin(User, User.id == AuditLog.user_id)
        .order_by(AuditLog.id.desc())
        .limit(max(1, min(limit, 200)))
    ).all()
    return [
        {
            "id": entry.id,
            "ts": iso(entry.ts),
            "username": username,
            "via": entry.via,
            "action": entry.action,
            "target": entry.target,
            "detail": entry.detail,
        }
        for entry, username in rows
    ]


def sweep_lost_jobs(db: Session, settings: HubSettings) -> int:
    """Fail running jobs whose node stopped heart-beating."""
    cutoff = utcnow() - timedelta(seconds=settings.node_lost_after_s)
    lost = 0
    for job in db.scalars(select(Job).where(Job.status == "running")):
        node = db.get(Node, job.node_id) if job.node_id else None
        if node is None or node.last_seen_at is None or node.last_seen_at < cutoff:
            job.status, job.finished_at = "failed", utcnow()
            job.error = "노드 연결이 끊겨 작업을 잃었습니다."
            lost += 1
    return lost
