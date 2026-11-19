"""Trial runs of model template versions before anyone is offered them (EXTENSION_PLAN 8).

Every uploaded version gets a short run on one server that trains (``template.check``): a small
dataset the node draws itself, the version's own environment, train and then predict · eval ·
export as declared. Its state, kept with the version (``Template.trial``):

    checking   the trial is queued or running            not offered
    ok         it passed                                 offered
    failed     a step failed (which, and why)            not offered
    unchecked  no server could run it                    not offered
    allowed    an admin let it through, with a reason    offered
    legacy     registered before trials existed          offered until its trial says otherwise

Built-in templates are tried before BADAMOA ships and are always ``ok``. Screens offer each
template's newest version that may be offered, so a new version that is still being tried, or
that failed, does not take away the one people were using. Admins see every version, with a
warning on those not offered, and may still start them.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core.templates import check_params

from . import registry, services
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import Template

CHECK_SINCE = (0, 53, 0)  # nodes that know the ``template.check`` job
OFFERED = ("ok", "allowed", "legacy")
NOT_YET = "이 모델 종류는 아직 확인되지 않았습니다. 관리자에게 문의하세요."


def _version(node: Node) -> tuple[int, ...]:
    return tuple(int(p) for p in (node.agent_version or "0").split(".")[:3] if p.isdigit())


def state(row: Template) -> str:
    return "ok" if row.builtin else str((row.trial or {}).get("state") or "legacy")


def offered(row: Template) -> bool:
    """Whether everyone may train with this version (admins may use any)."""
    check = row.trial or {}
    return state(row) in OFFERED or (state(row) == "checking" and bool(check.get("legacy")))


def view(row: Template) -> dict[str, Any]:
    """The version's check as screens show it (the installed package list stays out: it is long)."""
    check = dict(row.trial or {})
    packages = check.pop("installed", None)
    return {
        **check,
        "state": state(row),
        "offered": offered(row),
        "packages": len(packages) if isinstance(packages, list) else check.get("packages"),
    }


def _settle(row: Template, **fields: Any) -> None:
    row.trial = {k: v for k, v in fields.items() if v not in (None, "", [], {})}


# ---- starting a trial ---------------------------------------------------------------------------


def _servers(db: Session, settings: HubSettings, manifest: Any, gpu: bool) -> tuple[list[tuple[Node, dict]], str]:
    """Servers that can try the template, the one already holding its environment first; or why none can."""
    found: list[tuple[int, str, Node, dict[str, Any]]] = []
    why = "켜져 있는 학습 서버가 없습니다 (노드 설정에서 '학습·오토라벨' 역할)."
    for node in db.scalars(select(Node).order_by(Node.name)):
        if not registry.trains(node):
            continue
        if _version(node) < CHECK_SINCE:
            need = ".".join(str(n) for n in CHECK_SINCE)
            why = f"학습 서버의 BADAMOA가 시험 실행을 모릅니다. {need} 이상으로 업데이트하세요."
            continue
        if services.node_status(node, settings) != "online":
            continue
        try:
            spec = registry.environment_for_run(db, manifest, "", node)
        except ServiceError as e:
            why = e.message
            continue
        if spec.get("kind") == "docker" and not registry.has_docker(node):
            why = "이 모델 종류는 Docker 이미지로 돕니다. Docker가 있는 학습 서버가 없습니다."
            continue
        if gpu and not (node.info or {}).get("gpus"):
            why = "이 모델 종류는 시험 실행에 GPU가 필요합니다 ([check] gpu). GPU가 있는 학습 서버가 없습니다."
            continue
        ready = any(registry._same_env(e, spec) for e in (node.info or {}).get("envs") or [])
        found.append((0 if ready else 1, node.name, node, spec))
    found.sort(key=lambda f: (f[0], f[1]))
    return [(node, spec) for _, _, node, spec in found], why


def start(
    db: Session, settings: HubSettings, row: Template, user_id: str | None, via: str = "api", legacy: bool = False
) -> Job | None:
    """Queue the version's trial on a server that can run it. None when no server can (the reason is kept);
    a version from before trials keeps being offered meanwhile (``legacy``)."""
    if row.builtin:
        return None
    manifest = registry.manifest_of(row)
    servers, why = _servers(db, settings, manifest, manifest.check.gpu)
    if not servers:
        if not legacy:
            _settle(row, state="unchecked", reason=why, checked_at=iso(utcnow()))
        return None
    node, spec = servers[0]
    params = {
        "template_id": row.id,
        "template": {
            "name": row.name,
            "version": row.version,
            "builtin": row.builtin,
            "sha256": row.blob or "",
            "task": manifest.task,
            "data_format": manifest.data_format,
            "resume": manifest.resume,
            "entrypoints": manifest.entrypoints,
        },
        "environment": spec,
        "params": check_params(manifest),
        "check": manifest.check.model_dump(exclude={"params"}),
        "data": manifest.data.model_dump(),
    }
    gpus = 1 if manifest.check.gpu else 0
    job = Job(
        type="template.check",
        params=params,
        target_node_id=node.id,
        requirements={"roles": ["compute"], "gpus": gpus, "min_vram_mb": 0} if gpus else {},
        created_by=user_id,
        created_via=via,
    )
    db.add(job)
    db.flush()
    _settle(row, state="checking", job_id=job.id, node=node.name, legacy=legacy or None, started_at=iso(utcnow()))
    services.audit(db, user_id, via, "template.check", row.id, name=row.name, version=row.version, node=node.name)
    return job


def recheck(db: Session, settings: HubSettings, user: User, template_id: str, via: str = "api") -> Job:
    services.require_admin(user)
    row = registry.get_template(db, template_id)
    if row.builtin:
        raise ServiceError(400, "기본 제공 모델 종류는 BADAMOA를 낼 때 확인했습니다.")
    if state(row) == "checking":
        raise ServiceError(409, f"{row.title} {row.version}은(는) 이미 시험 실행 중입니다.")
    job = start(db, settings, row, user.id, via)
    if job is None:
        raise ServiceError(409, (row.trial or {}).get("reason") or "시험 실행할 서버가 없습니다.")
    return job


def allow(db: Session, user: User, template_id: str, reason: str, via: str = "api") -> Template:
    """Offer a version without a passed trial: an admin's decision, with the reason on record."""
    services.require_admin(user)
    row = registry.get_template(db, template_id)
    reason = reason.strip()
    if not reason:
        raise ServiceError(400, "허용하는 이유를 적어 주세요. 기록에 남습니다.")
    if row.builtin:
        raise ServiceError(400, "기본 제공 모델 종류는 이미 쓸 수 있습니다.")
    before = dict(row.trial or {})
    _settle(
        row,
        state="allowed",
        why=reason[:500],
        by=user.username,
        at=iso(utcnow()),
        was=before.get("state"),
        reason=before.get("reason"),
        failed=before.get("failed"),
    )
    services.audit(db, user.id, via, "template.allow", row.id, name=row.name, version=row.version, why=reason[:500])
    return row


def require_offered(row: Template, user: User) -> None:
    """Members train only with versions that passed (or were let through); admins with any."""
    if user.role != "admin" and not offered(row):
        raise ServiceError(403, NOT_YET)


# ---- what came back -----------------------------------------------------------------------------


def _passed_training(db: Session, row: Template) -> str:
    """A training with this version that succeeded, if any (its job id)."""
    jobs = db.scalars(select(Job).where(Job.type == "train.run", Job.status == "succeeded"))
    return next((j.id for j in jobs if ((j.params or {}).get("template") or {}).get("id") == row.id), "")


def on_job_finished(db: Session, job: Job) -> None:
    """Keep a trial's verdict with its version; a version a real training ran with is taken as passed."""
    if job.type == "train.run" and job.status == "succeeded":
        row = db.get(Template, ((job.params or {}).get("template") or {}).get("id", ""))
        if row is not None and state(row) in ("unchecked", "allowed", "legacy"):
            _settle(row, state="ok", via="run", job_id=job.id, checked_at=iso(utcnow()))
        return
    if job.type != "template.check":
        return
    row = db.get(Template, (job.params or {}).get("template_id", ""))
    if row is None or (row.trial or {}).get("job_id") != job.id:
        return  # another trial took its place
    node = db.get(Node, job.node_id) if job.node_id else None
    common = {"job_id": job.id, "node": node.name if node else "", "checked_at": iso(utcnow())}
    result = job.result or {}
    if job.status == "succeeded":
        _settle(
            row,
            **common,
            state="ok" if result.get("ok") else "failed",
            failed=result.get("failed"),
            reason=result.get("reason"),
            steps=result.get("steps"),
            notes=result.get("notes"),
            took_s=result.get("took_s"),
            environment=result.get("environment"),
            metrics=result.get("metrics"),
            installed=result.get("packages"),
        )
    else:  # stopped, or the job itself broke (a server that went away): no verdict on the code
        stopped = job.status in ("canceled", "cancelled")
        why = "시험 실행을 멈췄습니다." if stopped else (job.error or "시험 실행 작업이 끝나지 못했습니다.")
        if (row.trial or {}).get("legacy"):  # keeps being offered as before; an admin may try again
            _settle(row, **common, state="legacy", tried=iso(utcnow()), reason=why)
        else:
            _settle(row, **common, state="unchecked" if stopped else "failed", reason=why)


# ---- versions from before trials ---------------------------------------------------------------


def mark_legacy(db: Session) -> None:
    """When the Hub starts: a version registered before trials existed is ``ok`` if a training with it
    succeeded, else ``legacy`` (offered, and tried as soon as a server can)."""
    for row in db.scalars(select(Template).where(Template.builtin.is_(False), Template.deleted_at.is_(None))):
        if row.trial:
            continue
        done = _passed_training(db, row)
        if done:
            _settle(row, state="ok", via="run", job_id=done)
        else:
            _settle(row, state="legacy")


def queue_legacy(db: Session, settings: HubSettings, node: Node) -> None:
    """A server that can try templates just reported in: queue the trials of versions from before trials
    (each template's newest), at most once each."""
    if not registry.trains(node) or _version(node) < CHECK_SINCE:
        return
    rows = db.scalars(
        select(Template)
        .where(Template.deleted_at.is_(None), func.json_extract(Template.trial, "$.state") == "legacy")
        .order_by(Template.created_at.desc())
    ).all()
    newest: dict[str, Template] = {}
    for row in rows:
        newest.setdefault(row.name, row)
    for row in newest.values():
        if (row.trial or {}).get("tried"):
            continue
        job = start(db, settings, row, None, "system", legacy=True)
        if job is None:
            row.trial = {**(row.trial or {}), "tried": iso(utcnow())}  # nobody can yet: not on every heartbeat
