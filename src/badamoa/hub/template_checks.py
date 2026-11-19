"""Trial runs of model template versions, started and approved by an admin (EXTENSION_PLAN 8).

An uploaded version waits (``pending``) until an admin starts its trial (``template.check``): one
server that trains runs it briefly on a small dataset the node draws itself, in the version's own
environment — train, then predict · eval · export as declared. A pass does not publish it: the
admin approves it (decided 2026-11-19: nothing runs or goes out on its own). Its state, kept with
the version (``Template.trial``):

    pending    uploaded, not tried yet                          not offered
    checking   its trial is queued or running                   not offered
    passed     the trial passed; waiting for an admin's "허용"   not offered
    ok         approved after passing                           offered
    failed     a step failed (which, and why)                   not offered
    allowed    let through without a pass, with a reason        offered
    legacy     registered before trials existed                 offered, whatever a trial says

(``unchecked`` from 0.53.0 reads as ``pending``.) Built-in templates are tried before BADAMOA ships
and are always ``ok``. Screens offer each template's newest version that may be offered, so a new
version waiting, being tried or failed does not take away the one people were using. Admins see
every version, with a warning on those not offered, and may still start them.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from badamoa.core.templates import check_params

from . import registry, resources, services
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import Template

CHECK_SINCE = (0, 53, 0)  # nodes that know the ``template.check`` job
OFFERED = ("ok", "allowed", "legacy")
NOT_YET = "이 모델 종류는 아직 관리자가 허용하지 않았습니다. 관리자에게 문의하세요."
SAID = {"pending": "시험 전", "checking": "시험 실행 중", "passed": "통과 · 허용 대기", "failed": "시험 실행 실패"}


def _version(node: Node) -> tuple[int, ...]:
    return tuple(int(p) for p in (node.agent_version or "0").split(".")[:3] if p.isdigit())


def state(row: Template) -> str:
    if row.builtin:
        return "ok"
    found = str((row.trial or {}).get("state") or "legacy")
    return "pending" if found == "unchecked" else found


def offered(row: Template) -> bool:
    """Whether everyone may train with this version (admins may use any). A version from before trials
    stays offered while an admin tries it, and after."""
    return state(row) in OFFERED or bool((row.trial or {}).get("legacy"))


def view(row: Template) -> dict[str, Any]:
    """The version's trial as screens show it (the installed package list stays out: it is long)."""
    trial = dict(row.trial or {})
    packages = trial.pop("installed", None)
    return {
        **trial,
        "state": state(row),
        "offered": offered(row),
        "packages": len(packages) if isinstance(packages, list) else trial.get("packages"),
        # what the other servers install with: the versions the passed trial installed
        "pinned": len(registry.pinned_for(row)),
    }


def _settle(row: Template, **fields: Any) -> None:
    row.trial = {k: v for k, v in fields.items() if v not in (None, "", [], {})}


def uploaded(row: Template) -> None:
    """A new version: it waits for an admin to try it."""
    if not row.builtin and not row.trial:
        _settle(row, state="pending")


# ---- starting a trial (an admin) -----------------------------------------------------------------


# A node that tries each task of a kind of several tasks (older ones would try the first only).
SEVERAL_TASKS_SINCE = (0, 60, 0)


def _task_pass(manifest: Any, task: str) -> dict[str, Any]:
    """One task's pass of a trial: its data format and settings."""
    mine = manifest.for_task(task)
    return {"task": task, "data_format": mine.data_format, "params": check_params(mine)}


def _servers(db: Session, settings: HubSettings, manifest: Any, gpu: bool) -> tuple[list[tuple[Node, dict, bool]], str]:
    """Servers that can try the template, with whether one of their GPUs is free now: those with a free GPU
    first, then the one already holding its environment; or why none can."""
    found: list[tuple[int, int, str, Node, dict[str, Any], bool]] = []
    assigned = resources.assignments(db)
    why = "켜져 있는 학습 서버가 없습니다 (노드 설정에서 '학습용' 역할)."
    for node in db.scalars(select(Node).order_by(Node.name)):
        if not registry.trains(node):
            continue
        since = SEVERAL_TASKS_SINCE if len(manifest.trains()) > 1 else CHECK_SINCE
        if _version(node) < since:
            need = ".".join(str(n) for n in since)
            what = "여러 태스크의 시험 실행을" if since == SEVERAL_TASKS_SINCE else "시험 실행을"
            why = f"학습 서버의 BADAMOA가 {what} 모릅니다. {need} 이상으로 업데이트하세요."
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
        free = bool(resources.free_gpus(node, settings, assigned.get(node.id)))
        found.append((0 if free else 1, 0 if ready else 1, node.name, node, spec, free))
    found.sort(key=lambda f: (f[0], f[1], f[2]))
    return [(node, spec, free) for _, _, _, node, spec, free in found], why


def start(db: Session, settings: HubSettings, user: User, template_id: str, via: str = "api") -> Job:
    """An admin starts the version's trial on a server that can run it (refused, with why, when none can)."""
    services.require_admin(user)
    row = registry.get_template(db, template_id)
    if row.builtin:
        raise ServiceError(400, "기본 제공 모델 종류는 BADAMOA를 낼 때 확인했습니다.")
    before = state(row)
    if before == "checking":
        raise ServiceError(409, f"{row.title} {row.version}은(는) 이미 시험 실행 중입니다.")
    manifest = registry.manifest_of(row)
    servers, why = _servers(db, settings, manifest, manifest.check.gpu)
    if not servers:
        raise ServiceError(409, why)
    node, spec, free = servers[0]
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
            "tasks": manifest.trains(),
            "autolabel": manifest.autolabel,
            "entrypoints": manifest.entrypoints,
        },
        "environment": spec,
        "params": check_params(manifest.for_task(manifest.task)),
        # a kind of several tasks is tried on each of them in turn, with that task's data format and settings
        **({"tasks": [_task_pass(manifest, t) for t in manifest.trains()]} if len(manifest.trains()) > 1 else {}),
        "check": manifest.check.model_dump(exclude={"params"}),
        "data": manifest.data.model_dump(),
    }
    # one GPU when the server has an idle one as the job starts (quicker, and what trainings use), else the
    # CPU; ``[check] gpu = true``: code that needs a GPU waits for one
    requirements: dict[str, Any] = {}
    if manifest.check.gpu or free:
        requirements = {"roles": ["compute"], "gpus": 1, "min_vram_mb": 0}
        if not manifest.check.gpu:
            requirements["gpu_if_free"] = True
    job = Job(
        type="template.check",
        params=params,
        target_node_id=node.id,
        requirements=requirements,
        created_by=user.id,
        created_via=via,
    )
    db.add(job)
    db.flush()
    legacy = before == "legacy" or bool((row.trial or {}).get("legacy"))
    _settle(
        row,
        state="checking",
        job_id=job.id,
        node=node.name,
        before=before,
        legacy=legacy or None,
        started_at=iso(utcnow()),
        by=user.username,
    )
    services.audit(db, user.id, via, "template.check", row.id, name=row.name, version=row.version, node=node.name)
    return job


def allow(db: Session, user: User, template_id: str, reason: str = "", via: str = "api") -> Template:
    """An admin offers the version to everyone: one that passed its trial as it is, any other only with
    a reason, which is kept with it and in the audit log."""
    services.require_admin(user)
    row = registry.get_template(db, template_id)
    if row.builtin:
        raise ServiceError(400, "기본 제공 모델 종류는 이미 쓸 수 있습니다.")
    before = dict(row.trial or {})
    reason = reason.strip()
    now = iso(utcnow())
    if state(row) == "passed":
        kept = {k: v for k, v in before.items() if k not in ("state", "legacy")}
        _settle(row, **kept, state="ok", approved_by=user.username, approved_at=now)
        services.audit(db, user.id, via, "template.allow", row.id, name=row.name, version=row.version, passed=True)
        return row
    if state(row) == "checking":
        raise ServiceError(409, "시험 실행이 끝난 뒤에 허용하세요.")
    if not reason:
        raise ServiceError(400, "시험 실행을 통과하지 않은 버전입니다. 허용하는 이유를 적어 주세요. 기록에 남습니다.")
    _settle(
        row,
        state="allowed",
        why=reason[:500],
        approved_by=user.username,
        approved_at=now,
        was=state(row),
        reason=before.get("reason"),
        failed=before.get("failed"),
        job_id=before.get("job_id"),
    )
    services.audit(db, user.id, via, "template.allow", row.id, name=row.name, version=row.version, why=reason[:500])
    return row


def require_offered(row: Template, user: User) -> None:
    """Members train only with versions an admin offered, and with their own drafts; admins with any."""
    if user.role != "admin" and not offered(row) and row.created_by != user.id:
        raise ServiceError(403, NOT_YET)


# ---- what came back -----------------------------------------------------------------------------


def _passed_training(db: Session, row: Template) -> str:
    """A training with this version that succeeded, if any (its job id)."""
    jobs = db.scalars(select(Job).where(Job.type == "train.run", Job.status == "succeeded"))
    return next((j.id for j in jobs if ((j.params or {}).get("template") or {}).get("id") == row.id), "")


def on_job_finished(db: Session, job: Job) -> None:
    """Keep a trial's verdict with its version. A pass waits for an admin to offer it."""
    if job.type != "template.check":
        return
    row = db.get(Template, (job.params or {}).get("template_id", ""))
    trial = dict(row.trial or {}) if row is not None else {}
    if row is None or trial.get("job_id") != job.id:
        return  # another trial took its place
    node = db.get(Node, job.node_id) if job.node_id else None
    result = job.result or {}
    common = {
        "job_id": job.id,
        "node": node.name if node else "",
        "device": ("CPU" if str(result.get("device")) == "cpu" else "GPU") if result.get("device") else "",
        "checked_at": iso(utcnow()),
        "by": trial.get("by"),
        "legacy": trial.get("legacy"),
    }
    if job.status == "succeeded":
        _settle(
            row,
            **common,
            state="passed" if result.get("ok") else "failed",
            failed=result.get("failed"),
            reason=result.get("reason"),
            steps=result.get("steps"),
            notes=result.get("notes"),
            took_s=result.get("took_s"),
            environment=result.get("environment"),
            metrics=result.get("metrics"),
            installed=result.get("packages"),
            sources=result.get("sources"),
            contract=result.get("contract"),
        )
    else:  # stopped, or the job itself broke (a server that went away): no verdict on the code
        stopped = job.status in ("canceled", "cancelled")
        why = "시험 실행을 멈췄습니다." if stopped else (job.error or "시험 실행 작업이 끝나지 못했습니다.")
        _settle(row, **common, state=trial.get("before") or "pending", reason=why)


# ---- versions from before trials ---------------------------------------------------------------


def mark_legacy(db: Session) -> None:
    """When the Hub starts: a version registered before trials existed stays offered — ``ok`` if a training
    with it succeeded, else ``legacy``. An admin may try it by hand."""
    for row in db.scalars(select(Template).where(Template.builtin.is_(False), Template.deleted_at.is_(None))):
        if row.trial:
            continue
        done = _passed_training(db, row)
        if done:
            _settle(row, state="ok", via="run", job_id=done)
        else:
            _settle(row, state="legacy")
