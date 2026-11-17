"""What an admin sets up once so everyone else only picks by name: model templates and environments."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa import templates as shipped
from badamoa.core import tasks as task_names
from badamoa.core.storage import Storage
from badamoa.core.templates import TemplateError, TemplateManifest, load_manifest, read_bundle

from . import cleanup, services
from .catalog_models import Project
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import Environment, ModelJob, ModelVersion, Run, RunFile, Template

DEFAULT_TEMPLATE = "yolo"
# What a project trains with when nobody chose: the built-in template for its task.
DEFAULT_BY_TASK = {"detect": "yolo", "segment": "yolo-seg", "classify": "classify"}
AUTO = {"kind": "auto", "name": "자동 설치"}
_ENV_NAME_RE = re.compile(r"^[A-Za-z0-9가-힣][A-Za-z0-9가-힣 ._-]{0,62}$")


# ---- templates ---------------------------------------------------------------------


def sync_builtin(db: Session) -> None:
    """Make the templates shipped in the package visible in the registry (called when the Hub starts)."""
    for name, folder in shipped.builtin().items():
        manifest = load_manifest(folder)
        row = db.scalar(select(Template).where(Template.name == name, Template.version == manifest.version))
        if row is None:
            row = Template(name=name, version=manifest.version, builtin=True)
            db.add(row)
        row.title, row.task, row.hidden = manifest.title, manifest.task, manifest.hidden
        row.manifest, row.builtin, row.deleted_at = manifest.model_dump(), True, None
        db.flush()
        # A node only carries the version shipped with it, so earlier built-in versions leave the list.
        older = select(Template).where(Template.name == name, Template.builtin, Template.id != row.id)
        for previous in db.scalars(older):
            previous.deleted_at = previous.deleted_at or utcnow()


def register_template(db: Session, user: User, data: bytes, storage: Storage, via: str = "api") -> Template:
    """Add a template version from an uploaded bundle (the template folder as a zip)."""
    services.require_admin(user)
    try:
        manifest = read_bundle(data)
    except TemplateError as e:
        raise ServiceError(400, str(e)) from e
    if manifest.name in shipped.builtin():
        raise ServiceError(409, f"{manifest.name}은 내장 템플릿의 이름입니다. 다른 name을 쓰세요.")
    blob = storage.put_blob(data)
    row = db.scalar(select(Template).where(Template.name == manifest.name, Template.version == manifest.version))
    if row is not None and row.blob != blob:
        same = f"{manifest.name} {manifest.version}"
        raise ServiceError(409, f"{same}은 이미 다른 내용으로 등록되어 있습니다. badamoa.toml의 version을 올리세요.")
    if row is None:
        row = Template(name=manifest.name, version=manifest.version, blob=blob, bytes=len(data), created_by=user.id)
        db.add(row)
    row.title, row.task, row.hidden = manifest.title, manifest.task, manifest.hidden
    row.manifest, row.deleted_at = manifest.model_dump(), None
    db.flush()
    services.audit(db, user.id, via, "template.register", row.id, name=row.name, version=row.version)
    return row


def template_for(db: Session, settings: dict[str, Any] | None, task: str, ref: str = "") -> Template:
    """The template a run on data of ``task`` uses when none is named: the project's last choice
    if it trains that task, else the task's built-in one, else any template that declares the task."""
    if ref:
        return get_template(db, ref)
    saved = (settings or {}).get("template")
    if saved:
        try:
            row = get_template(db, saved)
        except ServiceError:
            row = None
        if row is not None and task_names.same(row.task, task):
            return row
    if task in DEFAULT_BY_TASK:
        return get_template(db, DEFAULT_BY_TASK[task])
    offered = list_templates(db, task=task)
    if offered:
        return get_template(db, offered[0]["id"])
    if task == "pose":
        raise ServiceError(
            400,
            "Kpt 학습 템플릿은 아직 없습니다(지금은 가져오기 · 보기까지). "
            "박스로 학습하려면 같은 폴더를 태스크 Det에 두고 새 데이터셋으로 가져오세요.",
        )
    raise ServiceError(
        400, f"{task_names.short(task)} 태스크를 학습하는 템플릿이 없습니다. 이 태스크를 적은 모델 템플릿을 등록하세요."
    )


def get_template(db: Session, ref: str) -> Template:
    """By id, or by name (the newest version)."""
    row = db.get(Template, ref)
    if row is None:
        row = db.scalar(
            select(Template)
            .where(Template.name == ref, Template.deleted_at.is_(None))
            .order_by(Template.created_at.desc(), Template.id)
            .limit(1)
        )
    if row is None or row.deleted_at is not None:
        raise ServiceError(404, f"모델 템플릿을 찾을 수 없습니다: {ref}")
    return row


def manifest_of(row: Template) -> TemplateManifest:
    return TemplateManifest(**row.manifest)


def template_view(db: Session, row: Template, versions: int = 1, models: int = 0) -> dict[str, Any]:
    manifest = manifest_of(row)
    return {
        "id": row.id,
        "name": row.name,
        "title": row.title,
        "version": row.version,
        "versions": versions,
        "task": row.task,
        "description": manifest.description,
        "license": manifest.license,
        "builtin": row.builtin,
        "hidden": row.hidden,
        "data_format": manifest.data_format,
        "entrypoints": sorted(manifest.entrypoints),
        "variant": manifest.variant,
        "params": [{"name": name, **spec.model_dump()} for name, spec in manifest.params.items()],
        "environment": {
            "default": manifest.environment.default,
            "requirements": manifest.environment.requirements,
            **environment_status(db, row, manifest),
        },
        "bytes": row.bytes,
        "models": models,
        "created_at": iso(row.created_at),
    }


def list_templates(db: Session, include_hidden: bool = False, task: str = "") -> list[dict[str, Any]]:
    """The newest version of each template, optionally only those for one task."""
    rows = db.scalars(
        select(Template).where(Template.deleted_at.is_(None)).order_by(Template.created_at.desc(), Template.id)
    ).all()
    counts: dict[str, int] = {}
    newest: dict[str, Template] = {}
    for row in rows:
        counts[row.name] = counts.get(row.name, 0) + 1
        newest.setdefault(row.name, row)
    models = dict(
        db.execute(
            select(ModelVersion.trainer, func.count())
            .where(ModelVersion.deleted_at.is_(None))
            .group_by(ModelVersion.trainer)
        ).all()
    )
    shown = [
        t for t in newest.values() if (include_hidden or not t.hidden) and (not task or task_names.same(t.task, task))
    ]
    first = list(dict.fromkeys(DEFAULT_BY_TASK.values()))  # each task's default template leads, detection first
    shown.sort(key=lambda t: (first.index(t.name) if t.name in first else len(first), not t.builtin, t.title.lower()))
    return [template_view(db, t, counts[t.name], models.get(t.name, 0)) for t in shown]


def delete_template(db: Session, user: User, template_id: str) -> None:
    """Remove every version of a template from the list. Runs and models made with it keep their record."""
    services.require_admin(user)
    row = get_template(db, template_id)
    if row.builtin:
        raise ServiceError(400, "내장 템플릿은 지울 수 없습니다.")
    for version in db.scalars(select(Template).where(Template.name == row.name, Template.deleted_at.is_(None))):
        version.deleted_at = utcnow()
    services.audit(db, user.id, "api", "template.delete", row.id, name=row.name)


def restore_template(db: Session, user: User, template_id: str) -> Template:
    """Back on the list from the trash, every version of it."""
    services.require_admin(user)
    row = db.get(Template, template_id)
    if row is None or row.deleted_at is None or row.builtin:
        raise ServiceError(404, "휴지통에 그 모델 종류가 없습니다.")
    for version in db.scalars(select(Template).where(Template.name == row.name, Template.deleted_at.is_not(None))):
        version.deleted_at = None
    services.audit(db, user.id, "api", "template.restore", row.id, name=row.name)
    return row


# ---- environments ------------------------------------------------------------------


def _environment(db: Session, ref: str) -> Environment | None:
    return db.get(Environment, ref) or db.scalar(select(Environment).where(Environment.name == ref))


def get_environment(db: Session, ref: str) -> Environment:
    env = _environment(db, ref)
    if env is None:
        raise ServiceError(404, f"환경을 찾을 수 없습니다: {ref}")
    return env


def _clean_paths(db: Session, paths: dict[str, str], previous: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for node_id, path in paths.items():
        path = (path or "").strip()
        if not path:
            continue
        services.get_node(db, node_id)
        old = previous.get(node_id) or {}
        # A changed path has not been checked yet.
        nodes[node_id] = old if old.get("path") == path else {"path": path, "ok": None, "detail": {}, "checked_at": None}
    return nodes


def save_environment(
    db: Session, user: User, *, name: str, description: str = "", paths: dict[str, str] | None = None, env_id: str = ""
) -> Environment:
    """Create or update an "existing Python" environment: per node, the path of its interpreter."""
    services.require_admin(user)
    name = name.strip()
    if not _ENV_NAME_RE.match(name) or name == AUTO["name"]:
        raise ServiceError(400, "환경 이름은 1~63자의 글자·숫자·공백·`-`·`_`·`.`로 지어 주세요.")
    env = db.get(Environment, env_id) if env_id else None
    if env_id and env is None:
        raise ServiceError(404, "환경을 찾을 수 없습니다.")
    clash = db.scalar(select(Environment).where(Environment.name == name))
    if clash is not None and (env is None or clash.id != env.id):
        raise ServiceError(409, "같은 이름의 환경이 이미 있습니다.")
    if env is None:
        env = Environment(name=name, kind="python", created_by=user.id)
        db.add(env)
    env.name, env.description = name, description.strip()
    env.nodes = _clean_paths(db, paths or {}, env.nodes or {})
    db.flush()
    services.audit(db, user.id, "api", "environment.save", env.id, name=name)
    return env


def delete_environment(db: Session, user: User, env_id: str) -> None:
    services.require_admin(user)
    env = get_environment(db, env_id)
    services.audit(db, user.id, "api", "environment.delete", env.id, name=env.name)
    db.delete(env)


def environment_view(db: Session, settings: HubSettings, env: Environment) -> dict[str, Any]:
    names = services.node_name_map(db)
    return {
        "id": env.id,
        "name": env.name,
        "kind": env.kind,
        "description": env.description,
        "created_at": iso(env.created_at),
        "nodes": [
            {
                "node_id": node_id,
                "node_name": names.get(node_id, "삭제된 노드"),
                "path": entry.get("path", ""),
                "ok": entry.get("ok"),
                "detail": entry.get("detail") or {},
                "checked_at": entry.get("checked_at"),
            }
            for node_id, entry in (env.nodes or {}).items()
        ],
    }


# ---- automatically installed environments, ahead of the first training ------------------


def _auto_spec(manifest: TemplateManifest) -> dict[str, Any]:
    return {**AUTO, **manifest.environment.model_dump(exclude={"default"})}


def _same_env(found: dict[str, Any], spec: dict[str, Any]) -> bool:
    """An environment a node reported is the one a template installs (older reports carry no torch flag)."""
    return (
        sorted(found.get("requirements") or []) == sorted(spec.get("requirements") or [])
        and (found.get("python") or "3.12") == (spec.get("python") or "3.12")
        and found.get("torch") in (None, bool(spec.get("torch")))
    )


def environment_status(db: Session, row: Template, manifest: TemplateManifest | None = None) -> dict[str, Any]:
    """Where a template's automatically installed environment is ready, and where it is being prepared.

    ``installs``: whether there is anything to install at all (a template that names an
    environment an admin registered, or needs only the standard library, has nothing to prepare).
    """
    manifest = manifest or manifest_of(row)
    spec = _auto_spec(manifest)
    if manifest.environment.default or not spec.get("requirements"):
        return {"installs": False, "ready": [], "preparing": []}
    nodes = [n for n in db.scalars(select(Node).order_by(Node.name)) if n.enabled]
    ready = [n.name for n in nodes if any(_same_env(e, spec) for e in (n.info or {}).get("envs") or [])]
    active = db.scalars(select(Job).where(Job.type == "env.prepare", Job.status.in_(("queued", "running"))))
    names = {n.id: n.name for n in nodes}
    mine = [j for j in active if (j.params or {}).get("template") == row.name]
    preparing = sorted({names.get(j.target_node_id or j.node_id or "", "") for j in mine} - {""})
    return {"installs": True, "ready": ready, "preparing": preparing, "nodes": len(nodes)}


def prepare_environment(
    db: Session, user: User, template_id: str, node_ids: list[str] | None = None, via: str = "api"
) -> list[Job]:
    """Install a template's environment on servers before anyone trains with it (all enabled ones by default)."""
    services.require_admin(user)
    row = get_template(db, template_id)
    manifest = manifest_of(row)
    if not environment_status(db, row, manifest)["installs"]:
        reason = "등록된 환경이나 서버의 Python을 씁니다"
        raise ServiceError(400, f"{row.title}은(는) 자동으로 설치할 환경이 없습니다 ({reason}).")
    nodes = [db.get(Node, n) for n in node_ids] if node_ids else list(db.scalars(select(Node).order_by(Node.name)))
    nodes = [n for n in nodes if n is not None and n.enabled]
    if not nodes:
        raise ServiceError(400, "환경을 준비할 서버가 없습니다.")
    params = {"template": row.name, "version": row.version, "environment": _auto_spec(manifest)}
    jobs = [services.create_job(db, user, "env.prepare", params, n.id, via=via, gpus=0) for n in nodes]
    services.audit(db, user.id, via, "environment.prepare", row.id, template=row.name, nodes=[n.name for n in nodes])
    return jobs


def list_environments(db: Session, settings: HubSettings) -> list[dict[str, Any]]:
    return [environment_view(db, settings, e) for e in db.scalars(select(Environment).order_by(Environment.name))]


def check_environment(db: Session, user: User, env_id: str) -> list[Job]:
    """Ask every node the environment is set up on to try it. Results arrive when the jobs finish."""
    services.require_admin(user)
    env = get_environment(db, env_id)
    jobs = []
    nodes = dict(env.nodes or {})
    for node_id, entry in nodes.items():
        node = db.get(Node, node_id)
        if node is None or not node.enabled:
            continue
        job = services.create_job(
            db, user, "env.check", {"environment_id": env.id, "python": entry["path"]}, node_id, via="ui", gpus=0
        )
        nodes[node_id] = {**entry, "ok": None, "job_id": job.id}
        jobs.append(job)
    env.nodes = nodes
    if not jobs:
        raise ServiceError(400, "이 환경에 경로를 적어 둔 노드가 없습니다.")
    return jobs


def on_job_finished(db: Session, job: Job) -> None:
    """Record what an ``env.check`` job found."""
    if job.type != "env.check" or not job.node_id:
        return
    env = db.get(Environment, (job.params or {}).get("environment_id", ""))
    entry = (env.nodes or {}).get(job.node_id) if env else None
    if env is None or entry is None or entry.get("job_id") != job.id:
        return
    result = job.result or {}
    ok = job.status == "succeeded" and bool(result.get("ok"))
    detail = result if job.status == "succeeded" else {"error": job.error or "확인 작업이 끝나지 못했습니다."}
    env.nodes = {**env.nodes, job.node_id: {**entry, "ok": ok, "detail": detail, "checked_at": iso(utcnow())}}


def environment_for_run(db: Session, manifest: TemplateManifest, wanted: str, node: Node) -> dict[str, Any]:
    """The environment a run will use on ``node``, as a snapshot the node can act on.

    ``wanted`` is an environment id or name; empty means the template's default, and
    when the template names none, the automatically installed one.
    """
    auto = {**AUTO, **manifest.environment.model_dump(exclude={"default"})}
    if wanted in ("auto", AUTO["name"]):
        return auto
    env = _environment(db, wanted or manifest.environment.default) if (wanted or manifest.environment.default) else None
    if env is None:
        if wanted:
            raise ServiceError(404, f"환경을 찾을 수 없습니다: {wanted}")
        return auto  # the template's default is not registered on this Hub: fall back to installing
    entry = (env.nodes or {}).get(node.id)
    if not entry or not entry.get("path"):
        where = f"'{env.name}' 환경이 {node.name} 서버에"
        raise ServiceError(400, f"{where} 준비되어 있지 않습니다. 관리자에게 설정 → 환경에서 등록을 요청하세요.")
    return {"kind": env.kind, "id": env.id, "name": env.name, "path": entry["path"]}


# ---- where everything is kept ---------------------------------------------------------


def storage_usage(db: Session, settings: HubSettings, user: User) -> dict[str, Any]:
    """How much the Hub keeps and of what. The Hub's folder holds every record: nodes keep only caches."""
    stored = files = 0
    for folder, _, names in os.walk(settings.storage_dir):
        for name in names:
            try:
                stored += os.path.getsize(os.path.join(folder, name))
                files += 1
            except OSError:
                pass  # removed meanwhile
    database = sum(p.stat().st_size for p in settings.home.glob("hub.db*") if p.is_file())
    # What only the trash holds: deleted runs and models, and those of deleted projects.
    gone = set(db.scalars(select(Project.id).where(Project.deleted_at.is_not(None))))
    models = outputs = checkpoints = trash = 0
    binned_models: set[str] = set()
    for model_id, project_id, deleted_at, size in db.execute(
        select(ModelVersion.id, ModelVersion.project_id, ModelVersion.deleted_at, ModelVersion.bytes)
    ):
        if deleted_at is not None or project_id in gone:
            binned_models.add(model_id)
            trash += size or 0
        else:
            models += size or 0
    binned_runs: set[str] = set()
    for run_id, project_id, deleted_at, checkpoint in db.execute(
        select(Run.id, Run.project_id, Run.deleted_at, Run.checkpoint)
    ):
        size = int((checkpoint or {}).get("bytes") or 0)
        if deleted_at is not None or project_id in gone:
            binned_runs.add(run_id)
            trash += size
        else:
            checkpoints += size
    for run_id, size in db.execute(select(RunFile.run_id, RunFile.bytes)):
        if run_id in binned_runs:
            trash += size or 0
        else:
            outputs += size or 0
    for op in db.scalars(select(ModelJob)):
        size = sum(int(f.get("bytes") or 0) for f in op.files or [])
        if op.model_id in binned_models:
            trash += size
        else:
            outputs += size
    disk = shutil.disk_usage(settings.home)
    return {
        "policy": cleanup.get_policy(db),
        "projects": project_usage(db),
        "path": str(settings.home.resolve()) if user.role == "admin" else None,
        "bytes": stored + database,
        "files": files,
        "free": disk.free,
        "disk": disk.total,
        # The same bytes recorded twice are stored once, so the parts can add up to more than the whole.
        "parts": {
            "models": models,
            "outputs": outputs,
            "checkpoints": checkpoints,
            "trash": trash,
            "other": max(0, stored - models - outputs - checkpoints - trash) + database,
        },
    }


def project_usage(db: Session) -> list[dict[str, Any]]:
    """How much each project keeps: its models, its runs' and model jobs' outputs, its checkpoints."""
    usage: dict[str, dict[str, Any]] = {
        p.id: {"id": p.id, "name": p.name, "models": 0, "outputs": 0, "checkpoints": 0}
        for p in db.scalars(select(Project).where(Project.deleted_at.is_(None)))
    }
    model_project: dict[str, str] = {}
    for model in db.scalars(select(ModelVersion).where(ModelVersion.deleted_at.is_(None))):
        model_project[model.id] = model.project_id
        if model.project_id in usage:
            usage[model.project_id]["models"] += model.bytes or 0
    run_project = {r.id: r.project_id for r in db.scalars(select(Run).where(Run.deleted_at.is_(None)))}
    for run_id, size in db.execute(select(RunFile.run_id, RunFile.bytes)):
        project = run_project.get(run_id)
        if project in usage:
            usage[project]["outputs"] += size or 0
    for op in db.scalars(select(ModelJob)):
        project = model_project.get(op.model_id)
        if project in usage:
            usage[project]["outputs"] += sum(int(f.get("bytes") or 0) for f in op.files or [])
    for run in db.scalars(select(Run).where(Run.deleted_at.is_(None))):
        if run.project_id in usage:
            usage[run.project_id]["checkpoints"] += int((run.checkpoint or {}).get("bytes") or 0)
    rows = [{**u, "total": u["models"] + u["outputs"] + u["checkpoints"]} for u in usage.values()]
    return sorted(rows, key=lambda u: -u["total"])

