"""What an admin sets up once so everyone else only picks by name: model templates and environments."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa import templates as shipped
from badamoa.core.storage import Storage
from badamoa.core.templates import TemplateError, TemplateManifest, load_manifest, read_bundle

from . import services
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
    shown = [t for t in newest.values() if (include_hidden or not t.hidden) and (not task or t.task == task)]
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
    kept = ModelVersion.deleted_at.is_(None)
    models = int(db.scalar(select(func.coalesce(func.sum(ModelVersion.bytes), 0)).where(kept)) or 0)
    outputs = int(db.scalar(select(func.coalesce(func.sum(RunFile.bytes), 0))) or 0)
    outputs += sum(int(f.get("bytes") or 0) for op in db.scalars(select(ModelJob)) for f in op.files or [])
    checkpoints = sum(int((c or {}).get("bytes") or 0) for c in db.scalars(select(Run.checkpoint)))
    disk = shutil.disk_usage(settings.home)
    return {
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
            "other": max(0, stored - models - outputs - checkpoints) + database,
        },
    }
