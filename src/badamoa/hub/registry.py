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
from badamoa.core.templates import IMAGE_RE, TemplateError, TemplateManifest, load_manifest, read_bundle

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
        "data": manifest.data.model_dump(),
        "notes": manifest.notes,  # what the checks found worth saying
        "entrypoints": sorted(manifest.entrypoints),
        "variant": manifest.variant,
        "params": [{"name": name, **spec.model_dump()} for name, spec in manifest.params.items()],
        "environment": {
            "default": manifest.environment.default,
            "requirements": manifest.environment.requirements,
            # an image declared next to the requirements: offered as the environment "docker" in the dialog
            "declared_image": manifest.environment.image if manifest.environment.requirements else "",
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
    db: Session,
    user: User,
    *,
    name: str,
    description: str = "",
    paths: dict[str, str] | None = None,
    env_id: str = "",
    kind: str = "python",
    image: str = "",
) -> Environment:
    """Create or update a registered environment.

    ``python``: an interpreter already on each server (``paths``: per node, its path).
    ``docker``: an image (``image``) every server with Docker runs; nothing to set per server.
    """
    services.require_admin(user)
    name = name.strip()
    if not _ENV_NAME_RE.match(name) or name == AUTO["name"]:
        raise ServiceError(400, "환경 이름은 1~63자의 글자·숫자·공백·`-`·`_`·`.`로 지어 주세요.")
    if kind not in ("python", "docker"):
        raise ServiceError(400, "환경의 방식은 python(서버의 Python) 또는 docker(Docker 이미지)입니다.")
    image = image.strip()
    if kind == "docker" and not IMAGE_RE.match(image):
        raise ServiceError(400, "Docker 이미지 이름을 적어 주세요 (예: registry.local:5000/team/train:1.2).")
    env = db.get(Environment, env_id) if env_id else None
    if env_id and env is None:
        raise ServiceError(404, "환경을 찾을 수 없습니다.")
    clash = db.scalar(select(Environment).where(Environment.name == name))
    if clash is not None and (env is None or clash.id != env.id):
        raise ServiceError(409, "같은 이름의 환경이 이미 있습니다.")
    if env is None:
        env = Environment(name=name, kind=kind, created_by=user.id)
        db.add(env)
    elif env.kind != kind:
        raise ServiceError(400, "등록한 환경의 방식은 바꿀 수 없습니다. 새 환경으로 등록하세요.")
    env.name, env.description = name, description.strip()
    if kind == "docker":
        if env.image != image:
            env.nodes = {}  # another image: what was checked no longer holds
        env.image = image
    else:
        env.nodes = _clean_paths(db, paths or {}, env.nodes or {})
    db.flush()
    services.audit(db, user.id, "api", "environment.save", env.id, name=name, kind=kind, image=image or None)
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
        "image": env.image or "",
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


# ---- installing environments on servers ahead of the first training ----------------------

TRAINS = "compute"  # the node role that trains (shown as 학습·오토라벨): environments go only there
REMOVE_SINCE = (0, 47, 0)  # nodes from here on remove an installed environment (``env.remove``)


def trains(node: Node) -> bool:
    return node.enabled and TRAINS in (node.roles or [])


def has_docker(node: Node) -> bool:
    return bool((node.info or {}).get("docker"))


def _auto_spec(manifest: TemplateManifest) -> dict[str, Any]:
    """What a template installs by itself: a virtual environment from its requirements, or the image it
    declares when it declares nothing else to install."""
    declared = manifest.environment
    if declared.image and not declared.requirements:
        return {"kind": "docker", "name": "Docker 이미지", "image": declared.image}
    return {**AUTO, **declared.model_dump(exclude={"default", "image"})}


def install_spec(db: Session, manifest: TemplateManifest) -> dict[str, Any] | None:
    """What "미리 준비" puts on a server for a template: its virtual environment, its image, or the image of
    the registered Docker environment it names. None when there is nothing to install (a server's own
    Python, or code that needs only the standard library)."""
    if manifest.environment.default:
        env = _environment(db, manifest.environment.default)
        if env is not None:
            if env.kind == "docker":
                return {"kind": "docker", "id": env.id, "name": env.name, "image": env.image}
            return None
    spec = _auto_spec(manifest)
    return spec if spec["kind"] == "docker" or spec.get("requirements") else None


def _same_env(found: dict[str, Any], spec: dict[str, Any]) -> bool:
    """An environment a node reported is the one a template installs (older reports carry no torch flag)."""
    if spec.get("kind") == "docker" or found.get("kind") == "docker":
        return found.get("kind") == spec.get("kind") == "docker" and found.get("image") == spec.get("image")
    return (
        sorted(found.get("requirements") or []) == sorted(spec.get("requirements") or [])
        and (found.get("python") or "3.12") == (spec.get("python") or "3.12")
        and found.get("torch") in (None, bool(spec.get("torch")))
    )


def _why_not(node: Node, spec: dict[str, Any]) -> str:
    """Why an environment cannot be installed on a server ("" when it can)."""
    if not trains(node):
        return "학습 역할 아님"
    if spec.get("kind") == "docker" and not has_docker(node):
        return "Docker 없음"
    return ""


def environment_status(db: Session, row: Template, manifest: TemplateManifest | None = None) -> dict[str, Any]:
    """Where a template's environment is installed, where it is being installed, and on which servers it may be.

    ``installs``: whether there is anything to install at all. ``servers``: every server switched on,
    with ``state`` (ready · preparing · "") and ``why`` it cannot have the environment ("" when it can).
    """
    manifest = manifest or manifest_of(row)
    spec = install_spec(db, manifest)
    if spec is None:
        return {"installs": False, "ready": [], "preparing": [], "servers": []}
    nodes = [n for n in db.scalars(select(Node).order_by(Node.name)) if n.enabled]
    ready = {n.id for n in nodes if any(_same_env(e, spec) for e in (n.info or {}).get("envs") or [])}
    active = db.scalars(select(Job).where(Job.type == "env.prepare", Job.status.in_(("queued", "running"))))
    preparing = {j.target_node_id or j.node_id for j in active if (j.params or {}).get("template") == row.name}
    now = utcnow()
    servers = [
        {
            "id": n.id,
            "name": n.name,
            "status": "online" if n.last_seen_at and (now - n.last_seen_at).total_seconds() < 120 else "offline",
            "trains": trains(n),
            "why": _why_not(n, spec),
            "state": "preparing" if n.id in preparing else "ready" if n.id in ready else "",
        }
        for n in nodes
    ]
    return {
        "installs": True,
        "kind": spec["kind"],
        "image": spec.get("image", ""),
        "ready": [s["name"] for s in servers if s["state"] == "ready"],
        "preparing": [s["name"] for s in servers if s["state"] == "preparing"],
        "nodes": sum(1 for s in servers if not s["why"]),
        "servers": servers,
    }


def prepare_environment(
    db: Session, user: User, template_id: str, node_ids: list[str] | None = None, via: str = "api"
) -> list[Job]:
    """Install a template's environment on servers before anyone trains with it.

    ``node_ids``: the servers to install on (none: every server that trains and can take it). A server
    without the training role, or without Docker for an image, is turned away with the reason.
    """
    services.require_admin(user)
    row = get_template(db, template_id)
    manifest = manifest_of(row)
    spec = install_spec(db, manifest)
    if spec is None:
        reason = "등록된 서버의 Python이나 표준 라이브러리만 씁니다"
        raise ServiceError(400, f"{row.title}은(는) 미리 설치할 환경이 없습니다 ({reason}).")
    if node_ids:
        nodes = [services.get_node(db, n) for n in dict.fromkeys(node_ids)]
        for node in nodes:
            why = "작업을 받지 않도록 꺼 둠" if not node.enabled else _why_not(node, spec)
            if why == "학습 역할 아님":
                hint = "노드 설정에서 '학습·오토라벨'을 켜세요"
                raise ServiceError(400, f"{node.name}은(는) 학습 역할이 아니라 학습 환경을 두지 않습니다. {hint}.")
            if why:
                raise ServiceError(400, f"{node.name}에는 이 환경을 설치할 수 없습니다: {why}.")
    else:
        nodes = [n for n in db.scalars(select(Node).order_by(Node.name)) if n.enabled and not _why_not(n, spec)]
    if not nodes:
        need = "학습 역할이 있고 Docker가 있는" if spec["kind"] == "docker" else "학습 역할이 있는"
        raise ServiceError(400, f"환경을 준비할 서버가 없습니다. {need} 서버가 있어야 합니다.")
    params = {"template": row.name, "version": row.version, "environment": spec}
    jobs = [services.create_job(db, user, "env.prepare", params, n.id, via=via, gpus=0) for n in nodes]
    services.audit(db, user.id, via, "environment.prepare", row.id, template=row.name, nodes=[n.name for n in nodes])
    return jobs


def _runs_here(job: Job, node_id: str) -> bool:
    if job.node_id:
        return job.node_id == node_id
    return job.target_node_id == node_id or node_id in ((job.requirements or {}).get("nodes") or [])


def remove_environment(db: Session, user: User, template_id: str, node_id: str, via: str = "api") -> Job:
    """Take a template's installed environment off a server: the virtual environment's folder, or the image
    if BADAMOA pulled it (an image that was there before is only forgotten). Refused while a job there uses it."""
    services.require_admin(user)
    row = get_template(db, template_id)
    spec = install_spec(db, manifest_of(row))
    if spec is None:
        raise ServiceError(400, f"{row.title}은(는) 서버에 설치하는 환경이 없습니다.")
    node = services.get_node(db, node_id)
    version = tuple(int(p) for p in (node.agent_version or "0").split(".")[:3] if p.isdigit())
    if version < REMOVE_SINCE:
        need = ".".join(str(n) for n in REMOVE_SINCE)
        raise ServiceError(400, f"{node.name}의 BADAMOA는 설치한 환경을 지울 수 없습니다. {need} 이상으로 업데이트하세요.")
    kinds = ("train.run", "model.run", "env.prepare")
    for job in db.scalars(select(Job).where(Job.type.in_(kinds), Job.status.in_(("queued", "running")))):
        if _runs_here(job, node.id) and _same_env((job.params or {}).get("environment") or {}, spec):
            raise ServiceError(409, f"{node.name}에서 이 환경을 쓰는 작업이 있습니다. 끝난 뒤에 지우세요.")
    params = {"template": row.name, "environment": spec}
    job = services.create_job(db, user, "env.remove", params, node.id, via=via, gpus=0)
    services.audit(db, user.id, via, "environment.remove", row.id, template=row.name, node=node.name)
    return job


def list_environments(db: Session, settings: HubSettings) -> list[dict[str, Any]]:
    return [environment_view(db, settings, e) for e in db.scalars(select(Environment).order_by(Environment.name))]


def check_environment(db: Session, user: User, env_id: str) -> list[Job]:
    """Ask the servers to try a registered environment. Results arrive when the jobs finish.

    A server's Python: every server it is set up on. A Docker image: every server that trains and has
    Docker (it pulls the image if it has not, and runs Python in it).
    """
    services.require_admin(user)
    env = get_environment(db, env_id)
    jobs = []
    nodes = dict(env.nodes or {})
    if env.kind == "docker":
        targets = [n for n in db.scalars(select(Node).order_by(Node.name)) if trains(n) and has_docker(n)]
        for node in targets:
            params = {"environment_id": env.id, "kind": "docker", "image": env.image}
            job = services.create_job(db, user, "env.check", params, node.id, via="ui", gpus=0)
            nodes[node.id] = {**(nodes.get(node.id) or {}), "ok": None, "job_id": job.id}
            jobs.append(job)
        env.nodes = nodes
        if not jobs:
            raise ServiceError(400, "학습 역할이 있고 Docker가 있는 서버가 없습니다.")
        return jobs
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


DECLARED_IMAGE = "docker"  # a run's ``environment``: the image the template declares next to its requirements


def environment_for_run(db: Session, manifest: TemplateManifest, wanted: str, node: Node) -> dict[str, Any]:
    """The environment a run will use on ``node``, as a snapshot the node can act on.

    ``wanted`` is an environment id or name; empty means the template's default, and
    when the template names none, the one it installs itself (``_auto_spec``).
    ``docker``: the image the template declares, when it declares requirements as well.
    """
    auto = _auto_spec(manifest)
    if wanted in ("auto", AUTO["name"]):
        return auto
    if wanted == DECLARED_IMAGE:
        if not manifest.environment.image:
            raise ServiceError(400, "이 모델 종류에는 Docker 이미지가 적혀 있지 않습니다.")
        return {"kind": "docker", "name": "Docker 이미지", "image": manifest.environment.image}
    env = _environment(db, wanted or manifest.environment.default) if (wanted or manifest.environment.default) else None
    if env is None:
        if wanted:
            raise ServiceError(404, f"환경을 찾을 수 없습니다: {wanted}")
        return auto  # the template's default is not registered on this Hub: fall back to installing
    if env.kind == "docker":
        return {"kind": "docker", "id": env.id, "name": env.name, "image": env.image}
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

