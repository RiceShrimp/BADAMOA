"""What an admin sets up once so everyone else only picks by name: model templates and environments."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa import templates as shipped
from badamoa.core import tasks as task_names
from badamoa.core.storage import Storage
from badamoa.core.templates import (
    IMAGE_RE,
    TemplateError,
    TemplateManifest,
    load_manifest,
    pinned_versions,
    read_bundle,
)

from . import cleanup, services
from .catalog_models import Item, Project
from .config import HubSettings
from .db import iso, utcnow
from .models import HubSetting, Job, Node, User
from .services import ServiceError
from .train_models import Environment, ModelJob, ModelVersion, Package, Run, RunFile, Template

DEFAULT_TEMPLATE = "yolo"
# What a project trains with when nobody chose: the built-in template for its task (Ultralytics YOLO trains
# Det · Seg · Kpt; classification stays with torchvision, as before 0.60.0).
DEFAULT_BY_TASK = {"detect": "yolo", "segment": "yolo", "pose": "yolo", "classify": "torchvision"}
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


UPLOAD_NOT_ALLOWED = "모델 종류를 올릴 권한이 없습니다. 관리자에게 설정 → 사용자에서 '모델 종류 올리기'를 켜 달라고 하세요."
NEEDS_REPORT = (
    "먼저 badamoa template check . --run을 통과하세요. 이 PC에서 학습 · 테스트 · 평가 · 내보내기를 해 보고, "
    "그 보고서를 badamoa template push가 함께 보냅니다."
)


def read_report(report: dict[str, Any] | None, data: bytes, manifest: TemplateManifest) -> dict[str, Any]:
    """The uploader's ``check --run`` report, if it is for this very bundle and it passed (else why not)."""
    if not report:
        return {}
    if report.get("bundle_sha256") != hashlib.sha256(data).hexdigest():
        again = "코드를 바꿨다면 badamoa template check . --run을 다시 하세요."
        raise ServiceError(400, f"시험 실행 보고서가 이 묶음의 것이 아닙니다. {again}")
    if (report.get("name"), report.get("version")) != (manifest.name, manifest.version):
        raise ServiceError(400, "시험 실행 보고서의 이름 · 버전이 묶음과 다릅니다.")
    if not report.get("ok"):
        said = f"{report.get('failed') or ''}: {report.get('reason') or ''}".strip(": ")
        raise ServiceError(400, f"이 PC의 시험 실행이 통과하지 않았습니다 ({said}). 고친 뒤 다시 check --run 하세요.")
    return {k: v for k, v in report.items() if k in REPORT_KEYS}


REPORT_KEYS = (
    "ok", "steps", "notes", "took_s", "sdk", "contract", "python", "os", "gpus", "device", "packages", "sources",
    "bundle_sha256", "name", "version", "at",
)  # fmt: skip


def register_template(
    db: Session, user: User, data: bytes, storage: Storage, via: str = "api", report: dict[str, Any] | None = None
) -> Template:
    """Add a template version from an uploaded bundle (the template folder as a zip).

    Admins, and members an admin lets upload (``template_upload``). A member's upload is a draft only they
    see until an admin tries and offers it, and it comes with their own ``check --run`` of the same bundle.
    """
    if not services.can_upload_templates(user):
        raise ServiceError(403, UPLOAD_NOT_ALLOWED)
    try:
        manifest = read_bundle(data)
    except TemplateError as e:
        raise ServiceError(400, str(e)) from e
    proof = read_report(report, data, manifest)
    if user.role != "admin" and not proof:
        raise ServiceError(400, NEEDS_REPORT)
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
    if proof:
        row.report = {**proof, "by": user.username}
    db.flush()
    services.audit(db, user.id, via, "template.register", row.id, name=row.name, version=row.version, report=bool(proof))
    return row


def template_for(db: Session, settings: dict[str, Any] | None, task: str, ref: str = "") -> Template:
    """The template a run on data of ``task`` uses when none is named: the project's last choice
    if it trains that task, else the task's built-in one, else any template that declares the task."""
    if ref:  # a kind merged into another (``replaced_by``) gives way to it, as for the project's last choice
        return successor(db, get_template(db, ref, offered_only=True))
    saved = (settings or {}).get("template")
    if saved:
        try:
            row = get_template(db, saved, offered_only=True)
        except ServiceError:
            row = None
        row = successor(db, row) if row is not None else None
        if row is not None and supports(row, task):
            return row
    if task in DEFAULT_BY_TASK:
        return get_template(db, DEFAULT_BY_TASK[task])
    offered = list_templates(db, task=task)
    if offered:
        return get_template(db, offered[0]["id"])
    raise ServiceError(
        400, f"{task_names.short(task)} 태스크를 학습하는 템플릿이 없습니다. 이 태스크를 적은 모델 템플릿을 등록하세요."
    )


def get_template(db: Session, ref: str, offered_only: bool = False) -> Template:
    """By id, or by name: the newest version, or with ``offered_only`` the newest one offered to everyone
    (a version still being tried, or that failed, leaves the one before in use)."""
    from . import template_checks

    row = db.get(Template, ref)
    if row is None:
        versions = db.scalars(
            select(Template)
            .where(Template.name == ref, Template.deleted_at.is_(None))
            .order_by(Template.created_at.desc(), Template.id)
        ).all()
        row = next((v for v in versions if template_checks.offered(v)), None) if offered_only else None
        row = row or (versions[0] if versions else None)
    if row is None or row.deleted_at is not None:
        raise ServiceError(404, f"모델 템플릿을 찾을 수 없습니다: {ref}")
    return row


def manifest_of(row: Template, task: str = "") -> TemplateManifest:
    """The declaration; with ``task``, as a run on data of that task sees it (a kind of several tasks)."""
    manifest = TemplateManifest(**row.manifest)
    return manifest.for_task(task) if task else manifest


def supports(row: Template, task: str) -> bool:
    """Whether the kind trains data of ``task`` (one of the tasks it declares)."""
    return manifest_of(row).supports(task)


def successor(db: Session, row: Template) -> Template:
    """The kind new runs use instead of one merged into another (``replaced_by``), else the kind itself."""
    seen = {row.name}
    while (after := manifest_of(row).replaced_by) and after not in seen:
        seen.add(after)
        try:
            row = get_template(db, after, offered_only=True)
        except ServiceError:
            break
    return row


# ---- what admins write on the Hub: a model kind's description, a version's memo -----------------

ABOUT_KEY = "template_about"  # {template name: description written on the Hub}
MAX_ABOUT = 4000


def _about(db: Session) -> dict[str, str]:
    row = db.get(HubSetting, ABOUT_KEY)
    return dict((row.value or {}).get("about") or {}) if row is not None else {}


def description_of(db: Session, row: Template, written: dict[str, str] | None = None) -> str:
    """What the model kind is: written on the Hub, else what its newest bundle says."""
    found = (written if written is not None else _about(db)).get(row.name, "")
    return found or manifest_of(row).description


def set_description(db: Session, user: User, template_id: str, text: str) -> str:
    """Describe a model kind on the Hub (every version shows it). Empty: back to what the bundle says."""
    services.require_admin(user)
    row = get_template(db, template_id)
    text = text.strip()[:MAX_ABOUT]
    setting = db.get(HubSetting, ABOUT_KEY)
    if setting is None:
        setting = HubSetting(key=ABOUT_KEY, value={})
        db.add(setting)
    about = dict((setting.value or {}).get("about") or {})
    if text:
        about[row.name] = text
    else:
        about.pop(row.name, None)
    setting.value = {"about": about}
    services.audit(db, user.id, "ui", "template.describe", row.id, name=row.name, cleared=not text)
    return description_of(db, row, about)


def set_memo(db: Session, user: User, template_id: str, text: str) -> Template:
    """A memo on one version (what changed in it, what to watch for)."""
    services.require_admin(user)
    row = get_template(db, template_id)
    row.memo = text.strip()[:MAX_ABOUT]
    services.audit(db, user.id, "ui", "template.memo", row.id, name=row.name, version=row.version)
    return row


def versions_of(db: Session, template_id: str, user: User) -> list[dict[str, Any]]:
    """Every version of a model kind, newest first: when and by whom it came, its trial, its memo.
    Members see the versions offered to everyone and their own drafts."""
    from . import template_checks

    row = get_template(db, template_id)
    rows = db.scalars(
        select(Template)
        .where(Template.name == row.name, Template.deleted_at.is_(None))
        .order_by(Template.created_at.desc(), Template.id)
    ).all()
    shown = []
    for version in rows:
        offered = version.builtin or template_checks.offered(version)
        if user.role != "admin" and not offered and version.created_by != user.id:
            continue
        manifest = manifest_of(version)
        shown.append(
            {
                "id": version.id,
                "version": version.version,
                "title": version.title,
                "description": manifest.description,  # what that bundle says
                "memo": version.memo or "",
                "created_at": iso(version.created_at),
                "uploaded_by": _username(db, version.created_by),
                "builtin": version.builtin,
                "offered": offered,
                "check": template_checks.view(version),
            }
        )
    return shown


def template_view(
    db: Session, row: Template, versions: int = 1, models: int = 0, offered_version: str = "", task: str = ""
) -> dict[str, Any]:
    """``task``: the settings, data format and checks as a run on data of that task sees them."""
    from . import template_checks

    whole = manifest_of(row)
    manifest = whole.for_task(task) if task else whole
    return {
        "id": row.id,
        "name": row.name,
        "title": row.title,
        "version": row.version,
        "versions": versions,
        "task": manifest.task,
        "tasks": whole.trains(),  # every task it trains
        "data_formats": {t: whole.for_task(t).data_format for t in whole.trains()},
        "replaced_by": whole.replaced_by,
        "description": description_of(db, row),
        "declared_description": whole.description,  # what the bundle says (the above may be written on the Hub)
        "memo": row.memo or "",  # this version's memo, written on the Hub
        "license": manifest.license,
        "builtin": row.builtin,
        "hidden": row.hidden,
        "data_format": manifest.data_format,
        "data": manifest.data.model_dump(),
        "notes": manifest.notes,  # what the checks found worth saying
        "entrypoints": sorted(manifest.entrypoints),
        "variant": manifest.variant,
        "reads_weights_from": manifest.reads_weights_from,  # earlier versions whose models this one works with
        "params": [{"name": name, **spec.model_dump()} for name, spec in manifest.params.items()],
        "environment": {
            "default": manifest.environment.default,
            "requirements": manifest.environment.requirements,
            # an image declared next to the requirements: offered as the environment "docker" in the dialog
            "declared_image": manifest.environment.image if manifest.environment.requirements else "",
            **environment_status(db, row, manifest),
        },
        # the library it trains with and the version of it the servers run (or would install)
        "library": library_versions(db, row, manifest),
        "bytes": row.bytes,
        "models": models,
        "created_at": iso(row.created_at),
        # the trial run before it is offered; for admins, the version everyone else gets when this one is not
        "check": template_checks.view(row),
        "offered_version": offered_version,
        "draft": not row.builtin and not template_checks.offered(row),
        "uploaded_by": _username(db, row.created_by),
        # the uploader's own check --run of this bundle, without the long package list
        "local_check": {k: v for k, v in (row.report or {}).items() if k not in ("packages", "sources")}
        | ({"packages": len(row.report["packages"])} if (row.report or {}).get("packages") else {}),
    }


# ---- the library a model kind is built on, and the version of it servers have -----------------------

LIBRARIES = ("ultralytics", "torchvision", "timm", "transformers", "mmdet", "detectron2")  # named before the rest
_newest: dict[str, tuple[float, str]] = {}  # PyPI's newest release per requirement, and when it was asked
_asking: set[str] = set()


def library_of(manifest: TemplateManifest) -> tuple[str, str]:
    """The library a model kind trains with (its name, and the requirement that names it): a well-known one
    among its requirements, else the first. Empty for one that brings its own environment (an image)."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    found: list[tuple[str, str]] = []
    for line in manifest.environment.requirements:
        try:
            found.append((canonicalize_name(Requirement(line).name), line))
        except InvalidRequirement:
            continue
    for wanted in LIBRARIES:
        for name, line in found:
            if name == wanted:
                return name, line
    return found[0] if found else ("", "")


def _pypi_newest(requirement: str) -> str:
    """The newest release on PyPI that ``requirement`` allows, asked in the background (the Hub may have no
    internet: then nothing, and it asks again later). Kept six hours."""
    import threading
    import time

    now = time.time()
    at, version = _newest.get(requirement, (0.0, ""))
    if now - at < (6 * 3600 if version else 1800) or requirement in _asking:
        return version

    def ask() -> None:
        import httpx
        from packaging.requirements import Requirement
        from packaging.version import InvalidVersion, Version

        found = ""
        try:
            req = Requirement(requirement)
            data = httpx.get(f"https://pypi.org/pypi/{req.name}/json", timeout=5).json()
            versions = []
            for text, files in (data.get("releases") or {}).items():
                try:
                    v = Version(text)
                except InvalidVersion:
                    continue
                if files and not v.is_prerelease and req.specifier.contains(v):
                    versions.append(v)
            found = str(max(versions)) if versions else ""
        except Exception:  # noqa: BLE001 - no internet, no answer: said as "unknown"
            found = ""
        finally:
            _newest[requirement] = (time.time(), found)
            _asking.discard(requirement)

    _asking.add(requirement)
    threading.Thread(target=ask, name="pypi-newest", daemon=True).start()
    return version


def library_versions(db: Session, row: Template, manifest: TemplateManifest) -> dict[str, Any] | None:
    """Which version of its library a model kind runs on: what its recent runs installed, per server (the
    newest of each), else what its trial installed; and what a server installing it now would get (the newest
    the requirement allows of the Hub's wheels and PyPI's releases)."""
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    name, requirement = library_of(manifest)
    if not name:
        return None

    def version_in(packages: list[Any]) -> str:
        for line in packages or []:
            pkg, _, version = str(line).partition("==")
            if version and canonicalize_name(pkg) == name:
                return version.split("+")[0]
        return ""

    seen: dict[str, dict[str, Any]] = {}
    runs = db.execute(
        select(Run.info, Run.created_at, Job.node_id)
        .join(Job, Job.id == Run.job_id, isouter=True)
        .where(func.json_extract(Run.template, "$.name") == row.name, Run.deleted_at.is_(None))
        .order_by(Run.created_at.desc())
        .limit(60)
    ).all()
    for info, at, node_id in runs:
        version = version_in((info or {}).get("packages") or [])
        if version and (node_id or "") not in seen:
            node = db.get(Node, node_id) if node_id else None
            seen[node_id or ""] = {
                "node": node.name if node else "",
                "node_id": node_id or "",
                "version": version,
                "at": iso(at),
            }
    trial = row.trial or {}
    if not seen and version_in(trial.get("installed") or []):
        seen["trial"] = {"node": str(trial.get("node") or ""), "version": version_in(trial["installed"]), "trial": True}

    def key(text: str) -> Any:
        try:
            return Version(text)
        except InvalidVersion:
            return Version("0")

    # what a server installing it now gets: uv reads the Hub's wheels and the index together and takes the newest
    # the requirement allows, so an in-house build (a fork, 8.4.138.1) is what a kind gets only where it pins it
    # (or the index is out of reach); the built-in YOLO beside it still gets PyPI's 8.4.216
    in_house = db.scalars(select(Package.version).where(Package.name == name)).all()
    candidates: list[str] = []
    if in_house:
        from packaging.requirements import Requirement

        spec = Requirement(requirement).specifier
        candidates += [v for v in in_house if spec.contains(v, prereleases=True)]
    if in_house or row.builtin or seen:
        newest = _pypi_newest(requirement)
        if newest:
            candidates.append(newest)
    available = max(candidates, key=key) if candidates else ""

    installed = sorted(seen.values(), key=lambda s: key(s["version"]), reverse=True)
    return {
        "name": name,
        "requirement": requirement,
        "installed": installed,  # per server, the newest first
        "newest": installed[0]["version"] if installed else "",
        "available": available,  # what installing it now gets
    }


def _username(db: Session, user_id: str | None) -> str:
    found = db.get(User, user_id) if user_id else None
    return found.username if found is not None else ""


def list_templates(
    db: Session, include_hidden: bool = False, task: str = "", admin: bool = False, user_id: str = ""
) -> list[dict[str, Any]]:
    """Each template once, optionally only those for one task: the newest version offered to everyone, or
    for ``admin`` the newest version whatever its trial says (with the one everyone else gets)."""
    from . import template_checks

    rows = db.scalars(
        select(Template).where(Template.deleted_at.is_(None)).order_by(Template.created_at.desc(), Template.id)
    ).all()
    counts: dict[str, int] = {}
    newest: dict[str, Template] = {}
    usable: dict[str, Template] = {}
    for row in rows:
        counts[row.name] = counts.get(row.name, 0) + 1
        if admin:
            newest.setdefault(row.name, row)
        if template_checks.offered(row):
            usable.setdefault(row.name, row)
            if not admin:
                newest.setdefault(row.name, row)
        elif not admin and user_id and row.created_by == user_id:  # their own draft
            newest.setdefault(row.name, row)
    models = dict(
        db.execute(
            select(ModelVersion.trainer, func.count())
            .where(ModelVersion.deleted_at.is_(None))
            .group_by(ModelVersion.trainer)
        ).all()
    )
    shown = [t for t in newest.values() if (include_hidden or not t.hidden) and (not task or supports(t, task))]
    # the task's default kind leads (asked for one task), else each task's default, detection first
    first = [DEFAULT_BY_TASK[task]] if task in DEFAULT_BY_TASK else list(dict.fromkeys(DEFAULT_BY_TASK.values()))
    shown.sort(key=lambda t: (first.index(t.name) if t.name in first else len(first), not t.builtin, t.title.lower()))
    return [
        template_view(
            db,
            t,
            counts[t.name],
            models.get(t.name, 0),
            usable[t.name].version if admin and t.name in usable and usable[t.name] is not t else "",
            task,
        )
        for t in shown
    ]


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

TRAINS = "compute"  # 학습용: training runs only there
AUTOLABELS = "autolabel"  # 오토라벨용: a set's images are auto-labelled on the server holding them only with this role
REMOVE_SINCE = (0, 47, 0)  # nodes from here on remove an installed environment (``env.remove``)
NOT_FOR_MODELS = "학습용·오토라벨용 아님"


def has_role(node: Node, role: str) -> bool:
    return node.enabled and role in (node.roles or [])


def trains(node: Node) -> bool:
    return has_role(node, TRAINS)


def runs_models(node: Node) -> bool:
    """Templates' environments go to these servers: they train or auto-label."""
    return trains(node) or has_role(node, AUTOLABELS)


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
    if not runs_models(node):
        return NOT_FOR_MODELS
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
            "trains": runs_models(n),
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

    ``node_ids``: the servers to install on (none: every server that trains or auto-labels and can take it).
    A server with neither role, or without Docker for an image, is turned away with the reason.
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
            if why == NOT_FOR_MODELS:
                hint = "노드 설정에서 '학습용'이나 '오토라벨용'을 켜세요"
                what = "학습용·오토라벨용 서버가 아니라 모델 환경을 두지 않습니다"
                raise ServiceError(400, f"{node.name}은(는) {what}. {hint}.")
            if why:
                raise ServiceError(400, f"{node.name}에는 이 환경을 설치할 수 없습니다: {why}.")
    else:
        nodes = [n for n in db.scalars(select(Node).order_by(Node.name)) if n.enabled and not _why_not(n, spec)]
    if not nodes:
        need = "학습용·오토라벨용이고 Docker가 있는" if spec["kind"] == "docker" else "학습용·오토라벨용"
        raise ServiceError(400, f"환경을 준비할 서버가 없습니다. {need} 서버가 있어야 합니다.")
    params = {"template": row.name, "version": row.version, "environment": with_packages(db, spec, row)}
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
        raise ServiceError(400, f"{node.name}의 MOA는 설치한 환경을 지울 수 없습니다. {need} 이상으로 업데이트하세요.")
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

    A server's Python: every server it is set up on. A Docker image: every server that trains or auto-labels and has
    Docker (it pulls the image if it has not, and runs Python in it).
    """
    services.require_admin(user)
    env = get_environment(db, env_id)
    jobs = []
    nodes = dict(env.nodes or {})
    if env.kind == "docker":
        targets = [n for n in db.scalars(select(Node).order_by(Node.name)) if runs_models(n) and has_docker(n)]
        for node in targets:
            params = {"environment_id": env.id, "kind": "docker", "image": env.image}
            job = services.create_job(db, user, "env.check", params, node.id, via="ui", gpus=0)
            nodes[node.id] = {**(nodes.get(node.id) or {}), "ok": None, "job_id": job.id}
            jobs.append(job)
        env.nodes = nodes
        if not jobs:
            raise ServiceError(400, "학습용·오토라벨용이고 Docker가 있는 서버가 없습니다.")
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


def environment_for_run(
    db: Session, manifest: TemplateManifest, wanted: str, node: Node, row: Template | None = None
) -> dict[str, Any]:
    """The environment a run will use on ``node``, as a snapshot the node can act on.

    ``wanted`` is an environment id or name; empty means the template's default, and
    when the template names none, the one it installs itself (``_auto_spec``).
    ``docker``: the image the template declares, when it declares requirements as well.
    An environment installed from requirements also gets the Hub's in-house wheels, and with ``row``
    the versions its trial installed (``with_packages``).
    """
    return with_packages(db, _environment_spec(db, manifest, wanted, node), row)


def pinned_for(row: Template | None) -> list[str]:
    """The versions a template version's passed trial installed, to install the same on other servers."""
    trial = (row.trial or {}) if row is not None else {}
    if trial.get("state") not in ("ok", "passed", "allowed") or not trial.get("installed"):
        return []
    if not manifest_of(row).environment.requirements:  # nothing installed for it: the server's own Python
        return []
    return pinned_versions(list(trial["installed"]), trial.get("sources") or {})


def with_packages(db: Session, spec: dict[str, Any], row: Template | None = None) -> dict[str, Any]:
    """An environment installed from requirements: the in-house wheels to look in, and the versions to keep to."""
    if spec.get("kind") != "auto" or not spec.get("requirements"):
        return spec
    out = dict(spec)
    wheels = list_packages(db)
    if wheels:
        out["wheels"] = [{"filename": w["filename"], "sha256": w["sha256"]} for w in wheels]
    pinned = pinned_for(row)
    if pinned:
        out["pinned"] = pinned
    return out


def _environment_spec(db: Session, manifest: TemplateManifest, wanted: str, node: Node) -> dict[str, Any]:
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


def code_with_environment(db: Session, row: Template, bundle: bytes, storage: Storage) -> bytes:
    """A version's code as uploaded, plus what installing it takes, so it runs outside BADAMOA the same way:
    ``requirements.txt`` (what the version declares), ``requirements.lock.txt`` (the exact versions its passed
    trial installed, when there is one) and ``wheels/`` (the Hub's in-house wheels the requirements name: a
    custom fork, a package of the team; ``pip install -r requirements.txt --find-links wheels``). A file of the
    same name the code already has is left as it is; read-me files (``README*``) are left out."""
    import io
    import zipfile

    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    env = manifest_of(row).environment
    wanted: dict[str, Any] = {}
    for line in env.requirements:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        wanted[canonicalize_name(req.name)] = req.specifier
    wheels: list[Package] = []
    for package in db.scalars(select(Package).order_by(Package.created_at.desc())):
        name = canonicalize_name(package.name)
        spec = wanted.get(name)
        if spec is None or any(canonicalize_name(w.name) == name for w in wheels):
            continue  # not named, or a newer matching wheel of it is in already
        if spec and not spec.contains(package.version, prereleases=True):
            continue
        wheels.append(package)
    added: dict[str, bytes] = {}
    if env.requirements:
        added["requirements.txt"] = ("\n".join(env.requirements) + "\n").encode("utf-8")
    pinned = pinned_for(row)
    if pinned:
        head = f"# {row.name} {row.version}: what its trial on the Hub installed (torch: the server's driver picks)\n"
        added["requirements.lock.txt"] = (head + "\n".join(pinned) + "\n").encode("utf-8")
    for package in wheels:
        if storage.has_blob(package.blob):
            added[f"wheels/{package.filename}"] = storage.get_blob(package.blob)

    source = zipfile.ZipFile(io.BytesIO(bundle))
    have = set(source.namelist())
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            if info.filename.rsplit("/", 1)[-1].lower().startswith("readme"):
                continue  # the code and what installing it takes, not its read-me
            target.writestr(info, source.read(info))
        for name, data in added.items():
            if name not in have:
                target.writestr(name, data, zipfile.ZIP_STORED if name.endswith(".whl") else zipfile.ZIP_DEFLATED)
    return out.getvalue()


# ---- in-house packages (wheels the servers install from the Hub) ------------------------------------


def list_packages(db: Session) -> list[dict[str, Any]]:
    rows = db.scalars(select(Package).order_by(Package.name, Package.created_at.desc())).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "version": p.version,
            "filename": p.filename,
            "sha256": p.blob,
            "bytes": p.bytes,
            "created_at": iso(p.created_at),
        }
        for p in rows
    ]


def add_package(db: Session, user: User, filename: str, data: bytes, storage: Storage, via: str = "api") -> Package:
    """Keep a wheel on the Hub. The file name says what it is (``mylib-1.2.0-py3-none-any.whl``)."""
    services.require_admin(user)
    name = os.path.basename(filename.strip())
    # {name}-{version}[-{build}]-{python}-{abi}-{platform}.whl (PEP 427)
    parts = name[:-4].split("-") if name.endswith(".whl") else []
    if len(parts) not in (5, 6) or not re.match(r"^\d[\w.+!]*$", parts[1]) or not parts[0]:
        raise ServiceError(400, f"wheel 파일 이름이 아닙니다 ({name}): 예) mylib-1.2.0-py3-none-any.whl")
    dist, version = parts[0], parts[1]
    if not data[:4] == b"PK\x03\x04":
        raise ServiceError(400, f"{name}은(는) wheel(zip) 파일이 아닙니다.")
    blob = storage.put_blob(data)
    row = db.scalar(select(Package).where(Package.filename == name))
    if row is not None and row.blob != blob:
        raise ServiceError(409, f"{name}은 이미 다른 내용으로 올라와 있습니다. 버전을 올려 새로 만드세요.")
    if row is None:
        normal = re.sub(r"[-_.]+", "-", dist).lower()
        row = Package(name=normal, version=version, filename=name, blob=blob, bytes=len(data))
        row.created_by = user.id
        db.add(row)
        db.flush()
    services.audit(db, user.id, via, "package.add", row.id, filename=name)
    return row


def delete_package(db: Session, user: User, package_id: str, via: str = "api") -> None:
    """Off the list: servers stop finding it. Environments already built with it keep it."""
    services.require_admin(user)
    row = db.get(Package, package_id)
    if row is None:
        raise ServiceError(404, "그 패키지가 없습니다.")
    services.audit(db, user.id, via, "package.delete", row.id, filename=row.filename)
    db.delete(row)


def package_blob(db: Session, digest: str) -> str:
    """A listed wheel's storage key, for a server installing it."""
    row = db.scalar(select(Package).where(Package.blob == digest))
    if row is None:
        raise ServiceError(404, "Hub의 사내 패키지 목록에 없는 파일입니다.")
    return row.blob


# ---- where everything is kept ---------------------------------------------------------


def storage_usage(db: Session, settings: HubSettings, user: User) -> dict[str, Any]:
    """How much the Hub keeps and of what. The Hub's folder holds every record: nodes keep only caches.

    The folder itself is counted in the background (``hub_storage.measure``, every ten minutes): with a
    dataset of hundreds of thousands of thumbnails a count takes too long for a screen to wait on it."""
    from . import hub_storage

    counted = hub_storage.measured()
    database = hub_storage.database_usage(db, settings)
    images = hub_storage.image_usage(db, settings)  # counted in the background: it reads every image record
    # What only the trash holds: deleted runs and models, and those of deleted projects.
    gone = set(db.scalars(select(Project.id).where(Project.deleted_at.is_not(None))))
    datasets = trash = 0
    for project_id, usage in images["by_project"].items():  # each image's thumbnail
        if project_id in gone:
            trash += usage["thumbs"]
        else:
            datasets += usage["thumbs"]
    models = outputs = checkpoints = 0
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
    known = models + outputs + checkpoints + trash + datasets
    # before the first count: what the records say (the files nothing records come with the count)
    stored = counted["bytes"] if counted["at"] else known
    return {
        "policy": cleanup.get_policy(db),
        "projects": project_usage(db, images["by_project"]),
        "path": str(settings.home.resolve()) if user.role == "admin" else None,
        "bytes": stored + database["bytes"],
        "files": counted["files"],
        "counted_at": counted["at"] or None,  # when the folder was last counted (epoch seconds)
        "free": disk.free,
        "disk": disk.total,
        "keep_free": hub_storage.keep_free(settings),  # an import that would leave less is refused
        "per_image": images["per_image"],  # what one more registered image takes here
        # images whose thumbnail is still the older, larger kind (made smaller in the background, between imports)
        "old_thumbs": images["old_thumbs"],
        "shrinking": hub_storage.shrinking(),
        # The same bytes recorded twice are stored once, so the parts can add up to more than the whole.
        "parts": {
            "datasets": datasets,
            "models": models,
            "outputs": outputs,
            "checkpoints": checkpoints,
            "trash": trash,
            "database": database["bytes"],
            "other": max(0, stored - known),
        },
        "database": database,  # its size, its empty pages (given back by compacting), compacting now
    }


def project_usage(db: Session, images: dict[str, dict[str, int]] | None = None) -> list[dict[str, Any]]:
    """How much each project keeps: its datasets' thumbnails (and how many images), its models, its runs' and
    model jobs' outputs, its checkpoints."""
    usage: dict[str, dict[str, Any]] = {
        p.id: {"id": p.id, "name": p.name, "images": 0, "datasets": 0, "models": 0, "outputs": 0, "checkpoints": 0}
        for p in db.scalars(select(Project).where(Project.deleted_at.is_(None)))
    }
    if images is None:  # what the images take per project: hub_storage counts it in the background
        images = {
            project_id: {"images": count or 0, "thumbs": int(thumbs or 0)}
            for project_id, count, thumbs in db.execute(
                select(Item.project_id, func.count(Item.id), func.sum(func.max(Item.thumb_bytes, 0))).group_by(
                    Item.project_id
                )
            )
        }
    for project_id, counted in images.items():
        if project_id in usage:
            usage[project_id]["images"], usage[project_id]["datasets"] = counted["images"], counted["thumbs"]
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
    rows = [{**u, "total": u["datasets"] + u["models"] + u["outputs"] + u["checkpoints"]} for u in usage.values()]
    return sorted(rows, key=lambda u: -u["total"])
