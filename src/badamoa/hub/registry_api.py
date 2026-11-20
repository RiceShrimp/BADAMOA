"""Settings endpoints: model templates, environments, per-project defaults.

Everyone may read them (the training dialog needs the list); only admins change them.
"""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from badamoa.core.templates import MAX_BUNDLE_BYTES

from . import address, catalog, cleanup, install, registry, services, template_checks
from .deps import Hub, admin_user, bearer_token, current_node, current_user, db_session, get_hub
from .models import Job, Node, User
from .services import ServiceError
from .train_models import Template

router = APIRouter(prefix="/api")
MAX_WHEEL_BYTES = 1024**3
CHECK_HEADER = "X-Badamoa-Check"  # the uploader's own trial report with an upload
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/check")  # a server trying a template version
packages_router = APIRouter(prefix="/api/agent/packages")  # a server installing an in-house wheel
media_router = APIRouter(prefix="/media")  # a template version's code, by a signed link


def _with_links(view: dict[str, Any], user: User, hub: Hub) -> dict[str, Any]:
    """Admins get a link to the version's code as it was uploaded."""
    if user.role == "admin" and not view.get("builtin"):
        view["code_url"] = catalog.media_url(hub.settings, "template", view["id"])
    return view


# ---- templates -----------------------------------------------------------------------


@router.get("/templates")
def list_templates(
    hidden: bool = False,
    task: str = "",
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> list[dict[str, Any]]:
    """Each template's newest version offered to everyone; admins get the newest of all, with its trial."""
    views = registry.list_templates(db, include_hidden=hidden, task=task, admin=user.role == "admin", user_id=user.id)
    return [_with_links(view, user, hub) for view in views]


@router.get("/templates/kit")
def template_kit(request: Request, user: User = Depends(current_user), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """What someone who may upload model kinds runs on their own PC, with this Hub's address filled in."""
    if not services.can_upload_templates(user):
        raise HTTPException(status_code=403, detail=registry.UPLOAD_NOT_ALLOWED)
    hub_url, _ = address.resolve(request, hub.settings)
    host = hub_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    wheel = install.node_wheel(hub.settings)
    return {
        "hub_url": hub_url,
        "install_uv": f"uv tool install --python 3.12 --index {hub_url}/install/simple/ badamoa",
        "install_pip": f"pip install --index-url {hub_url}/install/simple/ --trusted-host {host} badamoa",
        "wheel_url": f"{hub_url}/install/wheels/{wheel.name}" if wheel is not None else "",
        "init": "badamoa template init my-model",
        "check": "badamoa template check my-model --run",
        "push": f"badamoa template push my-model --hub {hub_url} --token <토큰>",
        "draft": user.role != "admin",
    }


@router.get("/templates/{template_id}")
def get_template(
    template_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return _with_links(registry.template_view(db, registry.get_template(db, template_id)), user, hub)


class TextBody(BaseModel):
    text: str = ""


@router.put("/templates/{template_id}/description")
def describe_template(
    template_id: str, body: TextBody, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, str]:
    """The model kind's description, written on the Hub (empty: what its bundle says)."""
    text = registry.set_description(db, user, template_id, body.text)
    db.commit()
    return {"description": text}


@router.put("/templates/{template_id}/memo")
def template_memo(
    template_id: str, body: TextBody, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, str]:
    """A memo on this version."""
    row = registry.set_memo(db, user, template_id, body.text)
    db.commit()
    return {"memo": row.memo}


@router.get("/templates/{template_id}/versions")
def template_versions(
    template_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    """Every version of this model kind, newest first, with its trial and memo."""
    return registry.versions_of(db, template_id, user)


@media_router.get("/template/{template_id}")
def template_code(
    template_id: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> Response:
    """The version's code as it was uploaded (the link comes from the template's view, for admins)."""
    if not catalog.verify_media(hub.settings, "template", template_id, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    row = db.get(Template, template_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if row is None or not row.blob or not storage.has_blob(row.blob):
        raise HTTPException(status_code=404, detail="템플릿 묶음이 저장소에 없습니다.")
    name = f"{row.name}-{row.version}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{name}"'}
    # with requirements.txt, the trial's exact versions and the in-house wheels it names (ENVIRONMENT.md says how)
    data = registry.code_with_environment(db, row, storage.get_blob(row.blob), storage)
    return Response(data, media_type="application/zip", headers=headers)


# ---- in-house packages -----------------------------------------------------------------------


@router.get("/packages")
def list_packages(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    """The wheels the servers install from the Hub, besides the package index."""
    return registry.list_packages(db)


@router.post("/packages")
async def upload_package(request: Request, filename: str, hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Add a wheel (the request body is the .whl file; ``filename`` its name, which says what it is)."""
    token = bearer_token(request)
    data = await request.body()
    if len(data) > MAX_WHEEL_BYTES:
        raise HTTPException(status_code=413, detail="패키지 파일이 너무 큽니다 (1 GB까지).")

    def work() -> dict[str, Any]:
        with hub.db.session() as db:
            user = services.authenticate(db, token)
            if user is None:
                raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
            storage = catalog.storage_for(str(hub.settings.storage_dir))
            row = registry.add_package(db, user, filename, data, storage, via="ui")
            return next(p for p in registry.list_packages(db) if p["id"] == row.id)

    return await anyio.to_thread.run_sync(work)


@router.delete("/packages/{package_id}")
def delete_package(package_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    registry.delete_package(db, user, package_id, via="ui")
    db.commit()
    return {"ok": True}


@packages_router.get("/{digest}")
def package_file(
    digest: str, _: Node = Depends(current_node), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> Response:
    """A listed wheel, for a server building an environment that may need it."""
    blob = registry.package_blob(db, digest)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(blob):
        raise HTTPException(status_code=404, detail="패키지 파일이 저장소에 없습니다.")
    return Response(storage.get_blob(blob), media_type="application/octet-stream")


@router.post("/templates")
async def upload_template(request: Request, hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Register a template version. The request body is the template folder as a zip; ``X-Badamoa-Check``
    carries the uploader's ``badamoa template check --run`` report (gzip + base64), required from members."""
    token = bearer_token(request)
    try:
        header = request.headers.get(CHECK_HEADER, "")
        report = json.loads(gzip.decompress(base64.b64decode(header))) if header else None
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail="시험 실행 보고서를 읽을 수 없습니다.") from e
    if int(request.headers.get("content-length") or 0) > MAX_BUNDLE_BYTES:
        raise HTTPException(status_code=413, detail="템플릿 파일이 너무 큽니다.")
    data = await request.body()
    if len(data) > MAX_BUNDLE_BYTES:
        raise HTTPException(status_code=413, detail="템플릿 파일이 너무 큽니다.")

    def work() -> dict[str, Any]:
        with hub.db.session() as db:
            user = services.authenticate(db, token)
            if user is None:
                raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
            storage = catalog.storage_for(str(hub.settings.storage_dir))
            row = registry.register_template(db, user, data, storage, report=report if isinstance(report, dict) else None)
            template_checks.uploaded(row)  # a new version waits for an admin to try it (the same content keeps its trial)
            return registry.template_view(db, row)

    return await anyio.to_thread.run_sync(work)


@router.delete("/templates/{template_id}")
def delete_template(template_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    registry.delete_template(db, user, template_id)
    db.commit()
    return {"ok": True}


@router.post("/templates/{template_id}/check")
def check_template(
    template_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Start the version's trial on a server that trains (an admin; nothing starts one on its own)."""
    template_checks.start(db, hub.settings, user, template_id, via="ui")
    db.commit()
    return registry.template_view(db, registry.get_template(db, template_id))


class AllowBody(BaseModel):
    reason: str = ""  # needed unless the version passed its trial


@router.post("/templates/{template_id}/allow")
def allow_template(
    template_id: str, body: AllowBody, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Offer the version to everyone: one that passed as it is, any other with a reason (kept, and audited)."""
    row = template_checks.allow(db, user, template_id, body.reason, via="ui")
    db.commit()
    return registry.template_view(db, row)


@agent_router.get("/template")
def check_bundle(
    job_id: str, node: Node = Depends(current_node), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> Response:
    """The code of the version a server is trying, as the zip it was registered with."""
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.type != "template.check":
        raise ServiceError(404, "이 노드에 할당된 시험 실행이 아닙니다.")
    row = db.get(Template, (job.params or {}).get("template_id", ""))
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if row is None or not row.blob or not storage.has_blob(row.blob):
        raise HTTPException(status_code=404, detail="템플릿 묶음이 저장소에 없습니다.")
    return Response(storage.get_blob(row.blob), media_type="application/zip")


@router.post("/templates/{template_id}/restore")
def restore_template(template_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    row = registry.restore_template(db, user, template_id)
    db.commit()
    return {"id": row.id, "name": row.name, "note": ""}


# ---- environments --------------------------------------------------------------------


class EnvironmentBody(BaseModel):
    name: str
    description: str = ""
    kind: str = "python"  # python: a server's Python · docker: a Docker image
    paths: dict[str, str] = Field(default_factory=dict)  # python: node id → path of that node's Python
    image: str = ""  # docker: the image, e.g. registry.local:5000/team/train:1.2


@router.get("/environments")
def list_environments(
    user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    environments = registry.list_environments(db, hub.settings)
    if user.role != "admin":  # where things live on the servers is the admin's business
        for env in environments:
            for node in env["nodes"]:
                node.pop("path", None)
    return environments


@router.post("/environments")
def create_environment(
    body: EnvironmentBody, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    env = registry.save_environment(
        db, user, name=body.name, description=body.description, paths=body.paths, kind=body.kind, image=body.image
    )
    db.commit()
    return registry.environment_view(db, hub.settings, env)


@router.put("/environments/{env_id}")
def update_environment(
    env_id: str,
    body: EnvironmentBody,
    user: User = Depends(admin_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    env = registry.save_environment(
        db,
        user,
        name=body.name,
        description=body.description,
        paths=body.paths,
        env_id=env_id,
        kind=body.kind,
        image=body.image,
    )
    db.commit()
    return registry.environment_view(db, hub.settings, env)


class PrepareBody(BaseModel):
    node_ids: list[str] = []  # empty: every server that trains and can take the environment


@router.post("/templates/{template_id}/prepare")
def prepare_environment(
    template_id: str, body: PrepareBody, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Install the template's environment on servers now, so their first training does not wait for it."""
    jobs = registry.prepare_environment(db, user, template_id, body.node_ids or None, via="ui")
    db.commit()
    return {"jobs": [job.id for job in jobs]}


@router.delete("/templates/{template_id}/environment/{node_id}")
def remove_environment(
    template_id: str, node_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Take the template's installed environment off a server (refused while a job there uses it)."""
    job = registry.remove_environment(db, user, template_id, node_id, via="ui")
    db.commit()
    return {"job": job.id}


@router.post("/environments/{env_id}/check")
def check_environment(
    env_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    jobs = registry.check_environment(db, user, env_id)
    db.commit()
    return {"jobs": [job.id for job in jobs]}


@router.delete("/environments/{env_id}")
def delete_environment(env_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    registry.delete_environment(db, user, env_id)
    db.commit()
    return {"ok": True}


# ---- project defaults ----------------------------------------------------------------


@router.get("/storage")
def storage(
    user: User = Depends(current_user), hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """What the Hub keeps and how much room is left. The folder itself is shown to admins only."""
    return registry.storage_usage(db, hub.settings, user)


@router.post("/storage/compact")
def storage_compact(
    user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Give the database's empty pages back to the disk (VACUUM), in the background; only while no job runs."""
    from . import hub_storage

    services.audit(db, user.id, "ui", "storage.compact", "")
    db.commit()  # the audit row is written before the compaction holds writes off
    hub_storage.compact_in_background(hub.db, hub.settings)
    return hub_storage.compacting() | {"compacting": True}


class StoragePolicy(BaseModel):
    outputs_days: int | None = None  # 0: keep for ever; left out: as it is
    checkpoint_days: int | None = None
    trash_days: int | None = None  # what is in the trash, removed for good after this long
    workdir_days: int | None = None  # work folders runs left on the nodes


@router.put("/storage/policy")
def storage_policy(
    body: StoragePolicy, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    policy = cleanup.set_policy(db, user, **body.model_dump(exclude_unset=True))
    db.commit()
    return policy


@router.get("/storage/cleanup")
def storage_cleanup_preview(
    _: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """What a cleanup would remove now under the saved policy (nothing is removed)."""
    return cleanup.cleanup(db, hub.settings, apply=False)


@router.post("/storage/cleanup")
def storage_cleanup(
    user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    with cleanup.lock:
        report = cleanup.cleanup(db, hub.settings, apply=True, user=user)
        db.commit()
    return report


class ProjectSettings(BaseModel):
    template: str = ""  # the model template the training dialog opens with


@router.put("/projects/{project_id}/settings")
def project_settings(
    project_id: str, body: ProjectSettings, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    if body.template:
        registry.get_template(db, body.template)
    project = catalog.update_project_settings(db, user, project_id, {"template": body.template})
    db.commit()
    return project.settings
