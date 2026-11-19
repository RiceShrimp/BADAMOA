"""Settings endpoints: model templates, environments, per-project defaults.

Everyone may read them (the training dialog needs the list); only admins change them.
"""

from __future__ import annotations

from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from badamoa.core.templates import MAX_BUNDLE_BYTES

from . import catalog, cleanup, registry, services, template_checks
from .deps import Hub, admin_user, bearer_token, current_node, current_user, db_session, get_hub
from .models import Job, Node, User
from .services import ServiceError
from .train_models import Template

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/check")  # a server trying a template version


# ---- templates -----------------------------------------------------------------------


@router.get("/templates")
def list_templates(
    hidden: bool = False, task: str = "", user: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    """Each template's newest version offered to everyone; admins get the newest of all, with its trial."""
    return registry.list_templates(db, include_hidden=hidden, task=task, admin=user.role == "admin")


@router.get("/templates/{template_id}")
def get_template(template_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return registry.template_view(db, registry.get_template(db, template_id))


@router.post("/templates")
async def upload_template(request: Request, hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Register a template version. The request body is the template folder as a zip."""
    token = bearer_token(request)
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
            row = registry.register_template(db, user, data, storage)
            if not row.trial:  # a new version (the same content again keeps its trial)
                template_checks.start(db, hub.settings, row, user.id, "ui")
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
    """Try the version again on a server (after a fix on the server's side, or a failed trial)."""
    template_checks.recheck(db, hub.settings, user, template_id, via="ui")
    db.commit()
    return registry.template_view(db, registry.get_template(db, template_id))


class AllowBody(BaseModel):
    reason: str


@router.post("/templates/{template_id}/allow")
def allow_template(
    template_id: str, body: AllowBody, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Offer the version without a passed trial; the reason is kept with it and in the audit log."""
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
