"""REST API used by the web UI and scripts (``/api/*``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import resources, services
from .db import iso
from .deps import Hub, admin_user, current_user, db_session, get_hub, public_base_url
from .install import join_commands
from .models import User
from .release_notes import load_release_notes

router = APIRouter(prefix="/api")


# ---- health & setup ---------------------------------------------------------


@router.get("/health")
def health(db: Session = Depends(db_session)) -> dict[str, Any]:
    return {"status": "ok", "version": services.HUB_VERSION, "needs_setup": services.needs_setup(db)}


class Credentials(BaseModel):
    username: str
    password: str


@router.get("/setup/status")
def setup_status(db: Session = Depends(db_session)) -> dict[str, Any]:
    return {"needs_setup": services.needs_setup(db), "version": services.HUB_VERSION}


@router.post("/setup")
def setup(body: Credentials, db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    if not services.needs_setup(db):
        raise services.ServiceError(409, "이미 초기 설정이 완료되었습니다.")
    user = services.create_user(db, body.username, body.password, role="admin")
    token, _ = services.login(db, body.username, body.password, hub.settings)
    services.audit(db, user.id, "ui", "setup.admin_created", user.id)
    db.commit()
    return {"token": token, "user": services.serialize_user(user)}


# ---- auth ---------------------------------------------------------------------


@router.post("/auth/login")
def login(body: Credentials, db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    token, user = services.login(db, body.username, body.password, hub.settings)
    db.commit()
    return {"token": token, "user": services.serialize_user(user)}


@router.get("/auth/me")
def me(user: User = Depends(current_user)) -> dict[str, Any]:
    return services.serialize_user(user)


class NewToken(BaseModel):
    name: str = Field(min_length=1, max_length=128)


@router.get("/tokens")
def list_tokens(user: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    return services.list_api_tokens(db, user)


@router.post("/tokens")
def create_token(
    body: NewToken, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    token, row = services.issue_token(db, user, "api", body.name)
    services.audit(db, user.id, "ui", "token.create", row.id, name=body.name)
    db.commit()
    return {"id": row.id, "name": row.name, "token": token, "created_at": iso(row.created_at)}


@router.delete("/tokens/{token_id}")
def delete_token(token_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    services.revoke_token(db, user, token_id)
    db.commit()
    return {"ok": True}


@router.get("/release-notes")
def release_notes(_: User = Depends(current_user)) -> dict[str, Any]:
    return {"version": services.HUB_VERSION, "releases": load_release_notes()}


@router.get("/activity")
def activity(
    limit: int = 20, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    return services.list_activity(db, limit)


# ---- users (admin) -------------------------------------------------------------


class NewUser(Credentials):
    role: str = "member"


@router.get("/users")
def list_users(_: User = Depends(admin_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    from sqlalchemy import select

    return [services.serialize_user(u) for u in db.scalars(select(User).order_by(User.created_at))]


@router.post("/users")
def create_user(body: NewUser, admin: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    user = services.create_user(db, body.username, body.password, body.role)
    services.audit(db, admin.id, "ui", "user.create", user.id, username=user.username, role=user.role)
    db.commit()
    return services.serialize_user(user)


# ---- nodes ----------------------------------------------------------------------


@router.get("/nodes")
def list_nodes(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    return services.list_nodes(db, hub.settings)


@router.get("/nodes/{node_id}")
def get_node(
    node_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    node = services.get_node(db, node_id)
    data = services.node_view(db, hub.settings, node)
    data["disk_forecast"] = resources.disk_forecast(db, node)
    names = {node.id: node.name}
    data["recent_jobs"] = [services.serialize_job(j, names) for j in services.list_jobs(db, None, node.id, 10)]
    return data


@router.get("/nodes/{node_id}/metrics")
def node_metrics(
    node_id: str, range: str = "1h", _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    services.get_node(db, node_id)
    try:
        return resources.history(db, node_id, range)
    except ValueError as e:
        raise services.ServiceError(400, str(e)) from e


class NodePatch(BaseModel):
    name: str | None = None
    roles: list[str] | None = None
    enabled: bool | None = None
    labels: dict[str, Any] | None = None


@router.patch("/nodes/{node_id}")
def patch_node(
    node_id: str,
    body: NodePatch,
    user: User = Depends(admin_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    node = services.update_node(db, user, node_id, **body.model_dump(exclude_unset=True))
    db.commit()
    return services.node_view(db, hub.settings, node)


@router.delete("/nodes/{node_id}")
def delete_node(node_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    services.delete_node(db, user, node_id)
    db.commit()
    return {"ok": True}


class JoinTokenRequest(BaseModel):
    roles: list[str] | None = None
    ttl_minutes: int = 60


@router.post("/nodes/join-tokens")
def create_join_token(
    body: JoinTokenRequest,
    request: Request,
    user: User = Depends(admin_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    token, row = services.create_join_token(db, user, body.roles, body.ttl_minutes)
    services.audit(db, user.id, "ui", "node.join_token", row.id, roles=row.roles)
    db.commit()
    hub_url = public_base_url(request, hub)
    return {
        "token": token,
        "roles": row.roles,
        "expires_at": iso(row.expires_at),
        "hub_url": hub_url,
        "commands": join_commands(hub_url, token),
    }


# ---- jobs -------------------------------------------------------------------------


@router.get("/job-types")
def job_types(_: User = Depends(current_user)) -> list[dict[str, Any]]:
    return services.list_job_types()


class NewJob(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    node_id: str | None = None
    gpus: int | None = None  # None → the job type's default
    min_vram_mb: int = 0


@router.get("/jobs")
def list_jobs(
    status: str | None = None,
    node_id: str | None = None,
    limit: int = 100,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> list[dict[str, Any]]:
    names = services.node_name_map(db)
    return [services.job_view(db, hub.settings, j, names) for j in services.list_jobs(db, status, node_id, limit)]


@router.post("/jobs")
def create_job(
    body: NewJob, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict:
    job = services.create_job(
        db, user, body.type, body.params, body.node_id, via="ui", gpus=body.gpus, min_vram_mb=body.min_vram_mb
    )
    db.commit()
    return services.job_view(db, hub.settings, job)


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict:
    return services.job_view(db, hub.settings, services.get_job(db, job_id))


@router.get("/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str,
    after: int = 0,
    limit: int = 1000,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
) -> list[dict[str, Any]]:
    return services.get_logs(db, job_id, after, limit)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict:
    job = services.cancel_job(db, user, job_id, via="ui")
    db.commit()
    return services.job_view(db, hub.settings, job)
