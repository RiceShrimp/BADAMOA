"""The Hub's Docker registry for admins (설정 → 환경): where it is, what it holds, images it lacks."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import docker_registry
from .deps import Hub, admin_user, db_session, get_hub
from .models import User

router = APIRouter(prefix="/api/docker-registry")


@router.get("")
def registry_view(_: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return docker_registry.view(db)


@router.post("/check")
def registry_check(
    _: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Look again now: find, adopt or make the registry (it may take a minute the first time)."""
    docker_registry.setup(db, hub.settings)
    return docker_registry.view(db)


class RegistrySetting(BaseModel):
    address: str | None = Field(default=None, max_length=255)  # another registry's (external)
    off: bool = False


@router.put("")
def registry_set(
    body: RegistrySetting,
    user: User = Depends(admin_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """An address of its own, off, or (neither) back to finding one by itself."""
    docker_registry.configure(db, user, hub.settings, body.address, body.off)
    return docker_registry.view(db)


@router.get("/images")
def registry_images(
    _: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    return docker_registry.images(db, hub.settings)


class PushBody(BaseModel):
    image: str = Field(min_length=1, max_length=512)


@router.post("/push")
def registry_push(
    body: PushBody, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    job = docker_registry.push(db, user, hub.settings, body.image)
    return {"job_id": job.id}
