"""The 오토라벨 screen: runs across projects, what they can run over, and starting one (see ``autolabel``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import autolabel
from .deps import Hub, current_user, db_session, get_hub
from .models import User
from .services import ServiceError

router = APIRouter(prefix="/api/autolabel")


@router.get("/board")
def board(_: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """The runs going on now, then each project's runs, sets waiting and models that can label."""
    return autolabel.board(db, hub.settings)


@router.get("/runs")
def runs(
    project_id: str = "",
    set_id: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> list[dict[str, Any]]:
    """Every run (of a project, of a labeling set), newest first."""
    return autolabel.runs(db, hub.settings, project_id, set_id)


@router.get("/runs/{op_id}")
def run(
    op_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return autolabel.run_view(db, hub.settings, op_id)


@router.get("/targets")
def targets(project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """A project's labeling sets and its datasets' images (by split and label state) to run over."""
    return autolabel.targets(db, project_id)


@router.get("/models")
def models(
    set_id: str = "",
    release_id: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """The models that can label a set's (or a dataset version's) images, and the server it runs on."""
    if not set_id and not release_id:
        raise ServiceError(400, "라벨링 묶음이나 데이터셋 버전을 고르세요.")
    return autolabel.offer(db, hub.settings, set_id, release_id)


class Start(BaseModel):
    model_id: str
    # what it runs over: a labeling set's images not labelled yet, or a dataset version's (made into a set first)
    set_id: str = ""
    release_id: str = ""
    split: str = ""  # with release_id: one split's images ("" all)
    name: str = ""  # with release_id: the new set's name
    part_size: int = 50
    review: bool = True
    keep: float = autolabel.KEEP  # predictions below are dropped
    accept: float | None = None  # images whose predictions all reach it are saved and sent on at once
    gpus: int | None = None  # 0: on the CPU (by default a GPU when the server has one)


@router.post("")
def start(body: Start, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """Start one: over a set, or over a dataset's images nobody labelled yet (a set of them is made first)."""
    if body.set_id:
        op = autolabel.start(db, user, body.set_id, body.model_id, keep=body.keep, accept=body.accept, gpus=body.gpus)
        set_id = body.set_id
    elif body.release_id:
        found, op = autolabel.start_on_images(
            db,
            user,
            body.release_id,
            body.model_id,
            name=body.name,
            split=body.split,
            part_size=body.part_size,
            review=body.review,
            keep=body.keep,
            accept=body.accept,
            gpus=body.gpus,
        )
        set_id = found.id
    else:
        raise ServiceError(400, "라벨링 묶음이나 데이터셋 버전을 고르세요.")
    db.commit()
    return {"id": op.id, "job_id": op.job_id, "set_id": set_id}
