"""Labeling sets: images handed to people to label and review, part by part (see ``labeling``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import labeling
from .deps import current_user, db_session
from .models import User

router = APIRouter(prefix="/api")


@router.get("/label-sets")
def list_sets(project_id: str = "", user: User = Depends(current_user), db: Session = Depends(db_session)) -> list:
    return labeling.list_sets(db, user, project_id)


class NewSet(BaseModel):
    release_id: str
    name: str
    instructions: str = ""
    # the version screen's filters: split, state, class_id, site, camera, q, edited, suspect (an evaluation)
    filter: dict[str, Any] | None = None
    part_size: int = 50
    review: bool = True


@router.post("/label-sets")
def create_set(body: NewSet, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    found = labeling.create_set(
        db,
        user,
        release_id=body.release_id,
        name=body.name,
        instructions=body.instructions,
        match=body.filter,
        part_size=body.part_size,
        review=body.review,
    )
    db.commit()
    return labeling.set_view(db, user, found.id)


@router.get("/label-sets/{set_id}")
def get_set(set_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return labeling.set_view(db, user, set_id)


class Assignment(BaseModel):
    assignee: str | None = None  # a user id, "" to take it back, left out to keep
    reviewer: str | None = None


@router.patch("/label-sets/{set_id}/parts/{part}")
def assign(
    set_id: str, part: int, body: Assignment, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = labeling.assign(db, user, set_id, part, assignee=body.assignee, reviewer=body.reviewer)
    db.commit()
    return result


@router.post("/label-sets/{set_id}/items/{item_id}/done")
def done(set_id: str, item_id: int, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """Looked at and left as it is: labelled."""
    result = labeling.mark_labelled(db, user, set_id, item_id)
    db.commit()
    return result


class Review(BaseModel):
    approve: bool
    note: str = ""


@router.post("/label-sets/{set_id}/items/{item_id}/review")
def review(
    set_id: str, item_id: int, body: Review, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = labeling.review_item(db, user, set_id, item_id, body.approve, body.note)
    db.commit()
    return result


@router.delete("/label-sets/{set_id}")
def remove(set_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    result = labeling.remove_set(db, user, set_id)
    db.commit()
    return result
