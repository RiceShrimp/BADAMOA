"""The first screens of 라벨링, 검수, 학습 and 모델: every project at a glance (see ``boards``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from . import boards
from .deps import Hub, current_user, db_session, get_hub
from .models import User

router = APIRouter(prefix="/api/boards")


@router.get("/labeling")
def labeling(
    user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """The parts given to me with what is left, then every project's labeling sets."""
    return boards.labeling_board(db, hub.settings, user)


@router.get("/review")
def review(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Each dataset's newest labels with how many images the checks flag, the most first."""
    result = boards.review_board(db, hub.settings)
    db.commit()  # the checks that ran are kept until the labels change
    return result


@router.get("/training")
def training(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """The runs going on now, then each project's runs summed up."""
    return boards.training_board(db, hub.settings)


@router.get("/models")
def models(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Each project's models: the ones with an alias, the best, the newest."""
    return boards.models_board(db, hub.settings)
