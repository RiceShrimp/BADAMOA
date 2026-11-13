"""FastAPI dependencies: hub context, DB session, authenticated user/node."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from . import services
from .config import HubSettings
from .db import Database
from .models import Node, User


@dataclass
class Hub:
    settings: HubSettings
    db: Database


def get_hub(request: Request) -> Hub:
    return request.app.state.hub


def db_session(hub: Hub = Depends(get_hub)) -> Iterator[Session]:
    with hub.db.session() as s:
        yield s


def bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def current_user(request: Request, db: Session = Depends(db_session)) -> User:
    user = services.authenticate(db, bearer_token(request))
    if user is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return user


def current_node(request: Request, db: Session = Depends(db_session)) -> Node:
    node = services.authenticate_node(db, bearer_token(request))
    if node is None:
        raise HTTPException(status_code=401, detail="노드 인증 실패")
    return node


def public_base_url(request: Request, hub: Hub) -> str:
    if hub.settings.public_url:
        return hub.settings.public_url.rstrip("/")
    return str(request.base_url).rstrip("/")
