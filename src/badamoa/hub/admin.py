"""What an admin sets on the settings screen: people's accounts and when a GPU counts as busy."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import resources, services
from .config import HubSettings
from .db import iso, utcnow
from .models import ApiToken, HubSetting, Node, User
from .security import hash_password, hash_token, verify_password
from .services import ServiceError

# ---- accounts ---------------------------------------------------------------------------------


def list_users(db: Session) -> list[dict[str, Any]]:
    """Everyone, with when they were last seen (any of their tokens used) and how many API tokens they hold."""
    seen = dict(db.execute(select(ApiToken.user_id, func.max(ApiToken.last_used_at)).group_by(ApiToken.user_id)).all())
    tokens = dict(
        db.execute(
            select(ApiToken.user_id, func.count(ApiToken.id)).where(ApiToken.kind == "api").group_by(ApiToken.user_id)
        ).all()
    )
    return [
        {**services.serialize_user(u), "last_seen_at": iso(seen.get(u.id)), "api_tokens": tokens.get(u.id, 0)}
        for u in db.scalars(select(User).order_by(User.created_at))
    ]


def _active_admins(db: Session) -> int:
    return db.scalar(select(func.count(User.id)).where(User.role == "admin", User.disabled_at.is_(None))) or 0


def _sign_out(db: Session, user_id: str, keep: str = "") -> None:
    """End the person's browser sessions (API tokens stay), except the one with hash ``keep``."""
    query = delete(ApiToken).where(ApiToken.user_id == user_id, ApiToken.kind == "session")
    if keep:
        query = query.where(ApiToken.token_hash != keep)
    db.execute(query)


def update_user(
    db: Session,
    admin: User,
    user_id: str,
    *,
    role: str | None = None,
    disabled: bool | None = None,
    password: str | None = None,
) -> User:
    """Change someone's role, stop or restart their account, or set a new password for them.

    There is always at least one admin who can sign in: the last one cannot be demoted or stopped,
    and nobody stops their own account.
    """
    services.require_admin(admin)
    user = db.get(User, user_id)
    if user is None:
        raise ServiceError(404, "사용자를 찾을 수 없습니다.")
    last_admin = user.role == "admin" and user.disabled_at is None and _active_admins(db) <= 1
    changed: dict[str, Any] = {}
    if role is not None and role != user.role:
        if role not in ("admin", "member"):
            raise ServiceError(400, "역할은 admin 또는 member입니다.")
        if last_admin:
            raise ServiceError(409, "관리자가 한 명은 있어야 합니다. 다른 사람을 먼저 관리자로 만드세요.")
        user.role = changed["role"] = role
    if disabled is not None and disabled != (user.disabled_at is not None):
        if disabled and user.id == admin.id:
            raise ServiceError(409, "자기 계정은 중지할 수 없습니다.")
        if disabled and last_admin:
            raise ServiceError(409, "관리자가 한 명은 있어야 합니다. 다른 사람을 먼저 관리자로 만드세요.")
        user.disabled_at = utcnow() if disabled else None
        changed["disabled"] = disabled
        if disabled:
            _sign_out(db, user.id)
    if password is not None:
        if len(password) < 8:
            raise ServiceError(400, "비밀번호는 8자 이상이어야 합니다.")
        user.password_hash = hash_password(password)
        changed["password"] = True  # that it changed, never the value
        _sign_out(db, user.id)
    if changed:
        services.audit(db, admin.id, "ui", "user.update", user.id, username=user.username, **changed)
    return user


def change_own_password(db: Session, user: User, current: str, new: str, token: str) -> None:
    """Anyone, their own password: the current one first. Other browser sessions are signed out."""
    if not verify_password(current, user.password_hash):
        raise ServiceError(400, "지금 비밀번호가 맞지 않습니다.")
    if len(new) < 8:
        raise ServiceError(400, "새 비밀번호는 8자 이상이어야 합니다.")
    if new == current:
        raise ServiceError(400, "새 비밀번호가 지금 것과 같습니다.")
    user.password_hash = hash_password(new)
    _sign_out(db, user.id, keep=hash_token(token))
    services.audit(db, user.id, "ui", "user.password", user.id, username=user.username)


# ---- when a GPU counts as busy ------------------------------------------------------------------
#
# A GPU BADAMOA did not hand out still counts as busy when someone uses it directly: above a share
# of its memory or of its utilisation. The Hub starts with the values of its launch options; an
# admin can change them on the screen, and those are kept in the database and win from then on.

GPU_KEY = "gpu.busy"
LAUNCH_KEY = "gpu_busy_launch"


def load_gpu_busy(db: Session, settings: HubSettings) -> None:
    """At start-up: remember the launch values, then put the screen's values (if any) over them."""
    settings.extra.setdefault(LAUNCH_KEY, (settings.gpu_busy_memory_pct, settings.gpu_busy_util_pct))
    row = db.get(HubSetting, GPU_KEY)
    if row and row.value:
        settings.gpu_busy_memory_pct = float(row.value["memory_pct"])
        settings.gpu_busy_util_pct = float(row.value["util_pct"])


def gpu_busy_view(db: Session, settings: HubSettings) -> dict[str, Any]:
    """The thresholds, where they come from, and every GPU's load now (for trying values before saving)."""
    launch = settings.extra.get(LAUNCH_KEY, (settings.gpu_busy_memory_pct, settings.gpu_busy_util_pct))
    gpus = []
    assigned = resources.assignments(db)
    for node in db.scalars(select(Node).where(Node.enabled.is_(True)).order_by(Node.name)):
        if services.node_status(node, settings) != "online":
            continue
        live = {g.get("index"): g for g in (node.metrics or {}).get("gpus") or []}
        names = {g.get("index"): g.get("name") or "" for g in (node.info or {}).get("gpus") or []}
        for state in resources.gpu_states(node, settings, assigned.get(node.id)):
            now = live.get(state["index"], {})
            used, total = now.get("memory_used_mb"), now.get("memory_total_mb")
            gpus.append(
                {
                    "node": node.name,
                    "index": state["index"],
                    "name": names.get(state["index"], ""),
                    "memory_pct": round(used / total * 100, 1) if isinstance(used, int | float) and total else 0.0,
                    "util_pct": now.get("utilization_pct") or 0,
                    "state": state["state"],
                    "reserved": bool(state.get("reserved")),
                    "users": state.get("users") or [],
                }
            )
    return {
        "memory_pct": settings.gpu_busy_memory_pct,
        "util_pct": settings.gpu_busy_util_pct,
        "source": "screen" if db.get(HubSetting, GPU_KEY) else "launch",
        "launch": {"memory_pct": launch[0], "util_pct": launch[1]},
        "gpus": gpus,
    }


def set_gpu_busy(
    db: Session, user: User, settings: HubSettings, memory_pct: float | None, util_pct: float | None
) -> dict[str, Any]:
    """Save new thresholds (101 means "never busy"), or with both None go back to the launch values."""
    services.require_admin(user)
    row = db.get(HubSetting, GPU_KEY)
    if memory_pct is None and util_pct is None:
        if row is not None:
            db.delete(row)
        launch = settings.extra.get(LAUNCH_KEY, (settings.gpu_busy_memory_pct, settings.gpu_busy_util_pct))
        settings.gpu_busy_memory_pct, settings.gpu_busy_util_pct = float(launch[0]), float(launch[1])
        services.audit(db, user.id, "ui", "gpu.busy", "", reset=True)
        db.flush()
        return gpu_busy_view(db, settings)
    if memory_pct is None or util_pct is None or not (1 <= memory_pct <= 101 and 1 <= util_pct <= 101):
        raise ServiceError(400, "기준값은 1에서 100 사이의 %입니다 (101은 보지 않음).")
    value = {"memory_pct": float(memory_pct), "util_pct": float(util_pct)}
    if row is None:
        db.add(HubSetting(key=GPU_KEY, value=value))
    else:
        row.value = value
    settings.gpu_busy_memory_pct, settings.gpu_busy_util_pct = value["memory_pct"], value["util_pct"]
    services.audit(db, user.id, "ui", "gpu.busy", "", **value)
    db.flush()
    return gpu_busy_view(db, settings)
