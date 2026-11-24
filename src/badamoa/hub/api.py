"""REST API used by the web UI and scripts (``/api/*``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import (
    address,
    admin,
    asks,
    autostart,
    disk_space,
    gpus,
    guides,
    migration,
    node_storage,
    prometheus,
    resources,
    samples,
    services,
)
from .db import iso
from .deps import Hub, admin_user, bearer_token, current_user, db_session, get_hub
from .install import join_commands, node_wheel
from .models import User
from .release_notes import load_release_notes

router = APIRouter(prefix="/api")
public_router = APIRouter()  # outside /api: /metrics, where scrapers look for it


# ---- health & setup ---------------------------------------------------------


@router.get("/health")
def health(db: Session = Depends(db_session)) -> dict[str, Any]:
    # hub_id: the same on a Hub imported from this one, so a Hub being moved can tell its new place answers
    return {
        "status": "ok",
        "version": services.HUB_VERSION,
        "needs_setup": services.needs_setup(db),
        "hub_id": migration.hub_id(db),
    }


class Credentials(BaseModel):
    username: str
    password: str


@router.get("/setup/status")
def setup_status(db: Session = Depends(db_session)) -> dict[str, Any]:
    return {"needs_setup": services.needs_setup(db), "version": services.HUB_VERSION, "migration": migration.view(db)}


# ---- moving the Hub to another PC (admin) ----------------------------------------------------


class ExportRequest(BaseModel):
    folder: str


class HandOver(BaseModel):
    url: str


@router.get("/migration")
def migration_overview(
    _: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """How big an export is, where it could go, what blocks it and what stays on this PC; the state."""
    return migration.overview(db, hub.settings)


@router.post("/migration/export")
def migration_export(
    body: ExportRequest, request: Request, who: User = Depends(admin_user), db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:  # fmt: skip
    """Make this Hub read-only and pack it into one file in a folder of this PC."""
    agent = getattr(request.app.state, "local_agent", None)
    return migration.start_export(db, who, hub.settings, body.folder, agent)


@router.post("/migration/cancel")
def migration_cancel(who: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """This Hub writable again."""
    done = migration.cancel(db, who)
    db.commit()
    return done


@router.post("/migration/hand-over")
def migration_hand_over(body: HandOver, who: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    """Send nodes and browsers to the new Hub (it must answer as this Hub's import)."""
    done = migration.hand_over(db, who, body.url)
    db.commit()
    return done


@public_router.get("/media/hub-export/{name}")
def migration_file(name: str, exp: int, sig: str, hub: Hub = Depends(get_hub)) -> FileResponse:
    """The exported file by a signed link from the moving screen, to fetch from the new PC's browser (ranges are
    served, so a broken download resumes)."""
    from .catalog import verify_media

    path = migration.exported_file(hub)
    if not verify_media(hub.settings, "hub-export", name, exp, sig) or path is None or path.name != name:
        raise HTTPException(status_code=404, detail="내보낸 파일이 없거나 링크가 지났습니다. 화면에서 링크를 다시 받으세요.")
    return FileResponse(path, filename=path.name, media_type="application/x-tar")


@router.post("/setup")
def setup(body: Credentials, db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    if not services.needs_setup(db):
        raise services.ServiceError(409, "이미 초기 설정이 완료되었습니다.")
    user = services.create_user(db, body.username, body.password, role="admin")
    token, _ = services.login(db, body.username, body.password, hub.settings)
    services.audit(db, user.id, "ui", "setup.admin_created", user.id)
    db.commit()
    made = samples.safely(db, user, hub.settings)  # COCO8 for Det · Seg · Kpt and a 검수 example
    db.commit()
    return {"token": token, "user": services.serialize_user(user), "samples": made}


# ---- auth ---------------------------------------------------------------------


@router.post("/auth/login")
def login(body: Credentials, db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    token, user = services.login(db, body.username, body.password, hub.settings)
    db.commit()
    return {"token": token, "user": services.serialize_user(user)}


@router.get("/auth/me")
def me(user: User = Depends(current_user)) -> dict[str, Any]:
    return services.serialize_user(user)


# ---- each person's order of cards and lists ----------------------------------------------

ORDER_KEYS = 200  # screens one person can have an order for (a project's datasets is one per project)
ORDER_IDS = 5000


class Order(BaseModel):
    ids: list[str] = Field(max_length=ORDER_IDS)  # first to last; empty: back to the screen's own order


@router.get("/me/orders")
def my_orders(user: User = Depends(current_user)) -> dict[str, list[str]]:
    """The order this person dragged each screen's cards or rows into (screens not dragged are not listed)."""
    return dict(user.orders or {})


@router.put("/me/orders/{key}")
def set_my_order(
    key: str, body: Order, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, list[str]]:
    """Keep the order of one screen's cards (``key`` names the screen, e.g. ``review.projects``)."""
    if not 0 < len(key) <= 128:
        raise services.ServiceError(400, "화면 이름이 너무 깁니다")
    orders = dict(user.orders or {})
    orders.pop(key, None)  # a screen dragged again goes last: the oldest are let go first
    if body.ids:
        orders[key] = [str(i)[:64] for i in dict.fromkeys(body.ids)]
    while len(orders) > ORDER_KEYS:
        orders.pop(next(iter(orders)))
    user.orders = orders  # a new dict: a JSON column changed in place is not saved
    db.commit()
    return orders


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


@router.get("/guides")
def guide_list(_: User = Depends(current_user)) -> list[dict[str, Any]]:
    """The guides the app shows: the user manual and the template guide."""
    return guides.listing()


@router.get("/guides/images/{file}")
def guide_image(file: str) -> FileResponse:
    """A picture in the manual. Plain <img> tags fetch it, so no sign-in: it shows the sample Hub's screens only."""
    path = guides.image_path(file)
    return FileResponse(path, media_type=guides.IMAGE_TYPES[path.suffix], headers={"Cache-Control": "public, max-age=86400"})


@router.get("/guides/{name}")
def guide(name: str, _: User = Depends(current_user)) -> dict[str, Any]:
    return guides.read(name)


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
    return admin.list_users(db)


@router.get("/users/online")
def online_users(_: User = Depends(admin_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    """Who uses the Hub now (a token used in the last three minutes), for admins."""
    return admin.online(db)


@router.post("/users")
def create_user(body: NewUser, who: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    user = services.create_user(db, body.username, body.password, body.role)
    services.audit(db, who.id, "ui", "user.create", user.id, username=user.username, role=user.role)
    db.commit()
    return services.serialize_user(user)


class UserPatch(BaseModel):
    role: str | None = None
    disabled: bool | None = None
    template_upload: bool | None = None  # a member may upload model kinds (as drafts)
    password: str | None = None  # a new password set by the admin; the person's browser sessions end


@router.delete("/users/{user_id}")
def delete_user(user_id: str, who: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """Remove an account for good (not one's own, not the last admin); what the person made stays."""
    done = admin.delete_user(db, who, user_id)
    db.commit()
    return done


@router.patch("/users/{user_id}")
def update_user(
    user_id: str, body: UserPatch, who: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    admin.update_user(db, who, user_id, **body.model_dump(exclude_unset=True))
    db.commit()
    return next(u for u in admin.list_users(db) if u["id"] == user_id)


class PasswordChange(BaseModel):
    current: str
    new: str


@router.post("/auth/password")
def change_password(
    body: PasswordChange, request: Request, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict:
    """Your own password; your other browser sessions are signed out, this one stays."""
    admin.change_own_password(db, user, body.current, body.new, bearer_token(request))
    db.commit()
    return {"ok": True}


# ---- when a GPU counts as busy (admin) ---------------------------------------------


class GpuBusy(BaseModel):
    memory_pct: float | None = None  # both None: back to the Hub's launch options
    util_pct: float | None = None


@router.get("/settings/gpu-busy")
def gpu_busy(_: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict:
    return admin.gpu_busy_view(db, hub.settings)


@router.put("/settings/gpu-busy")
def set_gpu_busy(
    body: GpuBusy, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict:
    view = admin.set_gpu_busy(db, user, hub.settings, body.memory_pct, body.util_pct)
    db.commit()
    return view


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


@router.get("/nodes/{node_id}/storage")
def node_storage_view(node_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """What BADAMOA keeps on the server, by kind, as it last measured, and what each item is for."""
    return node_storage.view(db, services.get_node(db, node_id))


class StorageClean(BaseModel):
    kind: str = ""  # envs · containers · models · templates · template_data · runs · uv; "" = only measure again
    items: list[str] = Field(default_factory=list)
    all: bool = False  # every item of the kind no job uses


@router.post("/nodes/{node_id}/storage/clean")
def node_storage_clean(
    node_id: str, body: StorageClean, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Remove what BADAMOA kept on the server (admins), or with no ``kind`` have it measured again."""
    if body.kind:
        job = node_storage.clean(db, user, node_id, body.kind, body.items, body.all, via="ui")
    else:
        job = node_storage.measure(db, user, node_id, via="ui")
    db.commit()
    return {"job_id": job.id, **node_storage.view(db, services.get_node(db, node_id))}


@router.get("/nodes/{node_id}/docker-move")
def docker_move_state(node_id: str, _: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """Whether BADAMOA can move this server's Docker images now (and why not), where they are, where they could go."""
    node = services.get_node(db, node_id)
    return disk_space.docker_move_state(node, services.node_busy(db, node))


class DockerMove(BaseModel):
    target: str
    remove_old: bool = True  # the old folder goes once Docker runs from the new one with every image
    stop_containers: bool = False  # go on even though containers run (they stop while Docker is down)


@router.post("/nodes/{node_id}/docker-move")
def docker_move(
    node_id: str, body: DockerMove, user: User = Depends(admin_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Move this server's Docker images to another disk: Docker stops for the copy, so every container of the server
    stops meanwhile. Admins only, on a Linux node installed with sudo, while it runs no BADAMOA job."""
    job = services.start_docker_move(db, user, node_id, body.target, body.remove_old, body.stop_containers, via="ui")
    db.commit()
    return {"job_id": job.id}


@router.get("/nodes/{node_id}/browse")
def browse_node(
    node_id: str,
    path: str = "",
    hidden: bool = False,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """A node's folders (names, image counts, the dataset layout they look like), asked of the node itself."""
    node = services.get_node(db, node_id)
    if services.node_status(node, hub.settings) == "offline":
        raise services.ServiceError(409, "노드가 꺼져 있어 폴더를 볼 수 없습니다. 경로를 직접 적으세요.")
    if not asks.answers_questions(node.agent_version or ""):
        raise services.ServiceError(
            409, f"이 노드의 프로그램({node.agent_version})은 폴더 보기를 모릅니다. 노드 화면에서 업데이트하세요."
        )
    db.close()  # nothing more from the database while waiting for the node
    return asks.ask(node.id, "browse", {"path": path, "hidden": hidden})


@router.get("/nodes/{node_id}/metrics")
def node_metrics(
    node_id: str,
    range: str = "1h",
    start: datetime | None = Query(None, alias="from"),
    end: datetime | None = Query(None, alias="to"),
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
) -> dict[str, Any]:
    """A node's history: a named ``range``, or the moments ``from`` … ``to`` picked on a chart (ISO, UTC)."""
    services.get_node(db, node_id)
    try:
        if start is not None and end is not None:
            return resources.zoom(db, node_id, _naive_utc(start), _naive_utc(end))
        return resources.history(db, node_id, range)
    except ValueError as e:
        raise services.ServiceError(400, str(e)) from e


def _naive_utc(moment: datetime) -> datetime:
    """The Hub stores naive UTC; a time with an offset is converted, one without is taken as UTC."""
    return moment.astimezone(UTC).replace(tzinfo=None) if moment.tzinfo else moment


@public_router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> str:
    """Nodes, GPUs, disks and the job queue in the Prometheus text format (for an existing Prometheus · Grafana).

    Sign in with an API token: ``authorization: {credentials: <token>}`` in the scrape config.
    """
    return prometheus.render(db, hub.settings)


class NodePatch(BaseModel):
    name: str | None = None
    roles: list[str] | None = None
    enabled: bool | None = None
    labels: dict[str, Any] | None = None
    reserved_gpus: list[int] | None = None  # GPU indices kept for people
    cache_dir: str | None = None  # where images copied from other nodes go ("" = the node picks)
    work_dir: str | None = None  # where environments, work folders and fetched files go ("" = the node picks)


@router.get("/node-update-commands")
def node_update_commands(request: Request, _: User = Depends(admin_user), hub: Hub = Depends(get_hub)) -> dict:
    """For nodes from before they updated themselves: the one-time update, the registration stays."""
    from .install import update_commands

    hub_url = address.resolve(request, hub.settings)[0]
    return {"hub_url": hub_url, "version": services.HUB_VERSION, "commands": update_commands(hub_url)}


@router.get("/gpu-board")
def gpu_board(_: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict:
    """Every GPU of every server with what holds it, and the jobs waiting for one."""
    return gpus.board(db, hub.settings)


@router.get("/nodes/{node_id}/timeline")
def node_timeline(
    node_id: str,
    range: str = "24h",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """One server's GPUs over time: utilization, and the BADAMOA jobs and people that held each."""
    return gpus.timeline(db, hub.settings, services.get_node(db, node_id), range)


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
def delete_node(
    node_id: str,
    request: Request,
    cleanup: bool = True,
    user: User = Depends(admin_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict:
    """Remove a node; with ``cleanup`` an online node also takes BADAMOA off its machine (data stays).

    When it cannot (off, or a program from before 0.32.0) the answer carries the commands to do it by hand.
    """
    from .install import remove_commands

    node = services.get_node(db, node_id)
    services.require_admin(user)
    removed, reason = False, ""
    if cleanup and not node.is_local:
        if services.node_status(node, hub.settings) == "offline":
            reason = "노드가 꺼져 있어 서버를 정리하지 못했습니다."
        elif not asks.answers_questions(node.agent_version or "", asks.REMOVE_SINCE):
            reason = f"노드 프로그램({node.agent_version})이 스스로 정리하는 기능을 모릅니다."
        else:
            try:
                asks.ask(node.id, "remove", {}, wait=10)
                removed = True
            except services.ServiceError as e:
                reason = f"서버를 정리하지 못했습니다: {e.message}"
    services.delete_node(db, user, node_id)
    db.commit()
    hub_url = address.resolve(request, hub.settings)[0]
    return {"ok": True, "removed_on_server": removed, "reason": reason, "commands": remove_commands(hub_url)}


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
    where = address.view(request, hub.settings)
    hub_url = where["url"]
    return {
        "token": token,
        "roles": row.roles,
        "expires_at": iso(row.expires_at),
        "hub_url": hub_url,
        "address": where,  # where the address came from, and others this machine has
        # Without the wheel of this Hub's version on its index the install fails half-way.
        "node_package": {"version": services.HUB_VERSION, "ready": node_wheel(hub.settings) is not None},
        "commands": join_commands(hub_url, token),
    }


class HubAddress(BaseModel):
    url: str | None = None  # None: find it by itself again


@router.get("/hub-url")
def hub_url(request: Request, _: User = Depends(current_user), hub: Hub = Depends(get_hub)) -> dict[str, str]:
    """The address to put in commands run elsewhere (MCP, template push, Prometheus): never the browser's localhost."""
    return {"url": address.resolve(request, hub.settings)[0]}


@router.get("/settings/hub-address")
def hub_address(request: Request, _: User = Depends(admin_user), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """The address nodes are told to reach this Hub at, where it comes from, and the ones this machine has."""
    return address.view(request, hub.settings)


@router.put("/settings/hub-address")
def set_hub_address(
    body: HubAddress,
    request: Request,
    user: User = Depends(admin_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    address.save(db, user, hub.settings, body.url)
    db.commit()
    return address.view(request, hub.settings)


class HubAutostart(BaseModel):
    on: bool


@router.get("/settings/hub-autostart")
def hub_autostart(_: User = Depends(admin_user), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Whether this Hub starts by itself when its machine starts, and how (boot · log-on)."""
    return autostart.status(hub.settings.home, hub.settings.port)


@router.put("/settings/hub-autostart")
def set_hub_autostart(
    body: HubAutostart, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """On: record how this Hub runs now and register the machine to start it that way. Off: take that away."""
    settings = hub.settings
    if body.on:
        autostart.save_launch(settings, settings.extra.get(admin.LAUNCH_KEY))
        why = autostart.turn_on(settings.home)
    else:
        why = autostart.turn_off(settings.home, settings.port)
    if why:
        raise services.ServiceError(400, f"자동 실행을 {'켜지' if body.on else '끄지'} 못했습니다: {why}")
    view = autostart.status(settings.home, settings.port)
    services.audit(db, user.id, "ui", "hub.autostart", view["name"], on=view["on"], when=view["when"])
    db.commit()
    return view


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
