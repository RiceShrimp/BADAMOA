"""Hub application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from . import (
    address,
    admin,
    agent_api,
    api,
    boards_api,
    catalog,
    catalog_api,
    cleanup,
    docker_registry,
    docker_registry_api,
    exports_api,
    hub_storage,
    install,
    labeling_api,
    modelops_api,
    overview,
    qa,
    registry,
    registry_api,
    resources,
    services,
    sweeps,
    template_checks,
    training_api,
)
from .config import HubSettings
from .db import Database
from .deps import Hub
from .mcp_server import build_mcp

log = logging.getLogger("badamoa.hub")

STATIC_DIR = Path(__file__).parent / "static"
_SWEEP_INTERVAL_S = 15.0


def compact_if_worth(hub: Hub) -> None:
    """After the trash pass: give the database's empty pages back when they are a good part of it and no job runs."""
    with hub.db.session() as db:
        if hub_storage.busy(db) or not hub_storage.worth_compacting(db, hub.settings):
            return
    hub_storage.compact(hub.db, hub.settings)


async def _sweeper(hub: Hub) -> None:
    def sweep() -> int:
        with hub.db.session() as db:
            lost = services.sweep_lost_jobs(db, hub.settings)
            catalog.drop_unfinished_imports(db, hub.settings)  # imports that ended unfinished, also from before
            return lost

    def prune() -> None:
        with hub.db.session() as db:
            resources.prune(db)

    def fingerprint() -> int:
        with hub.db.session() as db:
            return catalog.fill_digests(db)

    def charts() -> int:
        with hub.db.session() as db:
            return overview.fill_charts(db)

    def label_checks() -> int:
        with hub.db.session() as db:
            return qa.fill_runs(db)

    ticks = 0
    while True:
        await anyio.sleep(_SWEEP_INTERVAL_S)
        ticks += 1
        try:  # versions made before fingerprints: a few each round until none is left
            await anyio.to_thread.run_sync(fingerprint)
        except Exception:
            log.exception("version fingerprints failed")
        try:  # finished versions' statistics, one each round, so the statistics tab opens at once
            await anyio.to_thread.run_sync(charts)
        except Exception:
            log.exception("version statistics failed")
        if hub_storage.images_due(hub.settings):  # what the images take: after a change, or every ten minutes
            try:
                await anyio.to_thread.run_sync(hub_storage.refresh_images, hub.db, hub.settings)
            except Exception:
                log.exception("image storage count failed")
        try:  # and their label checks (검수), likewise
            await anyio.to_thread.run_sync(label_checks)
        except Exception:
            log.exception("version label checks failed")
        if ticks % int(cleanup.EVERY_S // _SWEEP_INTERVAL_S) == 0:
            try:
                await anyio.to_thread.run_sync(cleanup.scheduled, hub.db, hub.settings)
                await anyio.to_thread.run_sync(compact_if_worth, hub)
            except Exception:
                log.exception("storage cleanup failed")
        if ticks % int(hub_storage.MEASURE_EVERY_S // _SWEEP_INTERVAL_S) == 1:  # soon after start, then every 10 min
            try:
                await anyio.to_thread.run_sync(hub_storage.measure, hub.db, hub.settings)
                hub_storage.start_shrinking(hub.db, hub.settings)  # older, larger thumbnails: in a thread of its own
            except Exception:
                log.exception("storage count failed")
        if ticks == 2 or ticks % 40 == 0:  # the Docker registry: set up soon after start, then checked every 10 min
            try:
                await anyio.to_thread.run_sync(docker_registry.ensure, hub.db, hub.settings)
            except Exception:
                log.exception("Docker registry check failed")
        if ticks % 40 == 0:  # about every 10 minutes
            try:
                await anyio.to_thread.run_sync(prune)
            except Exception:
                log.exception("metric prune failed")
        try:
            lost = await anyio.to_thread.run_sync(sweep)
            if lost:
                log.warning("marked %d job(s) as lost", lost)
        except Exception:  # keep the sweeper alive
            log.exception("sweep failed")


async def _searches(hub: Hub) -> None:
    """Start the next runs of hyperparameter searches: soon after a training ends, else every 15 s."""
    ticks = 0
    while True:
        await anyio.sleep(1.0)
        ticks += 1
        if not sweeps.pending() and ticks % 15:
            continue
        try:
            await anyio.to_thread.run_sync(sweeps.advance_all, hub.db)
        except Exception:  # keep the loop alive
            log.exception("advancing searches failed")


def create_app(settings: HubSettings) -> FastAPI:
    db = Database(settings.database_url or "")
    db.create_all()
    hub = Hub(settings=settings, db=db)
    with db.session() as session:
        registry.sync_builtin(session)
        template_checks.mark_legacy(session)
        admin.load_gpu_busy(session, settings)
        address.load(session, settings)
    mcp, mcp_endpoint = build_mcp(hub)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp.session_manager.run(), anyio.create_task_group() as tg:
            tg.start_soon(_sweeper, hub)
            tg.start_soon(_searches, hub)
            yield
            tg.cancel_scope.cancel()

    app = FastAPI(
        title="BADAMOA Hub",
        version=services.HUB_VERSION,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.hub = hub

    @app.exception_handler(services.ServiceError)
    async def _service_error(_: Request, exc: services.ServiceError) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    app.include_router(api.router)
    app.include_router(api.public_router)
    app.include_router(agent_api.router)
    app.include_router(install.router)
    app.include_router(catalog_api.router)
    app.include_router(catalog_api.agent_router)
    app.include_router(catalog_api.media_router)
    app.include_router(labeling_api.router)
    app.include_router(boards_api.router)
    app.include_router(exports_api.router)
    app.include_router(exports_api.agent_router)
    app.include_router(exports_api.media_router)
    app.include_router(training_api.router)
    app.include_router(training_api.agent_router)
    app.include_router(training_api.media_router)
    app.include_router(registry_api.router)
    app.include_router(docker_registry_api.router)
    app.include_router(registry_api.agent_router)
    app.include_router(registry_api.packages_router)
    app.include_router(registry_api.media_router)
    app.include_router(modelops_api.router)
    app.include_router(modelops_api.agent_router)
    app.include_router(modelops_api.media_router)
    app.router.routes.append(Route("/mcp", endpoint=mcp_endpoint))

    web_dir = settings.web_dir or STATIC_DIR
    index = web_dir / "index.html"
    if index.is_file():
        if (web_dir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=web_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith(("api/", "install/", "mcp")):
                raise HTTPException(status_code=404, detail="not found")
            candidate = (web_dir / path).resolve()
            if path and candidate.is_file() and web_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

    else:

        @app.get("/", include_in_schema=False)
        def no_ui() -> JSONResponse:
            return JSONResponse(
                {
                    "name": "BADAMOA Hub",
                    "version": services.HUB_VERSION,
                    "detail": "웹 UI가 빌드되지 않았습니다. `npm --prefix web run build`를 실행하세요.",
                    "api_docs": "/api/docs",
                }
            )

    return app
