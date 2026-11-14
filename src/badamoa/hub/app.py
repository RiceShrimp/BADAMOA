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

from . import agent_api, api, catalog_api, install, resources, services, training_api
from .config import HubSettings
from .db import Database
from .deps import Hub
from .mcp_server import build_mcp

log = logging.getLogger("badamoa.hub")

STATIC_DIR = Path(__file__).parent / "static"
_SWEEP_INTERVAL_S = 15.0


async def _sweeper(hub: Hub) -> None:
    def sweep() -> int:
        with hub.db.session() as db:
            return services.sweep_lost_jobs(db, hub.settings)

    def prune() -> None:
        with hub.db.session() as db:
            resources.prune(db)

    ticks = 0
    while True:
        await anyio.sleep(_SWEEP_INTERVAL_S)
        ticks += 1
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


def create_app(settings: HubSettings) -> FastAPI:
    db = Database(settings.database_url or "")
    db.create_all()
    hub = Hub(settings=settings, db=db)
    mcp, mcp_endpoint = build_mcp(hub)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp.session_manager.run(), anyio.create_task_group() as tg:
            tg.start_soon(_sweeper, hub)
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
    app.include_router(agent_api.router)
    app.include_router(install.router)
    app.include_router(catalog_api.router)
    app.include_router(catalog_api.agent_router)
    app.include_router(catalog_api.media_router)
    app.include_router(training_api.router)
    app.include_router(training_api.agent_router)
    app.include_router(training_api.media_router)
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
