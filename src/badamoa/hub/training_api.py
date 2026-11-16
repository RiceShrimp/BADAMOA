"""Training endpoints: runs and models for users, the training protocol for nodes, weights download."""

from __future__ import annotations

import mimetypes
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from badamoa.core.protocol import TrainFinish, TrainReport

from . import catalog, lineage, modelops, services, sweeps, training
from .deps import Hub, admin_user, bearer_token, current_node, current_user, db_session, get_hub
from .models import Node, User

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/train")
media_router = APIRouter(prefix="/media")

MAX_WEIGHTS_BYTES = 8 * 1024**3
MAX_FILE_BYTES = MAX_WEIGHTS_BYTES  # the last weights of a run are one of its files
_CHUNK = 1024 * 1024


# ---- runs ----------------------------------------------------------------------------


class NewRun(BaseModel):
    release_id: str
    template: str = ""  # id or name; empty = the project's default
    environment: str = ""  # id or name; empty = the template's default
    params: dict[str, Any] = Field(default_factory=dict)  # the template's own settings
    name: str = ""
    model_name: str = ""
    classes: list[int] = Field(default_factory=list)
    unlabelled: str = "exclude"
    val_fraction: float = 0.2
    gpus: int | None = None
    min_vram_mb: int = 0
    base_model_id: str = ""  # continue from this registered model's weights
    tags: list[str] = Field(default_factory=list)
    note: str = ""


class RunUpdate(BaseModel):
    name: str | None = None
    note: str | None = None
    tags: list[str] | None = None


@router.get("/projects/{project_id}/runs")
def list_runs(
    project_id: str,
    tag: str = "",
    q: str = "",
    sweep: str = "",
    limit: int = 200,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> list[dict[str, Any]]:
    """The project's runs, newest first; ``tag`` keeps one tag, ``q`` searches names, notes and tags."""
    return training.list_runs(db, hub.settings, project_id, limit=limit, tag=tag, q=q, sweep=sweep)


# ---- sweeps (hyperparameter search) -----------------------------------------------------


class NewSweep(BaseModel):
    release_id: str
    space: dict[str, Any]  # {"lr0": {"values": [0.01, 0.001]}, "momentum": {"min": 0.8, "max": 0.98}}
    template: str = ""
    environment: str = ""
    params: dict[str, Any] = Field(default_factory=dict)  # the settings every run shares
    method: str = "grid"  # grid | random | bayes
    metric: str = ""  # empty: the task's main metric
    goal: str = ""  # maximize | minimize; empty: minimize for a loss, else maximize
    max_runs: int = 10  # random: how many draws
    parallel: int = 1
    name: str = ""
    classes: list[int] = Field(default_factory=list)
    unlabelled: str = "exclude"
    val_fraction: float = 0.2
    gpus: int | None = None
    min_vram_mb: int = 0
    base_model_id: str = ""
    tags: list[str] = Field(default_factory=list)


@router.post("/sweeps")
def start_sweep(
    body: NewSweep, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    with sweeps.lock:
        sweep = sweeps.start_sweep(db, user, via="ui", **body.model_dump())
        db.commit()
    return sweeps.sweep_view(db, hub.settings, sweep)


@router.get("/projects/{project_id}/sweeps")
def list_sweeps(
    project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    return sweeps.list_sweeps(db, hub.settings, project_id)


@router.get("/sweeps/{sweep_id}")
def get_sweep(
    sweep_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return sweeps.sweep_view(db, hub.settings, sweeps.get_sweep(db, sweep_id), detail=True)


@router.post("/sweeps/{sweep_id}/stop")
def stop_sweep(
    sweep_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    with sweeps.lock:
        sweep = sweeps.stop_sweep(db, user, sweep_id, via="ui")
        db.commit()
    return sweeps.sweep_view(db, hub.settings, sweep, detail=True)


@router.get("/projects/{project_id}/tags")
def project_tags(project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return training.project_tags(db, project_id)


@router.post("/runs")
def start_run(
    body: NewRun, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    run = training.start_run(db, user, via="ui", **body.model_dump())
    db.commit()
    return training.run_view(db, hub.settings, run)


@router.get("/runs/{run_id}")
def get_run(
    run_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return training.run_view(db, hub.settings, training.get_run(db, run_id), detail=True)


@router.patch("/runs/{run_id}")
def update_run(
    run_id: str,
    body: RunUpdate,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    run = training.update_run(db, user, run_id, **body.model_dump(), via="ui")
    db.commit()
    return training.run_view(db, hub.settings, run)


@router.get("/runs/{run_id}/lineage")
def run_lineage(run_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return lineage.for_run(db, training.get_run(db, run_id))


@router.get("/models/{model_id}/lineage")
def model_lineage(model_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return lineage.for_model(db, training.get_model(db, model_id))


@router.get("/runs/{run_id}/system")
def run_system(run_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return training.system(db, training.get_run(db, run_id))


@router.post("/runs/{run_id}/rerun")
def rerun(
    run_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    run = training.rerun(db, user, run_id, via="ui")
    db.commit()
    return training.run_view(db, hub.settings, run)


@router.post("/runs/{run_id}/resume")
def resume(
    run_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    run = training.resume(db, user, run_id, via="ui")
    db.commit()
    return training.run_view(db, hub.settings, run)


@router.delete("/runs/{run_id}")
def delete_run(
    run_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict:
    training.delete_run(db, user, run_id, catalog.storage_for(str(hub.settings.storage_dir)))
    db.commit()
    return {"ok": True}


# ---- models --------------------------------------------------------------------------


class AliasRequest(BaseModel):
    alias: str = ""


class ModelUpdate(BaseModel):
    note: str | None = None
    tags: list[str] | None = None


@router.get("/projects/{project_id}/models")
def list_models(
    project_id: str,
    tag: str = "",
    q: str = "",
    name: str = "",
    alias: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> list[dict[str, Any]]:
    """Model versions; ``name`` and ``alias`` together find the version an alias points at."""
    return training.list_models(db, hub.settings, project_id, tag=tag, q=q, name=name, alias=alias)


@router.get("/model-aliases")
def model_aliases(_: User = Depends(current_user)) -> list[str]:
    return list(training.SUGGESTED_ALIASES)


@router.get("/models/{model_id}")
def get_model(
    model_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    model = training.get_model(db, model_id)
    view = training.model_view(db, hub.settings, model, detail=True)
    view["actions"] = modelops.actions(db, model)  # what this model's template can do: eval, export, predict
    return view


@router.post("/models/{model_id}/alias")
def set_alias(
    model_id: str,
    body: AliasRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    model = training.set_alias(db, user, model_id, body.alias, via="ui")
    db.commit()
    return training.model_view(db, hub.settings, model, detail=True)


@router.delete("/models/{model_id}/alias/{alias}")
def remove_alias(
    model_id: str,
    alias: str,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    model = training.set_alias(db, user, model_id, alias, via="ui", remove=True)
    db.commit()
    return training.model_view(db, hub.settings, model, detail=True)


@router.patch("/models/{model_id}")
def update_model(
    model_id: str,
    body: ModelUpdate,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    model = training.update_model(db, user, model_id, **body.model_dump(), via="ui")
    db.commit()
    return training.model_view(db, hub.settings, model, detail=True)


@router.delete("/models/{model_id}")
def delete_model(model_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    training.delete_model(db, user, model_id)
    db.commit()
    return {"ok": True}


# ---- node protocol -------------------------------------------------------------------


@agent_router.get("/manifest")
def manifest(
    job_id: str, after: int = 0, limit: int = 2000, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    return training.manifest(db, node, job_id, after, limit)


@agent_router.get("/template")
def template_bundle(
    job_id: str, node: Node = Depends(current_node), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> Response:
    """The code of the run's template, as the zip it was registered with."""
    blob = training.template_bundle(db, node, job_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(blob):
        raise HTTPException(status_code=404, detail="템플릿 묶음이 저장소에 없습니다.")
    return Response(storage.get_blob(blob), media_type="application/zip")


@agent_router.get("/checkpoint")
def checkpoint(
    job_id: str, node: Node = Depends(current_node), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> Response:
    """The checkpoint of the interrupted run this job continues."""
    blob = training.checkpoint_to_resume(db, node, job_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(blob):
        raise HTTPException(status_code=404, detail="체크포인트가 저장소에 없습니다.")
    return Response(storage.get_blob(blob), media_type="application/octet-stream")


@agent_router.get("/init-weights")
def init_weights(
    job_id: str, node: Node = Depends(current_node), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> Response:
    """The weights of the registered model this run continues from."""
    blob = training.init_weights(db, node, job_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(blob):
        raise HTTPException(status_code=404, detail="가중치 파일이 저장소에 없습니다.")
    return Response(storage.get_blob(blob), media_type="application/octet-stream")


@agent_router.post("/report")
def report(
    job_id: str, body: TrainReport, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = training.report(db, node, job_id, body)
    db.commit()
    return result


@agent_router.put("/artifact")
async def artifact(
    job_id: str,
    request: Request,
    name: str = "best.pt",
    kind: str = "weights",
    epoch: int = 0,
    key: str = "",
    step: int | None = None,
    caption: str = "",
    media: str = "",
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    """Receive a file as a raw request body, streamed to disk, and store it by content hash.

    ``kind=weights`` (the model, registered when the run finishes), ``kind=file`` (something the
    code recorded with the run: a plot, sample predictions) or ``kind=checkpoint`` (the state of
    an unfinished run after ``epoch``, replacing the previous one).
    """
    limit = MAX_FILE_BYTES if kind == "file" else MAX_WEIGHTS_BYTES
    token = bearer_token(request)

    def check() -> None:
        with hub.db.session() as db:
            node = services.authenticate_node(db, token)
            if node is None:
                raise HTTPException(status_code=401, detail="노드 인증 실패")
            training._running(db, node, job_id)

    await anyio.to_thread.run_sync(check)
    incoming = hub.settings.storage_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=incoming, suffix=".part")
    size = 0
    try:
        with os.fdopen(handle, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        storage = catalog.storage_for(str(hub.settings.storage_dir))
        blob = await anyio.to_thread.run_sync(storage.put_blob_file, temp)
    finally:
        Path(temp).unlink(missing_ok=True)
    if kind == "file":

        def record() -> dict[str, Any]:
            with hub.db.session() as db:
                node = services.authenticate_node(db, token)
                return training.add_file(db, node, job_id, name, blob, size, key, step, caption, media)

        return await anyio.to_thread.run_sync(record)
    if kind == "checkpoint":

        def keep() -> dict[str, Any]:
            with hub.db.session() as db:
                node = services.authenticate_node(db, token)
                return training.set_checkpoint(db, node, job_id, name, blob, size, epoch, storage)

        return await anyio.to_thread.run_sync(keep)
    return {"blob": blob, "bytes": size, "file_name": Path(name).name}


@agent_router.post("/finish")
def finish(
    job_id: str,
    body: TrainFinish,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    result = training.finish(db, node, job_id, body, catalog.storage_for(str(hub.settings.storage_dir)))
    db.commit()
    return result


# ---- files a run recorded (signed URL, like images) -------------------------------------


@media_router.get("/file/{file_id}")
def run_file(file_id: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)) -> Response:
    if not catalog.verify_media(hub.settings, "file", file_id, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    row = training.get_file(db, file_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(row.blob):
        raise HTTPException(status_code=404, detail="파일이 저장소에 없습니다.")
    return blob_response(storage, row.blob, row.name, row.kind in ("image", "table-image"))


def blob_response(storage, blob: str, name: str, image: bool) -> StreamingResponse:
    """A stored file as a response: streamed (weights are large), shown in place only when it is an image."""
    kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
    # Only images are shown in place; anything else is offered as a download, whatever it contains.
    inline = image and kind.startswith("image/") and kind != "image/svg+xml"

    def stream():
        with storage.open(storage.blob_key(blob)) as source:
            while chunk := source.read(_CHUNK):
                yield chunk

    headers = {
        "Content-Disposition": f"{'inline' if inline else 'attachment'}; filename*=UTF-8''{quote(Path(name).name)}",
        "Content-Length": str(storage.blob_size(blob)),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=3600",
    }
    return StreamingResponse(stream(), media_type=kind if inline else "application/octet-stream", headers=headers)


@media_router.get("/run-metrics/{run_id}")
def run_metrics(run_id: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)) -> Response:
    """The run's metrics at every step as CSV (with a BOM, so spreadsheet programs read Korean names right)."""
    if not catalog.verify_media(hub.settings, "run-metrics", run_id, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    run = training.get_run(db, run_id)
    file_name = quote(f"{run.name or run.id}-metrics.csv")
    return Response(
        training.metrics_csv(db, run).encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=\"metrics.csv\"; filename*=UTF-8''{file_name}"},
    )


@media_router.get("/run-files/{run_id}")
def run_files(
    run_id: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> FileResponse:
    """Everything a run recorded, as one zip with the folders the files were written in."""
    if not catalog.verify_media(hub.settings, "run-files", run_id, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    run = training.get_run(db, run_id)
    rows = training.files_of(db, run_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    incoming = hub.settings.storage_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=incoming, suffix=".zip")
    with os.fdopen(handle, "wb") as out, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as bundle:
        for row in rows:
            if not storage.has_blob(row.blob):
                continue
            with storage.open(storage.blob_key(row.blob)) as source, bundle.open(row.name, "w", force_zip64=True) as target:
                while chunk := source.read(_CHUNK):
                    target.write(chunk)
    file_name = quote(f"{run.name or run.id}.zip")
    return FileResponse(
        temp,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=\"run.zip\"; filename*=UTF-8''{file_name}"},
        background=BackgroundTask(lambda: Path(temp).unlink(missing_ok=True)),
    )


# ---- weights download (signed URL, like images) ---------------------------------------


@media_router.get("/model/{model_id}")
def download_model(
    model_id: str, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> StreamingResponse:
    if not catalog.verify_media(hub.settings, "model", model_id, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    model = training.get_model(db, model_id)
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(model.blob):
        raise HTTPException(status_code=404, detail="가중치 파일이 저장소에 없습니다.")
    suffix = Path(model.file_name).suffix or ".pt"

    def stream():
        with storage.open(storage.blob_key(model.blob)) as source:
            while chunk := source.read(_CHUNK):
                yield chunk

    # Model names may be Korean; headers are latin-1, so the real name goes in the RFC 5987 form.
    file_name = quote(f"{model.name}-v{model.version}{suffix}")
    return StreamingResponse(
        stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=\"model{suffix}\"; filename*=UTF-8''{file_name}",
            "Content-Length": str(storage.blob_size(model.blob)),
        },
    )
