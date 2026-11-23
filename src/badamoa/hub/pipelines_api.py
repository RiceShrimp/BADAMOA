"""The 파이프라인 screen: saved pipelines and their runs, and the steps a training run goes on to (see ``pipelines``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import catalog, pipelines, training
from .deps import Hub, current_user, db_session, get_hub
from .models import User
from .services import ServiceError

router = APIRouter(prefix="/api")


class Step(BaseModel):
    kind: str
    config: dict[str, Any] = Field(default_factory=dict)


class NewPipeline(BaseModel):
    dataset_id: str = ""
    release_id: str = ""  # instead of the dataset: the version the form had picked (and the one ``run`` runs on)
    name: str = ""
    steps: list[Step]
    trigger: str = "manual"  # manual | new_version
    note: str = ""
    run: bool = False  # and run it now, on ``release_id`` (else the newest version)


class PipelineUpdate(BaseModel):
    name: str | None = None
    trigger: str | None = None
    note: str | None = None
    steps: list[Step] | None = None


class NewTemplate(BaseModel):
    name: str
    description: str = ""
    steps: list[Step]
    trigger: str = "manual"  # what it suggests: manual | new_version


class RunNow(BaseModel):
    release_id: str = ""  # empty: the dataset's newest version


class NewPipelineRun(BaseModel):
    """Steps on a version without saving them: a new training and what follows it (``train`` + ``after``), the
    steps after a training run already there (``run_id`` + ``after``), or ``steps`` as they are."""

    release_id: str = ""
    train: dict[str, Any] | None = None  # the new-run dialog's settings (without the version)
    run_id: str = ""  # follow this training run
    after: list[Step] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    save: dict[str, Any] | None = None  # {"name", "trigger"}: also keep these steps as a pipeline of the dataset


def _steps(steps: list[Step]) -> list[dict[str, Any]]:
    return [s.model_dump() for s in steps]


@router.get("/projects/{project_id}/pipelines")
def board(
    project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return pipelines.board(db, hub.settings, project_id)


@router.post("/pipelines")
def create_pipeline(
    body: NewPipeline, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    dataset_id = body.dataset_id
    if body.release_id:
        dataset_id = catalog.get_release(db, body.release_id).dataset_id
    if not dataset_id:
        raise ServiceError(400, "파이프라인이 돌 데이터셋(dataset_id)이나 버전(release_id)을 주세요.")
    with pipelines.lock:
        pipeline = pipelines.create_pipeline(
            db, user, dataset_id=dataset_id, steps=_steps(body.steps), name=body.name, trigger=body.trigger,
            note=body.note, via="ui",
        )  # fmt: skip
        run = (
            pipelines.run_pipeline(db, user, pipeline.id, body.release_id, settings=hub.settings, via="ui")
            if body.run
            else None
        )
        db.commit()
    return {
        **pipelines.pipeline_view(db, hub.settings, pipeline),
        "started": pipelines.run_view(db, hub.settings, run) if run else None,
    }


@router.get("/pipelines/{pipeline_id}")
def get_pipeline(
    pipeline_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return pipelines.pipeline_view(db, hub.settings, pipelines.get_pipeline(db, pipeline_id), detail=True)


@router.patch("/pipelines/{pipeline_id}")
def update_pipeline(
    pipeline_id: str,
    body: PipelineUpdate,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    if body.steps is not None:
        fields["steps"] = _steps(body.steps)
    pipeline = pipelines.update_pipeline(db, user, pipeline_id, via="ui", **fields)
    db.commit()
    return pipelines.pipeline_view(db, hub.settings, pipeline, detail=True)


@router.delete("/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    pipelines.delete_pipeline(db, user, pipeline_id, via="ui")
    return {"ok": True}


@router.post("/pipelines/{pipeline_id}/run")
def run_pipeline(
    pipeline_id: str,
    body: RunNow,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    with pipelines.lock:
        run = pipelines.run_pipeline(db, user, pipeline_id, body.release_id, settings=hub.settings, via="ui")
        db.commit()
    return pipelines.run_view(db, hub.settings, run)


@router.post("/pipeline-runs")
def start_run(
    body: NewPipelineRun, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """From 새 학습 ("학습 끝나면"), or added to a training run: the run follows one version through its steps."""
    after = _steps(body.after)
    release_id, train = body.release_id, None
    if body.run_id:  # the training is there: the run picks it up; a pipeline kept from it trains as it did
        followed = training.get_run(db, body.run_id)
        release_id = followed.release_id
        train = {"kind": "train", "config": pipelines.train_settings_of(db, followed)}
        steps = after
    elif body.train is not None:
        if not release_id:
            raise ServiceError(400, "학습할 버전(release_id)을 주세요.")
        train = {"kind": "train", "config": {k: v for k, v in body.train.items() if k != "release_id"}}
        steps = [train, *after]
    else:
        steps = _steps(body.steps)
    with pipelines.lock:
        pipeline = None
        if body.save is not None:
            pipeline = pipelines.create_pipeline(
                db, user, dataset_id=catalog.get_release(db, release_id).dataset_id,
                steps=[train, *after] if train else steps, name=str(body.save.get("name") or ""),
                trigger=str(body.save.get("trigger") or "manual"), via="ui",
            )  # fmt: skip
        run = pipelines.start_run(
            db, user, steps=steps, release_id=release_id, pipeline=pipeline, follow_run_id=body.run_id,
            trigger="training" if train else "manual", settings=hub.settings, via="ui",
        )  # fmt: skip
        db.commit()
    return pipelines.run_view(db, hub.settings, run)


@router.get("/pipeline-runs/{run_id}")
def get_run(
    run_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    return pipelines.run_view(db, hub.settings, pipelines.get_run(db, run_id))


@router.post("/pipeline-runs/{run_id}/stop")
def stop_run(
    run_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    with pipelines.lock:
        run = pipelines.stop_run(db, user, run_id, via="ui")
        db.commit()
    return pipelines.run_view(db, hub.settings, run)


@router.post("/pipeline-runs/{run_id}/retry")
def retry_run(
    run_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    with pipelines.lock:
        run = pipelines.retry_run(db, user, run_id, settings=hub.settings, via="ui")
        db.commit()
    return pipelines.run_view(db, hub.settings, run)


@router.get("/runs/{run_id}/follow")
def follow(
    run_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """A training run's way on: the pipeline run it starts, and whether its model was evaluated."""
    return pipelines.follow_view(db, hub.settings, training.get_run(db, run_id))


@router.get("/pipeline-templates")
def list_templates(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    """The built-in templates, then the ones people saved: steps to start a pipeline (or 학습 끝나면) from."""
    return pipelines.list_templates(db)


@router.post("/pipeline-templates")
def create_template(
    body: NewTemplate, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    pipelines.create_template(
        db, user, name=body.name, description=body.description, steps=_steps(body.steps), trigger=body.trigger, via="ui"
    )
    db.commit()
    return pipelines.list_templates(db)


@router.delete("/pipeline-templates/{template_id}")
def delete_template(template_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    pipelines.delete_template(db, user, template_id, via="ui")
    return {"ok": True}
