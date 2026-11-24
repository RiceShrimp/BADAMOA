"""Dataset catalog endpoints: user API, node import protocol, signed media URLs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from badamoa.core.protocol import ImportBegin, ImportChunk, ImportProgress, ImportSame

from . import catalog, checks, dataset_move, drafts, originals, overview, purge, qa, review, services
from . import trash as trash_box
from .catalog_models import Item
from .deps import Hub, admin_user, current_node, current_user, db_session, get_hub
from .models import Node, User

router = APIRouter(prefix="/api")
agent_router = APIRouter(prefix="/api/agent/jobs/{job_id}/import")
media_router = APIRouter(prefix="/media")


# ---- projects & tree -------------------------------------------------------------


class NewProject(BaseModel):
    name: str
    task: str = ""  # ignored since 0.34.0: each dataset has its own task, chosen when importing
    folder: str = ""
    description: str = ""


@router.get("/projects")
def list_projects(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    return catalog.list_projects(db)


@router.post("/projects")
def create_project(body: NewProject, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    project = catalog.create_project(db, user, body.name, body.folder, body.description)
    db.commit()
    return next(p for p in catalog.list_projects(db) if p["id"] == project.id)


class ProjectPatch(BaseModel):
    name: str | None = None
    folder: str | None = None  # "" takes it out of its folder
    description: str | None = None


@router.patch("/projects/{project_id}")
def update_project(
    project_id: str, body: ProjectPatch, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    catalog.update_project(db, user, project_id, **body.model_dump(exclude_unset=True))
    db.commit()
    return next(p for p in catalog.list_projects(db) if p["id"] == project_id)


class DatasetPatch(BaseModel):
    name: str


@router.patch("/datasets/{dataset_id}")
def rename_dataset(
    dataset_id: str, body: DatasetPatch, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Rename a dataset: every version of it goes by the new name (the project's maker or an admin)."""
    dataset = catalog.rename_dataset(db, user, dataset_id, body.name)
    db.commit()
    return {"id": dataset.id, "name": dataset.name}


class DatasetMove(BaseModel):
    project_id: str
    name: str = ""  # what it is called there, when that project has a dataset of its name already


@router.get("/datasets/{dataset_id}/move")
def dataset_move_plan(
    dataset_id: str, project_id: str = "", _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """What moving a dataset to another project takes along and leaves, and what holds it now."""
    return dataset_move.plan(db, dataset_id, project_id)


@router.post("/datasets/{dataset_id}/move")
def move_dataset(
    dataset_id: str, body: DatasetMove, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Move a dataset, every version of it and its labeling sets, to another project (the maker of both or an
    admin); training runs and models stay where they were made."""
    moved = dataset_move.move(db, user, dataset_id, body.project_id, body.name)
    db.commit()
    return moved


class ReleasePatch(BaseModel):
    version: str


@router.patch("/releases/{release_id}")
def rename_release(
    release_id: str, body: ReleasePatch, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Rename a version within its dataset (not a work in progress, which is named when it is made a version)."""
    release = catalog.rename_release(db, user, release_id, body.version)
    db.commit()
    return {"id": release.id, "version": release.version}


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """To the trash, with its datasets, runs and models; the person who made it or an admin."""
    catalog.set_project_deleted(db, user, project_id, True)
    db.commit()
    return {"ok": True}


@router.post("/projects/{project_id}/restore")
def restore_project(project_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    catalog.set_project_deleted(db, user, project_id, False)
    db.commit()
    return {"ok": True}


@router.get("/trash/count")
def trash_count(_: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, int]:
    return trash_box.count(db)


@router.get("/trash")
def trash(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """What was deleted (projects, dataset versions, runs, models, templates), to bring back."""
    return trash_box.listing(db, catalog.storage_for(str(hub.settings.storage_dir)))


class EmptyTrash(BaseModel):
    projects: list[str] | None = None  # all left out: everything in the trash
    releases: list[str] | None = None
    runs: list[str] | None = None
    models: list[str] | None = None
    templates: list[str] | None = None  # the id of any version of each


@router.post("/trash/empty")
def empty_trash(
    body: EmptyTrash, user: User = Depends(admin_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Remove what is in the trash for good, and the stored files only it used. Cannot be undone."""
    result = purge.empty_trash(db, hub.settings, user, **body.model_dump())
    db.commit()
    return result


class NewFolder(BaseModel):
    name: str
    projects: list[str] = []  # moved into it; none: an empty folder


class FolderRename(BaseModel):
    name: str
    new_name: str


@router.get("/folders")
def list_folders(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    """Every folder, empty ones included, with how many projects are in it."""
    return catalog.list_folders(db)


@router.post("/folders")
def create_folder(body: NewFolder, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    result = catalog.create_folder(db, user, body.name, body.projects)
    db.commit()
    return result


@router.patch("/folders")
def rename_folder(body: FolderRename, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    result = catalog.rename_folder(db, user, body.name, body.new_name)
    db.commit()
    return result


@router.delete("/folders")
def delete_folder(name: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """The folder goes; its projects stay, outside any folder. (The name is a query parameter: any text works.)"""
    result = catalog.delete_folder(db, user, name)
    db.commit()
    return result


@router.get("/project-cards")
def project_cards(
    _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    """Every project as a card: collage, task, counts, the best model."""
    return overview.project_cards(db, hub.settings)


@router.get("/home")
def home(_: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)) -> dict[str, Any]:
    """Totals, the projects touched last and the latest runs, for the home screen."""
    return overview.home(db, hub.settings)


@router.get("/projects/{project_id}/overview")
def project_page(
    project_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """A project with its datasets as cards and its best models."""
    return overview.project_page(db, hub.settings, project_id)


@router.get("/projects/{project_id}/datasets")
def project_tree(
    project_id: str, trash: bool = False, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> list[dict[str, Any]]:
    return catalog.tree(db, project_id, include_deleted=trash)


# ---- import ------------------------------------------------------------------------


@router.get("/imports")
def imports(_: User = Depends(current_user), db: Session = Depends(db_session)) -> list[dict[str, Any]]:
    """Dataset imports under way and those that ended in the last half hour, with how far they got."""
    return catalog.imports(db)


@router.post("/imports/{job_id}/pause")
def pause_import(job_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """Stop an import and keep what it registered, to go on with later (only the files not in it yet are read)."""
    result = catalog.pause_import(db, user, job_id, via="ui")
    db.commit()
    return result


@router.post("/imports/{job_id}/resume")
def resume_import(job_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """Go on with a paused import, on the same server and into the same version."""
    job = catalog.resume_import(db, user, job_id, via="ui")
    db.commit()
    return {"job_id": job.id}


@router.post("/imports/{job_id}/discard")
def discard_import(
    job_id: str, user: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> dict[str, Any]:
    """Throw a paused import away with what only it registered."""
    result = catalog.discard_import(db, user, hub.settings, job_id, via="ui")
    db.commit()
    return result


class ImportRequest(BaseModel):
    node_id: str
    path: str
    dataset: str
    version: str
    format: str = "auto"
    label_space: str = ""
    note: str = ""
    base_release_id: str = ""
    task: str = ""  # Det · Seg · Kpt · Cls, a named task, or "" to read it from the folder
    # An image without a label file: "background", "unlabelled" (to label), or "auto" (unlabelled when no file has labels)
    unlabelled: str = "auto"
    classes: list[str] = []  # class names for a folder that names none (images to label)
    full_read: bool = False  # read every file, even those the project holds unchanged


@router.post("/projects/{project_id}/imports")
def start_import(
    project_id: str,
    body: ImportRequest,
    user: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    job = catalog.start_import(
        db,
        user,
        project_id=project_id,
        node_id=body.node_id,
        path=body.path,
        dataset=body.dataset,
        version=body.version,
        fmt=body.format,
        unlabelled=body.unlabelled,
        classes=body.classes,
        full_read=body.full_read,
        label_space=body.label_space,
        note=body.note,
        base_release_id=body.base_release_id,
        task=body.task,
        via="ui",
    )
    db.commit()
    return services.job_view(db, hub.settings, job)


@agent_router.post("/progress")
def import_progress(
    job_id: str, body: ImportProgress, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, bool]:
    catalog.import_progress(db, node, job_id, body)
    db.commit()
    return {"ok": True}


@agent_router.post("/begin")
def import_begin(
    job_id: str,
    body: ImportBegin,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    result = catalog.import_begin(db, node, job_id, body, hub.settings)
    db.commit()
    return result


@agent_router.get("/known")
def import_known(
    job_id: str, after: str = "", limit: int = 50_000, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """The files the version of a resumed import holds already (a page in path order): the node skips them."""
    return catalog.import_known(db, node, job_id, after, limit)


@agent_router.post("/same")
def import_same(
    job_id: str, body: ImportSame, node: Node = Depends(current_node), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Which of these files the project holds unchanged (path, size, modification time): the node does not read them."""
    return catalog.import_same(db, node, job_id, body.files)


@agent_router.post("/items")
def import_items(
    job_id: str,
    body: ImportChunk,
    node: Node = Depends(current_node),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    result = catalog.import_items(db, node, job_id, body, storage, hub.settings)
    db.commit()
    return result


@agent_router.post("/finish")
def import_finish(job_id: str, node: Node = Depends(current_node), db: Session = Depends(db_session)) -> dict[str, Any]:
    result = catalog.import_finish(db, node, job_id)
    db.commit()
    return result


# ---- releases ----------------------------------------------------------------------


@router.get("/releases/{release_id}")
def get_release(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    return catalog.release_view(db, release_id)


@router.get("/releases/{release_id}/items")
def list_items(
    release_id: str,
    split: str | None = None,
    state: str = "",
    class_id: list[str] = Query(default=[]),  # one or several: class_id=3,5 or class_id=3&class_id=5
    site: str = "",
    camera: str = "",
    q: str = "",
    edited: bool = False,
    suspect: str = "",
    label_set: str = "",
    part: int | None = None,
    check: str = "",
    mistake: str = "",
    changed: str = "",
    let_be: bool = False,
    offset: int = 0,
    limit: int = 60,
    pred: str = "",
    classes: str = "",  # 검수: class ids, comma separated (the findings of these classes)
    conf: float | None = None,  # 검수: a model check's thresholds (FiftyOne's when left out)
    min_area: float | None = None,
    reclass_conf: float | None = None,
    missed_iou: float | None = None,
    rules: str = "",  # 검수: the box checks' thresholds, e.g. duplicate:0.9,tiny:5 (the defaults when left out)
    picks: str = "",  # 검수: several checks at once, e.g. rule:duplicate,model:extra (with suspect for a model's)
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    picked = [int(c) for c in classes.split(",") if c.strip().lstrip("-").isdigit()]
    return catalog.list_items(
        db,
        hub.settings,
        release_id,
        split=split,
        state=state,
        class_id=catalog.class_ids(class_id),
        site=site,
        camera=camera,
        q=q,
        edited=edited,
        suspect=suspect,
        label_set=label_set,
        part=part,
        check=check,
        mistake=mistake,
        changed=changed,
        let_be=let_be,
        offset=offset,
        limit=limit,
        classes=picked,
        counting=checks.rule(conf, min_area, reclass_conf, missed_iou),
        at=qa.limits(rules),
        picks=[p for p in picks.split(",") if p.strip()],
        pred=pred,
    )


@router.get("/releases/{release_id}/items/{item_id}")
def get_item(
    release_id: str,
    item_id: int,
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
    hub: Hub = Depends(get_hub),
) -> dict[str, Any]:
    return catalog.item_detail(db, hub.settings, release_id, item_id)


@router.get("/releases/{release_id}/compare")
def compare(
    release_id: str, other: str, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    return catalog.compare(db, release_id, other)


class NewRelease(BaseModel):
    sources: list[str]
    version: str = ""
    dataset: str = ""  # empty: the first source's dataset (a new version of it)
    note: str = ""
    filter: dict[str, Any] | None = None  # the version screen's filters: split, state, class_id (1 or more), site, camera, q
    exclude: bool = False  # all items but those matching the filter
    split: dict[str, float] | None = None  # deal out again, e.g. {"train": 80, "val": 10, "test": 10}
    group_by: str = ""  # "", "site" or "camera": one group, one split
    classes: dict[str, str] | None = None  # {old class id: new name}: same name merges, "" drops
    dry_run: bool = False


@router.post("/releases")
def new_release(body: NewRelease, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """A release made from the items of existing ones, or with ``dry_run`` how many items it would hold per split."""
    result = catalog.new_release(
        db,
        user,
        sources=body.sources,
        version=body.version,
        dataset=body.dataset,
        note=body.note,
        match=body.filter,
        exclude=body.exclude,
        split=body.split,
        group_by=body.group_by,
        classes=body.classes,
        dry_run=body.dry_run,
        via="ui",
    )
    if not body.dry_run:
        db.commit()
    return result


@router.get("/releases/{release_id}/charts")
def release_charts(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """Split and label states, classes, image sizes, where the boxes sit and how big they are."""
    return overview.release_charts(db, release_id)


@router.get("/releases/{release_id}/models")
def release_models(
    release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list[dict[str, Any]]:
    """The runs that trained on this release and the models they made."""
    return overview.release_models(db, hub.settings, release_id)


class ClassColor(BaseModel):
    color: str = ""  # #rrggbb; empty: back to the automatic colour


@router.put("/releases/{release_id}/classes/{class_id}")
def class_color(
    release_id: str, class_id: int, body: ClassColor, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = overview.set_class_color(db, release_id, class_id, body.color)
    db.commit()
    return result


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    """Every version of the dataset to the trash."""
    versions = catalog.delete_dataset(db, user, dataset_id)
    db.commit()
    return {"ok": True, "versions": versions}


# ---- likely wrong labels: what evaluated models predicted ---------------------------------


class LetBe(BaseModel):
    check: str  # a rule check's id, or extra / missed (what a model's evaluation found)
    note: str = ""


@router.put("/releases/{release_id}/items/{item_id}/let-be")
def let_be(
    release_id: str, item_id: int, body: LetBe, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """What the check flagged on this image is right ("문제 없음"): kept for these labels, into later versions."""
    result = qa.let_be(db, user, catalog.get_release(db, release_id), item_id, body.check, body.note)
    db.commit()
    return result


@router.delete("/releases/{release_id}/items/{item_id}/let-be")
def take_back(
    release_id: str, item_id: int, check: str, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Count it again."""
    result = qa.let_be(db, user, catalog.get_release(db, release_id), item_id, check, undo=True)
    db.commit()
    return result


@router.get("/releases/{release_id}/review")
def release_review(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict[str, Any]:
    """A version's 검수 record (kept when it was confirmed); a work in progress's, as it would be kept now."""
    result = review.view(db, catalog.get_release(db, release_id))
    db.commit()  # checks that ran are kept until the labels change
    return result


@router.get("/releases/{release_id}/changes")
def draft_changes(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """What the dataset's work in progress changed against its base: labels added, removed, moved, reclassed."""
    draft = drafts.fixing(db, catalog.get_release(db, release_id))
    if draft is None:
        raise services.ServiceError(404, "고친 것이 없습니다(작업 중이 없음).")
    return drafts.changes(db, draft)


@router.get("/releases/{release_id}/qa")
def label_checks(
    release_id: str,
    conf: float | None = None,
    min_area: float | None = None,
    reclass_conf: float | None = None,
    missed_iou: float | None = None,
    rules: str = "",
    _: User = Depends(current_user),
    db: Session = Depends(db_session),
) -> dict:
    """검수: how many images each label check flags (``items?check=<id>``; the box checks at the thresholds in
    ``rules``, e.g. ``duplicate:0.9,tiny:5``), and the model's view per evaluation, counted FiftyOne's way at these
    thresholds (FiftyOne's own when left out); each with the counts at the levels around its threshold."""
    counting = checks.rule(conf, min_area, reclass_conf, missed_iou)
    result = qa.summary(db, catalog.get_release(db, release_id), counting, qa.limits(rules))
    db.commit()
    return result


@router.get("/releases/{release_id}/qa/progress")
def label_check_progress(release_id: str, _: User = Depends(current_user)) -> dict:
    """How far the label checks of a version got while they run (검수 shows it while it waits for them)."""
    return qa.progress(release_id)


@router.get("/releases/{release_id}/qa/history")
def label_check_history(
    release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session), hub: Hub = Depends(get_hub)
) -> list:
    """검수's record of the model checks on this dataset's versions: each with how it went, where, the images it
    had sent first and why it failed, kept after it ends (the log: ``/jobs/{job_id}/logs``)."""
    return checks.history(db, hub.settings, catalog.get_release(db, release_id))


@router.get("/releases/{release_id}/checks")
def release_checks(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> list:
    """Evaluations that can sort this release's images by likely wrong labels (``items?suspect=<id>``)."""
    return checks.release_checks(db, catalog.get_release(db, release_id))


@router.get("/releases/{release_id}/prediction-runs")
def release_prediction_runs(release_id: str, _: User = Depends(current_user), db: Session = Depends(db_session)) -> list:
    """Model runs (evaluations, 오토라벨) whose predictions can be laid over this version's images."""
    return checks.release_prediction_runs(db, catalog.get_release(db, release_id))


@router.get("/releases/{release_id}/items/{item_id}/prediction-runs")
def item_prediction_runs(
    release_id: str, item_id: int, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> list:
    """Every model run that predicted this image, which model and what it was, with its predictions."""
    return checks.item_prediction_runs(db, catalog.get_release(db, release_id), item_id)


@router.get("/releases/{release_id}/items/{item_id}/predictions")
def item_predictions(
    release_id: str, item_id: int, op: str, _: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """What the evaluation's model predicted for this image, in the release's label form."""
    return checks.item_predictions(db, catalog.get_release(db, release_id), item_id, op)


# ---- fixing labels: the dataset's work in progress -----------------------------------


class Labels(BaseModel):
    boxes: list[list[float]]  # [[class, cx, cy, w, h, …], …]; [[class]] for classification; [] for background
    parent_revision_id: int | None = None  # the labels the fix started from: refused if someone saved in between
    force: bool = False  # save over a fix someone else made in between
    label_set: str = ""  # labelled as part of this labeling set: it counts as done there


@router.put("/releases/{release_id}/items/{item_id}/labels")
def save_labels(
    release_id: str, item_id: int, body: Labels, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """Fixed labels of one image, into the dataset's work in progress (made on the first fix)."""
    result = drafts.save_labels(
        db,
        user,
        release_id,
        item_id,
        body.boxes,
        parent_id=body.parent_revision_id,
        force=body.force,
        label_set=body.label_set,
    )
    db.commit()
    return result


@router.post("/releases/{release_id}/items/{item_id}/revert")
def revert_labels(
    release_id: str, item_id: int, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    result = drafts.revert_item(db, user, release_id, item_id)
    db.commit()
    return result


class Confirm(BaseModel):
    version: str
    note: str = ""


@router.post("/releases/{release_id}/confirm")
def confirm_draft(
    release_id: str, body: Confirm, user: User = Depends(current_user), db: Session = Depends(db_session)
) -> dict[str, Any]:
    """The work in progress becomes a version."""
    result = drafts.confirm(db, user, release_id, body.version, body.note)
    db.commit()
    return result


@router.post("/releases/{release_id}/rebase")
def rebase_draft(release_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """Move the work in progress onto the dataset's newest version."""
    result = drafts.rebase(db, user, release_id)
    db.commit()
    return result


@router.post("/releases/{release_id}/discard")
def discard_draft(release_id: str, user: User = Depends(current_user), db: Session = Depends(db_session)) -> dict:
    """Throw the work in progress away (the fixed labels stay in each image's history)."""
    result = drafts.discard(db, user, release_id)
    db.commit()
    return result


@router.delete("/releases/{release_id}")
def delete_release(release_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    catalog.set_release_deleted(db, user, release_id, True)
    db.commit()
    return {"ok": True}


@router.post("/releases/{release_id}/restore")
def restore_release(release_id: str, user: User = Depends(admin_user), db: Session = Depends(db_session)) -> dict:
    catalog.set_release_deleted(db, user, release_id, False)
    db.commit()
    return {"ok": True}


# ---- media (signed, no bearer header) ------------------------------------------------

_CACHE = {"Cache-Control": "private, max-age=3600"}


@media_router.get("/thumb/{key}")
def thumb(key: str, exp: int, sig: str, hub: Hub = Depends(get_hub)) -> Response:
    if not catalog.verify_media(hub.settings, "thumb", key, exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    storage = catalog.storage_for(str(hub.settings.storage_dir))
    if not storage.has_blob(key):
        raise HTTPException(status_code=404, detail="not found")
    data = storage.get_blob(key)
    # WebP since 0.60.0, JPEG before (and until the Hub has shrunk it)
    kind = "image/webp" if data[:4] == b"RIFF" and data[8:12] == b"WEBP" else "image/jpeg"
    return Response(data, media_type=kind, headers=_CACHE)


@media_router.get("/item/{item_id}")
def item_image(
    item_id: int, exp: int, sig: str, hub: Hub = Depends(get_hub), db: Session = Depends(db_session)
) -> FileResponse:
    if not catalog.verify_media(hub.settings, "item", str(item_id), exp, sig):
        raise HTTPException(status_code=403, detail="링크가 만료되었거나 올바르지 않습니다.")
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="not found")
    path = catalog.item_file(db, item)
    if path is None:  # on another server: that node sends it (kept in the cache afterwards)
        try:
            path = originals.fetch(db, hub.settings, item)
        except services.ServiceError as e:
            raise HTTPException(status_code=e.status, detail=e.message) from e
    return FileResponse(path, headers=_CACHE)
