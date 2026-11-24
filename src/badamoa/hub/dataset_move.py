"""Moving a dataset to another project, with every version of it, its work in progress and its labeling sets.

What a project holds per project, and how it follows the dataset:

- Images (``items``) are the project's, one per file content (sha256); datasets of one project share them. An image
  only this dataset holds moves as it is (same id, its label history with it). One another dataset of the project
  holds too, or whose content the other project has already, is copied: the other project's own image (or a new one
  with the same file, path and thumbnail) gets the label revisions this dataset pins, and the dataset's versions,
  work in progress, labeling sets, 검수 results and per-image predictions point at it. The revisions only the moved
  dataset pinned then leave the old image's history, and an image no version of the old project holds any more
  goes, as when a version is removed from the trash. Paths, sizes and modification times come along, so a later
  import into the new project knows the files unchanged (``catalog.import_same``).
- Label spaces are the project's too. The other project's own space of the same name and classes is taken when it has
  one; else a space nothing in the old project uses any more moves as it is (renamed when the name is taken there),
  and one still used there is copied. The moved labels' own fingerprints follow a space they change to; a version
  keeps its fingerprint, its content being what it was.
- Labeling sets, imports' batches and saved pipelines (with their past runs) go with the dataset: a pipeline acts on
  its dataset and makes models in its dataset's project.
- Training runs, searches, models and evaluations stay: they are the old project's record (a model family may hold
  models of several datasets) and keep pointing at the moved versions, which still open by the same ids.

Nothing moves while something works on the dataset: an import (or one paused, to go on from), a training run, a
search, an evaluation or 오토라벨, an export, images on their way to a server, a pipeline run going or waiting, or
검수's checks of one of its versions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import Table, bindparam, delete, func, insert, select, update
from sqlalchemy.orm import Session

from . import catalog, hub_storage, qa, services
from .catalog_models import (
    Batch,
    Dataset,
    Item,
    ItemPrediction,
    LabelRevision,
    LabelSet,
    LabelSetItem,
    LabelSpace,
    QaDismissal,
    QaResult,
    QaRun,
    QaScore,
    Release,
    ReleaseItem,
    RevisionClass,
)
from .db import write_lock
from .models import Job, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Pipeline, PipelineRun, Run, Sweep

CHUNK = 800  # values bound at once; SQLite takes up to 999
JOB_WHAT = {"train.run": "학습", "data.export": "내려받기", "data.sync": "서버로 이미지 보내기"}
OP_WHAT = {"eval": "평가", "autolabel": "오토라벨", "export": "모델 내보내기", "predict": "모델 테스트"}


def _chunks(values: Iterable[Any], size: int = CHUNK) -> list[list[Any]]:
    values = list(values)
    return [values[i : i + size] for i in range(0, len(values), size)]


def _dataset(db: Session, dataset_id: str) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise ServiceError(404, "데이터셋을 찾을 수 없습니다.")
    return dataset


def _releases(dataset: Dataset):
    """Every version of the dataset, its work in progress and those in the trash (a subquery)."""
    return select(Release.id).where(Release.dataset_id == dataset.id)


# ---- what holds it ---------------------------------------------------------------------------


def blockers(db: Session, dataset: Dataset) -> list[str]:
    """What works on the dataset now, e.g. ["학습 1개", "가져오기 1개"]: it does not move until they end."""
    release_ids = set(db.scalars(_releases(dataset)))
    counts: dict[str, int] = {}

    def add(what: str, n: int = 1) -> None:
        if n:
            counts[what] = counts.get(what, 0) + n

    building = db.scalar(
        select(func.count(Release.id)).where(
            Release.dataset_id == dataset.id, Release.status == "building", Release.deleted_at.is_(None)
        )
    )
    add("가져오기(일시정지 포함)", building or 0)
    for job in db.scalars(select(Job).where(Job.status.in_(catalog.ACTIVE))):
        params = job.params or {}
        if job.type == "data.import":
            # before its version exists, an import names its project and dataset
            if params.get("project_id") == dataset.project_id and params.get("dataset") == dataset.name and not building:
                add("가져오기(일시정지 포함)")
        elif params.get("release_id") in release_ids:
            add(JOB_WHAT.get(job.type, "작업"))
    ops = db.execute(
        select(ModelJob.kind)
        .join(Job, Job.id == ModelJob.job_id)
        .where(ModelJob.release_id.in_(_releases(dataset)), Job.status.in_(catalog.ACTIVE))
    ).scalars()
    for kind in ops:
        add(OP_WHAT.get(kind, "모델 작업"))
    add(
        "탐색",
        db.scalar(select(func.count(Sweep.id)).where(Sweep.release_id.in_(_releases(dataset)), Sweep.status == "running"))
        or 0,
    )
    pipelines = select(Pipeline.id).where(Pipeline.dataset_id == dataset.id)
    going = db.scalar(
        select(func.count(PipelineRun.id)).where(
            PipelineRun.status.in_(("running", "held")),
            (PipelineRun.release_id.in_(_releases(dataset))) | (PipelineRun.pipeline_id.in_(pipelines)),
        )
    )
    add("파이프라인 실행", going or 0)
    add("검수 검사", sum(1 for rid in release_ids if qa.progress(rid).get("running")))
    return [f"{what} {n}개" for what, n in counts.items()]


def _held_elsewhere(db: Session, dataset: Dataset, item_ids: list[int]) -> set[int]:
    """Of these images, those another dataset of the project holds too (in a version or a judgement kept)."""
    found: set[int] = set()
    for chunk in _chunks(item_ids):
        found.update(
            db.scalars(
                select(ReleaseItem.item_id)
                .where(ReleaseItem.item_id.in_(chunk), ReleaseItem.release_id.not_in(_releases(dataset)))
                .distinct()
            )
        )
        found.update(
            db.scalars(
                select(QaDismissal.item_id)
                .where(QaDismissal.item_id.in_(chunk), QaDismissal.dataset_id != dataset.id)
                .distinct()
            )
        )
    return found


def _known_there(db: Session, project_id: str, shas: dict[int, str]) -> dict[int, int]:
    """Of these images (id → sha256), those the other project has already: id → that project's image id."""
    by_sha = {sha: item_id for item_id, sha in shas.items()}
    found: dict[int, int] = {}
    for chunk in _chunks(by_sha):
        for there, sha in db.execute(
            select(Item.id, Item.sha256).where(Item.project_id == project_id, Item.sha256.in_(chunk))
        ):
            found[by_sha[sha]] = there
    return found


def _items(db: Session, dataset: Dataset) -> dict[int, str]:
    """The dataset's images (id → sha256): those its versions and work in progress hold."""
    held = select(ReleaseItem.item_id).where(ReleaseItem.release_id.in_(_releases(dataset))).distinct().subquery()
    return dict(db.execute(select(Item.id, Item.sha256).join(held, held.c.item_id == Item.id)).all())


def plan(db: Session, dataset_id: str, project_id: str = "") -> dict[str, Any]:
    """What moving the dataset takes along and leaves, what holds it now, and (with ``project_id``) whether that
    project has a dataset of its name already and how many of its images it holds."""
    dataset = _dataset(db, dataset_id)
    source = catalog.get_project(db, dataset.project_id)
    releases = db.scalars(select(Release).where(Release.dataset_id == dataset.id)).all()
    items = _items(db, dataset)
    shared = _held_elsewhere(db, dataset, list(items))
    count = lambda query: db.scalar(query) or 0  # noqa: E731
    in_releases = _releases(dataset)
    out: dict[str, Any] = {
        "dataset": {"id": dataset.id, "name": dataset.name, "task": catalog.dataset_task(dataset)},
        "project": {"id": source.id, "name": source.name},
        "versions": sum(1 for r in releases if r.status != "draft" and r.deleted_at is None),
        "trashed": sum(1 for r in releases if r.deleted_at is not None),
        "draft": any(r.status == "draft" for r in releases),
        "items": len(items),
        "shared": len(shared),  # also in another dataset of this project: copied, the others keep theirs
        "label_sets": count(select(func.count(LabelSet.id)).where(LabelSet.dataset_id == dataset.id)),
        "pipelines": count(
            select(func.count(Pipeline.id)).where(Pipeline.dataset_id == dataset.id, Pipeline.deleted_at.is_(None))
        ),
        # what stays in this project, pointing at the moved versions
        "stays": {
            "runs": count(select(func.count(Run.id)).where(Run.release_id.in_(in_releases), Run.deleted_at.is_(None))),
            "models": count(
                select(func.count(ModelVersion.id)).where(
                    ModelVersion.release_id.in_(in_releases), ModelVersion.deleted_at.is_(None)
                )
            ),
            "evaluations": count(
                select(func.count(ModelJob.id)).where(ModelJob.release_id.in_(in_releases), ModelJob.kind == "eval")
            ),
            "sweeps": count(select(func.count(Sweep.id)).where(Sweep.release_id.in_(in_releases))),
        },
        "busy": blockers(db, dataset),
    }
    if project_id:
        target = catalog.get_project(db, project_id)
        out["target"] = {
            "id": target.id,
            "name": target.name,
            "name_taken": _name_taken(db, target.id, dataset.name),
            "known": len(_known_there(db, target.id, items)),  # images the other project has already
        }
    return out


def _name_taken(db: Session, project_id: str, name: str) -> bool:
    return db.scalar(select(Dataset.id).where(Dataset.project_id == project_id, Dataset.name == name)) is not None


# ---- moving -----------------------------------------------------------------------------------


def _remap(db: Session, table: Table, where: Any, items: dict[int, int], revisions: dict[int, int] | None = None) -> int:
    """Point the rows of ``table`` matching ``where`` at the copied images (and revisions): taken out and put back,
    since the image is part of their key. Returns how many rows moved."""
    moved = 0
    for chunk in _chunks(items):
        condition = [where, table.c.item_id.in_(chunk)]
        rows = [dict(r) for r in db.execute(select(table).where(*condition)).mappings()]
        if not rows:
            continue
        for row in rows:
            row["item_id"] = items[row["item_id"]]
            if revisions is not None:
                row["revision_id"] = revisions.get(row["revision_id"], row["revision_id"])
        db.execute(delete(table).where(*condition))
        db.execute(insert(table), rows)
        moved += len(rows)
    return moved


def _wanted_revisions(db: Session, dataset: Dataset, copied: list[int]) -> dict[int, list[LabelRevision]]:
    """The label revisions of the copied images the dataset needs (pinned by a version or the work in progress,
    judged in 검수) and the ones they were fixed from, oldest first per image."""
    wanted: set[int] = set()
    for chunk in _chunks(copied):
        wanted.update(
            db.scalars(
                select(ReleaseItem.revision_id).where(
                    ReleaseItem.release_id.in_(_releases(dataset)), ReleaseItem.item_id.in_(chunk)
                )
            )
        )
        wanted.update(
            db.scalars(
                select(QaDismissal.revision_id).where(QaDismissal.dataset_id == dataset.id, QaDismissal.item_id.in_(chunk))
            )
        )
    found: dict[int, LabelRevision] = {}
    images = set(copied)
    while wanted:
        for chunk in _chunks(wanted):
            for rev in db.scalars(select(LabelRevision).where(LabelRevision.id.in_(chunk))):
                if rev.item_id in images:  # what another image was fixed from is no history of this one
                    found[rev.id] = rev
        wanted = {r.parent_id for r in found.values() if r.parent_id and r.parent_id not in found} - wanted
    by_item: dict[int, list[LabelRevision]] = {}
    for rev in sorted(found.values(), key=lambda r: (r.item_id, r.rev)):
        by_item.setdefault(rev.item_id, []).append(rev)
    return by_item


def _copy_images(
    db: Session, dataset: Dataset, target_id: str, copied: list[int], known: dict[int, int]
) -> tuple[dict[int, int], dict[int, int]]:
    """Give each copied image a place in the other project (the image it has, or a new one) with the revisions the
    dataset needs. Returns (image id → its id there, revision id → the copy's id)."""
    new_items = {
        item.id: Item(
            project_id=target_id,
            sha256=item.sha256,
            file_name=item.file_name,
            width=item.width,
            height=item.height,
            bytes=item.bytes,
            site=item.site,
            camera=item.camera,
            location_id=item.location_id,
            rel_path=item.rel_path,
            mtime_ns=item.mtime_ns,
            thumb_key=item.thumb_key,
            thumb_bytes=item.thumb_bytes,
            thumb_v=item.thumb_v,
            batch_id=item.batch_id,
            created_at=item.created_at,
        )
        for chunk in _chunks(i for i in copied if i not in known)
        for item in db.scalars(select(Item).where(Item.id.in_(chunk)))
    }
    db.add_all(new_items.values())
    db.flush()
    item_map = {**known, **{old: item.id for old, item in new_items.items()}}

    by_item = _wanted_revisions(db, dataset, copied)
    last_rev: dict[int, int] = {}
    for chunk in _chunks(known.values()):
        rows = db.execute(
            select(LabelRevision.item_id, func.max(LabelRevision.rev))
            .where(LabelRevision.item_id.in_(chunk))
            .group_by(LabelRevision.item_id)
        )
        last_rev.update({item_id: rev for item_id, rev in rows})
    copies: dict[int, LabelRevision] = {}
    for old_item, revisions in by_item.items():
        there = item_map[old_item]
        after = last_rev.get(there)  # an image the other project had: the copies come after its own revisions
        for n, rev in enumerate(revisions, start=1):
            copies[rev.id] = LabelRevision(
                item_id=there,
                rev=(after or 0) + n,  # numbered on from the image's own revisions there, if it has any
                label_space_id=rev.label_space_id,  # the project's own label space is settled afterwards
                source=rev.source,
                state=rev.state,
                box_count=rev.box_count,
                boxes=rev.boxes,
                labelled_class_ids=rev.labelled_class_ids,
                digest=rev.digest,
                batch_id=rev.batch_id,
                created_by=rev.created_by,
                created_at=rev.created_at,
            )
    db.add_all(copies.values())
    db.flush()
    rev_map = {old: copy.id for old, copy in copies.items()}
    for chunk in _chunks(rev_map):
        old_revs = db.scalars(select(LabelRevision).where(LabelRevision.id.in_(chunk))).all()
        for old in old_revs:
            if old.parent_id in rev_map:
                copies[old.id].parent_id = rev_map[old.parent_id]
        counts = db.execute(select(RevisionClass).where(RevisionClass.revision_id.in_(chunk))).scalars().all()
        if counts:
            db.execute(
                insert(RevisionClass),
                [{"revision_id": rev_map[c.revision_id], "class_id": c.class_id, "count": c.count} for c in counts],
            )
    db.flush()
    return item_map, rev_map


def _leave_behind(db: Session, copied: list[int], rev_map: dict[int, int]) -> int:
    """In the old project: the copied revisions nothing there points at any more leave the images' history, and the
    images no version there holds go. Returns how many images went."""
    candidates = set(rev_map)
    keep: set[int] = set()
    for chunk in _chunks(candidates):
        keep.update(db.scalars(select(ReleaseItem.revision_id).where(ReleaseItem.revision_id.in_(chunk)).distinct()))
        keep.update(db.scalars(select(QaDismissal.revision_id).where(QaDismissal.revision_id.in_(chunk)).distinct()))
    parents: dict[int, int | None] = {}
    for chunk in _chunks(copied):
        parents.update(
            db.execute(select(LabelRevision.id, LabelRevision.parent_id).where(LabelRevision.item_id.in_(chunk))).all()
        )
    while True:  # a revision that stays keeps the one it was fixed from
        staying = (set(parents) - candidates) | keep
        more = {parents[r] for r in staying if parents.get(r) in candidates} - keep
        if not more:
            break
        keep |= more
    for chunk in _chunks(candidates - keep):
        db.execute(delete(LabelRevision).where(LabelRevision.id.in_(chunk)))  # with their class counts (cascade)
    gone: list[int] = []
    for chunk in _chunks(copied):
        held = set(db.scalars(select(ReleaseItem.item_id).where(ReleaseItem.item_id.in_(chunk)).distinct()))
        gone.extend(i for i in chunk if i not in held)
    for chunk in _chunks(gone):
        db.execute(delete(Item).where(Item.id.in_(chunk)))  # with what hangs off them (cascade)
    return len(gone)


def _space_name(db: Session, project_id: str, name: str) -> str:
    taken = set(db.scalars(select(LabelSpace.name).where(LabelSpace.project_id == project_id)))
    found, n = name, 2
    while found in taken:
        found, n = f"{name}-{n}", n + 1
    return found


def _settle_spaces(db: Session, dataset: Dataset, source_id: str, target_id: str) -> dict[str, str]:
    """The label spaces of the moved versions and labels, made the other project's: the one it has already with the
    same name and classes, else the space itself when the old project no longer uses it, else a copy. Returns old id
    → the other project's id for those not moved as they were (their labels' fingerprints follow)."""
    source_spaces = select(LabelSpace.id).where(LabelSpace.project_id == source_id)
    twins = list(db.scalars(select(LabelSpace).where(LabelSpace.project_id == target_id)))
    used = set(db.scalars(select(Release.label_space_id).where(Release.dataset_id == dataset.id).distinct()))
    used.update(
        db.scalars(
            select(LabelRevision.label_space_id)
            .join(Item, Item.id == LabelRevision.item_id)
            .where(Item.project_id == target_id, LabelRevision.label_space_id.in_(source_spaces))
            .distinct()
        )
    )
    copied: dict[str, str] = {}
    unused: list[LabelSpace] = []  # in the old project with nothing there pointing at them once the labels moved
    for space in db.scalars(select(LabelSpace).where(LabelSpace.id.in_(used), LabelSpace.project_id == source_id)).all():
        twin = next((s for s in twins if s.name == space.name and s.classes == space.classes), None)
        by_release = db.scalar(
            select(Release.id)
            .join(Dataset, Dataset.id == Release.dataset_id)
            .where(Dataset.project_id == source_id, Dataset.id != dataset.id, Release.label_space_id == space.id)
            .limit(1)
        )
        by_label = db.scalar(
            select(LabelRevision.id)
            .join(Item, Item.id == LabelRevision.item_id)
            .where(Item.project_id == source_id, LabelRevision.label_space_id == space.id)
            .limit(1)
        )
        left = by_release is None and by_label is None
        if twin is None and left:
            space.name = _space_name(db, target_id, space.name)
            space.project_id = target_id
            continue
        if twin is None:
            twin = LabelSpace(project_id=target_id, name=_space_name(db, target_id, space.name), classes=space.classes)
            db.add(twin)
            db.flush()
            twins.append(twin)
        elif left:
            unused.append(space)
        copied[space.id] = twin.id
    db.flush()
    for old, new in copied.items():
        db.execute(
            update(Release).where(Release.dataset_id == dataset.id, Release.label_space_id == old).values(label_space_id=new)
        )
        moved = db.execute(
            select(LabelRevision.id, LabelRevision.state, LabelRevision.boxes)
            .join(Item, Item.id == LabelRevision.item_id)
            .where(Item.project_id == target_id, LabelRevision.label_space_id == old)
        ).all()
        for chunk in _chunks(moved):
            db.connection().execute(
                update(LabelRevision.__table__)
                .where(LabelRevision.__table__.c.id == bindparam("rid"))
                .values(label_space_id=new, digest=bindparam("fingerprint")),
                [{"rid": rid, "fingerprint": catalog._digest(new, state, boxes or [])} for rid, state, boxes in chunk],
            )
    for space in unused:
        db.delete(space)
    db.flush()
    return copied


def move(db: Session, user: User, dataset_id: str, project_id: str, name: str = "", via: str = "api") -> dict[str, Any]:
    """Move a dataset to another project (see the module's doc). ``name``: what it is called there, when that
    project has a dataset of its name already. The maker of both projects, or an admin."""
    dataset = _dataset(db, dataset_id)
    source = catalog.get_project(db, dataset.project_id)
    target = catalog.get_project(db, project_id)
    catalog._may_edit(source, user)
    catalog._may_edit(target, user)
    if target.id == source.id:
        raise ServiceError(400, "이미 그 프로젝트에 있는 데이터셋입니다.")
    name = (name or dataset.name).strip()
    if not 1 <= len(name) <= 128:
        raise ServiceError(400, "데이터셋 이름은 1~128자여야 합니다.")
    write_lock(db)  # nothing starts on it or lands in either project meanwhile
    busy = blockers(db, dataset)
    if busy:
        raise ServiceError(409, f"이 데이터셋에서 {', '.join(busy)}가 진행 중입니다. 끝나거나 멈춘 뒤에 옮기세요.")
    if _name_taken(db, target.id, name):
        raise ServiceError(
            409, f"{target.name} 프로젝트에 {name} 데이터셋이 이미 있습니다(휴지통에 있는 것 포함). 다른 이름으로 옮기세요."
        )

    items = _items(db, dataset)
    shared = _held_elsewhere(db, dataset, list(items))
    known = _known_there(db, target.id, items)
    copied = sorted(set(shared) | set(known))
    item_map, rev_map = _copy_images(db, dataset, target.id, copied, known) if copied else ({}, {})
    if item_map:
        moved_rel = ReleaseItem.__table__
        _remap(db, moved_rel, moved_rel.c.release_id.in_(_releases(dataset)), item_map, rev_map)
        sets = select(LabelSet.id).where(LabelSet.dataset_id == dataset.id)
        _remap(db, LabelSetItem.__table__, LabelSetItem.__table__.c.set_id.in_(sets), item_map)
        for table in (QaResult.__table__, QaScore.__table__):
            _remap(db, table, table.c.release_id.in_(_releases(dataset)), item_map)
        _remap(db, QaDismissal.__table__, QaDismissal.__table__.c.dataset_id == dataset.id, item_map, rev_map)
        ops = select(ModelJob.id).where(ModelJob.release_id.in_(_releases(dataset)))
        _remap(db, ItemPrediction.__table__, ItemPrediction.__table__.c.op_id.in_(ops), item_map)
    dropped = _leave_behind(db, copied, rev_map) if copied else 0
    alone = [i for i in items if i not in item_map]
    for chunk in _chunks(alone):
        db.execute(update(Item).where(Item.id.in_(chunk)).values(project_id=target.id))
    db.flush()
    spaces = _settle_spaces(db, dataset, source.id, target.id)

    old_name = dataset.name
    dataset.project_id, dataset.name = target.id, name
    batches = select(Release.batch_id).where(Release.dataset_id == dataset.id, Release.batch_id.is_not(None))
    db.execute(update(Batch).where(Batch.id.in_(batches)).values(project_id=target.id))
    db.execute(update(LabelSet).where(LabelSet.dataset_id == dataset.id).values(project_id=target.id))
    pipelines = db.scalars(select(Pipeline).where(Pipeline.dataset_id == dataset.id)).all()
    if pipelines:
        from . import pipelines as pipeline_names  # noqa: PLC0415  (it imports this module's neighbours)

        for pipeline in pipelines:
            if pipeline.deleted_at is None:
                pipeline.name = pipeline_names._unique_name(db, target.id, pipeline.name, pipeline.id)
            pipeline.project_id = target.id
        db.flush()
        db.execute(
            update(PipelineRun).where(PipelineRun.pipeline_id.in_([p.id for p in pipelines])).values(project_id=target.id)
        )
    db.flush()
    releases = db.scalars(select(Release).where(Release.dataset_id == dataset.id)).all()
    for release in releases:
        # A version keeps its fingerprint (``Release.digest``): its images, splits and labels are what they were.
        done = db.get(QaRun, release.id)
        if done is not None and (item_map or spaces):
            done.fingerprint = qa._fingerprint(db, release)  # the same checks, now of the copied images
    hub_storage.mark_changed()
    services.audit(
        db,
        user.id,
        via,
        "dataset.move",
        dataset.id,
        name=name,
        old=old_name if old_name != name else None,
        source=source.id,
        source_name=source.name,
        project=target.id,
        project_name=target.name,
        versions=len(releases),
        items=len(items),
        copied=len(copied),
        dropped=dropped,
        label_spaces_copied=len(spaces),
        pipelines=len(pipelines),
    )
    latest = max(
        (r for r in releases if r.deleted_at is None), key=lambda r: (r.status != "draft", r.created_at), default=None
    )
    return {
        "id": dataset.id,
        "name": dataset.name,
        "project": {"id": target.id, "name": target.name},
        "from": {"id": source.id, "name": source.name},
        "release_id": latest.id if latest else None,
        "versions": len(releases),
        "items": len(items),
        "copied": len(copied),
    }
