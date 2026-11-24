"""Training logic: starting runs, the node-side protocol, metric history, the model registry."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
from badamoa.core.protocol import TERMINAL_STATUSES, TrainFinish, TrainReport
from badamoa.core.storage import Storage
from badamoa.core.templates import RESERVED, TemplateError, resolve_params

from . import catalog, node_data, registry, resources, services, template_checks, transfers
from .catalog_models import (
    Dataset,
    Item,
    LabelRevision,
    LabelSpace,
    Release,
    ReleaseItem,
    RevisionClass,
    StorageLocation,
)
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelVersion, Run, RunFile, RunMetric, Tag, Template

# Offered first in the screen; any other name works too (MLflow aliases are free names).
SUGGESTED_ALIASES = ("SOTA", "challenger", "production", "staging")
ALIAS_RE = re.compile(r"^[0-9A-Za-z가-힣][0-9A-Za-z가-힣._-]{0,31}$")
MAX_ALIASES = 10
MAX_TAGS = 20
TAG_LENGTH = 48
MAX_NOTE = 20000
STAGES = ("data", "env", "train", "upload")
MANIFEST_MAX = 5000
# The number a run is judged by: decides which step counts as "best so far".
PRIMARY_METRIC = {"detect": "map50_95", "segment": "mask_map50_95", "classify": "top1", "pose": "pose_map50_95"}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
MAX_FILES_PER_RUN = 5000  # per-step images add up: 16 pictures × 100 epochs
HISTORY_POINTS = 600  # more steps than this are thinned out when read
_NAME_RE = re.compile(r"[^0-9A-Za-z가-힣._-]+")


def _slug(text: str) -> str:
    return _NAME_RE.sub("-", text.strip()).strip("-")


def _unique(base: str, taken: set[str]) -> str:
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


# ---- run names: one per Hub ---------------------------------------------------------------
# A run's name says which run it is across the whole Hub (the Hub's lists, links, a model's "trained by"), not only
# in its project: two projects' runs never share one.

_DATED = re.compile(r"-\d{8}(-\d+)?$")  # a name's date (and its number that day): coco8-v1-yolov8s-20261123-2


def _today() -> str:
    """The day a run is named after, by the Hub's own clock (where its people are): 20261123."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d")


def run_name(db: Session, given: str, auto: str, again: bool = False) -> str:
    """A new run's name, kept apart from every run's on the Hub (any project, the trash too, which gives them back).
    Unnamed, it is ``auto`` (dataset-version-variant) with today's date: coco8-v1-yolov8s-20261123, the next that
    day -2. A run started again (다시 실행, 이어서 실행) takes its source's name with today's date in place of its
    day. A name given as it is (a sweep's trials too) gets a number only when the Hub has it already."""
    from .db import write_lock

    base = _slug(given)
    if not base or again:
        base = f"{_DATED.sub('', (base or _slug(auto)))[:140]}-{_today()}"
    write_lock(db)  # two starts at once read the names one after the other
    taken = set(db.scalars(select(Run.name).where(Run.name.startswith(base, autoescape=True))))
    return _unique(base[:150], taken)


def _name_holder(db: Session, name: str, run_id: str = "") -> Run | None:
    """A run on the Hub (not in the trash) already called ``name``, other than ``run_id``."""
    return db.scalars(select(Run).where(Run.name == name, Run.deleted_at.is_(None), Run.id != run_id).limit(1)).first()


# ---- starting runs -----------------------------------------------------------------


def data_node(db: Session, release_id: str) -> Node:
    """The node that holds the release's image files; training has to run there."""
    node_ids = db.scalars(
        select(StorageLocation.node_id)
        .join(Item, Item.location_id == StorageLocation.id)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release_id)
        .distinct()
    ).all()
    if not node_ids:
        raise ServiceError(400, "이 버전의 이미지가 어느 노드에 있는지 알 수 없습니다.")
    if len(node_ids) > 1:
        raise ServiceError(400, "이 버전의 이미지가 여러 노드에 흩어져 있습니다. 한 노드에 모인 버전만 학습할 수 있습니다.")  # noqa: E501
    node = db.get(Node, node_ids[0])
    if node is None:
        raise ServiceError(400, "이 버전의 이미지가 있던 노드가 삭제되었습니다.")
    return node


def placement(db: Session, release: Release, nodes: list[str] | None) -> tuple[list[Node], Node | None]:
    """Where a run on ``release`` may go, and the node that must receive the images first (if any).

    ``nodes``: the servers allowed (none: the one holding the images). Nothing is created here.
    """
    candidates = [data_node(db, release.id)]
    if not nodes:
        return candidates, None
    allowed = transfers.allowed_nodes(db, nodes)
    have = transfers.holders(db, release.id)
    ready = [n for n in allowed if n.id in have]
    if ready:
        return ready, None
    receiver = transfers.pick_receiver(allowed)  # none has the images: one receives them first
    transfers.check_route(candidates[0], receiver)
    return [receiver], receiver


# The node writes each format from this BADAMOA on: an older node cannot hand the data to the code.
def _tasks_text(manifest: Any) -> str:
    """The tasks a kind trains, as shown: Det, or Det · Seg · Kpt."""
    return " · ".join(task_names.short(t) for t in manifest.trains())


FORMAT_SINCE = {"yolo": (0, 5, 0), "imagefolder": (0, 8, 0), "yolo-seg": (0, 11, 0), "yolo-pose": (0, 60, 0)}


def _version(text: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(p) for p in (text or "").split(".")[:3])
    except ValueError:
        return None


def data_checks(
    db: Session, release: Release, manifest: Any, chosen: list[int], nodes: list[Node], val_fraction: float = 0.2
) -> tuple[list[str], list[str]]:
    """Whether a version fits what a template takes: reasons not to start, and notes worth seeing first.

    From the version's own records only (no node is asked), so the dialog can show them as one picks.
    """
    problems: list[str] = []
    notes: list[str] = []
    task = catalog.release_task(db, release)
    space = db.get(LabelSpace, release.label_space_id)
    names = {int(c["id"]): str(c["name"]) for c in space.classes}
    ids = list(chosen) or list(names)
    least, most = manifest.data.min_classes(task), manifest.data.classes.max
    if len(ids) < least:
        more = " 학습할 클래스를 더 고르세요." if chosen else ""
        problems.append(f"이 모델은 클래스가 {least}개 이상이어야 합니다. 지금 {len(ids)}개입니다.{more}")
    if most and len(ids) > most:
        problems.append(f"이 모델은 클래스를 {most}개까지 받습니다. 지금 {len(ids)}개입니다. 학습할 클래스를 고르세요.")
    labelled = (
        db.scalar(
            select(func.count(func.distinct(ReleaseItem.item_id)))
            .join(RevisionClass, RevisionClass.revision_id == ReleaseItem.revision_id)
            .where(ReleaseItem.release_id == release.id, ReleaseItem.state == "labelled", RevisionClass.class_id.in_(ids))
        )
        or 0
    )
    need = max(1, manifest.data.min_images)
    if labelled < need:
        what = "고른 클래스로 " if chosen else ""
        problems.append(
            f"{what}라벨된 이미지가 없습니다."
            if not labelled
            else f"{what}학습에 쓸 라벨 있는 이미지가 {labelled:,}장입니다 ({need:,}장 이상 필요)."
        )
    if manifest.data.keypoints and task == "pose":
        counts = {len(c.get("keypoints") or []) for c in space.classes if int(c["id"]) in ids}
        if counts and counts != {manifest.data.keypoints}:
            have = ", ".join(str(n) for n in sorted(counts))
            problems.append(f"이 모델은 물체마다 점 {manifest.data.keypoints}개를 받습니다. 이 데이터셋은 {have}개입니다.")
    since = FORMAT_SINCE.get(manifest.data_format)
    for node in nodes:
        version = _version(node.agent_version or "")
        if since and version and version < since:
            need_v = ".".join(str(n) for n in since)
            problems.append(
                f"{node.name}의 MOA({node.agent_version})는 이 모델이 받는 데이터 형식을 만들 수 없습니다. "
                f"{need_v} 이상으로 업데이트하세요."
            )
    splits = (release.stats or {}).get("splits") or {}
    if not (splits.get("val") or {}).get("items"):
        if manifest.data_format == "imagefolder" and (splits.get("test") or {}).get("items"):
            notes.append("이 버전에는 val이 없어 test로 검증합니다.")
        elif val_fraction > 0:
            notes.append(f"이 버전에는 val이 없어 train의 {val_fraction:.0%}를 떼어 검증에 씁니다.")
    if task == "classify":
        trained = set(
            db.scalars(
                select(RevisionClass.class_id)
                .join(ReleaseItem, ReleaseItem.revision_id == RevisionClass.revision_id)
                .where(ReleaseItem.release_id == release.id, ReleaseItem.split == "train", RevisionClass.class_id.in_(ids))
                .distinct()
            )
        )
        empty = [names.get(i, str(i)) for i in ids if i not in trained]
        if empty and (splits.get("train") or {}).get("items"):
            shown = ", ".join(empty[:5]) + (f" 외 {len(empty) - 5}개" if len(empty) > 5 else "")
            notes.append(f"train에 이미지가 없는 클래스가 있습니다: {shown}")
    return problems, notes


def check_run(
    db: Session,
    *,
    release_id: str,
    template: str = "",
    params: dict[str, Any] | None = None,
    classes: list[int] | None = None,
    nodes: list[str] | None = None,
    val_fraction: float = 0.2,
    space: dict[str, Any] | None = None,
    environment: str = "",
    gpus: int | None = None,
    user: User | None = None,
) -> dict[str, list[str]]:
    """What starting this run would run into, without starting it: ``problems`` stop it, ``notes`` do not.

    The new-run dialog asks as one picks the version, the model, the classes and the servers.
    ``space``: the settings a search would try (left out of the check of the settings).
    """
    problems: list[str] = []
    warnings: list[str] = []
    try:
        release = catalog.get_release(db, release_id)
        catalog.require_ready(release)
        dataset = db.get(Dataset, release.dataset_id)
        project = catalog.get_project(db, dataset.project_id)
        task = catalog.dataset_task(dataset)
        row = registry.template_for(db, project.settings, task, template)
        manifest = registry.manifest_of(row, task)
        if not manifest.supports(task):
            why = f"{row.title}은(는) {_tasks_text(manifest)}용이고 이 데이터셋은 {task_names.short(task)}입니다."
            return {"problems": [why], "notes": []}
        own = user is not None and row.created_by == user.id
        if not template_checks.offered(row) and own and user.role != "admin":
            warnings.append(f"{row.title} {row.version}은(는) 내 초안입니다. 관리자가 허용하기 전에는 나만 씁니다.")
        elif not template_checks.offered(row):
            if user is None or user.role != "admin":
                return {"problems": [template_checks.NOT_YET], "notes": []}
            trial = template_checks.view(row)
            said = template_checks.SAID.get(trial["state"], trial["state"])
            said += f": {trial['reason']}" if trial.get("reason") else ""
            warnings.append(f"{row.title} {row.version}은(는) 아직 허용되지 않았습니다 ({said}). 관리자만 쓸 수 있습니다.")
        try:
            resolve_params(manifest, {k: v for k, v in (params or {}).items() if k not in (space or {})})
        except TemplateError as e:
            problems.append(str(e))
        known = {int(c["id"]) for c in db.get(LabelSpace, release.label_space_id).classes}
        chosen = sorted({int(c) for c in classes or []} & known)
        candidates, _ = placement(db, release, nodes)
        env = registry.environment_for_run(db, manifest, environment, candidates[0], row)
    except ServiceError as e:
        return {"problems": [*problems, e.message], "notes": []}
    found, notes = data_checks(db, release, manifest, chosen, candidates, val_fraction)
    return {"problems": problems + found + fitting(candidates, env, gpus)[1], "notes": warnings + notes}


def fitting(
    nodes: list[Node], env: dict[str, Any], gpus: int | None, role: str = registry.TRAINS
) -> tuple[list[Node], list[str]]:
    """Of the servers a run may go to, those that can run it: they have the role (학습용, or 오토라벨용 for an
    autolabel), and for a Docker environment they have Docker (with GPUs when the run takes some). The reasons
    when none can."""
    autolabel = role == registry.AUTOLABELS
    why: dict[str, list[str]] = {}
    for node in nodes:
        found = why.setdefault(node.id, [])
        if not registry.has_role(node, role):
            if autolabel:
                hint = "오토라벨은 이미지가 있는 서버에서 돕니다. 노드 설정에서 '오토라벨용'을 켜세요"
                found.append(f"{node.name}은(는) 오토라벨용 서버가 아닙니다. {hint}.")
            else:
                hint = "학습할 노드를 고르거나 노드 설정에서 '학습용'을 켜세요"
                found.append(f"{node.name}은(는) 학습용 서버가 아닙니다. {hint}.")
        if env.get("kind") == "docker":
            docker = (node.info or {}).get("docker")
            wants_gpu = bool((node.info or {}).get("gpus")) if gpus is None else gpus > 0
            if not docker:
                verb = "돌릴" if autolabel else "학습할"
                found.append(f"{node.name}에는 Docker가 없어 '{env.get('name')}' 환경으로 {verb} 수 없습니다.")
            elif wants_gpu and not docker.get("gpu"):
                found.append(
                    f"{node.name}의 Docker는 GPU를 넘기지 못합니다(NVIDIA Container Toolkit 필요). "
                    "GPU 0개로 학습하거나 다른 노드를 고르세요."
                )
    usable = [n for n in nodes if not why[n.id]]
    if usable:
        return usable, []
    return nodes, list(dict.fromkeys(reason for reasons in why.values() for reason in reasons))


def pick_gpus(node: Node, candidates: list[Node], wanted: list[int] | None) -> list[int]:
    """The GPUs a run asked for by number: on the one server it trains on, there, and not kept for people."""
    picked = sorted({int(i) for i in wanted or []})
    if not picked:
        return []
    if len(candidates) != 1:
        names = ", ".join(n.name for n in candidates)
        raise ServiceError(400, f"GPU를 고르려면 학습할 노드가 하나여야 합니다 (지금 {names}). 노드를 하나만 고르세요.")
    known = {g.get("index") for g in (node.info or {}).get("gpus") or []}
    if missing := [i for i in picked if i not in known]:
        raise ServiceError(400, f"{node.name}에 GPU {', '.join(map(str, missing))}번이 없습니다.")
    if kept := [i for i in picked if i in (node.reserved_gpus or [])]:
        raise ServiceError(400, f"{node.name}의 GPU {', '.join(map(str, kept))}번은 사람 전용입니다 (노드 설정).")
    return picked


def start_run(
    db: Session,
    user: User,
    *,
    release_id: str,
    template: str = "",
    environment: str = "",
    params: dict[str, Any] | None = None,
    name: str = "",
    model_name: str = "",
    classes: list[int] | None = None,
    unlabelled: str = "exclude",
    val_fraction: float = 0.2,
    gpus: int | None = None,
    gpu_indices: list[int] | None = None,
    min_vram_mb: int = 0,
    source_run_id: str | None = None,
    base_model_id: str = "",
    resume_run_id: str = "",
    tags: list[str] | None = None,
    note: str = "",
    sweep: dict[str, Any] | None = None,
    nodes: list[str] | None = None,
    data_after: str = "",
    embed: bool = False,
    via: str = "api",
) -> Run:
    """Queue a training run: this release, with this template's code, in this environment.

    ``nodes`` are the servers it may train on (none: the one holding the images). One of them
    that has the images takes it as it is; if none has, one receives them first from the node
    that holds them (``transfers``), without holding a GPU meanwhile. ``data_after`` says what
    becomes of the copy such a node received once the run is over: ``keep`` it or ``delete`` it
    (only after a run that succeeded; ``node_data``). ``embed``: the server it trains on also makes the version's
    image vectors for 이미지 탐색 (only those not made before), from the copy it has anyway (``embeddings``).

    ``template`` and ``environment`` may be left empty: the project's default
    template and the template's default environment are used. ``base_model_id``
    starts from the weights of a registered model instead of pretrained ones
    (training on with new data). ``gpu_indices`` are the GPUs it takes on the one server it trains on
    (none: as many free ones as ``gpus`` asks, whichever they are).
    """
    release = catalog.get_release(db, release_id)
    catalog.require_ready(release)
    dataset = db.get(Dataset, release.dataset_id)
    project = catalog.get_project(db, dataset.project_id)
    task = catalog.dataset_task(dataset)
    row = registry.template_for(db, project.settings, task, template)
    template_checks.require_offered(row, user)
    manifest = registry.manifest_of(row, task)
    if not manifest.supports(task):
        why = f"{row.title} 템플릿은 {_tasks_text(manifest)}용이고 이 데이터셋은 {task_names.short(task)}입니다."
        raise ServiceError(400, why)
    try:
        settings = resolve_params(manifest, params or {})
    except TemplateError as e:
        raise ServiceError(400, str(e)) from e
    if unlabelled not in ("exclude", "background"):
        raise ServiceError(400, "unlabelled는 exclude 또는 background여야 합니다.")
    if not 0 <= val_fraction <= 0.9:
        raise ServiceError(400, "val_fraction은 0~0.9여야 합니다.")
    if data_after not in node_data.CHOICES:
        raise ServiceError(400, "data_after는 keep(남겨 두기) 또는 delete(지우기)입니다.")
    space = db.get(LabelSpace, release.label_space_id)
    known = {int(c["id"]) for c in space.classes}
    chosen = sorted({int(c) for c in classes or []})
    if not set(chosen) <= known:
        raise ServiceError(400, f"라벨 공간 {space.name}에 없는 클래스 id가 있습니다: {sorted(set(chosen) - known)}")
    candidates, receiver = placement(db, release, nodes)
    problems, _ = data_checks(db, release, manifest, chosen, candidates, val_fraction)
    env = registry.environment_for_run(db, manifest, environment, candidates[0], row)
    candidates, unfit = fitting(candidates, env, gpus)
    problems += unfit
    if problems:
        raise ServiceError(400, " ".join(problems))
    after = transfers.sync_job(db, user, release.id, receiver, via).id if receiver else None
    node = candidates[0]
    if env.get("path"):  # a registered environment is a folder on one server
        candidates = [node]
    interrupted = get_run(db, resume_run_id) if resume_run_id else None
    base = None
    if base_model_id:
        base = get_model(db, base_model_id)
        if base.project_id != project.id or base.trainer != row.name:
            raise ServiceError(400, f"{row.title}로 만든 이 프로젝트의 모델에서만 이어서 학습할 수 있습니다.")
    picked = pick_gpus(node, candidates, gpu_indices)
    gpu_count = (1 if any((n.info or {}).get("gpus") for n in candidates) else 0) if gpus is None else gpus
    gpu_count = len(picked) if picked else gpu_count

    # What the model is called: the chosen variant (yolo11n.pt → yolo11n), else the template itself.
    variant = Path(str(settings.get(manifest.variant, ""))).stem if manifest.variant else ""
    variant = _slug(variant) or row.name
    epochs = settings.get("epochs")
    run = Run(
        project_id=project.id,
        name=run_name(db, name, f"{dataset.name}-{release.version}-{variant}", again=bool(source_run_id or resume_run_id)),
        release_id=release.id,
        trainer=row.name,
        template={
            "id": row.id,
            "name": row.name,
            "title": row.title,
            "version": row.version,
            "sha256": row.blob or "",
            "builtin": row.builtin,
            "variant": variant,
        },
        environment=env,
        config={
            **settings,
            "classes": chosen,
            "unlabelled": unlabelled,
            "val_fraction": val_fraction,
            "gpus": gpu_count,
            **({"gpu_indices": picked} if picked else {}),
            "min_vram_mb": min_vram_mb,
            "model_name": _slug(model_name),
            **({"nodes": list(dict.fromkeys(nodes))} if nodes else {}),
            # only where the images may have to come to another server
            **({"data_after": data_after} if nodes and data_after and not sweep else {}),
            **({"base_model": {"id": base.id, "name": base.name, "version": base.version}} if base else {}),
            **({"resumed_from": {"id": interrupted.id, "name": interrupted.name}} if interrupted else {}),
            **({"sweep": sweep} if sweep else {}),
        },
        epochs=int(epochs) if isinstance(epochs, int | float) else 0,
        source_run_id=source_run_id,
        note=note.strip()[:MAX_NOTE],
        sweep_id=(sweep or {}).get("id"),
        created_by=user.id,
    )
    db.add(run)
    db.flush()
    if tags:
        _set_tags(db, project.id, "run", run.id, tags)
    job = services.create_job(
        db,
        user,
        "train.run",
        {
            "run_id": run.id,
            "release_id": release.id,
            "template": {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "builtin": row.builtin,
                "sha256": row.blob or "",
                "entrypoint": manifest.entrypoints["train"],
                "data_format": manifest.data_format,
                "task": manifest.task,  # the code reads it from its config (a kind of several tasks)
            },
            "environment": env,
            "params": settings,
            "classes": chosen,
            "unlabelled": unlabelled,
            "val_fraction": val_fraction,
            **({"init_weights": {"blob": base.blob, "file_name": base.file_name or "best.pt"}} if base else {}),
            **({"resume": {"job_id": interrupted.job_id or "", **interrupted.checkpoint}} if interrupted else {}),
        },
        node.id if len(candidates) == 1 else None,
        via=via,
        gpus=gpu_count,
        min_vram_mb=min_vram_mb,
        nodes=[n.id for n in candidates] if len(candidates) > 1 else None,
        after=after,
        gpu_indices=picked,
    )
    run.job_id = job.id
    if embed and not sweep:
        from . import embeddings  # noqa: PLC0415  (embeddings reads this module)

        made = embeddings.start(
            db, user, release,
            node_id=node.id if len(candidates) == 1 else None,
            nodes=[n.id for n in candidates] if len(candidates) > 1 else None,
            after=after, gpus=gpu_count, run_id=run.id, via=via,
        )  # fmt: skip
        known = {"model": embeddings.DEFAULT_MODEL, **({"job_id": made.id} if made else {"skipped": True})}
        run.config = {**run.config, "embed": known}  # a new dict: a JSON column changed in place is not saved
    services.audit(
        db, user.id, via, "run.create", run.id, name=run.name, release=f"{dataset.name} {release.version}", template=row.name
    )
    return run


def rerun(db: Session, user: User, run_id: str, via: str = "api") -> Run:
    """Start a new run with the same release, the same template version and the same settings."""
    source = get_run(db, run_id)
    c = source.config
    row = db.get(Template, (source.template or {}).get("id", "")) or registry.get_template(db, source.trainer)
    if row.deleted_at is not None and row.builtin:
        row = registry.get_template(db, row.name)  # built-in templates move on with the installed version
    if row.deleted_at is not None:
        raise ServiceError(409, f"이 실행이 쓴 템플릿({row.title} {row.version})이 삭제되었습니다.")
    declared = registry.manifest_of(row).params
    env = source.environment or {}
    return start_run(
        db,
        user,
        release_id=source.release_id,
        template=row.id,
        environment=env.get("id") or ("auto" if env.get("kind") == "auto" else ""),
        params={k: c[k] for k in declared if k in c},
        name=source.name,
        model_name=c.get("model_name", ""),
        classes=c.get("classes"),
        unlabelled=c.get("unlabelled", "exclude"),
        val_fraction=c.get("val_fraction", 0.2),
        gpus=c.get("gpus"),
        gpu_indices=c.get("gpu_indices"),
        min_vram_mb=c.get("min_vram_mb", 0),
        base_model_id=(c.get("base_model") or {}).get("id", ""),
        source_run_id=source.id,
        tags=tags_of(db, "run", [source.id]).get(source.id, []),
        nodes=c.get("nodes") or None,
        data_after=c.get("data_after", ""),
        embed=bool(c.get("embed")),
        via=via,
    )


def _can_resume(db: Session, run: Run, status: str) -> bool:
    """An interrupted run with a checkpoint on the Hub, whose template knows how to continue from one."""
    if status not in ("failed", "canceled") or not (run.checkpoint or {}).get("blob"):
        return False
    row = db.get(Template, (run.template or {}).get("id", ""))
    return bool(row and row.deleted_at is None and registry.manifest_of(row).resume)


def resume(db: Session, user: User, run_id: str, via: str = "api") -> Run:
    """Continue an interrupted run from its last checkpoint, as a new run with the same settings."""
    source = get_run(db, run_id)
    job = db.get(Job, source.job_id) if source.job_id else None
    if not _can_resume(db, source, job.status if job else "failed"):
        raise ServiceError(400, "이어서 실행할 수 없습니다. 중단되거나 실패한 실행이고 모델 종류가 재개를 지원해야 합니다.")
    c = source.config
    row = db.get(Template, source.template["id"])
    env = source.environment or {}
    return start_run(
        db,
        user,
        release_id=source.release_id,
        template=row.id,
        environment=env.get("id") or ("auto" if env.get("kind") == "auto" else ""),
        params={k: c[k] for k in registry.manifest_of(row).params if k in c},
        name=source.name,
        model_name=c.get("model_name", ""),
        classes=c.get("classes"),
        unlabelled=c.get("unlabelled", "exclude"),
        val_fraction=c.get("val_fraction", 0.2),
        gpus=c.get("gpus"),
        gpu_indices=c.get("gpu_indices"),
        min_vram_mb=c.get("min_vram_mb", 0),
        base_model_id=(c.get("base_model") or {}).get("id", ""),
        source_run_id=source.id,
        resume_run_id=source.id,
        tags=tags_of(db, "run", [source.id]).get(source.id, []),
        note=source.note or "",
        nodes=c.get("nodes") or None,
        data_after=c.get("data_after", ""),
        via=via,
    )


# ---- node protocol (the node running the job calls these) ---------------------------


def _running(db: Session, node: Node, job_id: str) -> tuple[Job, Run]:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or not job.type.startswith("train.") or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 학습 작업이 아닙니다.")
    run = db.scalar(select(Run).where(Run.job_id == job.id))
    if run is None:
        raise ServiceError(404, "이 작업에 연결된 학습 실행이 없습니다.")
    return job, run


def manifest(db: Session, node: Node, job_id: str, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    """What to train on: every item of the release with its split and the labels the release pins."""
    _, run = _running(db, node, job_id)
    return release_manifest(db, node, db.get(Release, run.release_id), after, limit)


def release_manifest(
    db: Session, node: Node, release: Release, after: int = 0, limit: int = 2000, only: Any = None
) -> dict[str, Any]:
    """A page of a release's items as the node needs them to lay the data out for a template.

    ``only``: a query of item ids to keep (a labeling set's images to 오토라벨)."""
    dataset = db.get(Dataset, release.dataset_id)
    space = db.get(LabelSpace, release.label_space_id)
    limit = max(1, min(limit, MANIFEST_MAX))
    # the columns the node needs, not whole records: building three objects a row took 0.37 s a page of 2,000
    # (684,049 images: two minutes before a training could start), the columns 0.02 s
    query = (
        select(
            ReleaseItem.item_id,
            Item.sha256,
            Item.rel_path,
            Item.bytes,
            Item.width,
            Item.height,
            Item.location_id,
            ReleaseItem.split,
            LabelRevision.state,
            LabelRevision.boxes,
            LabelRevision.labelled_class_ids,
        )
        .join(Item, Item.id == ReleaseItem.item_id)
        .join(LabelRevision, LabelRevision.id == ReleaseItem.revision_id)
        .where(ReleaseItem.release_id == release.id, ReleaseItem.item_id > after)
    )
    if only is not None:
        query = query.where(ReleaseItem.item_id.in_(only))
    rows = db.execute(query.order_by(ReleaseItem.item_id).limit(limit)).all()
    locations = {
        loc.id: loc.root for loc in db.scalars(select(StorageLocation).where(StorageLocation.node_id == node.id))
    }
    keys = ("id", "sha256", "rel_path", "bytes", "width", "height", "location_id", "split", "state", "boxes")
    return {
        "dataset": dataset.name,
        "version": release.version,
        "classes": space.classes,
        "locations": locations,
        "items": [dict(zip(keys, row[:10], strict=True), labelled_class_ids=row[10]) for row in rows],
        "next": rows[-1][0] if len(rows) == limit else None,
    }


def init_weights(db: Session, node: Node, job_id: str) -> str:
    """The storage key of the weights this run starts from, when it continues a registered model."""
    job, _ = _running(db, node, job_id)
    blob = ((job.params or {}).get("init_weights") or {}).get("blob")
    if not blob:
        raise ServiceError(404, "이 실행은 등록된 모델에서 시작하지 않습니다.")
    return blob


def system(db: Session, run: Run) -> dict[str, Any]:
    """CPU, memory and GPU load of the server while the run was on it."""
    job = db.get(Job, run.job_id) if run.job_id else None
    if job is None or job.node_id is None or job.started_at is None:
        return {"timestamps": [], "series": {}, "gpu_indices": [], "step_s": 60, "from": None, "to": None}
    history = resources.window(db, job.node_id, job.started_at, job.finished_at or utcnow())
    return {**history, "gpu_indices": job.gpu_indices or []}


def template_bundle(db: Session, node: Node, job_id: str) -> str:
    """The storage key (sha256) of the template code this run was started with."""
    _, run = _running(db, node, job_id)
    blob = (run.template or {}).get("sha256")
    if not blob:
        raise ServiceError(404, "이 실행의 템플릿은 내려받을 묶음이 없습니다 (내장 템플릿).")
    return blob


def report(db: Session, node: Node, job_id: str, body: TrainReport) -> dict[str, Any]:
    _, run = _running(db, node, job_id)
    data = body.data
    if body.event == "stage":
        if data.get("stage") in STAGES:
            run.stage = data["stage"]
    elif body.event == "export":
        run.export = data
    elif body.event == "info":
        run.info = {**(run.info or {}), **data}
    elif body.event == "epoch":
        primary = PRIMARY_METRIC.get(catalog.release_task(db, run.release_id), "map50_95")
        rows: dict[int, RunMetric] = {}
        # One entry, or several the node gathered while the code logged faster than once a second.
        for entry in data["batch"] if isinstance(data.get("batch"), list) else [data]:
            # ``step`` is where the values go on the x axis; ``epoch`` / ``epochs`` only move the progress bar.
            metrics = {k: v for k, v in (entry.get("metrics") or {}).items() if isinstance(v, int | float)}
            if entry.get("epoch") is not None:
                run.epoch = max(run.epoch, int(entry["epoch"]))
            if entry.get("epochs"):
                run.epochs = int(entry["epochs"])
            if not metrics:
                continue
            step = int(entry.get("step", entry.get("epoch", 0)) or 0)
            row = rows.get(step) or db.get(RunMetric, (run.id, step))
            if row is None:
                row = RunMetric(run_id=run.id, step=step, data=metrics)
                db.add(row)
            else:
                row.data = {**row.data, **metrics}  # several reports for one step add up
            rows[step] = row
            # Until the run finishes, its summary is the best step so far.
            if metrics.get(primary, 0.0) >= (run.summary or {}).get(primary, -1.0):
                run.summary = {**(run.summary or {}), **metrics}
    return {"ok": True}


def _drop_blob(db: Session, storage: Storage | None, blob: str) -> None:
    """Remove a checkpoint from storage unless something else (a model, a recorded file) is the same bytes."""
    if not blob or storage is None:
        return
    used = db.scalar(select(ModelVersion.id).where(ModelVersion.blob == blob).limit(1)) or db.scalar(
        select(RunFile.id).where(RunFile.blob == blob).limit(1)
    )
    if used is None and storage.has_blob(blob):
        try:
            storage.delete(storage.blob_key(blob))
        except OSError:
            pass  # a leftover file is harmless


def set_checkpoint(
    db: Session, node: Node, job_id: str, name: str, blob: str, size: int, epoch: int, storage: Storage
) -> dict[str, Any]:
    """Keep the run's newest checkpoint (and only that one) so the run can be resumed from any server."""
    _, run = _running(db, node, job_id)
    previous = (run.checkpoint or {}).get("blob", "")
    run.checkpoint = {"blob": blob, "bytes": size, "file_name": Path(name).name[:128], "epoch": int(epoch)}
    if previous != blob:
        _drop_blob(db, storage, previous)
    return {"epoch": int(epoch)}


def checkpoint_to_resume(db: Session, node: Node, job_id: str) -> str:
    """The storage key of the checkpoint this job continues from."""
    job, _ = _running(db, node, job_id)
    blob = ((job.params or {}).get("resume") or {}).get("blob")
    if not blob:
        raise ServiceError(404, "이 실행은 이어서 실행하는 것이 아닙니다.")
    return blob


def clean_file_name(name: str) -> str:
    """``weights\\last.pt`` → ``weights/last.pt``: a name may carry folders, never a way out of them."""
    parts = [part for part in re.split(r"[\\/]+", name.strip()) if part and part not in (".", "..")]
    return "/".join(parts)[-255:].lstrip("/") or "file"


def add_file(
    db: Session,
    node: Node,
    job_id: str,
    name: str,
    blob: str,
    size: int,
    key: str = "",
    step: int | None = None,
    caption: str = "",
    media: str = "",
) -> dict[str, Any]:
    """Attach a file to its run: one the code recorded, or one the node found in the output folder.

    With a ``key`` it is a per-step image or (``media="table"``) table. ``media="table-image"`` is a
    picture a table cell draws on (``bm.overlay``): shown in its table only. Recording the same name again replaces it.
    """
    _, run = _running(db, node, job_id)
    name = clean_file_name(name)
    existing = db.scalars(select(RunFile).where(RunFile.run_id == run.id)).all()
    row = next((f for f in existing if f.name == name), None)
    if row is None:
        if len(existing) >= MAX_FILES_PER_RUN:
            raise ServiceError(400, f"한 실행에 기록할 수 있는 파일은 {MAX_FILES_PER_RUN}개까지입니다.")
        row = RunFile(run_id=run.id, name=name)
        db.add(row)
    row.blob, row.bytes = blob, size
    picture = name.lower().endswith(IMAGE_SUFFIXES)
    if media == "table":
        row.kind = "table"
    elif media == "table-image" and picture:
        row.kind = "table-image"
    else:
        row.kind = "image" if picture else "file"
    row.key, row.step, row.caption = key.strip()[:64], step, caption.strip()[:255]
    db.flush()
    return {"id": row.id, "name": row.name}


def files_of(db: Session, run_id: str) -> list[RunFile]:
    get_run(db, run_id)
    return list(db.scalars(select(RunFile).where(RunFile.run_id == run_id).order_by(RunFile.name)))


def get_file(db: Session, file_id: str) -> RunFile:
    row = db.get(RunFile, file_id)
    if row is None:
        raise ServiceError(404, "파일을 찾을 수 없습니다.")
    return row


def finish(
    db: Session, node: Node, job_id: str, body: TrainFinish, storage: Storage | None = None
) -> dict[str, Any]:
    """Record the final metrics and, when weights were uploaded, register them as a new model version."""
    job, run = _running(db, node, job_id)
    # A finished run has nothing left to resume, and neither have the interrupted runs it continued.
    done, seen = run, set()
    while done is not None and done.id not in seen:
        seen.add(done.id)
        if done.checkpoint:
            spent, done.checkpoint = done.checkpoint.get("blob", ""), {}
            db.flush()
            _drop_blob(db, storage, spent)
        done = db.get(Run, ((done.config or {}).get("resumed_from") or {}).get("id") or "")
    existing = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id))
    if existing is not None:  # a retried call
        return {"run_id": run.id, "model": {"id": existing.id, "name": existing.name, "version": existing.version}}
    if body.metrics:
        run.summary = body.metrics
    run.class_metrics = body.classes
    blob = str(body.weights.get("blob", ""))
    if not blob:  # the code reported results but no weights: metrics only
        return {"run_id": run.id, "model": None}
    release = db.get(Release, run.release_id)
    dataset = db.get(Dataset, release.dataset_id)
    variant = (run.template or {}).get("variant") or Path(str(run.config.get("model", ""))).stem or run.trainer
    name = run.config.get("model_name") or _slug(f"{dataset.name}-{variant}")
    latest = db.scalar(
        select(func.max(ModelVersion.version)).where(ModelVersion.project_id == run.project_id, ModelVersion.name == name)
    )
    imgsz = run.config.get("imgsz")
    model = ModelVersion(
        project_id=run.project_id,
        name=name,
        version=(latest or 0) + 1,
        run_id=run.id,
        release_id=run.release_id,
        task=catalog.dataset_task(dataset),
        arch=variant,
        trainer=run.trainer,
        template=run.template or {},
        blob=blob,
        bytes=int(body.weights.get("bytes", 0)),
        file_name=str(body.weights.get("file_name", "best.pt"))[:128],
        imgsz=int(imgsz) if isinstance(imgsz, int | float) else 0,
        classes=list((run.export or {}).get("classes", [])),
        metrics=run.summary or {},
        class_metrics=body.classes,
        created_by=job.created_by,
    )
    db.add(model)
    db.flush()
    services.audit(db, job.created_by, "agent", "model.create", model.id, name=name, version=model.version)
    return {"run_id": run.id, "model": {"id": model.id, "name": model.name, "version": model.version}}


# ---- runs: views ---------------------------------------------------------------------


def get_run(db: Session, run_id: str) -> Run:
    run = db.get(Run, run_id)
    if run is None or run.deleted_at is not None:
        raise ServiceError(404, "학습 실행을 찾을 수 없습니다.")
    return run


def _template_of(run: Run) -> dict[str, Any]:
    """The run's template; runs from before templates existed were always the built-in YOLO."""
    if run.template:
        return run.template
    title = "Ultralytics YOLO" if run.trainer == "yolo" else run.trainer
    return {"id": "", "name": run.trainer, "title": title, "version": "", "builtin": True}


def _settings_of(db: Session, run: Run) -> list[dict[str, Any]]:
    """The template settings of a run with their labels, in the template's order."""
    row = db.get(Template, (run.template or {}).get("id", ""))
    declared = registry.manifest_of(row).params if row else {}
    names = list(declared) + [k for k in run.config if k not in declared and k not in RESERVED]
    return [
        {
            "name": k,
            "label": (declared[k].label if k in declared else "") or k,
            "value": run.config[k],
            "group": declared[k].group if k in declared else "",
            "advanced": declared[k].advanced if k in declared else False,
            # Lets the screen show what was changed first and fold the untouched defaults away.
            "changed": k not in declared or run.config[k] != declared[k].default,
        }
        for k in names
        if k in run.config and not isinstance(run.config[k], dict)
    ]


def _sync_view(db: Session, job: Job, node_names: dict[str, str] | None) -> dict[str, Any] | None:
    first = transfers.waiting_for(db, job)
    if first is None or first.type != "data.sync":
        return None
    names = node_names if node_names is not None else services.node_name_map(db)
    p = (first.result or {}).get("progress") or {}
    return {
        "job_id": first.id,
        "status": first.status,
        "node_name": names.get(first.target_node_id or "", ""),
        "items": p.get("items"),
        "bytes": p.get("bytes"),
        "done_items": p.get("done_items"),
        "done_bytes": p.get("done_bytes"),
        "cached_items": p.get("cached_items"),
        "rate": p.get("rate"),
        "eta_s": p.get("eta_s"),
    }


def _environment_view(env: dict[str, Any]) -> dict[str, Any]:
    """A run's environment as screens show it: no server paths, the long lists as counts."""
    out = {k: v for k, v in env.items() if k not in ("path", "pinned", "wheels")}
    if env.get("pinned"):
        out["pinned"] = len(env["pinned"])
    if env.get("wheels"):
        out["wheels"] = len(env["wheels"])
    return out


def run_view(
    db: Session,
    settings: HubSettings,
    run: Run,
    detail: bool = False,
    node_names: dict[str, str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    job = db.get(Job, run.job_id) if run.job_id else None
    release = db.get(Release, run.release_id)
    dataset = db.get(Dataset, release.dataset_id) if release else None
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id, ModelVersion.deleted_at.is_(None)))
    job_data = services.job_view(db, settings, job, node_names) if job else None
    status = job_data["status"] if job_data else "failed"
    template = _template_of(run)
    view: dict[str, Any] = {
        "id": run.id,
        "name": run.name,
        "project_id": run.project_id,
        "task": catalog.dataset_task(dataset),
        "template": {k: template.get(k, "") for k in ("id", "name", "title", "version", "variant")},
        "trainer_title": template.get("title") or run.trainer,
        "environment": _environment_view(run.environment or {"kind": "auto", "name": "자동 설치"}),
        "status": status,
        # What a running job is doing right now: data → env → train → upload.
        "stage": (run.stage or "data") if status == "running" else "",
        "job_id": run.job_id,
        "node_name": job_data["node_name"] if job_data else None,
        "wait_reason": job_data["wait_reason"] if job_data else None,
        "error": job_data["error"] if job_data else None,
        "duration_s": job_data["duration_s"] if job_data else None,
        "gpu_indices": job_data["gpu_indices"] if job_data else [],
        "created_via": job_data["created_via"] if job_data else "",
        "created_at": iso(run.created_at),
        "epochs": run.epochs,
        "epoch": run.epoch,
        "config": run.config,
        "release": (
            {"id": release.id, "dataset": dataset.name, "version": release.version, "digest": release.digest}
            if release and dataset
            else None
        ),
        "metrics": run.summary or {},  # final metrics once finished, the best step so far before that
        "model": (
            {"id": model.id, "name": model.name, "version": model.version, "alias": model.alias, "aliases": _aliases(model)}
            if model
            else None
        ),
        # Images on their way from the node that holds them, before the training can start.
        "sync": _sync_view(db, job, node_names) if job and status == "queued" else None,
        "base_model": run.config.get("base_model"),  # the registered model this run started from, if any
        "resumed_from": run.config.get("resumed_from"),  # the interrupted run this one continues
        "sweep": run.config.get("sweep"),  # the search this run is part of: {"id", "name", "index"}
        "can_resume": _can_resume(db, run, status),
        "checkpoint_epoch": (run.checkpoint or {}).get("epoch"),  # how far a resumed run would start from
        "source_run_id": run.source_run_id,
        "note": run.note or "",
        "tags": tags if tags is not None else tags_of(db, "run", [run.id]).get(run.id, []),
    }
    if detail:
        rows = db.scalars(select(RunMetric).where(RunMetric.run_id == run.id).order_by(RunMetric.step)).all()
        if len(rows) > 2 * HISTORY_POINTS:  # logged every iteration: a chart needs far fewer points
            stride = -(-len(rows) // HISTORY_POINTS)
            rows = [r for i, r in enumerate(rows) if i % stride == 0 or i == len(rows) - 1]
        keys = sorted({k for r in rows for k in r.data})
        view["history"] = {"steps": [r.step for r in rows], "series": {k: [r.data.get(k) for r in rows] for k in keys}}
        view["export"] = run.export or {}
        view["info"] = run.info or {}
        stored = db.scalars(select(RunFile).where(RunFile.run_id == run.id).order_by(RunFile.created_at, RunFile.name)).all()
        url = {f.id: catalog.media_url(settings, "file", f.id) for f in stored}
        view["files"] = [
            {"id": f.id, "name": f.name, "kind": f.kind, "bytes": f.bytes, "url": url[f.id]}
            for f in stored
            if not f.key and f.kind != "table-image"
        ]
        # Pictures that table cells draw on: found by name from the tables, not listed on their own.
        view["table_images"] = {f.name: url[f.id] for f in stored if f.kind == "table-image"}
        # Per-step images and tables, by key and step (W&B media panels, MLflow logged images and tables).
        media: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for f in sorted((f for f in stored if f.key and f.kind != "table-image"), key=lambda f: (f.step or 0, f.name)):
            media.setdefault(f.key, {}).setdefault(f.step or 0, []).append(
                {"name": f.name, "kind": f.kind, "caption": f.caption, "url": url[f.id]}
            )
        view["media"] = [
            {"key": key, "steps": [{"step": step, "files": files} for step, files in steps.items()]}
            for key, steps in media.items()
        ]
        view["files_url"] = catalog.media_url(settings, "run-files", run.id) if view["files"] else None
        view["metrics_csv_url"] = catalog.media_url(settings, "run-metrics", run.id) if rows else None
        view["class_metrics"] = run.class_metrics or []
        view["settings"] = _settings_of(db, run)
    return view


def _search(query: Any, table: Any, kind: str, tag: str, q: str) -> Any:
    """Narrow a listing to one tag and/or a text found in the name, the note or a tag."""
    if tag:
        query = query.where(table.id.in_(select(Tag.object_id).where(Tag.kind == kind, Tag.name == tag)))
    if q.strip():
        like = f"%{q.strip()}%"
        tagged = select(Tag.object_id).where(Tag.kind == kind, Tag.name.ilike(like))
        query = query.where(or_(table.name.ilike(like), table.note.ilike(like), table.id.in_(tagged)))
    return query


def metrics_csv(db: Session, run: Run) -> str:
    """Every step the run logged, one row each (the screen thins long histories out; this does not)."""
    rows = db.scalars(select(RunMetric).where(RunMetric.run_id == run.id).order_by(RunMetric.step)).all()
    keys = sorted({k for r in rows for k in r.data})
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["step", "time", *keys])
    for r in rows:
        writer.writerow([r.step, iso(r.ts), *("" if r.data.get(k) is None else r.data[k] for k in keys)])
    return out.getvalue()


def list_runs(
    db: Session,
    settings: HubSettings,
    project_id: str,
    limit: int = 200,
    tag: str = "",
    q: str = "",
    sweep: str = "",
) -> list[dict[str, Any]]:
    catalog.get_project(db, project_id)
    query = select(Run).where(Run.project_id == project_id, Run.deleted_at.is_(None))
    if sweep:
        query = query.where(Run.sweep_id == sweep)
    query = _search(query, Run, "run", tag, q).order_by(Run.created_at.desc()).limit(max(1, min(limit, 500)))
    runs = db.scalars(query).all()
    names = services.node_name_map(db)
    tags = tags_of(db, "run", [r.id for r in runs])
    return [run_view(db, settings, run, node_names=names, tags=tags.get(run.id, [])) for run in runs]


# ---- notes and tags ---------------------------------------------------------------------


def clean_tags(tags: list[str]) -> list[str]:
    """Trimmed, single-spaced, without repeats (ignoring case)."""
    out: list[str] = []
    for raw in tags:
        tag = " ".join(str(raw).split())[:TAG_LENGTH]
        if tag and tag.lower() not in {t.lower() for t in out}:
            out.append(tag)
    if len(out) > MAX_TAGS:
        raise ServiceError(400, f"태그는 {MAX_TAGS}개까지 붙일 수 있습니다.")
    return out


def tags_of(db: Session, kind: str, ids: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        for row in db.scalars(select(Tag).where(Tag.kind == kind, Tag.object_id.in_(chunk)).order_by(Tag.name)):
            found.setdefault(row.object_id, []).append(row.name)
    return found


def _set_tags(db: Session, project_id: str, kind: str, object_id: str, tags: list[str]) -> list[str]:
    wanted = clean_tags(tags)
    existing = {t.name: t for t in db.scalars(select(Tag).where(Tag.kind == kind, Tag.object_id == object_id))}
    for name, row in existing.items():
        if name not in wanted:
            db.delete(row)
    for name in wanted:
        if name not in existing:
            db.add(Tag(project_id=project_id, kind=kind, object_id=object_id, name=name))
    db.flush()
    return wanted


def project_tags(db: Session, project_id: str) -> dict[str, list[dict[str, Any]]]:
    """Every tag used in a project, most used first: what the screen offers to filter by."""
    catalog.get_project(db, project_id)
    found: dict[str, list[dict[str, Any]]] = {"runs": [], "models": []}
    count = func.count(Tag.id)
    binned = select(Run.id).where(Run.deleted_at.is_not(None)).union(
        select(ModelVersion.id).where(ModelVersion.deleted_at.is_not(None))
    )
    rows = db.execute(
        select(Tag.kind, Tag.name, count)
        .where(Tag.project_id == project_id, Tag.object_id.not_in(binned))
        .group_by(Tag.kind, Tag.name)
        .order_by(count.desc(), Tag.name)
    )
    for kind, name, n in rows:
        found["runs" if kind == "run" else "models"].append({"name": name, "count": n})
    return found


def update_run(
    db: Session,
    user: User,
    run_id: str,
    *,
    name: str | None = None,
    note: str | None = None,
    tags: list[str] | None = None,
    via: str = "api",
) -> Run:
    """Rename a run, write its note, set its tags. Anyone in the team may: it is shared bookkeeping."""
    run = get_run(db, run_id)
    changed: dict[str, Any] = {}
    if name is not None:
        name = " ".join(name.split())[:160]
        if not name:
            raise ServiceError(400, "이름을 비울 수 없습니다.")
        if name != run.name:
            from .catalog_models import Project
            from .db import write_lock

            write_lock(db)  # two renames to one name at once: the second sees the first
            holder = _name_holder(db, name, run.id)
            if holder is not None:  # one name per Hub, whatever the project
                if holder.project_id == run.project_id:
                    raise ServiceError(409, f"이 프로젝트에 이미 '{name}' 실행이 있습니다. 다른 이름을 쓰세요.")
                other = db.get(Project, holder.project_id)
                where = f"'{other.name}' 프로젝트" if other is not None else "다른 프로젝트"
                raise ServiceError(409, f"{where}에 이미 '{name}' 실행이 있습니다. 실행 이름은 Hub 전체에서 하나씩입니다.")
            changed["name"], run.name = name, name
    if note is not None:
        run.note = note.strip()[:MAX_NOTE]
        changed["note"] = len(run.note)
    if tags is not None:
        changed["tags"] = _set_tags(db, run.project_id, "run", run.id, tags)
    services.audit(db, user.id, via, "run.update", run.id, **changed)
    return run


def delete_run(db: Session, user: User, run_id: str) -> None:
    """To the trash, with its metrics, files, checkpoint and tags: restoring brings it all back."""
    run = get_run(db, run_id)
    job = db.get(Job, run.job_id) if run.job_id else None
    if job is not None and job.status not in TERMINAL_STATUSES:
        raise ServiceError(409, "실행 중인 학습은 먼저 중단하세요.")
    if user.role != "admin" and run.created_by != user.id:
        raise ServiceError(403, "자신이 시작한 학습만 지울 수 있습니다.")
    run.deleted_at = utcnow()
    services.audit(db, user.id, "api", "run.delete", run.id, name=run.name)


def restore_run(db: Session, user: User, run_id: str, storage: Storage) -> dict[str, Any]:
    """Out of the trash. A run of the same name started meanwhile keeps its name; this one gets a number."""
    run = db.get(Run, run_id)
    if run is None or run.deleted_at is None:
        raise ServiceError(404, "휴지통에 그 학습이 없습니다.")
    catalog.get_project(db, run.project_id)  # a run in a deleted project comes back with its project
    if user.role != "admin" and run.created_by != user.id:
        raise ServiceError(403, "자신이 시작한 학습이나 관리자만 되살릴 수 있습니다.")
    notes = []
    base, name, n = run.name[:150], run.name, 2
    while _name_holder(db, name, run.id) is not None:  # one name per Hub: another project's run keeps it too
        name, n = f"{base}-{n}", n + 1
    if name != run.name:
        notes.append(f"같은 이름의 학습이 있어 이름을 바꿨습니다: {name}")
        run.name = name
    # Deleted before the trash kept files: what the cleanup already removed cannot come back.
    gone = [f for f in db.scalars(select(RunFile).where(RunFile.run_id == run.id)) if not storage.has_blob(f.blob)]
    for f in gone:
        db.delete(f)
    if gone:
        notes.append(f"파일 {len(gone)}개는 이미 지워져 함께 되살리지 못했습니다.")
    if (run.checkpoint or {}).get("blob") and not storage.has_blob(run.checkpoint["blob"]):
        run.checkpoint = {}
    run.deleted_at = None
    services.audit(db, user.id, "api", "run.restore", run.id, name=run.name)
    return {"id": run.id, "name": run.name, "note": " ".join(notes)}


# ---- models ---------------------------------------------------------------------------


def get_model(db: Session, model_id: str) -> ModelVersion:
    model = db.get(ModelVersion, model_id)
    if model is None or model.deleted_at is not None:
        raise ServiceError(404, "모델을 찾을 수 없습니다.")
    return model


def _aliases(model: ModelVersion) -> list[str]:
    return list(model.aliases or ([model.alias] if model.alias else []))


def _put_aliases(model: ModelVersion, aliases: list[str]) -> None:
    model.aliases = aliases
    model.alias = aliases[0] if aliases else ""


def model_view(
    db: Session, settings: HubSettings, model: ModelVersion, detail: bool = False, tags: list[str] | None = None
) -> dict[str, Any]:
    release = db.get(Release, model.release_id) if model.release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    run = db.get(Run, model.run_id) if model.run_id else None
    source = None
    if release and dataset:
        source = {
            "id": release.id,
            "dataset": dataset.name,
            "version": release.version,
            "digest": release.digest,
            "deleted": bool(release.deleted_at),
        }
    template = model.template or (_template_of(run) if run else {"name": model.trainer, "title": model.trainer})
    view: dict[str, Any] = {
        "id": model.id,
        "project_id": model.project_id,
        "name": model.name,
        "version": model.version,
        "alias": model.alias,
        "aliases": _aliases(model),
        "tags": tags if tags is not None else tags_of(db, "model", [model.id]).get(model.id, []),
        "note": model.note or "",
        "task": model.task,
        "arch": model.arch,
        "trainer": model.trainer,
        "template": {k: template.get(k, "") for k in ("name", "title", "version")},
        "bytes": model.bytes,
        "imgsz": model.imgsz,
        "class_count": len(model.classes or []),
        "metrics": model.metrics or {},
        "created_at": iso(model.created_at),
        "run": {"id": run.id, "name": run.name} if run and run.deleted_at is None else None,
        "release": source,
        # "upload": trained elsewhere and uploaded with its classes (``model_upload``); "train": a run of this Hub
        "source": (model.source or {}).get("kind") or "train",
    }
    if detail:
        view["upload"] = dict(model.source or {}) if view["source"] == "upload" else None
        view["classes"] = model.classes or []
        view["class_metrics"] = model.class_metrics or []
        view["file_name"] = model.file_name
        view["sha256"] = model.blob
        view["config"] = run.config if run else {}
        view["info"] = run.info if run else {}
        view["export"] = run.export if run else {}
        view["environment"] = _environment_view((run.environment if run else None) or {})
        view["download_url"] = catalog.media_url(settings, "model", model.id)
    return view


def list_models(
    db: Session, settings: HubSettings, project_id: str, tag: str = "", q: str = "", name: str = "", alias: str = ""
) -> list[dict[str, Any]]:
    """The project's model versions; ``name`` + ``alias`` finds the version an alias points at (MLflow ``name@alias``)."""
    catalog.get_project(db, project_id)
    query = select(ModelVersion).where(ModelVersion.project_id == project_id, ModelVersion.deleted_at.is_(None))
    if name:
        query = query.where(ModelVersion.name == name)
    query = _search(query, ModelVersion, "model", tag, q).order_by(ModelVersion.name, ModelVersion.version.desc())
    models = [m for m in db.scalars(query) if not alias or alias in _aliases(m)]
    tags = tags_of(db, "model", [m.id for m in models])
    return [model_view(db, settings, m, tags=tags.get(m.id, [])) for m in models]


def update_model(
    db: Session, user: User, model_id: str, *, note: str | None = None, tags: list[str] | None = None, via: str = "api"
) -> ModelVersion:
    """A version's description (what it is for, what changed) and its tags."""
    model = get_model(db, model_id)
    changed: dict[str, Any] = {}
    if note is not None:
        model.note = note.strip()[:MAX_NOTE]
        changed["note"] = len(model.note)
    if tags is not None:
        changed["tags"] = _set_tags(db, model.project_id, "model", model.id, tags)
    services.audit(db, user.id, via, "model.update", model.id, name=model.name, version=model.version, **changed)
    return model


MAX_MODEL_VERSION = 1_000_000


def rename_model(
    db: Session, user: User, model_id: str, *, name: str | None = None, version: int | None = None, via: str = "api"
) -> ModelVersion:
    """Rename a model (``name``: every version of its family, in the trash too) or give this version another number.

    Names are shown from the model itself everywhere (lineage, runs, evaluations, 검수 …); the few records that keep a
    copy of a name follow here: a run continued from a model, a labeling set's 오토라벨, a version's 검수 record, the
    labels 오토라벨 wrote (their source). A run of the family registers its next version under the new name."""
    model = get_model(db, model_id)
    catalog._may_edit(catalog.get_project(db, model.project_id), user)
    changes: dict[str, tuple[str, int, str, int]] = {}  # model id → (old name, old version, new name, new version)
    old_name = model.name
    if name is not None:  # the rule names made by runs follow (spaces and signs become -): reruns keep the family
        name = _slug(name)[:128]
        if not name:
            raise ServiceError(400, "모델 이름을 적으세요.")
    if name is not None and name != model.name:
        taken = select(ModelVersion.id).where(ModelVersion.project_id == model.project_id, ModelVersion.name == name)
        if db.scalar(taken.limit(1)):
            raise ServiceError(409, f"이 프로젝트에 {name} 모델이 이미 있습니다(휴지통에 있는 것 포함). 다른 이름을 쓰세요.")
        family = db.scalars(
            select(ModelVersion).where(ModelVersion.project_id == model.project_id, ModelVersion.name == old_name)
        ).all()
        for row in family:
            changes[row.id] = (row.name, row.version, name, row.version)
            row.name = name
        # its runs register their next version (a rerun, a continued run) under the new name
        for run in db.scalars(select(Run).where(Run.id.in_([row.run_id for row in family if row.run_id]))):
            run.config = {**(run.config or {}), "model_name": name}
    if version is not None and int(version) != model.version:
        version = int(version)
        if not 1 <= version <= MAX_MODEL_VERSION:
            raise ServiceError(400, "버전은 1 이상의 정수여야 합니다.")
        taken = select(ModelVersion.id).where(
            ModelVersion.project_id == model.project_id,
            ModelVersion.name == model.name,
            ModelVersion.version == version,
            ModelVersion.id != model.id,
        )
        if db.scalar(taken.limit(1)):
            why = f"{model.name} v{version}이(가) 이미 있습니다(휴지통에 있는 것 포함). 다른 번호를 쓰세요."
            raise ServiceError(409, why)
        before = changes.get(model.id, (model.name, model.version, model.name, model.version))
        changes[model.id] = (before[0], before[1], model.name, version)
        model.version = version
    if not changes:
        return model
    _follow_rename(db, changes)
    old = changes[model.id]
    services.audit(
        db, user.id, via, "model.rename", model.id, old=f"{old[0]} v{old[1]}", name=model.name, version=model.version,
        versions=len(changes),
    )
    return model


def _follow_rename(db: Session, changes: dict[str, tuple[str, int, str, int]]) -> None:
    """The records that keep a copy of a model's name and version take the new ones."""
    from sqlalchemy import case, update

    from .catalog_models import LabelRevision, LabelSet

    def renamed(copy: Any) -> Any:
        if isinstance(copy, dict) and copy.get("id") in changes:
            _, _, name, number = changes[copy["id"]]
            return {**copy, "name": name, "version": number}
        return copy

    for run in db.scalars(select(Run)):  # a run continued from a model
        base = (run.config or {}).get("base_model")
        if isinstance(base, dict) and base.get("id") in changes:
            run.config = {**run.config, "base_model": renamed(base)}
    for found in db.scalars(select(LabelSet)):  # a labeling set's 오토라벨
        info = found.autolabel or {}
        if isinstance(info.get("model"), dict) and info["model"].get("id") in changes:
            found.autolabel = {**info, "model": renamed(info["model"])}
    for release in db.scalars(select(Release)):  # 검수 kept with a version: the model it went by
        record = release.review or {}
        basis = record.get("model") if isinstance(record, dict) else None
        if isinstance(basis, dict) and isinstance(basis.get("model"), dict) and basis["model"].get("id") in changes:
            release.review = {**record, "model": {**basis, "model": renamed(basis["model"])}}
    # the labels 오토라벨 wrote name the model they came from: one pass over the labels for every renamed version
    sources = {f"model:{a} v{b}": f"model:{c} v{d}" for a, b, c, d in changes.values() if (a, b) != (c, d)}
    if sources:
        db.execute(
            update(LabelRevision)
            .where(LabelRevision.source.in_(list(sources)))
            .values(source=case(sources, value=LabelRevision.source))
        )


def set_alias(db: Session, user: User, model_id: str, alias: str, via: str = "api", remove: bool = False) -> ModelVersion:
    """Point an alias at this version (MLflow model aliases), or take it off with ``remove``.

    Within a model family an alias sits on one version at a time, so it moves here from wherever
    it was; a version can carry several. An empty alias clears all of this version's.
    """
    model = get_model(db, model_id)
    alias = alias.strip()
    if not alias:
        _put_aliases(model, [])
    elif remove:
        _put_aliases(model, [a for a in _aliases(model) if a != alias])
    else:
        if not ALIAS_RE.match(alias) or alias.lower() == "latest" or re.fullmatch(r"[vV]\d+", alias):
            raise ServiceError(
                400, "alias는 32자 이내의 글자·숫자·._- 로 짓고, latest나 v1 같은 버전 표기는 쓸 수 없습니다."
            )
        family = db.scalars(
            select(ModelVersion).where(
                ModelVersion.project_id == model.project_id,
                ModelVersion.name == model.name,
                ModelVersion.id != model.id,
                ModelVersion.deleted_at.is_(None),
            )
        )
        for other in family:
            if alias in _aliases(other):
                _put_aliases(other, [a for a in _aliases(other) if a != alias])
        if alias not in _aliases(model):
            if len(_aliases(model)) >= MAX_ALIASES:
                raise ServiceError(400, f"한 버전에 alias는 {MAX_ALIASES}개까지입니다.")
            _put_aliases(model, [*_aliases(model), alias])
    services.audit(
        db, user.id, via, "model.alias", model.id, name=model.name, version=model.version, alias=alias, removed=remove
    )
    return model


def delete_model(db: Session, user: User, model_id: str) -> None:
    """To the trash, with its weights, evaluations and tags. Its aliases come off (they point at a listed version)."""
    services.require_admin(user)
    model = get_model(db, model_id)
    model.deleted_at = utcnow()
    _put_aliases(model, [])
    services.audit(db, user.id, "api", "model.delete", model.id, name=model.name, version=model.version)


def restore_model(db: Session, user: User, model_id: str, storage: Storage) -> dict[str, Any]:
    services.require_admin(user)
    model = db.get(ModelVersion, model_id)
    if model is None or model.deleted_at is None:
        raise ServiceError(404, "휴지통에 그 모델이 없습니다.")
    catalog.get_project(db, model.project_id)  # a model in a deleted project comes back with its project
    if not storage.has_blob(model.blob):
        raise ServiceError(409, "모델 파일이 이미 지워져 되살릴 수 없습니다. 휴지통에서 완전히 지우세요.")
    model.deleted_at = None
    services.audit(db, user.id, "api", "model.restore", model.id, name=model.name, version=model.version)
    return {"id": model.id, "name": model.name, "version": model.version, "note": ""}
