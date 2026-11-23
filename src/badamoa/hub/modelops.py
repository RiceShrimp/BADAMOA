"""What can be done with a registered model: evaluate it on a release, export it, try it on sample images.

Each is a job that runs an entry point (``eval`` / ``export`` / ``predict``) of the
template version that trained the model, so the code that reads the weights is
the code that wrote them. When that version has no such entry point, a newer offered
version of the same template that declares it reads those weights
(``reads_weights_from``) and has it does the job (``code_for``). Otherwise the
action is not offered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
from badamoa.core.storage import Storage
from badamoa.core.templates import TemplateManifest, reads_weights

from . import catalog, checks, drafts, registry, services, training, transfers
from .catalog_models import Dataset, ItemPrediction, LabelSpace, Release
from .config import HubSettings
from .db import iso
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Run, Template

ACTIONS = {"eval": "평가", "export": "내보내기", "predict": "테스트"}
# 오토라벨 runs the template's predict code in autolabel mode (see ``autolabel``)
ENTRY = {"autolabel": "predict"}
EXPORT_FORMATS = ("onnx", "torchscript")
MAX_TEST_IMAGES = 20
MAX_RESULT_ITEMS = 200
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
# A server evaluates on images it received from another one (laid out from its cache) from this BADAMOA on.
EVAL_RECEIVED_SINCE = (0, 69, 0)
SPLITS_SINCE = (0, 71, 0)  # a node evaluates several splits in one job
ORDERED_SINCE = (0, 81, 0)  # a node exports an evaluation's classes in the model's order


def _template_of(db: Session, model: ModelVersion) -> Template:
    row = db.get(Template, (model.template or {}).get("id", "")) or registry.get_template(db, model.trainer)
    if row.deleted_at is not None and row.builtin:
        row = registry.get_template(db, row.name)
    if row.deleted_at is not None:
        raise ServiceError(409, f"이 모델을 만든 템플릿({row.title} {row.version})이 삭제되었습니다.")
    return row


def _does(row: Template, kind: str) -> bool:
    manifest = registry.manifest_of(row)  # entry points and autolabel are the same for every task it trains
    return ENTRY.get(kind, kind) in manifest.entrypoints and (kind != "autolabel" or manifest.autolabel)


def _installs_env(row: Template) -> bool:
    """The version installs a virtual environment (it names packages), whether or not it declares an image too."""
    return bool(registry.manifest_of(row).environment.requirements)


def code_for(db: Session, model: ModelVersion, kind: str) -> Template | None:
    """The template version whose code does ``kind`` (eval · export · predict · autolabel) for this model: the
    version that trained it when that has the code, else the newest version offered to everyone that has it and
    declares it reads this model's weights (``reads_weights_from``). None when no version can.

    가상환경 먼저: when the version that trained it runs only in a Docker image (it names no packages), such a newer
    version that installs a virtual environment does it instead (a model trained with DCL 1.0.3, image only, is
    evaluated with 1.0.6's code in its virtual environment, as 1.0.6 says it reads 1.0.x weights)."""
    own = _template_of(db, model)
    if own.builtin:
        return own if _does(own, kind) else None
    if _does(own, kind) and _installs_env(own):
        return own
    from . import template_checks

    newer = db.scalars(
        select(Template)
        .where(Template.name == own.name, Template.deleted_at.is_(None), Template.id != own.id)
        .order_by(Template.created_at.desc(), Template.id)
    )
    for row in newer:
        if not (template_checks.offered(row) and _does(row, kind) and reads_weights(registry.manifest_of(row), own.version)):
            continue
        if _does(own, kind) and not _installs_env(row):
            continue  # the own code does it already: another version only to get a virtual environment
        return row
    return own if _does(own, kind) else None


def environment_choice(manifest: TemplateManifest, run: Run | None) -> str:
    """The environment a model's evaluation, export, test or auto-label asks for (``environment_for_run``'s
    ``wanted``). 가상환경 먼저, as for training: code that names packages runs in the virtual environment it installs
    (or the environment its template names as default) even when it declares a Docker image too, whatever the model
    was trained in; code that names none runs where the model was trained (a registered environment, or the image
    the code declares)."""
    if manifest.environment.requirements:
        return ""
    wanted = (run.environment or {}) if run else {}
    same = wanted.get("id") or {"auto": "auto", "docker": registry.DECLARED_IMAGE}.get(wanted.get("kind", ""), "")
    if same == registry.DECLARED_IMAGE and not manifest.environment.image:
        same = ""
    return same


def actions(db: Session, model: ModelVersion) -> list[str]:
    """Which of evaluate / export / test there is code for: this model's template version, or a newer one that
    reads its weights."""
    try:
        return [action for action in ACTIONS if code_for(db, model, action) is not None]
    except ServiceError:
        return []


def autolabels(db: Session, model: ModelVersion) -> bool:
    """Whether a template version that can work with this model declares its predict code also labels images."""
    try:
        return code_for(db, model, "autolabel") is not None
    except ServiceError:
        return False


def _training_node(db: Session, model: ModelVersion) -> Node | None:
    """The server that trained the model: its environment is already there."""
    run = db.get(Run, model.run_id) if model.run_id else None
    job = db.get(Job, run.job_id) if run and run.job_id else None
    node = db.get(Node, job.node_id) if job and job.node_id else None
    return node if node is not None and node.enabled else None


def _start(
    db: Session,
    user: User,
    model: ModelVersion,
    kind: str,
    node: Node | None,
    params: dict[str, Any],
    *,
    release_id: str | None = None,
    extra: dict[str, Any] | None = None,
    gpus: int = 0,
    via: str = "api",
    after: str | None = None,
) -> ModelJob:
    own = _template_of(db, model)
    row = code_for(db, model, kind)
    entry = ENTRY.get(kind, kind)
    if row is None:
        what = "오토라벨" if kind == "autolabel" else ACTIONS[entry]
        raise ServiceError(400, f"{own.title} {own.version}에는 {what} 코드가 없습니다 (이 가중치를 읽는 새 버전도 없음).")
    manifest = registry.manifest_of(row, model.task or "")  # a kind of several tasks: the model's
    if node is None:
        raise ServiceError(409, "이 모델을 학습한 서버가 연결되어 있지 않아 실행할 곳이 없습니다.")
    run = db.get(Run, model.run_id) if model.run_id else None
    env = registry.environment_for_run(db, manifest, environment_choice(manifest, run), node, row)
    role = registry.AUTOLABELS if kind == "autolabel" else registry.TRAINS
    _, unfit = training.fitting([node], env, gpus, role)
    if unfit:
        raise ServiceError(400, " ".join(unfit))
    declared = {**registry.manifest_of(own).params, **manifest.params}  # the settings it was trained with
    op = ModelJob(model_id=model.id, kind=kind, release_id=release_id, params=params, created_by=user.id)
    db.add(op)
    db.flush()
    job = services.create_job(
        db,
        user,
        "model.run",
        {
            "op_id": op.id,
            "model_id": model.id,
            "action": kind,
            "template": {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "builtin": row.builtin,
                "sha256": row.blob or "",
                "entrypoint": manifest.entrypoints[entry],
                "data_format": manifest.data_format,
                "task": manifest.task,
                # the version that trained the model, when the code is a newer one's
                **({"trained_with": own.version} if row.id != own.id else {}),
            },
            "environment": env,
            "params": params,
            # How the model was trained (architecture, image size …): the code needs it to rebuild the network.
            # Uploaded weights have no run: the image size read from them.
            "train_params": {k: v for k, v in run.config.items() if k in declared}
            if run
            else ({"imgsz": model.imgsz} if model.imgsz else {}),
            "classes": model.classes or [],
            "weights": {"blob": model.blob, "file_name": model.file_name or "best.pt"},
            **(extra or {}),
        },
        node.id,
        via=via,
        gpus=gpus,
        role=role,
        after=after,
    )
    op.job_id = job.id
    services.audit(db, user.id, via, f"model.{kind}", model.id, name=model.name, version=model.version, code=row.version)
    return op


def start_eval(
    db: Session,
    user: User,
    model_id: str,
    release_id: str,
    split: str = "val",
    gpus: int | None = None,
    via: str = "api",
    node_id: str = "",
    splits: list[str] | None = None,
) -> ModelJob:
    """Measure a model on a release: the way to compare models trained on different data.

    Like training, it runs where the release's images are (or on ``node_id``, which receives them first when it has
    none) and takes a GPU when that server has one (``gpus=0`` forces the CPU). ``splits``: several in one evaluation,
    one after the other; their per-image predictions make one basis for 검수.
    """
    model = training.get_model(db, model_id)
    release = catalog.get_release(db, release_id)
    catalog.require_ready(release)
    dataset = db.get(Dataset, release.dataset_id)
    # another project's version too: what counts is the task and the classes (checked below), so models of
    # different projects can be measured on the same data
    wanted = list(dict.fromkeys(splits or [split]))
    if not wanted or any(s not in ("val", "test", "train") for s in wanted):
        raise ServiceError(400, "split은 val, test, train 중에서 고르세요.")
    task = catalog.dataset_task(dataset)
    if model.task and not task_names.same(model.task, task):
        short = task_names.short
        raise ServiceError(400, f"이 모델은 {short(model.task)}용이고 이 데이터셋은 {short(task)}입니다.")
    for each in wanted:
        if not (((release.stats or {}).get("splits") or {}).get(each) or {}).get("items"):
            raise ServiceError(400, f"이 버전의 {each}에는 이미지가 없습니다. 다른 split이나 버전을 고르세요.")
    space = db.get(LabelSpace, release.label_space_id)
    ids = {str(c["name"]): int(c["id"]) for c in space.classes}
    missing = [name for name in model.classes or [] if name not in ids]
    if missing:
        raise ServiceError(400, f"이 버전에는 모델의 클래스가 없습니다: {', '.join(missing[:5])}")
    class_ids = [ids[name] for name in model.classes or []]
    holder = training.data_node(db, release.id)
    node, after = holder, None
    if node_id and node_id != holder.id:
        node = _picked(db, node_id)
        if node.id not in transfers.holders(db, release.id):  # the images come to it first
            if not _since(node, EVAL_RECEIVED_SINCE):
                need = ".".join(str(n) for n in EVAL_RECEIVED_SINCE)
                raise ServiceError(400, f"{node.name}에서 다른 서버의 이미지로 평가하려면 {need} 이상이어야 합니다.")
            transfers.check_route(holder, node)
            after = transfers.sync_job(db, user, release.id, node, via).id
    if len(wanted) > 1 and not _since(node, SPLITS_SINCE):  # an older node would evaluate only the first
        need = ".".join(str(n) for n in SPLITS_SINCE)
        raise ServiceError(400, f"{node.name}에서 split 여럿을 한 번에 평가하려면 {need} 이상이어야 합니다.")
    if class_ids != sorted(class_ids) and not _since(node, ORDERED_SINCE):  # the model numbers them otherwise
        need = ".".join(str(n) for n in ORDERED_SINCE)
        raise ServiceError(400, f"클래스 순서가 모델과 다른 버전을 평가하려면 {node.name}이(가) {need} 이상이어야 합니다.")
    gpus = (1 if (node.info or {}).get("gpus") else 0) if gpus is None else gpus
    extra = {"release_id": release.id, "split": wanted[0], "class_ids": class_ids}
    if len(wanted) > 1:
        extra["splits"] = wanted
    shown = {"split": " · ".join(wanted), **({"splits": wanted} if len(wanted) > 1 else {})}
    return _start(
        db, user, model, "eval", node, shown, release_id=release.id, extra=extra, gpus=gpus, via=via, after=after
    )


def _picked(db: Session, node_id: str) -> Node:
    node = db.get(Node, node_id)
    if node is None:
        raise ServiceError(404, "고른 서버를 찾을 수 없습니다. 목록을 새로 고친 뒤 다시 고르세요.")
    if not node.enabled:
        raise ServiceError(400, f"{node.name}은(는) 작업을 받지 않도록 되어 있습니다.")
    return node


def _since(node: Node, version: tuple[int, ...]) -> bool:
    found = training._version(node.agent_version or "")
    return found is not None and found >= version


def _default_node(db: Session, settings: HubSettings, model: ModelVersion, kind: str) -> Node | None:
    """Export and test: the server that trained the model (its environment is there), else one that is up."""
    trained = _training_node(db, model)
    if trained is not None and services.node_status(trained, settings) == "online":
        return trained
    for place in places(db, settings, model.id, kind)["nodes"]:
        if place["fits"]:
            return db.get(Node, place["id"])
    return trained


def places(db: Session, settings: HubSettings, model_id: str, kind: str, release_id: str = "") -> dict[str, Any]:
    """The servers a model's evaluation, export or test can run on: each with whether it can and why not, the one
    it runs on unless another is picked, and for an evaluation whether the version's images must be sent there first."""
    if kind not in ACTIONS:
        raise ServiceError(400, f"작업은 {', '.join(ACTIONS)} 중 하나여야 합니다.")
    model = training.get_model(db, model_id)
    row = code_for(db, model, kind)
    if row is None:
        own = _template_of(db, model)
        raise ServiceError(400, f"{own.title} {own.version}에는 {ACTIONS[kind]} 코드가 없습니다.")
    manifest = registry.manifest_of(row, model.task or "")
    run = db.get(Run, model.run_id) if model.run_id else None
    same = environment_choice(manifest, run)
    trained = _training_node(db, model)
    holder = holders = None
    if kind == "eval":
        release = catalog.get_release(db, release_id)
        holder = training.data_node(db, release.id)
        holders = transfers.holders(db, release.id)
    nodes = list(db.scalars(select(Node).where(Node.enabled.is_(True)).order_by(Node.name)))
    found = []
    for node in nodes:
        why: list[str] = []
        status = services.node_status(node, settings)
        if status != "online":
            why.append("응답 없음")
        gpus = 1 if kind == "eval" and (node.info or {}).get("gpus") else 0
        try:
            env = registry.environment_for_run(db, manifest, same, node, row)
            why += training.fitting([node], env, gpus, registry.TRAINS)[1]
        except ServiceError as e:
            why.append(e.message)
        transfer = kind == "eval" and holders is not None and node.id not in holders
        if transfer and holder is not None:
            if not _since(node, EVAL_RECEIVED_SINCE):
                need = ".".join(str(n) for n in EVAL_RECEIVED_SINCE)
                why.append(f"다른 서버의 이미지로 평가하려면 {need} 이상이어야 합니다")
            else:
                try:
                    transfers.check_route(holder, node)
                except ServiceError as e:
                    why.append(e.message)
        found.append(
            {
                "id": node.id,
                "name": node.name,
                "status": status,
                "gpus": len((node.info or {}).get("gpus") or []),
                "fits": not why,
                "why": " · ".join(dict.fromkeys(why)),
                "transfer": transfer,  # the version's images are sent there first
                "trained_here": trained is not None and node.id == trained.id,
                "holds_images": kind == "eval" and not transfer,
            }
        )
    if kind == "eval":
        default = holder.id if holder is not None else ""
    else:
        default = trained.id if trained is not None and any(f["id"] == trained.id and f["fits"] for f in found) else ""
        default = default or next((f["id"] for f in found if f["fits"]), "")
    found.sort(key=lambda f: (f["id"] != default, not f["fits"], f["name"]))
    return {"kind": kind, "default": default, "nodes": found}


ALIAS_ORDER = ("production", "SOTA", "staging", "challenger")


def eval_problem(db: Session, model: ModelVersion, release: Release) -> str:
    """Why the model cannot evaluate (and so check the labels of) this version, or "" when it can: a model of
    another task, classes the version does not have, no eval code. The classes are matched by name, in any
    order (the evaluation numbers them as the model does)."""
    task = catalog.release_task(db, release)
    if not task_names.same(model.task or "detect", task):
        return f"{task_names.short(model.task)} 모델이라 {task_names.short(task)} 데이터셋에는 쓸 수 없습니다"
    space = db.get(LabelSpace, release.label_space_id)
    ids = {str(c["name"]): int(c["id"]) for c in (space.classes if space else [])}
    missing = [name for name in model.classes or [] if name not in ids]
    if missing:
        more = f" 외 {len(missing) - 3}개" if len(missing) > 3 else ""
        return f"이 버전에 없는 클래스가 있습니다: {', '.join(missing[:3])}{more}"
    if "eval" not in actions(db, model):
        return "이 모델의 템플릿에는 평가 코드가 없습니다"
    return ""


def _trained_on(db: Session, model: ModelVersion, dataset: Dataset, target: Release | None) -> dict[str, Any]:
    """What the model learned from, against the version being checked: the same version, another version of
    the same dataset, another dataset of the project, another project; and the word of warning that goes with
    it (a model of other data counts more mistakes that are its own, not the labels')."""
    from .catalog_models import Project

    if (model.source or {}).get("kind") == "upload":
        return {
            "relation": "uploaded",
            "trained": {"dataset": "", "version": "", "project": ""},
            "warning": "밖에서 학습해 올린 모델입니다. 클래스 이름은 같아도 무엇을 그 클래스로 라벨했는지나 "
            "촬영 환경이 다르면, 모델이 더 찾거나 놓친 것이 라벨의 잘못이 아니라 모델의 잘못일 수 있어 "
            "미라벨 · 오라벨이 실제보다 많이 나옵니다. 걸린 이미지를 직접 보고 판단하세요.",
        }
    run_release = db.get(Release, model.release_id) if model.release_id else None
    run_dataset = db.get(Dataset, run_release.dataset_id) if run_release else None
    project = db.get(Project, run_dataset.project_id) if run_dataset else None
    where = {
        "dataset": run_dataset.name if run_dataset else "",
        "version": run_release.version if run_release else "",
        "project": project.name if project else "",
    }
    if run_release is not None and target is not None and run_release.id == target.id:
        return {"relation": "same_version", "trained": where, "warning": ""}
    if run_dataset is not None and run_dataset.id == dataset.id:
        return {
            "relation": "same_dataset",
            "trained": where,
            "warning": f"이 데이터셋의 다른 버전({where['version']})으로 학습한 모델입니다. 두 버전에 같은 이미지가 있으면 "
            "모델이 그 라벨을 외웠을 수 있어 틀린 라벨을 덜 찾습니다.",
        }
    near = run_dataset is not None and run_dataset.project_id == dataset.project_id
    other = "다른 데이터셋" if near else "다른 프로젝트의 데이터"
    named = f"({where['project']} · {where['dataset']} {where['version']})" if run_dataset else ""
    return {
        "relation": "same_project" if near else "other_project",
        "trained": where,
        "warning": f"{other}{named}로 학습한 모델입니다. 클래스 이름은 같아도 무엇을 그 클래스로 라벨했는지나 "
        "촬영 환경이 다르면, 모델이 더 찾거나 놓친 것이 라벨의 잘못이 아니라 모델의 잘못일 수 있어 미라벨 · 오라벨이 "
        "실제보다 많이 나옵니다. 걸린 이미지를 직접 보고 판단하세요.",
    }


RELATION_ORDER = ("same_version", "same_dataset", "same_project", "other_project", "uploaded")


def eval_choices(db: Session, release: Release) -> dict[str, Any]:
    """Models for checking this version's labels: this project's, and those of other projects with the same task
    and classes (by name). The ones that fit first — trained on this version, on another version of the dataset,
    on another dataset of the project, in another project — then an alias, then the best; each with why it does
    not fit when it does not (other projects' are listed only when they fit) and a warning when it learned from
    other data. A work in progress is evaluated as its base version."""
    from .catalog_models import Project

    dataset = db.get(Dataset, release.dataset_id)
    target = drafts.latest_ready(db, dataset.id) if release.status == "draft" else release
    splits: dict[str, int] = {}
    for name, part in ((target.stats or {}).get("splits") or {}).items() if target else []:
        key = name or "train"  # images without a split are trained on
        splits[key] = splits.get(key, 0) + int(part.get("items") or 0)
    rows = []
    gone = select(Project.id).where(Project.deleted_at.is_not(None))
    task = catalog.release_task(db, target or release)
    for model in db.scalars(
        select(ModelVersion)
        .where(ModelVersion.deleted_at.is_(None), ModelVersion.project_id.not_in(gone))
        .order_by(ModelVersion.created_at.desc())
    ):
        here = model.project_id == dataset.project_id
        if not here and not task_names.same(model.task or "detect", task):
            continue
        why = eval_problem(db, model, target or release)
        if why and not here:
            continue  # another project's model that cannot: not worth a line
        metric = training.PRIMARY_METRIC.get(model.task or "detect", "map50_95")
        value = (model.metrics or {}).get(metric)
        rows.append(
            {
                **checks.model_ref(model),
                "arch": model.arch,
                "metric": metric,
                "value": value if isinstance(value, int | float) else None,
                "created_at": iso(model.created_at),
                "trained_on": model.release_id == (target.id if target else None),  # its train split was learned
                "fits": not why,
                "why": why,
                # what it learned from, and a word of warning when that is not this dataset
                **_trained_on(db, model, dataset, target),
                "classes": len(model.classes or []),
            }
        )

    def rank(row: dict[str, Any]) -> tuple:
        known = [ALIAS_ORDER.index(a) for a in row["aliases"] if a in ALIAS_ORDER]
        near = RELATION_ORDER.index(row["relation"])
        return (not row["fits"], near, min(known) if known else len(ALIAS_ORDER), -(row["value"] or -1))

    rows.sort(key=rank)
    return {
        "release_id": release.id,
        # where the evaluation runs: the version itself, or a work in progress's base
        "evaluate": {"id": target.id, "version": target.version} if target else None,
        "task": catalog.dataset_task(dataset),
        "splits": splits,
        "models": rows,
    }


def start_export(
    db: Session,
    user: User,
    model_id: str,
    fmt: str = "onnx",
    via: str = "api",
    node_id: str = "",
    settings: HubSettings | None = None,
) -> ModelJob:
    """Export a model: on ``node_id``, else the server that trained it (or another that is up)."""
    model = training.get_model(db, model_id)
    if fmt not in EXPORT_FORMATS:
        raise ServiceError(400, f"형식은 {', '.join(EXPORT_FORMATS)} 중 하나여야 합니다.")
    node = _picked(db, node_id) if node_id else _place(db, settings, model, "export")
    return _start(db, user, model, "export", node, {"format": fmt}, via=via)


def _place(db: Session, settings: HubSettings | None, model: ModelVersion, kind: str) -> Node | None:
    return _default_node(db, settings, model, kind) if settings is not None else _training_node(db, model)


def start_predict(
    db: Session,
    user: User,
    model_id: str,
    images: list[dict[str, Any]],
    storage: Storage,
    conf: float = 0.25,
    via: str = "api",
    node_id: str = "",
    settings: HubSettings | None = None,
) -> ModelJob:
    """Run the model on a few uploaded images and keep what it found (on ``node_id``, else as ``start_export``)."""
    model = training.get_model(db, model_id)
    if not 1 <= len(images) <= MAX_TEST_IMAGES:
        raise ServiceError(400, f"이미지는 1~{MAX_TEST_IMAGES}장을 올려 주세요.")
    inputs = []
    for n, image in enumerate(images):
        blob, name = str(image.get("blob", "")), Path(str(image.get("name") or f"image{n}.jpg")).name
        if not blob or not storage.has_blob(blob):
            raise ServiceError(400, f"올린 이미지를 찾을 수 없습니다: {name}")
        inputs.append({"name": f"{n:02d}_{name}", "blob": blob})
    if not 0 <= conf <= 1:
        raise ServiceError(400, "conf는 0~1이어야 합니다.")
    node = _picked(db, node_id) if node_id else _place(db, settings, model, "predict")
    return _start(db, user, model, "predict", node, {"conf": conf, "images": inputs}, via=via)


# ---- node protocol -----------------------------------------------------------------


def _running(db: Session, node: Node, job_id: str) -> tuple[ModelJob, ModelVersion]:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.type != "model.run" or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 모델 작업이 아닙니다.")
    op = db.scalar(select(ModelJob).where(ModelJob.job_id == job.id))
    model = db.get(ModelVersion, op.model_id) if op else None
    if op is None or model is None:
        raise ServiceError(404, "이 작업에 연결된 모델이 없습니다.")
    return op, model


def manifest(db: Session, node: Node, job_id: str, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    op, _ = _running(db, node, job_id)
    if op.kind == "autolabel":
        from . import autolabel

        return autolabel.manifest(db, node, op, after, limit)
    release = db.get(Release, op.release_id) if op.release_id else None
    if release is None:
        raise ServiceError(404, "이 작업은 버전을 쓰지 않습니다.")
    return training.release_manifest(db, node, release, after, limit)


def blob_for(db: Session, node: Node, job_id: str, what: str, index: int = 0) -> str:
    """The storage key of something the node needs: the model's weights, the template's code, an input image."""
    op, model = _running(db, node, job_id)
    if what == "weights":
        return model.blob
    if what == "template":
        return _template_of(db, model).blob or ""
    images = (op.params or {}).get("images") or []
    if not 0 <= index < len(images):
        raise ServiceError(404, "그런 입력 이미지가 없습니다.")
    return images[index]["blob"]


def _drawn(name: str, media: str, picture: bool, blob: str, storage: Storage | None) -> bool:
    """A picture an evaluation drew: an image (Ultralytics' plots, sample mosaics), a table's picture, or a table of
    pictures (the images it got most wrong, with their boxes)."""
    if picture or media == "table-image":
        return True
    if media != "table" or storage is None:
        return False
    try:
        import json

        rows = json.loads(storage.get_blob(blob)).get("rows") or []
    except (OSError, ValueError, AttributeError):
        return False
    return any(isinstance(cell, dict) and "image" in cell for row in rows for cell in row)


def add_file(
    db: Session,
    node: Node,
    job_id: str,
    name: str,
    blob: str,
    size: int,
    media: str = "",
    caption: str = "",
    storage: Storage | None = None,
) -> dict[str, Any]:
    """A file the job recorded. ``media``: ``table`` (bm.log_table), ``table-image`` (a picture a table draws on)
    or ``predictions`` (bm.log_predictions: every evaluated image's predictions, stored per image)."""
    op, _ = _running(db, node, job_id)
    if media == "predictions":
        kept = op.kind in ("eval", "autolabel")  # per image, by image: a test's few uploads are not
        matched = checks.ingest(db, op, storage.get_blob(blob)) if storage is not None and kept else 0
        if kept:  # every split's images so far (an evaluation of several uploads once per split)
            db.flush()
            total = db.scalar(select(func.count()).select_from(ItemPrediction).where(ItemPrediction.op_id == op.id))
            matched = max(matched, int(total or 0))
        files = [f for f in (op.files or []) if f.get("kind") != "predictions"]
        entry = {"name": "predictions.json", "blob": blob, "bytes": size, "kind": "predictions", "images": matched}
        op.files = [*files, entry]
        return {"name": "predictions.json", "images": matched}
    name = training.clean_file_name(name)
    picture = name.lower().endswith(IMAGE_SUFFIXES)
    if op.kind == "eval" and _drawn(name, media, picture, blob, storage):
        # An evaluation keeps its numbers, per-image results and predictions, not pictures: what it found on each
        # image is seen on 검수 (the dataset's own image under its predictions). The upload is not recorded, and the
        # cleanup removes it as nothing's.
        return {"name": name, "kept": False}
    kind = "image" if picture else "file"
    if media == "table" or (media == "table-image" and picture):
        kind = media
    entry = {"name": name, "blob": blob, "bytes": size, "kind": kind, "caption": caption[:255]}
    files = [f for f in (op.files or []) if f["name"] != name]
    if len(files) >= MAX_RESULT_ITEMS:
        raise ServiceError(400, "기록할 수 있는 파일 수를 넘었습니다.")
    op.files = [*files, entry]
    return {"name": name}


def _named_as_model(result: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    """Per-class results named by the model's classes: uploaded weights may carry other names than the yaml that
    came with them, and the code reports the weights' own."""

    def renamed(rows: Any) -> Any:
        if not isinstance(rows, list):
            return rows
        out = []
        for row in rows:
            n = row.get("id") if isinstance(row, dict) else None
            out.append({**row, "name": classes[n]} if isinstance(n, int) and 0 <= n < len(classes) else row)
        return out

    found = {**result, "classes": renamed(result.get("classes"))} if "classes" in result else dict(result)
    splits = found.get("splits")
    if isinstance(splits, dict):
        found["splits"] = {
            k: ({**v, "classes": renamed(v.get("classes"))} if isinstance(v, dict) and "classes" in v else v)
            for k, v in splits.items()
        }
    return found


def finish(db: Session, node: Node, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
    op, model = _running(db, node, job_id)
    if (model.source or {}).get("renamed") and op.kind == "eval":
        result = _named_as_model(result, list(model.classes or []))
    predictions = result.get("predictions")
    if isinstance(predictions, list):
        result = {**result, "predictions": predictions[:MAX_RESULT_ITEMS]}
    if op.kind == "autolabel":
        from . import autolabel

        result = {**result, "autolabel": autolabel.apply(db, op)}
    op.result = result
    return {"ok": True}


# ---- views -------------------------------------------------------------------------


def get_op(db: Session, op_id: str) -> ModelJob:
    op = db.get(ModelJob, op_id)
    if op is None:
        raise ServiceError(404, "기록을 찾을 수 없습니다.")
    return op


def _code_of(job: Job | None) -> dict[str, Any]:
    ran = ((job.params or {}).get("template") or {}) if job is not None else {}
    found = {k: ran[k] for k in ("name", "version", "trained_with") if ran.get(k)}
    return found


def op_view(db: Session, settings: HubSettings, op: ModelJob, node_names: dict[str, str] | None = None) -> dict[str, Any]:
    job = db.get(Job, op.job_id) if op.job_id else None
    data = services.job_view(db, settings, job, node_names) if job else None
    listed = [(f, catalog.media_url(settings, "op", f"{op.id}-{n}")) for n, f in enumerate(op.files or [])]
    release = db.get(Release, op.release_id) if op.release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    return {
        "id": op.id,
        "kind": op.kind,
        "status": data["status"] if data else "failed",
        "error": data["error"] if data else None,
        "wait_reason": data["wait_reason"] if data else None,
        "duration_s": data["duration_s"] if data else None,
        "node_name": data["node_name"] if data else None,
        "job_id": op.job_id,
        "created_at": iso(op.created_at),
        "release": {"id": release.id, "dataset": dataset.name, "version": release.version} if release and dataset else None,
        "params": {k: v for k, v in (op.params or {}).items() if k != "images"},
        # which template version's code ran it, and whether that is a newer one than the model's own
        "code": _code_of(job),
        "result": op.result or {},
        "files": [
            {**{k: f[k] for k in ("name", "kind", "bytes")}, "url": url}
            for f, url in listed
            if f["kind"] not in ("table", "table-image", "predictions")
        ],
        # images whose predictions were kept: the dataset screen can sort them by likely wrong labels
        "predicted": sum(int(f.get("images") or 0) for f in op.files or [] if f["kind"] == "predictions"),
        # Tables the code recorded (per-image results), and the pictures their cells draw on
        "tables": [
            {"name": f["name"], "kind": "table", "caption": f.get("caption", ""), "url": url}
            for f, url in listed
            if f["kind"] == "table"
        ],
        "table_images": {f["name"]: url for f, url in listed if f["kind"] == "table-image"},
    }


def _model_ref(model: ModelVersion | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "task": model.task or "detect",
        "aliases": training._aliases(model),
        "project_id": model.project_id,
        "deleted": model.deleted_at is not None,
    }


def _splits_of(op: ModelJob) -> list[str]:
    params = op.params or {}
    return [str(s) for s in params.get("splits") or [params.get("split") or "val"]]


def op_detail(db: Session, settings: HubSettings, op_id: str) -> dict[str, Any]:
    """One evaluation (export, test) on its own screen: what ran where and how long, its numbers per class and per
    split, and what the job said (``log_count`` lines)."""
    op = get_op(db, op_id)
    job = db.get(Job, op.job_id) if op.job_id else None
    data = services.job_view(db, settings, job) if job else {}
    result = op.result or {}
    return {
        **op_view(db, settings, op),
        "model": _model_ref(db.get(ModelVersion, op.model_id)),
        "splits": _splits_of(op) if op.kind == "eval" else [],
        "split_metrics": result.get("splits") or {},  # several splits: each one's numbers
        "class_metrics": result.get("classes") or [],  # per class (of the first split when there are several)
        "node_id": data.get("node_id"),
        "started_at": data.get("started_at"),
        "finished_at": iso(job.finished_at) if job and job.finished_at else None,
        "log_count": job.log_count if job else 0,
        "gpus": data.get("gpus_requested"),
    }


def rerun(db: Session, user: User, settings: HubSettings, op_id: str, node_id: str = "", via: str = "api") -> ModelJob:
    """The same evaluation (export, test) again: same model, version, splits and settings, on the server it ran on
    unless another is picked. For one that was stopped or failed, or to measure again."""
    op = get_op(db, op_id)
    job = db.get(Job, op.job_id) if op.job_id else None
    if job is not None and job.status in ("queued", "running"):
        raise ServiceError(409, "아직 돌고 있습니다. 끝나거나 중단한 뒤에 다시 실행하세요.")
    if not node_id and job is not None and job.node_id:  # where it ran, while that server still takes jobs
        before = db.get(Node, job.node_id)
        node_id = before.id if before is not None and before.enabled else ""
    params = op.params or {}
    if op.kind == "eval":
        if not op.release_id:
            raise ServiceError(400, "평가한 버전이 지워져 다시 실행할 수 없습니다.")
        splits = _splits_of(op)
        # as many GPUs as it took (a job that took none keeps no "gpus" in its requirements)
        gpus = int((job.requirements or {}).get("gpus") or 0) if job is not None else None
        return start_eval(
            db, user, op.model_id, op.release_id, splits[0], gpus, via=via, node_id=node_id, splits=splits
        )
    if op.kind == "export":
        fmt = str(params.get("format") or "onnx")
        return start_export(db, user, op.model_id, fmt, via=via, node_id=node_id, settings=settings)
    if op.kind == "predict":
        from .catalog import storage_for

        storage = storage_for(str(settings.storage_dir))
        images = [{"name": i.get("name"), "blob": i.get("blob")} for i in params.get("images") or []]
        conf = float(params.get("conf", 0.25))
        return start_predict(db, user, op.model_id, images, storage, conf, via=via, node_id=node_id, settings=settings)
    raise ServiceError(400, "이 작업은 여기서 다시 실행할 수 없습니다.")


def delete_op(db: Session, user: User, op_id: str, via: str = "api") -> None:
    """Remove an evaluation (export, test) and what it kept: its numbers, files and per-image predictions (the 검수
    basis and grid overlays made of it go with it; a 검수 record already kept with a version stays). Its starter or an
    admin; not while it runs."""
    from sqlalchemy import delete

    from .catalog_models import ItemPrediction

    op = get_op(db, op_id)
    if op.kind not in ACTIONS:
        raise ServiceError(400, "오토라벨 기록은 라벨링 묶음에서 다룹니다.")
    if user.role != "admin" and op.created_by != user.id:
        raise ServiceError(403, "실행한 사람이나 관리자만 지울 수 있습니다.")
    job = db.get(Job, op.job_id) if op.job_id else None
    if job is not None and job.status in ("queued", "running"):
        raise ServiceError(409, "돌고 있습니다. 중단한 뒤에 지우세요.")
    model = db.get(ModelVersion, op.model_id)
    db.execute(delete(ItemPrediction).where(ItemPrediction.op_id == op.id))
    services.audit(
        db, user.id, via, f"model.{op.kind}.delete", op.id, model=f"{model.name} v{model.version}" if model else "",
        release=op.release_id or "",
    )
    db.delete(op)  # its files' blobs go with the next storage cleanup, as nothing names them any more


def find_models(db: Session, q: str = "", limit: int = 30) -> list[dict[str, Any]]:
    """Models of every project (not in the trash), newest first, for picking ones to compare: by model or project name."""
    from .catalog_models import Project

    query = (
        select(ModelVersion, Project)
        .join(Project, Project.id == ModelVersion.project_id)
        .where(ModelVersion.deleted_at.is_(None), Project.deleted_at.is_(None))
        .order_by(ModelVersion.created_at.desc())
    )
    needle = q.strip().lower()
    found = []
    for model, project in db.execute(query):
        if needle and needle not in f"{model.name} v{model.version} {project.name}".lower():
            continue
        metric = training.PRIMARY_METRIC.get(model.task or "detect", "map50_95")
        found.append(
            {
                **_model_ref(model),
                "project": {"id": project.id, "name": project.name},
                "metric": metric,
                "value": (model.metrics or {}).get(metric),
            }
        )
        if len(found) >= limit:
            break
    return found


def compare(db: Session, settings: HubSettings, ids: list[str]) -> dict[str, Any]:
    """Models from any project side by side: each one's own validation (on the data it was trained with, which may
    differ), and every version · split two or more of them were evaluated on (the same data: a fair comparison)."""
    from .catalog_models import Project

    models = [training.get_model(db, model_id) for model_id in list(dict.fromkeys(ids))[:6]]
    projects = {p.id: p.name for p in db.scalars(select(Project).where(Project.id.in_({m.project_id for m in models})))}
    listed = []
    for model in models:
        view = training.model_view(db, settings, model)
        listed.append(
            {
                **_model_ref(model),
                "project": {"id": model.project_id, "name": projects.get(model.project_id, "")},
                "metrics": model.metrics or {},
                "class_metrics": model.class_metrics or [],
                "release": view.get("release"),  # the version it was trained and validated on
            }
        )
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.model_id.in_([m.id for m in models]), ModelJob.release_id.is_not(None))
        .order_by(ModelJob.created_at.desc())
    ).all()
    jobs = {j.id: j for j in db.scalars(select(Job).where(Job.id.in_([op.job_id for op in ops if op.job_id])))}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for op in ops:
        job = jobs.get(op.job_id or "")
        result = op.result or {}
        if job is None or job.status != "succeeded" or not result.get("metrics"):
            continue
        splits = _splits_of(op)
        for n, each in enumerate(splits):
            numbers = (result.get("splits") or {}).get(each) if len(splits) > 1 else result.get("metrics")
            if not numbers:
                continue
            group = groups.setdefault((op.release_id, each), {"release_id": op.release_id, "split": each, "rows": {}})
            if op.model_id in group["rows"]:
                continue  # each model's newest on this data
            group["rows"][op.model_id] = {
                "op_id": op.id,
                "metrics": numbers,
                "class_metrics": (result.get("classes") or []) if n == 0 else [],
                "created_at": iso(op.created_at),
            }
    shown = []
    for group in groups.values():
        release = db.get(Release, group["release_id"])
        dataset = db.get(Dataset, release.dataset_id) if release else None
        if release is None or dataset is None:
            continue
        shown.append(
            {
                "release": {
                    "id": release.id,
                    "dataset": dataset.name,
                    "version": release.version,
                    "project": {"id": dataset.project_id, "name": db.get(Project, dataset.project_id).name},
                    "deleted": release.deleted_at is not None,
                },
                "split": group["split"],
                "rows": group["rows"],
            }
        )
    shown.sort(key=lambda g: (-len(g["rows"]), g["release"]["project"]["name"], g["release"]["dataset"], g["split"]))
    return {"models": listed, "evaluations": shown}


def release_evaluations(db: Session, settings: HubSettings, release_id: str, split: str = "") -> dict[str, Any]:
    """Every model evaluated on this version (on ``split``, or each split there is), newest first: the board to
    compare models on the same data. ``latest`` marks each model's newest evaluation on a split."""
    release = catalog.get_release(db, release_id)
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.release_id == release.id)
        .order_by(ModelJob.created_at.desc())
    ).all()
    jobs = {j.id: j for j in db.scalars(select(Job).where(Job.id.in_([op.job_id for op in ops if op.job_id])))}
    models = {m.id: m for m in db.scalars(select(ModelVersion).where(ModelVersion.id.in_({op.model_id for op in ops})))}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    found_splits: list[str] = []
    for op in ops:
        job = jobs.get(op.job_id or "")
        result = op.result or {}
        if job is None or job.status != "succeeded" or not result.get("metrics"):
            continue
        splits = _splits_of(op)
        per_split = result.get("splits") or {}
        for n, each in enumerate(splits):
            if each not in found_splits:
                found_splits.append(each)
            if split and each != split:
                continue
            numbers = per_split.get(each) if len(splits) > 1 else result.get("metrics")
            if not numbers:
                continue
            model = models.get(op.model_id)
            key = (op.model_id, each)
            rows.append(
                {
                    "op_id": op.id,
                    "model": _model_ref(model),
                    "split": each,
                    "metrics": numbers,
                    "class_metrics": (result.get("classes") or []) if n == 0 else [],
                    "created_at": iso(op.created_at),
                    "duration_s": (job.finished_at - job.started_at).total_seconds()
                    if job.finished_at and job.started_at
                    else None,
                    "latest": key not in seen,
                }
            )
            seen.add(key)
    dataset = db.get(Dataset, release.dataset_id)
    order = {"val": 0, "test": 1, "train": 2}
    return {
        "release": {"id": release.id, "dataset": dataset.name if dataset else "", "version": release.version},
        "task": catalog.dataset_task(dataset) if dataset else "detect",
        "splits": sorted(found_splits, key=lambda s: order.get(s, 9)),
        "rows": rows,
    }


def list_ops(db: Session, settings: HubSettings, model_id: str, limit: int = 50) -> list[dict[str, Any]]:
    training.get_model(db, model_id)
    ops = db.scalars(
        select(ModelJob).where(ModelJob.model_id == model_id).order_by(ModelJob.created_at.desc()).limit(limit)
    ).all()
    names = services.node_name_map(db)
    return [op_view(db, settings, op, names) for op in ops]


def file_of(db: Session, key: str) -> dict[str, Any]:
    """``<op id>-<index>`` → the recorded file."""
    op_id, _, index = key.rpartition("-")
    op = get_op(db, op_id)
    files = op.files or []
    if not index.isdigit() or int(index) >= len(files):
        raise ServiceError(404, "파일을 찾을 수 없습니다.")
    return files[int(index)]
