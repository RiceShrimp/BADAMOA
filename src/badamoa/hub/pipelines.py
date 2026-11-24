"""Pipelines (Phase 5 · Automate): steps the Hub runs one after another, each when the one before succeeded.

A step is something the Hub already does, started as a person would start it, by the person who started the run.
A run follows a version (the current one) and, once a step has made one, a model (the current one):

    import     the dataset's folder on a server imported again as a new version (files it holds unchanged are not
               read again) → the current version
    check      a gate on 검수's rule checks: the current version's flagged images at most ``max_flagged``
    sync       the current version's images sent to a server ahead of a training there
    autolabel  the current model (or the one holding an alias) labels the version's images nobody labelled yet,
               in a labeling set of them (오토라벨, Closed-Voca)
    label      people finish a labeling set: the one 오토라벨 made, else a new one of the images to label
    confirm    the work in progress (작업 중) confirmed as a new version → the current version
    train      a training run on the current version → the model it registers
    sweep      a hyperparameter search on the current version → the model of its best run
    eval       the model measured on a version's split; with ``review`` also every split of the current version,
               so 검수 sorts that version's images by likely wrong labels (its model check)
    promote    a gate: an alias (SOTA) moves to the evaluated model only if it beats the version holding it, both
               measured on the same version and split; otherwise the alias stays
    export     the model exported (ONNX)

Steps run in the order they are given. ``train``, ``sweep`` and ``eval`` become gates too with ``at_least`` (the
model's number must reach it). A gate that does not pass ends the run ``held``; a step that fails stops it and
says why; the steps after either are skipped. A run is started by a person, by a new version of the dataset
(``new_version``; not one the pipeline made itself), or by a new training run ("학습 끝나면"). The Hub's background
loop advances runs (``advance_all``): soon after a job ends, otherwise every 15 s.

Templates keep steps to start pipelines from: the built-in ones (``BUILTIN_TEMPLATES``) and the ones people save.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from datetime import datetime
from pathlib import PurePath
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
from badamoa.core.protocol import TERMINAL_STATUSES

from . import asks, autolabel, catalog, drafts, labeling, modelops, qa, registry, services, sweeps, training, transfers
from .catalog_models import Dataset, ItemPrediction, LabelSet, Project, Release
from .config import HubSettings
from .db import iso, utcnow, write_lock
from .models import Job, Node, User
from .services import ServiceError
from .train_models import ModelJob, ModelVersion, Pipeline, PipelineRun, PipelineTemplate, Run, Sweep, Template

log = logging.getLogger("badamoa.hub.pipelines")

# ``check`` (검수 확인, a gate on the rule checks) is no longer offered for new steps (0.104.0); saved ones still run.
KINDS = (
    "import", "check", "review", "sync", "autolabel", "label", "confirm", "train", "sweep", "eval", "promote", "export"
)  # fmt: skip
LABEL = {
    "import": "가져오기",
    "check": "검수 확인",
    "review": "검수",
    "sync": "서버로 보내기",
    "autolabel": "오토라벨",
    "label": "라벨링",
    "confirm": "버전 확정",
    "train": "학습",
    "sweep": "탐색",
    "eval": "평가",
    "promote": "승격",
    "export": "내보내기",
}
MAKES_MODEL = ("train", "sweep")
USES_MODEL = ("autolabel", "review", "eval", "export")
# How a labeling set is split (as 라벨링 묶음 만들기 does): images to a part, or all of it to one person.
DIVIDE = ("size", "one")
UNLABELLED = ("auto", "background", "unlabelled")  # 가져오기: an image without a label file
MAX_VERSION = 64
GATES = ("check", "promote")  # always; train · sweep · eval with ``at_least``
DEFAULT_ALIAS = "SOTA"
TRIGGERS = ("manual", "new_version")
SPLITS = ("auto", "val", "test", "train")
CHECK_GROUPS = {"any": "문제", "missing": "라벨 빠짐", "wrong": "틀린 라벨", "boxes": "박스 문제"}
GOING = ("pending", "running")
ENDED = {"failed": "failed", "stopped": "stopped", "held": "held"}  # a step's end → the run's
MAX_NAME = 160
MAX_STEPS = 20
MAX_RUNS_SHOWN = 50
# What a training step may carry: the new-run dialog's settings, without the version (the run's own)
TRAIN_KEYS = (
    "template", "environment", "params", "name", "model_name", "classes", "unlabelled", "val_fraction", "gpus",
    "gpu_indices", "min_vram_mb", "base_model_id", "tags", "note", "nodes", "data_after", "embed",
)  # fmt: skip
SWEEP_KEYS = (
    "template", "environment", "params", "space", "method", "metric", "goal", "max_runs", "parallel", "classes",
    "gpus", "nodes",
)  # fmt: skip
METRIC_LABEL = {
    "map50_95": "mAP50-95",
    "map50": "mAP50",
    "mask_map50_95": "mask mAP50-95",
    "pose_map50_95": "pose mAP50-95",
    "top1": "Top-1",
    "top5": "Top-5",
}

# The ways to start from. Blanks (the training's model kind, the model a SOTA check looks at, the settings a search
# tries) are filled in when a pipeline is made from one.
BUILTIN_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "builtin-train",
        "name": "학습 → 평가 → 승격",
        "description": "고른 버전으로 학습하고, 같은 데이터로 지금 SOTA와 비교해 더 나을 때만 SOTA로 지정합니다. "
        "평가는 검수의 모델 검사로도 쓰입니다.",
        "trigger": "manual",
        "steps": [
            {"kind": "train", "config": {}},
            {"kind": "eval", "config": {"split": "auto", "review": True}},
            {"kind": "promote", "config": {"alias": DEFAULT_ALIAS}},
            {"kind": "export", "config": {"format": "onnx"}},
        ],
    },
    {
        "id": "builtin-retrain",
        "name": "새 버전마다 다시 학습",
        "description": "데이터셋에 새 버전이 생기면(가져오기 · 작업 중 확정) 같은 설정으로 다시 학습하고, "
        "나을 때만 SOTA를 옮깁니다.",
        "trigger": "new_version",
        "steps": [
            {"kind": "train", "config": {}},
            {"kind": "eval", "config": {"split": "auto", "review": True}},
            {"kind": "promote", "config": {"alias": DEFAULT_ALIAS}},
        ],
    },
    {
        "id": "builtin-sota",
        "name": "새 버전마다 SOTA 평가",
        "description": "새 버전이 생기면 지금 SOTA를 그 버전에서 평가합니다. 지난 버전보다 얼마나 달라졌는지 보고, "
        "검수가 그 버전의 틀린 라벨 후보를 바로 찾습니다.",
        "trigger": "new_version",
        "steps": [
            {"kind": "eval", "config": {"model": {"name": "", "alias": DEFAULT_ALIAS}, "split": "auto", "review": True}}
        ],
    },
    {
        "id": "builtin-labeling",
        "name": "오토라벨 → 라벨링 → 새 버전 → 학습",
        "description": "SOTA가 라벨 안 한 이미지에 먼저 그리고, 사람이 라벨링 묶음에서 고쳐 끝내면 "
        "작업 중을 새 버전으로 확정해 다시 학습하고, 나을 때만 SOTA를 옮깁니다.",
        "trigger": "manual",
        "steps": [
            {"kind": "autolabel", "config": {"model": {"name": "", "alias": DEFAULT_ALIAS}}},
            {"kind": "label", "config": {}},
            {"kind": "confirm", "config": {}},
            {"kind": "train", "config": {}},
            {"kind": "eval", "config": {"split": "auto", "review": True}},
            {"kind": "promote", "config": {"alias": DEFAULT_ALIAS}},
        ],
    },
    {
        "id": "builtin-review",
        "name": "새 버전마다 SOTA로 검수",
        "description": "새 버전이 생기면 지금 SOTA가 그 버전의 모든 이미지를 예측해, 검수에 라벨이 틀렸을 수 있는 "
        "이미지를 올립니다(검수의 모델로 검사하기와 같음).",
        "trigger": "new_version",
        "steps": [{"kind": "review", "config": {"model": {"name": "", "alias": DEFAULT_ALIAS}}}],
    },
    {
        "id": "builtin-sweep",
        "name": "탐색 → 최고 모델 평가 → 내보내기",
        "description": "하이퍼파라미터 탐색이 끝나면 가장 좋은 실행의 모델을 평가해 나을 때만 SOTA로 지정하고 내보냅니다.",
        "trigger": "manual",
        "steps": [
            {"kind": "sweep", "config": {"space": {}, "method": "random", "max_runs": 6}},
            {"kind": "eval", "config": {"split": "auto"}},
            {"kind": "promote", "config": {"alias": DEFAULT_ALIAS}},
            {"kind": "export", "config": {"format": "onnx"}},
        ],
    },
]

# One advancing pass at a time (two would start the same step twice); requests that start or stop a run hold it
# until they have committed, as hyperparameter searches do.
lock = threading.Lock()
_dirty = threading.Event()


def wake() -> None:
    _dirty.set()


def pending() -> bool:
    return _dirty.is_set()


def on_job_finished(job: Job) -> None:
    if job.type in ("train.run", "model.run", "data.sync", "data.import"):
        wake()


# ---- what a pipeline is made of -----------------------------------------------------------


def _has(release: Release, split: str) -> bool:
    return bool((((release.stats or {}).get("splits") or {}).get(split) or {}).get("items"))


def _split_of(release: Release, wanted: str) -> str:
    """``auto``: the version's test split when it has one (data no model trained on), else val."""
    if wanted != "auto":
        return wanted
    return "test" if _has(release, "test") else "val"


def _check_alias(alias: str) -> str:
    alias = alias.strip()
    if not training.ALIAS_RE.match(alias) or alias.lower() == "latest" or re.fullmatch(r"[vV]\d+", alias):
        raise ServiceError(400, "alias는 32자 이내의 글자·숫자·._- 로 짓고, latest나 v1 같은 버전 표기는 쓸 수 없습니다.")
    return alias


def _number(value: Any, what: str, low: float | None = None, high: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ServiceError(400, f"{what}은(는) 숫자입니다.") from e
    if (low is not None and number < low) or (high is not None and number > high):
        span = f"{low if low is not None else ''}~{high if high is not None else ''}"
        raise ServiceError(400, f"{what}은(는) {span} 사이입니다.")
    return number


def _gate_of(config: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """``at_least``: the model's number must reach it to go on (a gate); ``metric``: which (else the task's main)."""
    if config.get("at_least") not in (None, ""):
        out["at_least"] = _number(config["at_least"], "기준값")
    if str(config.get("metric") or "").strip():
        out["metric"] = str(config["metric"]).strip()
    return out


def _gpus_of(config: dict[str, Any], out: dict[str, Any]) -> None:
    """GPUs a model step takes; left out: as many as the training took (none: one when the server has any)."""
    if config.get("gpus") is not None:
        if not isinstance(config["gpus"], int) or config["gpus"] < 0:
            raise ServiceError(400, "gpus는 0(CPU) 이상의 정수입니다.")
        out["gpus"] = config["gpus"]


def _model_choice(config: dict[str, Any]) -> dict[str, str] | None:
    """A model named by its family and alias (the one holding it when the step runs)."""
    model = config.get("model") or {}
    if not model:
        return None
    name, alias = str(model.get("name") or "").strip(), str(model.get("alias") or "").strip()
    if not name or not alias:
        raise ServiceError(400, f"모델을 고르세요: 모델 이름과 alias(예: {DEFAULT_ALIAS}).")
    return {"name": name, "alias": _check_alias(alias)}


def _node_id(db: Session, config: dict[str, Any], what: str, kind: str = "", dataset: Dataset | None = None) -> str:
    node_id = str(config.get("node_id") or "")
    if not node_id:
        raise ServiceError(400, f"{what}: 서버를 고르세요.")
    node = db.get(Node, node_id)
    if node is None:
        raise ServiceError(404, f"{what}: 고른 서버를 찾을 수 없습니다.")
    why = node_rules(db, node, kind, newest_release(db, dataset.id) if dataset is not None else None)
    if why:
        raise ServiceError(400, f"{what}: {node.name}에서는 할 수 없습니다: {' · '.join(why)}")
    return node_id


def _place_of(db: Session, config: dict[str, Any], out: dict[str, Any], kind: str, dataset: Dataset | None) -> None:
    """The server a model step runs on; left out: where the Hub would put it (the images' server, the model's). One
    its own screen would not offer is refused (``node_rules``)."""
    node_id = str(config.get("node_id") or "")
    if node_id:
        node = db.get(Node, node_id)
        if node is None:
            raise ServiceError(404, "고른 서버를 찾을 수 없습니다.")
        release = newest_release(db, dataset.id) if dataset is not None else None
        why = node_rules(db, node, kind, release)
        if why:
            raise ServiceError(400, f"{node.name}에서는 할 수 없습니다: {' · '.join(why)}")
        out["node_id"] = node_id


# Where a step's work runs, by the role a server needs for it (노드 설정의 학습용 · 오토라벨용).
PLACE_ROLE = {"autolabel": "autolabel", "review": "compute", "eval": "compute", "export": "compute"}  # AUTOLABELS · TRAINS


def node_rules(db: Session, node: Node, kind: str, release: Release | None) -> list[str]:
    """Why a step cannot run on ``node`` by the rules its own screen goes by, whatever the model (the model's own
    — its environment, Docker — are checked when the step starts, as by hand): a server that takes jobs, with the
    role the work needs (학습용 for an evaluation or export, 오토라벨용 for an 오토라벨), a MOA new enough, and when
    the version's images must come to it first, both ends able to send them. ``release``: the version it would work
    on now (None: none yet, so nothing about its images). 가져오기 goes to a data server (데이터 서버용), as 데이터셋
    가져오기 does; 서버로 보내기 to one that trains or auto-labels (where images are received)."""
    why: list[str] = []
    if not node.enabled:
        return ["작업을 받지 않도록 되어 있음"]
    if kind == "import" and not registry.has_role(node, registry.STORES):
        why.append("데이터 서버용 서버가 아님 (노드 설정에서 '데이터 서버용'을 켭니다)")
    if kind == "sync" and not registry.runs_models(node):
        why.append("학습용 · 오토라벨용 서버가 아님 (이미지는 학습하거나 오토라벨할 서버로 보냅니다)")
    role = PLACE_ROLE.get(kind)
    if role == registry.AUTOLABELS and not registry.has_role(node, role):
        why.append("오토라벨용 서버가 아님 (노드 설정에서 '오토라벨용'을 켭니다)")
    elif role == registry.TRAINS and not registry.has_role(node, role):
        why.append("학습용 서버가 아님 (노드 설정에서 '학습용'을 켭니다)")
    if kind == "autolabel" and not asks.answers_questions(node.agent_version or "", autolabel.SINCE):
        why.append(f"MOA {'.'.join(str(n) for n in autolabel.SINCE)} 이상 필요")
    if kind in ("autolabel", "review", "eval", "sync") and release is not None:
        try:
            holder = training.data_node(db, release.id)
        except ServiceError:
            holder = None
        if holder is not None and node.id not in transfers.holders(db, release.id):  # the images come to it first
            since = {"autolabel": autolabel.RECEIVED_SINCE, "sync": transfers.SYNC_SINCE}.get(
                kind, modelops.EVAL_RECEIVED_SINCE
            )
            if not asks.answers_questions(node.agent_version or "", since):
                why.append(f"다른 서버의 이미지를 받아 하려면 MOA {'.'.join(str(n) for n in since)} 이상 필요")
            else:
                try:
                    transfers.check_route(holder, node)
                except ServiceError as e:
                    why.append(e.message)
    return list(dict.fromkeys(why))


def places(
    db: Session, settings: HubSettings, dataset_id: str, kind: str, model_name: str = "", model_alias: str = ""
) -> dict[str, Any]:
    """The servers a step may run on, each with whether it can and why not, as its own screen lists them: 새 오토라벨's
    servers, 평가 · 모델로 검사하기 · 내보내기's (for the model named by its alias, when there is one now), the
    servers a version can go to (서버로 보내기), the ones taking jobs (가져오기). A server not answering now is
    said so but not refused: a pipeline runs later. ``default``: where it runs when 자동 is kept."""
    if kind not in ("import", "sync", *PLACE_ROLE):
        raise ServiceError(400, "서버를 고르는 단계는 가져오기 · 서버로 보내기 · 오토라벨 · 검수 · 평가 · 내보내기입니다.")
    dataset = _dataset_of(db, dataset_id)
    release = newest_release(db, dataset.id)
    model = _holder(db, dataset.project_id, model_name, model_alias) if model_name and model_alias else None
    listed: dict[str, Any] | None = None
    try:
        if kind == "autolabel" and release is not None:
            listed = autolabel._servers(db, settings, release)
        elif kind in ("review", "eval") and model is not None and release is not None:
            listed = modelops.places(db, settings, model.id, "eval", release.id)
        elif kind == "export" and model is not None:
            listed = modelops.places(db, settings, model.id, "export")
    except ServiceError as e:
        listed = {"default": "", "nodes": [], "problem": e.message}
    rows = []
    if listed is not None and listed.get("nodes"):
        for place in listed["nodes"]:
            node = db.get(Node, place["id"])
            reasons = [w for w in str(place.get("why") or "").split(" · ") if w and w != "응답 없음"]
            reasons += [w for w in node_rules(db, node, kind, release) if w not in reasons] if node else []
            rows.append({**place, "fits": not reasons, "why": " · ".join(reasons)})
        default = str(listed.get("default") or "")
    else:
        for node in db.scalars(select(Node).where(Node.enabled.is_(True)).order_by(Node.name)):
            reasons = node_rules(db, node, kind, release)
            held = release is not None and node.id in transfers.holders(db, release.id)
            rows.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "status": services.node_status(node, settings),
                    "gpus": len((node.info or {}).get("gpus") or []),
                    "fits": not reasons,
                    "why": " · ".join(reasons),
                    "transfer": kind in ("autolabel", "review", "eval", "sync") and release is not None and not held,
                    "holds_images": held and kind in ("autolabel", "review", "eval", "sync"),
                }
            )
        default = ""
        if release is not None and kind in ("autolabel", "review", "eval"):
            try:
                default = training.data_node(db, release.id).id
            except ServiceError:
                default = ""
    rows.sort(key=lambda r: (r["id"] != default, not r["fits"], r["name"]))
    return {
        "kind": kind,
        "default": default,
        "nodes": rows,
        "problem": (listed or {}).get("problem", ""),
        "release": {"id": release.id, "version": release.version} if release else None,
        "model": {"id": model.id, "name": model.name, "version": model.version} if model else None,
    }


def _version_pattern(db: Session, dataset: Dataset | None, config: dict[str, Any]) -> str:
    """A step's name for the version it makes: as typed, ``{n}`` the next free number and ``{date}`` the day it runs
    (20261124); empty: the next vN. A name with no ``{n}`` must not be the dataset's already."""
    pattern = str(config.get("version") or "").strip()
    if not pattern:
        return ""
    if len(pattern) > MAX_VERSION:
        raise ServiceError(400, f"버전 이름은 {MAX_VERSION}자까지입니다.")
    if pattern == drafts.DRAFT:
        raise ServiceError(400, f"'{drafts.DRAFT}'은 버전 이름으로 쓸 수 없습니다.")
    rest = pattern.replace("{n}", "").replace("{date}", "")
    if "{" in rest or "}" in rest or pattern.count("{n}") > 1:
        raise ServiceError(400, "버전 이름에는 {n}(다음 번호)과 {date}(그날 날짜)를 한 번씩만 넣을 수 있습니다.")
    if dataset is not None and "{n}" not in pattern:
        name = _version_name(db, dataset.id, pattern)
        if db.scalar(select(Release.id).where(Release.dataset_id == dataset.id, Release.version == name)) is not None:
            raise ServiceError(
                409, f"{dataset.name}에 {name} 버전이 이미 있습니다(휴지통 포함). 다른 이름을 쓰거나 {{n}}을 넣으세요."
            )
    return pattern


def _version_name(db: Session, dataset_id: str, pattern: str) -> str:
    """The name a step's version gets now (``_version_pattern``)."""
    if not pattern:
        return _next_version(db, dataset_id)
    text = pattern.replace("{date}", f"{datetime.now():%Y%m%d}")
    if "{n}" not in text:
        return text
    head, _, tail = text.partition("{n}")
    names = set(db.scalars(select(Release.version).where(Release.dataset_id == dataset_id)))
    shape = re.compile(re.escape(head) + r"(\d+)" + re.escape(tail))
    n = max((int(m.group(1)) for v in names if (m := shape.fullmatch(v or ""))), default=0) + 1
    while f"{head}{n}{tail}" in names:
        n += 1
    return f"{head}{n}{tail}"


def _clean_import(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    path = str(config.get("path") or "").strip()
    if not path:
        raise ServiceError(400, "가져오기: 서버의 폴더 경로를 적으세요.")
    fmt = str(config.get("format") or "auto")
    if fmt not in ("auto", "yolo", "coco", "imagefolder"):
        raise ServiceError(400, "가져오기 형식은 auto, yolo, coco, imagefolder 중 하나입니다.")
    unlabelled = str(config.get("unlabelled") or "auto")
    if unlabelled not in UNLABELLED:
        raise ServiceError(400, "라벨 파일이 없는 이미지는 auto · background · unlabelled 중 하나입니다.")
    out: dict[str, Any] = {
        "node_id": _node_id(db, config, "가져오기", "import"),
        "path": path,
        "format": fmt,
        "full_read": bool(config.get("full_read")),
        "unlabelled": unlabelled,
    }
    classes = config.get("classes") or []
    if isinstance(classes, str):
        classes = classes.replace("\n", ",").split(",")
    if [str(c).strip() for c in classes if str(c).strip()]:
        out["classes"] = [str(c).strip() for c in classes if str(c).strip()]
    _note_of(config, out)
    version = _version_pattern(db, dataset, config)
    if version:
        out["version"] = version
    return out


def _note_of(config: dict[str, Any], out: dict[str, Any]) -> None:
    """The note a version made by a step keeps (가져오기 · 버전 확정, as their dialogs take one)."""
    note = str(config.get("note") or "").strip()
    if note:
        out["note"] = note[:500]


def _set_of(config: dict[str, Any], out: dict[str, Any]) -> None:
    """A labeling set's settings, as 라벨링 묶음 만들기 and 새 오토라벨 take them: its name, images to a part (or all of
    it to one person), whether someone approves the work (검수하기, off unless asked)."""
    divide = str(config.get("divide") or "size")
    if divide not in DIVIDE:
        raise ServiceError(400, "나누기는 size(일감 하나에 몇 장) · one(한 사람에게 모두)입니다.")
    out.update(review=bool(config.get("review", False)), divide=divide)
    if divide == "size":
        out["part_size"] = int(_number(config.get("part_size") or 50, "일감 하나의 이미지 수", 1))
    name = str(config.get("name") or "").strip()
    if name:
        out["name"] = name[:128]


def _clean_review(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    """검수 (모델로 검사하기): the model predicts every image of the version, for 검수's label checks."""
    out: dict[str, Any] = {}
    splits = [str(x) for x in config.get("splits") or []]
    if any(x not in ("train", "val", "test") for x in splits):
        raise ServiceError(400, "평가할 부분은 train · val · test 중에서 고릅니다(비우면 모두).")
    if splits:
        out["splits"] = [x for x in ("train", "val", "test") if x in splits]
    _gpus_of(config, out)
    _place_of(db, config, out, "review", dataset)
    model = _model_choice(config)
    if model:
        out["model"] = model
    return out


def _clean_check(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    group = str(config.get("group") or "any")
    if group not in CHECK_GROUPS:
        raise ServiceError(400, f"검수 확인의 group은 {', '.join(CHECK_GROUPS)} 중 하나입니다.")
    return {"max_flagged": int(_number(config.get("max_flagged") or 0, "문제 이미지 수", 0)), "group": group}


def _clean_sync(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    return {"node_id": _node_id(db, config, "서버로 보내기", "sync", dataset)}


def _clean_autolabel(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    split = str(config.get("split") or "")
    if split not in ("", "train", "val", "test"):
        raise ServiceError(400, "오토라벨 split은 train, val, test 중 하나입니다(비우면 모두).")
    out: dict[str, Any] = {"split": split, "keep": _number(config.get("keep", autolabel.KEEP), "최소 확신도", 0, 1)}
    if config.get("accept") not in (None, ""):
        out["accept"] = _number(config["accept"], "자동 승인 확신도", 0, 1)
        if out["accept"] < out["keep"]:
            raise ServiceError(400, "자동 승인 확신도는 최소 확신도 이상입니다.")
    _set_of(config, out)
    _gpus_of(config, out)
    _place_of(db, config, out, "autolabel", dataset)
    model = _model_choice(config)
    if model:
        out["model"] = model
    return out


def _clean_label(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    """A labeling set of the version's images nobody labelled yet, as 라벨링 묶음 만들기 makes one (after an 오토라벨:
    the set that one made, as it made it)."""
    split = str(config.get("split") or "")
    if split not in ("", "train", "val", "test"):
        raise ServiceError(400, "라벨링 split은 train, val, test 중 하나입니다(비우면 모두).")
    out: dict[str, Any] = {"split": split}
    _set_of(config, out)
    instructions = str(config.get("instructions") or "").strip()
    if instructions:
        out["instructions"] = instructions[:2000]
    return out


def _clean_confirm(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    version = _version_pattern(db, dataset, config)
    if version:
        out["version"] = version
    _note_of(config, out)
    return out


def _clean_train(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    if config.get("attached"):  # follows a training run started on its own: repeated as "다시 실행" would
        return {"attached": True, "run_id": str(config.get("run_id") or "")}
    out = {k: v for k, v in config.items() if k in TRAIN_KEYS and v not in (None, "")}
    if not isinstance(out.get("params", {}), dict):
        raise ServiceError(400, "params는 설정 이름과 값입니다.")
    for key in ("classes", "gpu_indices"):
        if key in out:
            out[key] = [int(v) for v in out[key] or []]
    for key in ("nodes", "tags"):
        if key in out:
            out[key] = [str(v) for v in out[key] or []]
    return _gate_of(config, out)


def _clean_sweep(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    out = {k: v for k, v in config.items() if k in SWEEP_KEYS and v not in (None, "")}
    space = out.get("space")
    if not isinstance(space, dict) or not space:
        raise ServiceError(400, "탐색: 시험할 설정과 값(space)을 하나 이상 적으세요.")
    if out.get("method", "random") not in sweeps.METHODS:
        raise ServiceError(400, "탐색 방식은 grid · random · bayes 중 하나입니다.")
    out["max_runs"] = int(_number(out.get("max_runs", 10), "실행 수", 1, sweeps.MAX_RUNS))
    out["parallel"] = int(_number(out.get("parallel", 1), "동시 실행 수", 1, sweeps.MAX_PARALLEL))
    if not isinstance(out.get("params", {}), dict):
        raise ServiceError(400, "params는 설정 이름과 값입니다.")
    out.pop("metric", None)  # the search's metric is the gate's (below)
    return _gate_of(config, out)


def _clean_eval(
    db: Session, dataset: Dataset | None, config: dict[str, Any], model_ready: bool = True, **_: Any
) -> dict[str, Any]:
    release_id = str(config.get("release_id") or "")
    split = str(config.get("split") or "auto")
    if split not in SPLITS:
        raise ServiceError(400, "평가 split은 auto(test가 있으면 test, 없으면 val) · val · test · train 중 하나입니다.")
    if release_id:  # a fixed version (a test set): every run is measured on the same data
        release = catalog.get_release(db, release_id)
        catalog.require_ready(release)
        other = db.get(Dataset, release.dataset_id)
        if dataset is not None and not task_names.same(catalog.dataset_task(other), catalog.dataset_task(dataset)):
            raise ServiceError(400, "평가할 버전은 학습하는 데이터셋과 태스크가 같아야 합니다.")
        if not _has(release, _split_of(release, split)):
            raise ServiceError(400, f"{other.name} {release.version}의 {_split_of(release, split)}에는 이미지가 없습니다.")
    out: dict[str, Any] = {"release_id": release_id, "split": split, "review": bool(config.get("review"))}
    _gpus_of(config, out)
    _place_of(db, config, out, "eval", dataset)
    model = _model_choice(config)
    if model:
        out["model"] = model
    return _gate_of(config, out)


def _clean_promote(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    min_delta = _number(config.get("min_delta") or 0, "min_delta(이만큼 더 나아야 승격)", 0)
    return {
        "alias": _check_alias(str(config.get("alias") or DEFAULT_ALIAS)),
        "metric": str(config.get("metric") or "").strip(),
        "min_delta": min_delta,
    }


def _clean_export(db: Session, dataset: Dataset | None, config: dict[str, Any], **_: Any) -> dict[str, Any]:
    fmt = str(config.get("format") or "onnx")
    if fmt not in modelops.EXPORT_FORMATS:
        raise ServiceError(400, f"내보내기 형식은 {', '.join(modelops.EXPORT_FORMATS)} 중 하나입니다.")
    out: dict[str, Any] = {"format": fmt}
    _place_of(db, config, out, "export", dataset)
    model = _model_choice(config)
    if model:
        out["model"] = model
    return out


CLEAN = {
    "import": _clean_import,
    "check": _clean_check,
    "review": _clean_review,
    "sync": _clean_sync,
    "autolabel": _clean_autolabel,
    "label": _clean_label,
    "confirm": _clean_confirm,
    "train": _clean_train,
    "sweep": _clean_sweep,
    "eval": _clean_eval,
    "promote": _clean_promote,
    "export": _clean_export,
}


def clean_steps(db: Session, dataset: Dataset | None, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The steps as stored: known kinds, each with complete settings, in an order that can run — a step that uses a
    model comes after one that makes it (or names one by its alias), a promotion right after an evaluation of it.
    ``dataset`` None: a template's steps (no dataset to check them against)."""
    if not steps:
        raise ServiceError(400, "단계를 하나 이상 넣으세요.")
    if len(steps) > MAX_STEPS:
        raise ServiceError(400, f"단계는 {MAX_STEPS}개까지입니다.")
    out: list[dict[str, Any]] = []
    model_ready = measured = False  # a model made by an earlier step; an evaluation of it since
    for number, step in enumerate(steps, 1):
        kind = str((step or {}).get("kind") or "")
        if kind not in KINDS:
            raise ServiceError(400, f"모르는 단계입니다: {kind or '(없음)'}. 단계는 {' · '.join(KINDS)}입니다.")
        where = f"{number}번째 단계({LABEL[kind]})"
        try:
            config = CLEAN[kind](db, dataset, dict(step.get("config") or {}), model_ready=model_ready)
        except ServiceError as e:
            raise ServiceError(e.status, f"{where}: {e.message}") from e
        if kind in USES_MODEL and not model_ready and not config.get("model"):
            raise ServiceError(
                400, f"{where}: 앞에 학습이나 탐색이 없으면 쓸 모델을 고르세요(모델 이름과 alias, 예: {DEFAULT_ALIAS})."
            )
        if kind == "promote" and not measured:
            raise ServiceError(
                400, f"{where}: 승격은 바로 앞의 평가로 정합니다. 학습 · 탐색 뒤에 평가를 넣고 그다음에 두세요."
            )
        if kind in MAKES_MODEL:
            model_ready, measured = True, False
        elif kind == "eval":
            measured = not config.get("model")
        out.append({"kind": kind, "config": config})
    return out


def _check_code(db: Session, dataset: Dataset, steps: list[dict[str, Any]]) -> None:
    """The model kind a training or search uses has code for the steps after it: refused now, not after training."""
    task = catalog.dataset_task(dataset)
    row: Template | None = None
    for step in steps:
        c = step["config"]
        if step["kind"] in MAKES_MODEL:
            if c.get("attached"):  # the kind the followed run trained with
                followed = db.get(Run, c.get("run_id") or "")
                row = db.get(Template, (followed.template or {}).get("id", "")) if followed else None
            else:
                project = catalog.get_project(db, dataset.project_id)
                row = registry.template_for(db, project.settings, task, str(c.get("template") or ""))
            continue
        if row is None or c.get("model") or step["kind"] not in USES_MODEL:
            continue
        if step["kind"] == "autolabel" and not modelops._does(row, "autolabel"):
            raise ServiceError(400, f"{row.title}은(는) 오토라벨을 하지 않아 그 단계를 넣을 수 없습니다.")
        code = "eval" if step["kind"] == "review" else step["kind"]  # 검수 is an evaluation on every split
        if code in ("eval", "export") and code not in registry.manifest_of(row, task).entrypoints:
            what = LABEL[step["kind"]]
            raise ServiceError(400, f"{row.title}에는 {LABEL[code]} 코드가 없어 {what} 단계를 넣을 수 없습니다.")


def train_settings_of(db: Session, run: Run) -> dict[str, Any]:
    """A training run's settings as a training step's (what "다시 실행" repeats): to keep its way as a pipeline."""
    c = run.config or {}
    row = db.get(Template, (run.template or {}).get("id", ""))
    declared = registry.manifest_of(row).params if row is not None else {}
    env = run.environment or {}
    settings = {
        "template": (row.name if row is not None and row.builtin else row.id) if row is not None else run.trainer,
        "environment": env.get("id") or ("auto" if env.get("kind") == "auto" else ""),
        "params": {k: c[k] for k in declared if k in c},
        "model_name": c.get("model_name", ""),
        "classes": c.get("classes") or [],
        "unlabelled": c.get("unlabelled", "exclude"),
        "val_fraction": c.get("val_fraction", 0.2),
        "gpus": c.get("gpus"),
        "gpu_indices": c.get("gpu_indices") or [],
        "min_vram_mb": c.get("min_vram_mb", 0),
        "base_model_id": (c.get("base_model") or {}).get("id", ""),
        "nodes": c.get("nodes") or [],
        "data_after": c.get("data_after", ""),
        "embed": bool(c.get("embed")) or None,  # the image vectors made alongside
    }
    return {k: v for k, v in settings.items() if v not in (None, "", [], {})}


# ---- saved pipelines ------------------------------------------------------------------------


def _dataset_of(db: Session, dataset_id: str) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise ServiceError(404, "데이터셋을 찾을 수 없습니다.")
    return dataset


def newest_release(db: Session, dataset_id: str) -> Release | None:
    return db.scalar(
        select(Release)
        .where(Release.dataset_id == dataset_id, Release.status == "ready", Release.deleted_at.is_(None))
        .order_by(Release.created_at.desc())
        .limit(1)
    )


def _unique_name(db: Session, project_id: str, name: str, pipeline_id: str = "") -> str:
    taken = set(
        db.scalars(
            select(Pipeline.name).where(
                Pipeline.project_id == project_id, Pipeline.deleted_at.is_(None), Pipeline.id != pipeline_id
            )
        )
    )
    found, n = name, 2
    while found in taken:
        found, n = f"{name} {n}", n + 1
    return found


def _default_name(dataset: Dataset, steps: list[dict[str, Any]]) -> str:
    return f"{dataset.name} {' → '.join(LABEL[s['kind']] for s in steps)}"[:MAX_NAME]


def _check_training(db: Session, user: User, dataset: Dataset, steps: list[dict[str, Any]]) -> None:
    """A pipeline's first training (when no step before it makes a new version) is checked against the dataset's
    newest version as it is saved, as the new-run dialog checks."""
    newest = newest_release(db, dataset.id)
    for step in steps:
        if step["kind"] in ("import", "confirm"):
            return  # it trains on a version still to come
        if step["kind"] != "train" or newest is None:
            continue
        c = step["config"]
        found = training.check_run(
            db,
            release_id=newest.id,
            template=str(c.get("template") or ""),
            params=c.get("params") or {},
            classes=c.get("classes"),
            nodes=c.get("nodes"),
            val_fraction=float(c.get("val_fraction", 0.2)),
            environment=str(c.get("environment") or ""),
            gpus=c.get("gpus"),
            user=user,
        )
        if found["problems"]:
            raise ServiceError(400, " ".join(found["problems"]))
        return


def create_pipeline(
    db: Session,
    user: User,
    *,
    dataset_id: str,
    steps: list[dict[str, Any]],
    name: str = "",
    trigger: str = "manual",
    note: str = "",
    via: str = "api",
) -> Pipeline:
    """Save a pipeline for a dataset. Its training settings are checked against the dataset's newest version now."""
    dataset = _dataset_of(db, dataset_id)
    cleaned = clean_steps(db, dataset, steps)
    if any(s["config"].get("attached") for s in cleaned):
        raise ServiceError(400, "저장하는 파이프라인의 학습 단계에는 설정이 있어야 합니다.")
    _check_code(db, dataset, cleaned)
    if trigger not in TRIGGERS:
        raise ServiceError(400, "trigger는 manual(직접 실행) 또는 new_version(새 버전이 생기면)입니다.")
    _check_training(db, user, dataset, cleaned)
    pipeline = Pipeline(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        name=_unique_name(db, dataset.project_id, name.strip()[:MAX_NAME] or _default_name(dataset, cleaned)),
        steps=cleaned,
        trigger=trigger,
        watch_since=utcnow() if trigger == "new_version" else None,
        note=note.strip(),
        created_by=user.id,
    )
    db.add(pipeline)
    db.flush()
    services.audit(db, user.id, via, "pipeline.create", pipeline.id, name=pipeline.name, trigger=trigger)
    return pipeline


def get_pipeline(db: Session, pipeline_id: str) -> Pipeline:
    pipeline = db.get(Pipeline, pipeline_id)
    if pipeline is None or pipeline.deleted_at is not None:
        raise ServiceError(404, "파이프라인을 찾을 수 없습니다.")
    return pipeline


def update_pipeline(
    db: Session,
    user: User,
    pipeline_id: str,
    *,
    name: str | None = None,
    trigger: str | None = None,
    note: str | None = None,
    steps: list[dict[str, Any]] | None = None,
    via: str = "api",
) -> Pipeline:
    pipeline = get_pipeline(db, pipeline_id)
    if name is not None and name.strip():
        pipeline.name = _unique_name(db, pipeline.project_id, name.strip()[:MAX_NAME], pipeline.id)
    if trigger is not None:
        if trigger not in TRIGGERS:
            raise ServiceError(400, "trigger는 manual(직접 실행) 또는 new_version(새 버전이 생기면)입니다.")
        if trigger == "new_version" and pipeline.trigger != "new_version":
            pipeline.watch_since = utcnow()  # versions from now on: not the ones made while it was off
        pipeline.trigger = trigger
    if note is not None:
        pipeline.note = note.strip()
    if steps is not None:
        dataset = _dataset_of(db, pipeline.dataset_id)
        cleaned = clean_steps(db, dataset, steps)
        _check_code(db, dataset, cleaned)
        _check_training(db, user, dataset, cleaned)
        pipeline.steps = cleaned
    services.audit(db, user.id, via, "pipeline.update", pipeline.id, name=pipeline.name, trigger=pipeline.trigger)
    return pipeline


def delete_pipeline(db: Session, user: User, pipeline_id: str, via: str = "api") -> None:
    """It stops watching for new versions; its runs stay listed (a running one goes on)."""
    pipeline = get_pipeline(db, pipeline_id)
    pipeline.deleted_at = utcnow()
    services.audit(db, user.id, via, "pipeline.delete", pipeline.id, name=pipeline.name)


# ---- templates ------------------------------------------------------------------------------


def _for_template(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What of a pipeline's steps belongs to one dataset only stays out of a template: the classes picked by id."""
    out = copy.deepcopy(steps)
    for step in out:
        if step["kind"] in MAKES_MODEL:
            step["config"].pop("classes", None)
            step["config"].pop("name", None)  # a run name is the run's own
    return out


def create_template(
    db: Session,
    user: User,
    *,
    name: str,
    steps: list[dict[str, Any]],
    description: str = "",
    trigger: str = "manual",
    via: str = "api",
) -> PipelineTemplate:
    """Keep steps as a template everyone can start a pipeline from."""
    name = name.strip()[:MAX_NAME]
    if not name:
        raise ServiceError(400, "템플릿 이름을 적으세요.")
    if trigger not in TRIGGERS:
        raise ServiceError(400, "trigger는 manual(직접 실행) 또는 new_version(새 버전이 생기면)입니다.")
    cleaned = clean_steps(db, None, _for_template(steps))
    if any(s["config"].get("attached") for s in cleaned):
        raise ServiceError(400, "템플릿의 학습 단계에는 설정이 있어야 합니다.")
    taken = db.scalar(select(PipelineTemplate).where(PipelineTemplate.name == name, PipelineTemplate.deleted_at.is_(None)))
    if taken is not None or any(t["name"] == name for t in BUILTIN_TEMPLATES):
        raise ServiceError(409, f"'{name}' 템플릿이 이미 있습니다. 다른 이름을 쓰세요.")
    row = PipelineTemplate(name=name, description=description.strip(), steps=cleaned, trigger=trigger, created_by=user.id)
    db.add(row)
    db.flush()
    services.audit(db, user.id, via, "pipeline.template.create", row.id, name=name)
    return row


def delete_template(db: Session, user: User, template_id: str, via: str = "api") -> None:
    row = db.get(PipelineTemplate, template_id)
    if row is None or row.deleted_at is not None:
        raise ServiceError(404, "템플릿을 찾을 수 없습니다.")
    if row.created_by != user.id and user.role != "admin":
        raise ServiceError(403, "내가 만든 템플릿만 지울 수 있습니다(관리자는 모두).")
    row.deleted_at = utcnow()
    services.audit(db, user.id, via, "pipeline.template.delete", row.id, name=row.name)


def _steps_view(db: Session, project: Project | None, dataset: Dataset | None, steps: list[dict[str, Any]]) -> list[dict]:
    return [
        {
            "kind": s["kind"],
            "label": _label(s),
            "summary": _summary(db, project, dataset, s),
            "gate": _is_gate(s),
            "config": s["config"],
        }
        for s in steps
    ]


def list_templates(db: Session) -> list[dict[str, Any]]:
    """The built-in templates, then the saved ones (newest first)."""
    users = dict(db.execute(select(User.id, User.username)).all())
    out = [
        {**t, "builtin": True, "created_by": None, "created_at": None, "steps": _steps_view(db, None, None, t["steps"])}
        for t in BUILTIN_TEMPLATES
    ]
    rows = db.scalars(
        select(PipelineTemplate).where(PipelineTemplate.deleted_at.is_(None)).order_by(PipelineTemplate.created_at.desc())
    )
    for row in rows:
        out.append(
            {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "trigger": row.trigger,
                "builtin": False,
                "created_by": users.get(row.created_by or ""),
                "created_by_id": row.created_by,
                "created_at": iso(row.created_at),
                "steps": _steps_view(db, None, None, row.steps),
            }
        )
    for t in out:
        t["starts_with_train"] = bool(t["steps"]) and t["steps"][0]["kind"] == "train"
    return out


# ---- runs -----------------------------------------------------------------------------------


def _fresh(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": step["kind"],
        "config": step["config"],
        "status": "pending",
        "message": "",
        "refs": {},
        "started_at": None,
        "finished_at": None,
    }


def start_run(
    db: Session,
    user: User,
    *,
    steps: list[dict[str, Any]],
    release_id: str = "",
    pipeline: Pipeline | None = None,
    follow_run_id: str = "",
    name: str = "",
    trigger: str = "manual",
    settings: HubSettings | None = None,
    via: str = "api",
    strict: bool = True,
) -> PipelineRun:
    """Run steps on a version. The first step starts now: with ``strict`` what stops it is refused (nothing is
    made), else it is the run's failure. ``follow_run_id``: an existing training run is the first step (the
    steps after a run started on its own, or to finish what it began)."""
    followed = None
    if follow_run_id:
        followed = training.get_run(db, follow_run_id)
        if followed.deleted_at is not None:
            raise ServiceError(404, "학습 실행을 찾을 수 없습니다.")
        if following(db, followed.id, going_only=True) is not None:
            raise ServiceError(409, "이 학습에는 이미 이어서 할 일이 있습니다. 파이프라인 화면에서 보세요.")
        job = db.get(Job, followed.job_id) if followed.job_id else None
        if job is None or job.status in ("failed", "canceled"):
            raise ServiceError(409, "실패했거나 중단한 학습에는 이어서 할 수 없습니다. 다시 실행한 뒤에 붙이세요.")
        release_id = followed.release_id
        steps = [{"kind": "train", "config": {"attached": True, "run_id": followed.id}}, *steps]
    release = catalog.get_release(db, release_id)
    dataset = db.get(Dataset, release.dataset_id)
    if pipeline is not None and pipeline.dataset_id != dataset.id:
        raise ServiceError(400, "이 파이프라인은 다른 데이터셋의 버전에서 돕니다.")
    cleaned = clean_steps(db, dataset, steps)
    _check_code(db, dataset, cleaned)
    states = [_fresh(s) for s in cleaned]
    run = PipelineRun(
        project_id=dataset.project_id,
        pipeline_id=pipeline.id if pipeline else None,
        name=(name.strip() or (pipeline.name if pipeline else ""))[:MAX_NAME],
        release_id=release.id,
        steps=states,
        trigger=trigger if not followed else "training",
        created_by=user.id,
    )
    if followed is not None:  # the training is already there: the run picks it up where it stands
        run.train_run_id = followed.id
        run.name = run.name or followed.name
        states[0].update(
            status="running", refs={"run_id": followed.id, "job_id": followed.job_id}, started_at=iso(followed.created_at)
        )
    db.add(run)
    db.flush()
    _advance(db, run, settings, strict=strict, via=via)
    services.audit(
        db, user.id, via, "pipeline.run", run.id, name=run.name, release=f"{dataset.name} {release.version}", trigger=trigger
    )
    return run


def run_pipeline(
    db: Session,
    user: User,
    pipeline_id: str,
    release_id: str = "",
    settings: HubSettings | None = None,
    via: str = "api",
) -> PipelineRun:
    """A saved pipeline on a version of its dataset (the newest when none is named)."""
    pipeline = get_pipeline(db, pipeline_id)
    if not release_id:
        newest = newest_release(db, pipeline.dataset_id)
        if newest is None:
            raise ServiceError(400, "이 데이터셋에는 아직 쓸 수 있는 버전이 없습니다.")
        release_id = newest.id
    return start_run(db, user, steps=pipeline.steps, release_id=release_id, pipeline=pipeline, settings=settings, via=via)


def get_run(db: Session, run_id: str) -> PipelineRun:
    run = db.get(PipelineRun, run_id)
    if run is None:
        raise ServiceError(404, "파이프라인 실행을 찾을 수 없습니다.")
    return run


def following(db: Session, training_run_id: str, going_only: bool = False) -> PipelineRun | None:
    """The pipeline run a training run is the first step of (the newest)."""
    query = select(PipelineRun).where(PipelineRun.train_run_id == training_run_id)
    if going_only:
        query = query.where(PipelineRun.status == "running")
    return db.scalar(query.order_by(PipelineRun.created_at.desc()).limit(1))


def _jobs_of(step: dict[str, Any]) -> list[str]:
    refs = step.get("refs") or {}
    ids = [refs["job_id"]] if refs.get("job_id") else []
    return ids + [op_job for op_job in refs.get("op_jobs", []) if op_job]


def stop_run(db: Session, user: User, run_id: str, via: str = "api") -> PipelineRun:
    """Stop the step going (its training, search or evaluation is canceled) and skip the rest. What finished stays."""
    run = get_run(db, run_id)
    if run.status != "running":
        raise ServiceError(409, "이미 끝난 파이프라인 실행입니다.")
    steps = copy.deepcopy(run.steps)
    index = next((i for i, s in enumerate(steps) if s["status"] in GOING), len(steps) - 1)
    sweep_id = (steps[index].get("refs") or {}).get("sweep_id")
    if sweep_id:
        found = db.get(Sweep, sweep_id)
        if found is not None and found.status == "running":
            sweeps.stop_sweep(db, user, sweep_id, via=via)
    for job_id in _jobs_of(steps[index]):
        job = db.get(Job, job_id)
        if job is not None and job.status not in TERMINAL_STATUSES:
            services.cancel_job(db, user, job.id, via=via)
    _end(run, steps, index, "stopped", f"{user.username}이(가) 멈췄습니다.")
    run.steps = steps
    services.audit(db, user.id, via, "pipeline.stop", run.id, name=run.name)
    return run


def retry_run(db: Session, user: User, run_id: str, settings: HubSettings | None = None, via: str = "api") -> PipelineRun:
    """From the step that failed or was stopped: what succeeded before it stays (its training, its model)."""
    run = get_run(db, run_id)
    if run.status not in ("failed", "stopped"):
        raise ServiceError(409, "실패했거나 멈춘 실행만 이어서 다시 할 수 있습니다.")
    steps = copy.deepcopy(run.steps)
    index = next((i for i, s in enumerate(steps) if s["status"] in ("failed", "stopped")), 0)
    trained = (steps[index].get("refs") or {}).get("run_id") if steps[index]["kind"] == "train" else None
    for step in steps[index:]:
        step.update(_fresh(step))
    if trained:  # again with what that training ran
        steps[index]["config"] = {**steps[index]["config"], "again": trained}
    run.steps, run.status, run.error, run.finished_at = steps, "running", "", None
    run.created_by = user.id  # the steps from here on are started as this person
    _advance(db, run, settings, strict=True, via=via)
    services.audit(db, user.id, via, "pipeline.retry", run.id, name=run.name, step=steps[index]["kind"])
    return run


# ---- moving a run along ---------------------------------------------------------------------


def _model(db: Session, model_id: str | None) -> ModelVersion | None:
    model = db.get(ModelVersion, model_id) if model_id else None
    return model if model is not None and model.deleted_at is None else None


def _holder(db: Session, project_id: str, name: str, alias: str) -> ModelVersion | None:
    """The version of a model family carrying ``alias`` (one at a time)."""
    family = db.scalars(
        select(ModelVersion).where(
            ModelVersion.project_id == project_id, ModelVersion.name == name, ModelVersion.deleted_at.is_(None)
        )
    )
    return next((m for m in family if alias in training._aliases(m)), None)


def _numbers(op: ModelJob, split: str) -> dict[str, Any] | None:
    """An evaluation's numbers on one of its splits."""
    splits = modelops._splits_of(op)
    if split not in splits:
        return None
    result = op.result or {}
    return ((result.get("splits") or {}).get(split) if len(splits) > 1 else result.get("metrics")) or None


def _measured(db: Session, model_id: str, release_id: str, split: str) -> tuple[ModelJob, dict[str, Any]] | None:
    """The model's newest finished evaluation on this version's split."""
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.model_id == model_id, ModelJob.release_id == release_id)
        .order_by(ModelJob.created_at.desc())
    )
    for op in ops:
        job = db.get(Job, op.job_id) if op.job_id else None
        numbers = _numbers(op, split) if job is not None and job.status == "succeeded" else None
        if numbers:
            return op, numbers
    return None


def _measuring(db: Session, model_id: str, release_id: str, split: str) -> ModelJob | None:
    """An evaluation of the model on this version's split still waiting or going."""
    ops = db.scalars(
        select(ModelJob)
        .join(Job, Job.id == ModelJob.job_id)
        .where(
            ModelJob.kind == "eval",
            ModelJob.model_id == model_id,
            ModelJob.release_id == release_id,
            Job.status.in_(("queued", "running")),
        )
    )
    return next((op for op in ops if split in modelops._splits_of(op)), None)


def _before_step(steps: list[dict[str, Any]], index: int, kinds: tuple[str, ...]) -> dict[str, Any] | None:
    """The nearest step before ``index`` of one of ``kinds``."""
    return next((s for s in reversed(steps[:index]) if s["kind"] in kinds), None)


def _gate_after(steps: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    """The promotion that decides on this evaluation (the next step that is one, before another model or evaluation)."""
    for step in steps[index + 1 :]:
        if step["kind"] == "promote":
            return step
        if step["kind"] in (*MAKES_MODEL, "eval"):
            return None
    return None


def _release_now(run: PipelineRun, steps: list[dict[str, Any]], index: int) -> str:
    """The current version: the newest one an earlier step made (import, confirm), else the run's."""
    for step in reversed(steps[:index]):
        made = (step.get("refs") or {}).get("release_out")
        if made:
            return made
    return run.release_id


def _subject(db: Session, run: PipelineRun, steps: list[dict[str, Any]], index: int) -> ModelVersion:
    """The model a step acts on: the one it names by alias, else the one an earlier training or search made."""
    wanted = steps[index]["config"].get("model")
    if wanted:
        holder = _holder(db, run.project_id, wanted.get("name", ""), wanted.get("alias", ""))
        if holder is None:
            raise ServiceError(409, f"지금 {wanted.get('alias')}인 {wanted.get('name')} 모델이 없습니다.")
        return holder
    maker = _before_step(steps, index, MAKES_MODEL)
    model = _model(db, ((maker or {}).get("refs") or {}).get("model_id"))
    if model is None:
        raise ServiceError(409, f"{LABEL[maker['kind']]}이 등록한 모델이 지워졌습니다." if maker else "쓸 모델이 없습니다.")
    return model


def _gpus(db: Session, steps: list[dict[str, Any]], index: int) -> int | None:
    """An evaluation takes as many GPUs as the training did (a run on the CPU is measured on the CPU); without
    training, one when the server has any."""
    own = steps[index]["config"].get("gpus")
    if own is not None:
        return int(own)
    maker = _before_step(steps, index, MAKES_MODEL)
    refs = (maker or {}).get("refs") or {}
    if maker is not None and maker["kind"] == "sweep":
        found = db.get(Sweep, refs.get("sweep_id") or "")
        gpus = (found.settings or {}).get("gpus") if found else None
        return int(gpus) if gpus is not None else None
    made = db.get(Run, refs.get("run_id") or "") if maker else None
    return int((made.config or {}).get("gpus") or 0) if made is not None else None


def _set_settings(c: dict[str, Any]) -> dict[str, Any]:
    """A step's labeling-set settings (``_set_of``) as ``labeling.create_set`` takes them."""
    return {
        "name": c.get("name", ""),
        "instructions": c.get("instructions", ""),
        "review": bool(c.get("review", False)),
        "part_size": int(c.get("part_size") or 50),
        "parts": 1 if c.get("divide") == "one" else 0,  # one part: all of it to one person
    }


def _metric_key(model: ModelVersion | None, config: dict[str, Any] | None = None) -> str:
    task = (model.task if model else "") or "detect"
    return (config or {}).get("metric") or training.PRIMARY_METRIC.get(task, "map50_95")


def _next_version(db: Session, dataset_id: str) -> str:
    """The next version name: one past the highest vN, else after the count."""
    names = set(db.scalars(select(Release.version).where(Release.dataset_id == dataset_id)))
    numbers = [int(m.group(1)) for v in names if (m := re.fullmatch(r"[vV](\d+)", v or ""))]
    n = (max(numbers) if numbers else len(names)) + 1
    while f"v{n}" in names:
        n += 1
    return f"v{n}"


def _begin(
    db: Session,
    owner: User,
    run: PipelineRun,
    steps: list[dict[str, Any]],
    index: int,
    settings: HubSettings | None,
    via: str = "pipeline",
) -> None:
    """Start a step: what it runs is started as ``owner`` would start it by hand (``via``: the request that started
    it, else the pipeline)."""
    step = steps[index]
    kind, c = step["kind"], step["config"]
    release_id = _release_now(run, steps, index)
    refs: dict[str, Any] = {}
    if kind == "import":
        release = catalog.get_release(db, release_id)
        dataset = db.get(Dataset, release.dataset_id)
        version = _version_name(db, dataset.id, c.get("version", ""))
        job = catalog.start_import(
            db, owner, project_id=dataset.project_id, node_id=c["node_id"], path=c["path"], dataset=dataset.name,
            version=version, fmt=c.get("format", "auto"), full_read=bool(c.get("full_read")), via=via,
            unlabelled=c.get("unlabelled", "auto"), classes=c.get("classes") or None, note=c.get("note", ""),
        )  # fmt: skip
        refs = {"job_id": job.id, "version": version}
    elif kind == "check":
        refs = {"release_id": release_id}
    elif kind == "review":  # 검수's 모델로 검사하기: every split of the version in one evaluation
        model = _subject(db, run, steps, index)
        release = catalog.get_release(db, release_id)
        wanted = c.get("splits") or ("train", "val", "test")
        parts = [s for s in ("train", "val", "test") if s in wanted and _has(release, s)]
        if not parts:
            raise ServiceError(400, f"{release.version}의 {' · '.join(wanted)}에 검사할 이미지가 없습니다.")
        op = modelops.start_eval(
            db, owner, model.id, release.id, parts[0], _gpus(db, steps, index), via="pipeline",
            node_id=c.get("node_id", ""), splits=parts,
        )  # fmt: skip
        refs = {"model_id": model.id, "release_id": release.id, "op_id": op.id, "op_jobs": [op.job_id]}
    elif kind == "sync":
        node = services.get_node(db, c["node_id"])
        refs = {"node_id": node.id, "release_id": release_id}
        if node.id not in transfers.holders(db, release_id):
            transfers.check_route(training.data_node(db, release_id), node)
            refs["job_id"] = transfers.sync_job(db, owner, release_id, node, via).id
    elif kind == "autolabel":
        model = _subject(db, run, steps, index)
        place = c.get("node_id", "")
        if not place and settings is not None:  # 자동: the server 새 오토라벨 would pick (the holder when it can)
            place = autolabel._servers(db, settings, catalog.get_release(db, release_id))["default"]
        found, op = autolabel.start_on_images(
            db, owner, release_id, model.id, split=c.get("split", ""), keep=c.get("keep", autolabel.KEEP),
            accept=c.get("accept"), gpus=_gpus(db, steps, index), via=via, node_id=place,
            **_set_settings(c),
        )  # fmt: skip
        refs = {"model_id": model.id, "set_id": found.id, "op_id": op.id, "op_jobs": [op.job_id]}
    elif kind == "label":
        # the set an 오토라벨 right before made (no 라벨링 · 버전 확정 · 가져오기 between), else one of its own
        made = _before_step(steps, index, ("autolabel", "label", "confirm", "import"))
        set_id = ((made or {}).get("refs") or {}).get("set_id") if made and made["kind"] == "autolabel" else None
        if set_id and db.get(LabelSet, set_id) is not None:
            refs = {"set_id": set_id}
        else:  # a set of the version's images nobody labelled yet
            release = catalog.get_release(db, release_id)
            dataset = db.get(Dataset, release.dataset_id)
            match = {"state": "unlabelled", **({"split": c["split"]} if c.get("split") else {})}
            wanted = _set_settings(c)
            wanted["name"] = wanted["name"] or f"{dataset.name} 라벨링 {utcnow():%m/%d}"
            found = labeling.create_set(db, owner, release_id=release_id, match=match, via=via, **wanted)
            refs = {"set_id": found.id, "made": True}
    elif kind == "confirm":
        release = catalog.get_release(db, release_id)
        draft = drafts.draft_of(db, release.dataset_id)
        edited = drafts.edited_count(db, draft) if draft is not None else 0
        if edited:
            version = _version_name(db, release.dataset_id, c.get("version", ""))
            drafts.confirm(db, owner, draft.id, version, note=c.get("note") or f"파이프라인 {run.name}에서 확정", via=via)
            refs = {"release_out": draft.id, "version": version, "edited": edited}
        else:
            refs = {"edited": 0}
    elif kind == "train":
        again = c.get("again") or (c.get("run_id") if c.get("attached") else "")
        if again:  # the training it followed, again with the same settings
            made_run = training.rerun(db, owner, again, via=via)
        else:
            made_run = training.start_run(
                db, owner, release_id=release_id, via=via, **{k: v for k, v in c.items() if k in TRAIN_KEYS}
            )
        refs = {"run_id": made_run.id, "job_id": made_run.job_id}
        if _before_step(steps, index, ("train",)) is None:  # the run's training: its screen shows this run
            run.train_run_id = made_run.id
        run.name = run.name or made_run.name
    elif kind == "sweep":
        kept = {k: v for k, v in c.items() if k in SWEEP_KEYS and k != "metric"}
        found = sweeps.start_sweep(db, owner, release_id=release_id, metric=c.get("metric", ""), via=via, **kept)
        refs = {"sweep_id": found.id}
        run.name = run.name or found.name
    elif kind == "eval":
        model = _subject(db, run, steps, index)
        release = catalog.get_release(db, c.get("release_id") or release_id)
        split = _split_of(release, c.get("split") or "auto")
        if not _has(release, split):
            raise ServiceError(400, f"{release.version}의 {split}에는 이미지가 없습니다.")
        splits = [split]
        if c.get("review") and release.id == release_id:  # one evaluation: the gate's split and 검수's images
            splits += [s for s in ("train", "val", "test") if s != split and _has(release, s)]
        gpus = _gpus(db, steps, index)
        place = c.get("node_id", "")  # the server picked, else where the images are
        refs = {"model_id": model.id, "release_id": release.id, "split": split, "op_jobs": []}
        gate = _gate_after(steps, index)
        if gate is not None:  # the alias holder measured on the same data, alongside (and before, so the newest
            # evaluation 검수 offers is the new model's)
            holder = _holder(db, model.project_id, model.name, gate["config"]["alias"])
            if holder is not None and holder.id != model.id and _measured(db, holder.id, release.id, split) is None:
                if _measuring(db, holder.id, release.id, split) is None:
                    try:
                        with _savepoint(db):
                            op = modelops.start_eval(
                                db, owner, holder.id, release.id, split, gpus, via="pipeline", node_id=place
                            )
                        refs.update(holder_op_id=op.id)
                        refs["op_jobs"].append(op.job_id)
                    except ServiceError as e:  # the gate says so when it compares
                        refs["holder_problem"] = e.message
        op = modelops.start_eval(
            db, owner, model.id, release.id, split, gpus, via="pipeline", splits=splits, node_id=place
        )
        refs.update(op_id=op.id)
        refs["op_jobs"].append(op.job_id)
        if c.get("review") and release.id != release_id:  # 검수 looks at the current version
            own = catalog.get_release(db, release_id)
            parts = [s for s in ("train", "val", "test") if _has(own, s)]
            review = modelops.start_eval(
                db, owner, model.id, own.id, parts[0], gpus, via="pipeline", splits=parts, node_id=place
            )
            refs.update(review_op_id=review.id, review_release_id=own.id)
            refs["op_jobs"].append(review.job_id)
        else:
            refs["review_release_id"] = release.id
    elif kind == "promote":
        refs = {}  # decided when the evaluations are in (``_check``)
    elif kind == "export":
        model = _subject(db, run, steps, index)
        op = modelops.start_export(
            db, owner, model.id, c.get("format", "onnx"), via="pipeline", node_id=c.get("node_id", ""), settings=settings
        )
        refs = {"model_id": model.id, "op_id": op.id, "op_jobs": [op.job_id]}
    step.update(status="running", refs=refs, started_at=iso(utcnow()), message="")


def _savepoint(db: Session) -> Any:
    """A savepoint inside a real transaction (SQLite: take the write lock first, so it is not a transaction itself)."""
    write_lock(db)
    return db.begin_nested()


def _job_end(db: Session, job_id: str | None) -> tuple[str, str] | None:
    """A job that ended: (status, error); None while it waits or runs."""
    job = db.get(Job, job_id) if job_id else None
    if job is None:
        return "failed", "작업 기록을 찾을 수 없습니다."
    if job.status not in TERMINAL_STATUSES:
        return None
    return job.status, job.error or ""


def _ended(db: Session, job_id: str | None, what: str) -> tuple[str, str] | None:
    """A step that is one job: None while it goes, else how it went (succeeded with no words yet)."""
    ended = _job_end(db, job_id)
    if ended is None:
        return None
    status, error = ended
    if status == "canceled":
        return "stopped", f"{what}을(를) 중단했습니다."
    if status != "succeeded":
        return "failed", error or f"{what}이(가) 실패했습니다."
    return "succeeded", ""


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "–"


def _reaches(config: dict[str, Any], key: str, value: Any, said: str) -> tuple[str, str]:
    """A step's own gate: its number must reach ``at_least``."""
    floor = config.get("at_least")
    if floor is None:
        return "succeeded", said
    label = METRIC_LABEL.get(key, key)
    if not isinstance(value, int | float):
        return "held", f"{said} · {label}이(가) 없어 기준({_fmt(floor)})을 확인하지 못해 멈춤"
    if ("loss" in key and value > floor) or ("loss" not in key and value < floor):
        return "held", f"{said} · 기준 {_fmt(floor)}에 못 미쳐 멈춤"
    return "succeeded", f"{said} · 기준 {_fmt(floor)} 넘음"


def _check(db: Session, owner: User, run: PipelineRun, steps: list[dict[str, Any]], index: int) -> tuple[str, str] | None:
    """Whether a running step is over: (status, what to say), or None while it goes on."""
    step = steps[index]
    kind, refs, c = step["kind"], step["refs"], step["config"]
    if kind == "import":
        ended = _ended(db, refs.get("job_id"), "가져오기")
        if ended is None or ended[0] != "succeeded":
            return ended
        made = db.scalar(select(Release).where(Release.job_id == refs["job_id"]))
        if made is None or made.status != "ready":
            return "failed", "가져오기는 끝났지만 새 버전이 없습니다."
        refs["release_out"] = made.id
        return "succeeded", f"{made.version} · 이미지 {int((made.stats or {}).get('items') or 0):,}장"
    if kind == "check":
        release = catalog.get_release(db, refs["release_id"])
        if not qa.is_fresh(db, release):
            qa.run(db, release)  # the rule checks on the labels as they are now
        flagged = qa.groups(db, release.id).get(c.get("group", "any"), 0)
        refs["flagged"] = flagged
        what = CHECK_GROUPS.get(c.get("group", "any"), "문제")
        said = f"{release.version}의 {what} 이미지 {flagged:,}장 (기준 {c['max_flagged']:,}장 이하)"
        if flagged > c["max_flagged"]:
            return "held", f"{said}: 검수에서 고치고 버전을 확정한 뒤 다시"
        return "succeeded", said
    if kind == "review":
        op = db.get(ModelJob, refs.get("op_id") or "")
        ended = _ended(db, op.job_id if op else None, "모델로 검사하기")
        if ended is None or ended[0] != "succeeded":
            return ended
        base = select(func.count()).select_from(ItemPrediction).where(ItemPrediction.op_id == op.id)
        images, wrong = db.scalar(base) or 0, db.scalar(base.where(ItemPrediction.wrong > 0)) or 0
        refs["images"], refs["flagged"] = images, wrong
        model = db.get(ModelVersion, op.model_id)
        who = f"{model.name} v{model.version}" if model else "모델"
        return "succeeded", f"{who} · 이미지 {images:,}장 검사 · 라벨이 틀렸을 수 있는 이미지 {wrong:,}장"
    if kind == "sync":
        if not refs.get("job_id"):
            return "succeeded", "이미 그 서버에 있음"
        ended = _ended(db, refs["job_id"], "이미지 보내기")
        if ended is None or ended[0] != "succeeded":
            return ended
        node = db.get(Node, refs.get("node_id") or "")
        return "succeeded", f"{node.name if node else '서버'}에 보냄"
    if kind == "autolabel":
        op = db.get(ModelJob, refs.get("op_id") or "")
        ended = _ended(db, op.job_id if op else None, "오토라벨")
        if ended is None or ended[0] != "succeeded":
            return ended
        result = ((op.result or {}).get("autolabel") or op.result or {}) if op else {}
        said = f"이미지 {int(result.get('images') or 0):,}장 · 라벨 {int(result.get('labels') or 0):,}개"
        if c.get("accept") is not None:
            said += f" · 자동 승인 {int(result.get('accepted') or 0):,}장"
        return "succeeded", said
    if kind == "label":
        found = db.get(LabelSet, refs.get("set_id") or "")
        if found is None:
            return "failed", "라벨링 묶음이 지워졌습니다."
        done, total = _set_progress(db, found)
        refs["done"], refs["total"] = done, total
        if done < total:
            return None  # people are on it
        return "succeeded", f"{found.name} 끝남 · {total:,}장"
    if kind == "confirm":
        if not refs.get("edited"):
            return "succeeded", "고친 이미지가 없어 새 버전을 만들지 않음"
        return "succeeded", f"{refs['version']} 확정 · 고친 이미지 {refs['edited']:,}장"
    if kind == "train":
        ended = _ended(db, refs.get("job_id"), "학습")
        if ended is None or ended[0] != "succeeded":
            return ended
        model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == refs.get("run_id")))
        if model is None:
            return "failed", "학습은 끝났지만 등록된 모델이 없습니다 (코드가 가중치를 올리지 않았습니다)."
        refs["model_id"] = model.id
        key = _metric_key(model, c)
        return _reaches(c, key, (model.metrics or {}).get(key), f"{model.name} v{model.version} 등록")
    if kind == "sweep":
        return _sweep_end(db, step)
    if kind == "eval":
        return _eval_end(db, steps, index)
    if kind == "promote":
        return _decide(db, owner, steps, index)
    if kind == "export":
        op = db.get(ModelJob, refs.get("op_id") or "")
        ended = _ended(db, op.job_id if op else None, "내보내기")
        if ended is None or ended[0] != "succeeded":
            return ended
        return "succeeded", f"{(op.result or {}).get('file') or c.get('format', 'onnx')} 만듦"
    return "failed", f"모르는 단계입니다: {kind}"


def _set_progress(db: Session, found: LabelSet) -> tuple[int, int]:
    """A labeling set's images done (approved, or labelled without review) and in all."""
    counts = labeling._counts(db, found.id)
    total = sum(sum(c.values()) for c in counts.values())
    done = sum(c["approved"] + (0 if found.review else c["labelled"]) for c in counts.values())
    return done, total


def _sweep_end(db: Session, step: dict[str, Any]) -> tuple[str, str] | None:
    refs, c = step["refs"], step["config"]
    found = db.get(Sweep, refs.get("sweep_id") or "")
    if found is None:
        return "failed", "탐색 기록을 찾을 수 없습니다."
    if found.status == "running":
        return None
    if found.status == "stopped":
        return ("failed", found.error) if found.error else ("stopped", "탐색을 멈췄습니다.")
    runs = [r for r in sweeps._runs_of(db, found.id) if sweeps._status(db, r) == "succeeded"]
    scored = [r for r in runs if isinstance((r.summary or {}).get(found.metric), int | float)]
    if not scored:
        return "failed", "탐색이 끝났지만 지표가 있는 실행이 없어 고를 모델이 없습니다."
    pick = max if found.goal == "maximize" else min
    best = pick(scored, key=lambda r: r.summary[found.metric])
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == best.id))
    if model is None:
        return "failed", f"가장 좋은 실행 {best.name}이 모델을 등록하지 않았습니다."
    refs.update(model_id=model.id, run_id=best.id, value=best.summary[found.metric], metric=found.metric)
    tried = ", ".join(f"{k}={(best.config or {}).get(k)}" for k in found.space)
    said = f"{len(runs)}번 중 최고 {METRIC_LABEL.get(found.metric, found.metric)} {_fmt(refs['value'])} ({tried})"
    return _reaches({**c, "metric": found.metric}, found.metric, refs["value"], said)


def _eval_end(db: Session, steps: list[dict[str, Any]], index: int) -> tuple[str, str] | None:
    step = steps[index]
    refs, c = step["refs"], step["config"]
    mine = [refs.get("op_id"), refs.get("review_op_id")]
    ends = {}
    for op_id in [*mine, refs.get("holder_op_id")]:
        op = db.get(ModelJob, op_id) if op_id else None
        if op is None:
            continue
        ended = _job_end(db, op.job_id)
        if ended is None:
            return None  # the holder's evaluation too: the gate right after needs it
        ends[op_id] = ended
    for op_id in mine:
        status, error = ends.get(op_id, ("succeeded", ""))
        if status == "canceled":
            return "stopped", "평가를 중단했습니다."
        if status != "succeeded":
            return "failed", error or "평가가 실패했습니다."
    op = db.get(ModelJob, refs.get("op_id"))
    numbers = _numbers(op, refs["split"]) if op is not None else None
    if not numbers:
        return "failed", "평가는 끝났지만 지표가 없습니다."
    model = _model(db, refs.get("model_id"))
    gate = _gate_after(steps, index)
    key = _metric_key(model, c if c.get("metric") else gate["config"] if gate else None)
    refs["metrics"] = {k: v for k, v in numbers.items() if isinstance(v, int | float)}
    refs["metric"], refs["value"] = key, numbers.get(key)
    release = db.get(Release, refs["release_id"])
    said = f"{METRIC_LABEL.get(key, key)} {_fmt(numbers.get(key))} ({release.version if release else '?'} {refs['split']})"
    if c.get("model") and model is not None:  # the holder on a new version: how it holds up
        before = _before(db, model, release, refs["split"], key)
        if before is not None:
            refs["before"] = before
            change = (numbers.get(key) or 0) - before["value"]
            said += f" · {before['version']}보다 {'+' if change >= 0 else ''}{change:.3f}"
    return _reaches(c, key, numbers.get(key), said)


def _before(db: Session, model: ModelVersion, release: Release | None, split: str, key: str) -> dict[str, Any] | None:
    """The model's newest finished evaluation on another version of the same dataset (same split)."""
    if release is None:
        return None
    others = select(Release.id).where(Release.dataset_id == release.dataset_id, Release.id != release.id)
    ops = db.scalars(
        select(ModelJob)
        .where(ModelJob.kind == "eval", ModelJob.model_id == model.id, ModelJob.release_id.in_(others))
        .order_by(ModelJob.created_at.desc())
    )
    for op in ops:
        job = db.get(Job, op.job_id) if op.job_id else None
        numbers = _numbers(op, split) if job is not None and job.status == "succeeded" else None
        if numbers and isinstance(numbers.get(key), int | float):
            other = db.get(Release, op.release_id)
            return {"release_id": op.release_id, "version": other.version if other else "?", "value": numbers[key]}
    return None


def _decide(db: Session, owner: User, steps: list[dict[str, Any]], index: int) -> tuple[str, str] | None:
    """The gate: move the alias here when this model measured better than the version holding it."""
    step = steps[index]
    c, refs = step["config"], step["refs"]
    measured = (_before_step(steps, index, ("eval",)) or {}).get("refs") or {}
    model = _model(db, measured.get("model_id"))
    if model is None:
        return "failed", "평가한 모델이 지워졌습니다."
    alias, key = c["alias"], _metric_key(model, c)
    mine = (measured.get("metrics") or {}).get(key)
    if not isinstance(mine, int | float):
        return "failed", f"평가 결과에 {METRIC_LABEL.get(key, key)}이(가) 없어 비교할 수 없습니다."
    label = METRIC_LABEL.get(key, key)
    holder = _holder(db, model.project_id, model.name, alias)
    refs.update(alias=alias, metric=key, mine=mine, model_id=model.id)
    if holder is not None and holder.id == model.id:
        refs["outcome"] = "promoted"
        return "succeeded", f"이미 {alias}입니다."
    if holder is None:
        training.set_alias(db, owner, model.id, alias, via="pipeline")
        refs.update(outcome="promoted", holder=None)
        return "succeeded", f"{alias}이(가) 없어 {model.name} v{model.version}에 붙였습니다."
    release_id, split = measured["release_id"], measured["split"]
    found = _measured(db, holder.id, release_id, split)
    if found is None:
        if _measuring(db, holder.id, release_id, split) is not None:
            return None
        earlier = db.get(ModelJob, measured["holder_op_id"]) if measured.get("holder_op_id") else None
        if refs.get("holder_tried") == holder.id or (earlier is not None and earlier.model_id == holder.id):
            why = measured.get("holder_problem") or "평가가 실패했습니다"
            return "failed", f"지금 {alias}인 v{holder.version}을(를) 같은 데이터로 재지 못해 비교하지 못했습니다: {why}"
        refs["holder_tried"] = holder.id  # the holder changed since, or was never measured: measure it now
        try:
            with _savepoint(db):
                gpus = _gpus(db, steps, index)
                op = modelops.start_eval(db, owner, holder.id, release_id, split, gpus, via="pipeline")
        except ServiceError as e:
            return (
                "failed",
                f"지금 {alias}인 v{holder.version}을(를) 같은 데이터로 평가할 수 없어 비교하지 못했습니다: {e.message}",
            )
        refs["op_jobs"] = [op.job_id]
        return None
    theirs = found[1].get(key)
    refs["holder"] = {"id": holder.id, "name": holder.name, "version": holder.version, "op_id": found[0].id}
    refs["theirs"] = theirs
    if not isinstance(theirs, int | float):
        theirs = None
    lower = "loss" in key
    better = theirs is None or (mine < theirs - c["min_delta"] if lower else mine > theirs + c["min_delta"])
    if better:
        training.set_alias(db, owner, model.id, alias, via="pipeline")
        refs["outcome"] = "promoted"
        moved = f"{alias}을(를) v{model.version}(으)로 옮김"
        return "succeeded", f"{label} {_fmt(mine)} > v{holder.version} {_fmt(theirs)}: {moved}"
    refs["outcome"] = "held"
    return "held", f"{label} {_fmt(mine)} ≤ v{holder.version} {_fmt(theirs)}: {alias}은(는) v{holder.version} 그대로"


def _end(run: PipelineRun, steps: list[dict[str, Any]], index: int, status: str, message: str) -> None:
    now = utcnow()
    steps[index].update(status=status, message=message, finished_at=iso(now))
    for later in steps[index + 1 :]:
        if later["status"] in GOING:
            later["status"] = "skipped"
    run.status, run.finished_at = ENDED.get(status, "failed"), now
    run.error = f"{LABEL[steps[index]['kind']]} 단계: {message}" if status in ("failed", "stopped") else ""


def _advance(
    db: Session, run: PipelineRun, settings: HubSettings | None, strict: bool = False, via: str = "pipeline"
) -> None:
    """Move a run as far as it goes now: a step that ended hands over to the next, which starts at once."""
    if run.status != "running":
        return
    steps = copy.deepcopy(run.steps)  # a new value (the stored JSON is compared to it when saved)
    try:
        owner = db.get(User, run.created_by) if run.created_by else None
        for index, step in enumerate(steps):
            if step["status"] not in GOING:
                continue
            if owner is None:
                _end(run, steps, index, "failed", "실행을 시작한 사람의 계정이 없어 멈췄습니다.")
                return
            if step["status"] == "pending":
                try:
                    with _savepoint(db):  # a start that fails leaves nothing half-made
                        _begin(db, owner, run, steps, index, settings, via if strict else "pipeline")
                except ServiceError as e:
                    if strict:
                        raise ServiceError(e.status, f"{LABEL[step['kind']]} 단계: {e.message}") from e
                    _end(run, steps, index, "failed", e.message)
                    return
                strict = False  # only the first step a request starts is refused instead of failing the run
            outcome = _check(db, owner, run, steps, index)
            if outcome is None:
                return
            status, message = outcome
            if status != "succeeded":
                _end(run, steps, index, status, message)
                return
            step.update(status="succeeded", message=message, finished_at=iso(utcnow()))
        run.status, run.finished_at, run.error = "succeeded", utcnow(), ""
    finally:
        run.steps = steps


def _made_by(db: Session, pipeline: Pipeline) -> set[str]:
    """The versions a pipeline's own runs made (import, confirm): they start no run of it."""
    made = set()
    for steps in db.scalars(select(PipelineRun.steps).where(PipelineRun.pipeline_id == pipeline.id)):
        made |= {(s.get("refs") or {}).get("release_out") for s in steps or []}
    return made - {None}


def _watch(db: Session, pipeline: Pipeline, settings: HubSettings | None) -> None:
    """A version of the dataset that became ready since the pipeline last looked starts a run (the newest only)."""
    since = pipeline.watch_since or pipeline.created_at
    fresh = db.scalars(
        select(Release)
        .where(
            Release.dataset_id == pipeline.dataset_id,
            Release.status == "ready",
            Release.deleted_at.is_(None),
            Release.created_at > since,
        )
        .order_by(Release.created_at.desc())
        .limit(20)
    ).all()
    if not fresh:
        return
    write_lock(db)
    pipeline.watch_since = fresh[0].created_at
    own = _made_by(db, pipeline)
    newest = next((r for r in fresh if r.id not in own), None)
    owner = db.get(User, pipeline.created_by) if pipeline.created_by else None
    if newest is None or owner is None:
        return
    try:
        with _savepoint(db):
            start_run(
                db, owner, steps=pipeline.steps, release_id=newest.id, pipeline=pipeline, trigger="new_version",
                settings=settings, via="pipeline", strict=True,
            )  # fmt: skip
    except ServiceError as e:  # nothing started: a run that says why, so the screen shows it
        steps = [{**_fresh(s), "status": "skipped"} for s in pipeline.steps]
        if steps:
            steps[0].update(status="failed", message=e.message, finished_at=iso(utcnow()))
        db.add(
            PipelineRun(
                project_id=pipeline.project_id,
                pipeline_id=pipeline.id,
                name=pipeline.name,
                release_id=newest.id,
                steps=steps,
                status="failed",
                error=e.message,
                trigger="new_version",
                created_by=owner.id,
                finished_at=utcnow(),
            )
        )


def advance_all(database: Any, settings: HubSettings | None = None) -> None:
    """One pass: watched datasets' new versions, then every running run (the Hub's background loop)."""
    with lock:
        _dirty.clear()
        with database.session() as db:
            watched = select(Pipeline).where(Pipeline.trigger == "new_version", Pipeline.deleted_at.is_(None))
            for pipeline in db.scalars(watched).all():
                try:
                    _watch(db, pipeline, settings)
                except Exception:  # one pipeline's trouble does not hold up the others
                    log.exception("pipeline %s: watching for new versions failed", pipeline.id)
            for run in db.scalars(select(PipelineRun).where(PipelineRun.status == "running")).all():
                try:
                    _advance(db, run, settings)
                except Exception:
                    log.exception("pipeline run %s: advancing failed", run.id)


# ---- views ----------------------------------------------------------------------------------


def _release_ref(db: Session, release_id: str | None) -> dict[str, Any] | None:
    release = db.get(Release, release_id) if release_id else None
    dataset = db.get(Dataset, release.dataset_id) if release else None
    if release is None or dataset is None:
        return None
    return {
        "id": release.id,
        "version": release.version,
        "dataset": dataset.name,
        "dataset_id": dataset.id,
        "task": catalog.dataset_task(dataset),
        "deleted": release.deleted_at is not None,
    }


def _model_ref(model: ModelVersion | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "aliases": training._aliases(model),
        "deleted": model.deleted_at is not None,
    }


SPLIT_LABEL = {"auto": "test (없으면 val)", "val": "val", "test": "test", "train": "train", "": "모든 split"}
METHOD_LABEL = {"grid": "모든 조합", "random": "무작위", "bayes": "베이지안"}


def _label(step: dict[str, Any]) -> str:
    return LABEL[step["kind"]] + (" · 검수" if step["kind"] == "eval" and step["config"].get("review") else "")


def _is_gate(step: dict[str, Any]) -> bool:
    """Whether the step decides if the run goes on (else it goes on when the step succeeds)."""
    return step["kind"] in GATES or step["config"].get("at_least") is not None


def _node_name(db: Session, node_id: str | None) -> str:
    node = db.get(Node, node_id) if node_id else None
    return node.name if node else "지운 서버"


def _train_summary(db: Session, project: Project | None, dataset: Dataset | None, c: dict[str, Any]) -> str:
    task = catalog.dataset_task(dataset) if dataset else ""
    try:
        if task:
            row = registry.template_for(db, project.settings if project else {}, task, str(c.get("template") or ""))
        elif c.get("template"):
            row = registry.get_template(db, str(c["template"]))
        else:
            return "프로젝트 기본 모델 종류"
    except ServiceError:
        return "삭제된 모델 종류"
    manifest = registry.manifest_of(row, task) if task else registry.manifest_of(row)
    params = c.get("params") or {}
    parts = [row.title]
    if manifest.variant:
        spec = manifest.params.get(manifest.variant)
        value = params.get(manifest.variant, spec.default if spec else "")
        if value:
            parts.append(PurePath(str(value)).stem)
    epochs = params.get("epochs", manifest.params["epochs"].default if "epochs" in manifest.params else None)
    if epochs:
        parts.append(f"epoch {epochs}")
    return " · ".join(parts)


def _summary(db: Session, project: Project | None, dataset: Dataset | None, step: dict[str, Any]) -> str:
    """What a step does, in a few words (the cards and the steps not started yet)."""
    kind, c = step["kind"], step["config"]
    gate = ""
    if c.get("at_least") is not None:
        gate = f" · {METRIC_LABEL.get(c.get('metric', ''), c.get('metric') or '주 지표')} {c['at_least']} 이상이면 다음"
    model = c.get("model")
    who = f"{model['name'] or '모델'} {model['alias']}" if model else "앞 단계의 모델"
    where = f" · {_node_name(db, c['node_id'])}에서" if c.get("node_id") else ""
    if kind == "import":
        return f"{_node_name(db, c.get('node_id'))}: {c.get('path')} → 새 버전 {_version_said(c)}"
    if kind == "review":
        return f"{who} · 지금 버전의 {' · '.join(c.get('splits') or ['모든 이미지'])}{where}"
    if kind == "check":
        return f"{CHECK_GROUPS.get(c.get('group', 'any'), '문제')} 이미지 {c.get('max_flagged', 0):,}장 이하면 다음"
    if kind == "sync":
        return f"{_node_name(db, c.get('node_id'))}(으)로"
    if kind == "autolabel":
        said = f"{who} · {SPLIT_LABEL.get(c.get('split', ''), c.get('split'))}의 라벨 안 한 이미지{where}"
        return said + (f" · 확신도 {c['accept']} 이상 자동 승인" if c.get("accept") is not None else "")
    if kind == "label":
        parts = "한 사람에게 모두" if c.get("divide") == "one" else f"일감 하나에 {c.get('part_size', 50)}장"
        return f"{parts}{' · 검수하기' if c.get('review') else ''} · 사람이 끝낼 때까지 기다림"
    if kind == "confirm":
        return f"작업 중을 새 버전으로 {_version_said(c)}"
    if kind == "train":
        if c.get("attached"):
            return "따로 시작한 학습"
        return _train_summary(db, project, dataset, c) + gate
    if kind == "sweep":
        tried = ", ".join(c.get("space") or {}) or "시험할 설정 없음"
        return f"{METHOD_LABEL.get(c.get('method', 'random'), c.get('method'))} {c.get('max_runs', 10)}번 · {tried}{gate}"
    if kind == "eval":
        at = "그 버전" if c.get("model") else "학습한 버전"  # a model made before: on the version it trained on
        if c.get("release_id"):
            ref = _release_ref(db, c["release_id"])
            at = f"{ref['dataset']} {ref['version']}" if ref else "지운 버전"
        said = f"{at} · {SPLIT_LABEL.get(c.get('split', 'auto'), c.get('split'))}"
        if c.get("model"):
            said = f"{c['model']['name'] or '모델'} {c['model']['alias']} → {said}"
        return said + (" · 검수에도" if c.get("review") else "") + where + gate
    if kind == "promote":
        gain = f" (+{c['min_delta']} 이상)" if c.get("min_delta") else ""
        return f"더 나으면 {c['alias']}{gain}"
    if kind == "export":
        return str(c.get("format", "onnx")).upper() + where
    return ""


def _version_said(c: dict[str, Any]) -> str:
    """The name a step gives the version it makes, as the summaries say it."""
    return f"({c['version']})" if c.get("version") else "(이름은 다음 번호)"


def _job_line(db: Session, settings: HubSettings, job_id: str | None) -> dict[str, Any] | None:
    job = db.get(Job, job_id) if job_id else None
    if job is None:
        return None
    line = {"id": job.id, "status": job.status}
    if job.status == "queued":
        line["wait_reason"] = services.explain_wait(db, settings, job)
    return line


def _step_view(
    db: Session, settings: HubSettings, run: PipelineRun, step: dict[str, Any], names: tuple[Any, Any]
) -> dict[str, Any]:
    kind, refs = step["kind"], step.get("refs") or {}
    running = step["status"] == "running"
    view: dict[str, Any] = {
        "kind": kind,
        "label": _label(step),
        "status": step["status"],
        "message": step.get("message") or "",
        "summary": _summary(db, names[0], names[1], step),
        "gate": _is_gate(step),
        "config": step["config"],
        "started_at": step.get("started_at"),
        "finished_at": step.get("finished_at"),
    }
    if refs.get("release_out"):
        view["made_release"] = _release_ref(db, refs["release_out"])
    if kind in ("import", "sync") and running:
        view["job"] = _job_line(db, settings, refs.get("job_id"))
    elif kind == "check" and refs:
        view["release"] = _release_ref(db, refs.get("release_id"))
        view["flagged"] = refs.get("flagged")
    elif kind == "review" and refs:
        view["release"] = _release_ref(db, refs.get("release_id"))
        view["model"] = _model_ref(db.get(ModelVersion, refs.get("model_id") or ""))
        view["op_id"] = refs.get("op_id")
        view["flagged"] = refs.get("flagged")
        if running:
            op = db.get(ModelJob, refs.get("op_id") or "")
            view["job"] = _job_line(db, settings, op.job_id if op else None)
    elif kind in ("autolabel", "label") and refs.get("set_id"):
        found = db.get(LabelSet, refs["set_id"])
        view["set"] = {"id": found.id, "name": found.name} if found else None
        if kind == "label" and found is not None:
            view["set"]["done"], view["set"]["total"] = _set_progress(db, found)
        if kind == "autolabel":
            view["op_id"] = refs.get("op_id")
            view["model"] = _model_ref(db.get(ModelVersion, refs.get("model_id") or ""))
            if running:
                op = db.get(ModelJob, refs.get("op_id") or "")
                view["job"] = _job_line(db, settings, op.job_id if op else None)
    elif kind == "train" and refs.get("run_id"):
        made = db.get(Run, refs["run_id"])
        view["run"] = {"id": made.id, "name": made.name, "epoch": made.epoch, "epochs": made.epochs} if made else None
        view["job"] = _job_line(db, settings, refs.get("job_id")) if running else None
        view["model"] = _model_ref(db.get(ModelVersion, refs["model_id"])) if refs.get("model_id") else None
    elif kind == "sweep" and refs.get("sweep_id"):
        found = db.get(Sweep, refs["sweep_id"])
        if found is not None:
            runs = sweeps._runs_of(db, found.id)
            view["sweep"] = {"id": found.id, "name": found.name, "started": len(runs), "total": sweeps.total_of(found)}
        view["model"] = _model_ref(db.get(ModelVersion, refs["model_id"])) if refs.get("model_id") else None
    elif kind == "eval" and refs:
        view["model"] = _model_ref(db.get(ModelVersion, refs.get("model_id")))
        view["release"] = _release_ref(db, refs.get("release_id"))
        view["split"] = refs.get("split")
        view["op_id"] = refs.get("op_id")
        key = refs.get("metric")
        view["metric"] = {"key": key, "label": METRIC_LABEL.get(key or "", key), "value": refs.get("value")} if key else None
        view["before"] = refs.get("before")
        if step["config"].get("review"):  # 검수 shows this model's check on that version
            view["review_release_id"] = refs.get("review_release_id") or refs.get("release_id")
        if running:
            op = db.get(ModelJob, refs.get("op_id")) if refs.get("op_id") else None
            view["job"] = _job_line(db, settings, op.job_id if op else None)
    elif kind == "promote" and refs:
        view["alias"] = refs.get("alias") or step["config"].get("alias")
        view["outcome"] = refs.get("outcome")
        view["mine"], view["theirs"] = refs.get("mine"), refs.get("theirs")
        view["metric"] = {"key": refs.get("metric"), "label": METRIC_LABEL.get(refs.get("metric") or "", refs.get("metric"))}
        view["holder"] = refs.get("holder")
        view["model"] = _model_ref(db.get(ModelVersion, refs["model_id"])) if refs.get("model_id") else None
    elif kind == "export" and refs:
        view["op_id"] = refs.get("op_id")
        view["model"] = _model_ref(db.get(ModelVersion, refs.get("model_id")))
        if running:
            op = db.get(ModelJob, refs.get("op_id")) if refs.get("op_id") else None
            view["job"] = _job_line(db, settings, op.job_id if op else None)
    return view


def run_view(db: Session, settings: HubSettings, run: PipelineRun) -> dict[str, Any]:
    pipeline = db.get(Pipeline, run.pipeline_id) if run.pipeline_id else None
    release = _release_ref(db, run.release_id)
    dataset = db.get(Dataset, release["dataset_id"]) if release else None
    project = db.get(Project, run.project_id)
    owner = db.get(User, run.created_by) if run.created_by else None
    end = run.finished_at or utcnow()
    return {
        "id": run.id,
        "name": run.name or (pipeline.name if pipeline else ""),
        "status": run.status,
        "error": run.error or None,
        "trigger": run.trigger,
        "project": {"id": run.project_id, "name": project.name if project else ""},
        "pipeline": {"id": pipeline.id, "name": pipeline.name, "deleted": pipeline.deleted_at is not None}
        if pipeline
        else None,
        "release": release,
        "train_run_id": run.train_run_id,
        "created_by": owner.username if owner else None,
        "created_at": iso(run.created_at),
        "finished_at": iso(run.finished_at),
        "duration_s": (end - run.created_at).total_seconds(),
        "steps": [_step_view(db, settings, run, s, (project, dataset)) for s in run.steps],
    }


def pipeline_view(db: Session, settings: HubSettings, pipeline: Pipeline, detail: bool = False) -> dict[str, Any]:
    dataset = db.get(Dataset, pipeline.dataset_id)
    project = db.get(Project, pipeline.project_id)
    owner = db.get(User, pipeline.created_by) if pipeline.created_by else None
    runs = db.scalars(
        select(PipelineRun).where(PipelineRun.pipeline_id == pipeline.id).order_by(PipelineRun.created_at.desc())
    ).all()
    newest = newest_release(db, pipeline.dataset_id)
    view: dict[str, Any] = {
        "id": pipeline.id,
        "name": pipeline.name,
        "project_id": pipeline.project_id,
        "dataset": {"id": dataset.id, "name": dataset.name, "task": catalog.dataset_task(dataset)} if dataset else None,
        "steps": _steps_view(db, project, dataset, pipeline.steps),
        "trigger": pipeline.trigger,
        "watch_since": iso(pipeline.watch_since),
        "note": pipeline.note,
        "created_by": owner.username if owner else None,
        "created_at": iso(pipeline.created_at),
        "runs": len(runs),
        "going": sum(1 for r in runs if r.status == "running"),
        "last": run_view(db, settings, runs[0]) if runs else None,
        "newest_release": {"id": newest.id, "version": newest.version} if newest else None,
    }
    if detail:
        view["history"] = [run_view(db, settings, r) for r in runs[:MAX_RUNS_SHOWN]]
    return view


def board(db: Session, settings: HubSettings, project_id: str) -> dict[str, Any]:
    """The 파이프라인 screen for a project: its saved pipelines, and the runs (theirs and the ones without)."""
    catalog.get_project(db, project_id)
    pipelines = db.scalars(
        select(Pipeline)
        .where(Pipeline.project_id == project_id, Pipeline.deleted_at.is_(None))
        .order_by(Pipeline.created_at.desc())
    ).all()
    runs = db.scalars(
        select(PipelineRun)
        .where(PipelineRun.project_id == project_id)
        .order_by(PipelineRun.created_at.desc())
        .limit(MAX_RUNS_SHOWN)
    ).all()
    counts = dict(
        db.execute(
            select(PipelineRun.status, func.count()).where(PipelineRun.project_id == project_id).group_by(PipelineRun.status)
        ).all()
    )
    promoted = sum(
        1
        for r in runs
        for s in r.steps
        if s["kind"] == "promote" and (s.get("refs") or {}).get("outcome") == "promoted" and s["status"] == "succeeded"
    )
    return {
        "pipelines": [pipeline_view(db, settings, p) for p in pipelines],
        "runs": [run_view(db, settings, r) for r in runs],
        "totals": {
            "pipelines": len(pipelines),
            "watching": sum(1 for p in pipelines if p.trigger == "new_version"),
            "runs": sum(counts.values()),
            "going": counts.get("running", 0),
            "succeeded": counts.get("succeeded", 0),
            "held": counts.get("held", 0),
            "failed": counts.get("failed", 0) + counts.get("stopped", 0),
            "promoted": promoted,
        },
    }


def follow_view(db: Session, settings: HubSettings, run: Run) -> dict[str, Any]:
    """For a training run's screen: the pipeline run it is the first step of, and whether its model was evaluated
    (a finished run with no evaluation and no pipeline is nudged to go on)."""
    found = following(db, run.id)
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id))
    evaluations = (
        db.scalar(select(func.count()).select_from(ModelJob).where(ModelJob.kind == "eval", ModelJob.model_id == model.id))
        if model is not None
        else 0
    )
    return {"pipeline": run_view(db, settings, found) if found else None, "evaluations": evaluations or 0}
