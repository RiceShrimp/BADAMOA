"""Models trained elsewhere: pretrained or open Ultralytics YOLO weights uploaded with their classes.

The person uploads the ``.pt`` with a YOLO data yaml (the ``names:`` a dataset import reads). The Hub reads the
class count, task and image size from the checkpoint without torch: a ``.pt`` is a zip holding a pickle, which is
read here with every class and function it names replaced by an inert stand-in (nothing in it runs) and its tensors
left unread. The counts must agree, else nothing is stored. The upload becomes a model version like a trained one,
run by the built-in Ultralytics YOLO kind, so evaluation, 검수, 오토라벨, export, comparisons and pipelines take it.
"""

from __future__ import annotations

import pickle
import re
import zipfile
from pathlib import Path
from typing import IO, Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from badamoa.core import tasks as task_names
from badamoa.core.storage import Storage
from badamoa.node.importer import parse_yolo_yaml

from . import catalog, services, training
from .models import User
from .services import ServiceError
from .train_models import ModelVersion

TEMPLATE = "yolo"  # the built-in kind that evaluates, exports, tests and auto-labels Ultralytics YOLO weights
MAX_PICKLE_BYTES = 64 * 1024**2  # the pickle holds the structure; the tensors are separate files in the zip
MAX_YAML_BYTES = 1024**2
MAX_CLASSES = 10_000
# The model class Ultralytics saves for each task (when the checkpoint does not name the task itself).
TASK_OF_CLASS = {
    "DetectionModel": "detect",
    "SegmentationModel": "segment",
    "PoseModel": "pose",
    "ClassificationModel": "classify",
    "OBBModel": "obb",
    "WorldModel": "detect",
    "YOLOEModel": "detect",
    "YOLOESegModel": "segment",
    "RTDETRDetectionModel": "detect",
}
# The head a model's yaml ends with, for the same purpose.
TASK_OF_HEAD = {"Detect": "detect", "Segment": "segment", "Pose": "pose", "Classify": "classify", "OBB": "obb"}
# torch's legacy (pre-zip) format starts with this pickled number
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


# ---- reading a checkpoint without torch ---------------------------------------------


class _Stub:
    """Whatever the pickle names (a module class, torch's tensor rebuilders, OrderedDict …) becomes one of these:
    it only keeps what it is given, so loading runs no code but this."""

    _where = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__["_args"] = args

    def __call__(self, *args: Any, **kwargs: Any) -> _Stub:  # a stand-in called as a function
        return _Stub(*args)

    def __setstate__(self, state: Any) -> None:
        self.__dict__["_state"] = state

    def __setitem__(self, key: Any, value: Any) -> None:  # an OrderedDict filled item by item
        self.__dict__.setdefault("_items", {})[key] = value

    def append(self, value: Any) -> None:
        self.__dict__.setdefault("_list", []).append(value)

    def extend(self, values: Any) -> None:
        self.__dict__.setdefault("_list", []).extend(values)

    def add(self, value: Any) -> None:
        self.append(value)


class _Reader(pickle.Unpickler):
    def __init__(self, file: IO[bytes]) -> None:
        super().__init__(file)
        self._made: dict[tuple[str, str], type] = {}

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key not in self._made:
            self._made[key] = type(name[:64] or "Stub", (_Stub,), {"_where": f"{module}.{name}"})
        return self._made[key]

    def persistent_load(self, pid: Any) -> None:  # a tensor's storage: left in the zip, never read
        return None


def _load(file: IO[bytes]) -> Any:
    """The object ``torch.save`` wrote, with stand-ins for everything but plain Python data."""
    file.seek(0)
    try:
        if zipfile.is_zipfile(file):
            file.seek(0)
            with zipfile.ZipFile(file) as archive:
                found = [i for i in archive.infolist() if i.filename.endswith("data.pkl")]
                if not found:
                    raise ServiceError(400, "PyTorch 가중치(.pt)가 아닙니다: 파일 안에 data.pkl이 없습니다.")
                info = min(found, key=lambda i: i.filename.count("/"))
                if info.file_size > MAX_PICKLE_BYTES:
                    raise ServiceError(400, "가중치의 구조 정보가 너무 큽니다. Ultralytics YOLO 가중치인지 확인하세요.")
                with archive.open(info) as data:
                    return _Reader(data).load()
        file.seek(0)  # torch's legacy format: the magic number, the protocol, system info, then the object
        reader = _Reader(file)
        if reader.load() != _LEGACY_MAGIC:
            raise ValueError("not a torch file")
        reader.load()
        reader.load()
        return reader.load()
    except ServiceError:
        raise
    except Exception as e:  # noqa: BLE001 - whatever the file is, it is not weights we can read
        raise ServiceError(400, "PyTorch 가중치(.pt)로 읽을 수 없는 파일입니다.") from e


def _fields(obj: Any) -> dict[Any, Any]:
    """A loaded object's fields: a dict as is, a stand-in's state (a pickled module) or its items (an OrderedDict)."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, _Stub):
        state = obj.__dict__.get("_state")
        if isinstance(state, tuple) and state and isinstance(state[0], dict):  # (state, slots)
            state = state[0]
        if isinstance(state, dict):
            return state
        items = obj.__dict__.get("_items")
        if isinstance(items, dict):
            return items
    return {}


def _names(value: Any) -> list[str] | None:
    data = _fields(value) if isinstance(value, _Stub) else value
    if isinstance(data, dict) and data:
        try:
            return [str(data[k]) for k in sorted(data, key=int)]
        except (TypeError, ValueError):
            return [str(v) for v in data.values()]
    if isinstance(data, list | tuple) and data:
        return [str(v) for v in data]
    return None


def _size(value: Any) -> int:
    if isinstance(value, int | float) and value > 0:
        return int(value)
    if isinstance(value, list | tuple) and value and all(isinstance(v, int | float) for v in value):
        return int(max(value))
    return 0


def read_checkpoint(file: IO[bytes]) -> dict[str, Any]:
    """What an Ultralytics YOLO ``.pt`` says about itself: ``classes`` (its names in class order), ``task``
    ("" when it does not say), ``imgsz``, ``arch`` (yolo11n …) and ``ultralytics`` (the version that saved it)."""
    top = _fields(_load(file))
    model = top.get("model") or top.get("ema")
    plain = isinstance(model, dict) or getattr(model, "_where", "").endswith("OrderedDict")  # weights only
    state = {} if plain else _fields(model)
    if not state:
        raise ServiceError(
            400,
            "Ultralytics YOLO 가중치가 아닙니다: 체크포인트에 모델이 없습니다 "
            "(state_dict만 저장한 파일은 올릴 수 없습니다).",
        )
    names = _names(state.get("names"))
    nc = state.get("nc")
    spec = state.get("yaml") if isinstance(state.get("yaml"), dict) else {}
    if names is None and isinstance(nc, int):
        names = [str(i) for i in range(nc)]
    if names is None and isinstance(spec.get("nc"), int):
        names = [str(i) for i in range(spec["nc"])]
    if not names:
        raise ServiceError(400, "가중치에서 클래스 정보를 찾지 못했습니다 (model.names · nc가 없음).")
    args = _fields(state.get("args"))
    train_args = _fields(top.get("train_args"))
    task = str(state.get("task") or args.get("task") or train_args.get("task") or "")
    if not task:
        where = getattr(model, "_where", "")
        task = TASK_OF_CLASS.get(where.rsplit(".", 1)[-1], "")
    if not task:
        head = spec.get("head")
        last = head[-1] if isinstance(head, list) and head and isinstance(head[-1], list | tuple) else []
        task = TASK_OF_HEAD.get(str(last[2]) if len(last) > 2 else "", "")
    source = str(spec.get("yaml_file") or state.get("yaml_file") or train_args.get("model") or "")
    arch = Path(source.replace("\\", "/")).stem if source else ""
    version = top.get("version")
    return {
        "classes": names,
        "task": task,
        "imgsz": _size(args.get("imgsz")) or _size(train_args.get("imgsz")),
        "arch": arch[:64],
        "ultralytics": str(version)[:32] if isinstance(version, str) else "",
        "kpt_shape": list(spec.get("kpt_shape") or []) if isinstance(spec.get("kpt_shape"), list | tuple) else [],
    }


# ---- the classes that come with it -----------------------------------------------------


def read_yaml(text: str) -> dict[str, Any]:
    """The YOLO data yaml's class names (``names:`` as a list or as ``0: name``) and ``nc`` when it has one."""
    data = parse_yolo_yaml(text)
    names = [str(n).strip() for n in data.get("names") or []]
    nc = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if found := re.fullmatch(r"nc\s*:\s*(\d+)", line):
            nc = int(found.group(1))
    if not names:
        raise ServiceError(
            400, "yaml에 클래스 이름(names:)이 없습니다. YOLO 데이터셋의 data.yaml처럼 names: 아래에 클래스를 적으세요."
        )
    if nc is not None and nc != len(names):
        raise ServiceError(400, f"yaml의 nc는 {nc}인데 names에는 클래스가 {len(names)}개입니다.")
    if len(names) > MAX_CLASSES:
        raise ServiceError(400, f"클래스는 {MAX_CLASSES:,}개까지입니다.")
    if any(not n for n in names):
        raise ServiceError(400, "yaml에 이름이 빈 클래스가 있습니다.")
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ServiceError(400, f"yaml에 같은 이름의 클래스가 여럿 있습니다: {', '.join(twice[:5])}")
    return {"names": names, "kpt_shape": data.get("kpt_shape") or []}


def check(pt: dict[str, Any], classes: dict[str, Any], task: str = "") -> str:
    """The task the upload registers as, after checking the weights and the yaml agree; raises when they do not."""
    have, given = len(pt["classes"]), len(classes["names"])
    if have != given:
        why = f"클래스 수가 다릅니다: pt는 클래스 {have}개, yaml은 {given}개입니다."
        raise ServiceError(400, f"{why} 같은 모델의 yaml을 올리세요.")
    read = task_names.normalize(pt["task"]) if pt["task"] else ""
    chosen = task_names.normalize(task) if task else ""
    if read == "obb" or chosen == "obb":
        raise ServiceError(400, "회전 박스(OBB) 모델은 아직 올릴 수 없습니다.")
    if read and chosen and not task_names.same(read, chosen):
        short = task_names.short
        raise ServiceError(400, f"가중치는 {short(read)} 모델인데 태스크를 {short(chosen)}(으)로 골랐습니다.")
    found = read or chosen or ("pose" if classes["kpt_shape"] else "")
    if not found:
        raise ServiceError(400, "가중치에서 태스크를 알 수 없습니다. 세부 설정에서 태스크를 고르세요.")
    if found not in task_names.BUILTIN:
        raise ServiceError(400, f"{found} 태스크의 모델은 올릴 수 없습니다 (검출 · 분할 · 키포인트 · 분류만).")
    return found


# ---- registering -----------------------------------------------------------------------


def register(
    db: Session,
    user: User,
    storage: Storage,
    project_id: str,
    weights: Path,
    pt: dict[str, Any],
    classes: dict[str, Any],
    task: str,
    *,
    name: str = "",
    file_name: str = "",
    yaml_name: str = "",
    note: str = "",
    imgsz: int = 0,
    via: str = "api",
) -> ModelVersion:
    """Store the checked weights and make them the next version of ``name`` (the file's name when empty)."""
    from . import registry  # here: registry → cleanup → exports → training → registry when imported first

    catalog.get_project(db, project_id)
    template = registry.get_template(db, TEMPLATE)
    file_name = Path(file_name or weights.name).name[:128] or "model.pt"
    name = training._slug(name or Path(file_name).stem)[:128]
    if not name:
        raise ServiceError(400, "모델 이름을 적으세요.")
    if imgsz and not 32 <= imgsz <= 8192:
        raise ServiceError(400, "이미지 크기는 32~8192 사이여야 합니다.")
    size = weights.stat().st_size
    blob = storage.put_blob_file(weights)
    from .db import write_lock

    write_lock(db)  # the next version number, read and taken at once
    latest = db.scalar(
        select(func.max(ModelVersion.version)).where(ModelVersion.project_id == project_id, ModelVersion.name == name)
    )
    renamed = sum(1 for a, b in zip(pt["classes"], classes["names"], strict=True) if a != b)
    model = ModelVersion(
        project_id=project_id,
        name=name,
        version=(latest or 0) + 1,
        run_id=None,
        release_id=None,
        task=task,
        arch=pt["arch"] or Path(file_name).stem[:64],
        trainer=template.name,
        template={
            "id": template.id,
            "name": template.name,
            "title": template.title,
            "version": template.version,
            "builtin": template.builtin,
            "variant": pt["arch"],
        },
        blob=blob,
        bytes=size,
        file_name=file_name,
        imgsz=imgsz or pt["imgsz"] or (224 if task == "classify" else 640),
        classes=classes["names"],
        source={
            "kind": "upload",
            "file_name": file_name,
            "yaml": Path(yaml_name).name[:128],
            "ultralytics": pt["ultralytics"],
            # the names the weights carried, when the yaml gave the classes other names
            **({"weights_classes": pt["classes"], "renamed": renamed} if renamed else {}),
        },
        note=note.strip()[: training.MAX_NOTE],
        created_by=user.id,
    )
    db.add(model)
    db.flush()
    services.audit(
        db, user.id, via, "model.upload", model.id, name=model.name, version=model.version, task=task,
        classes=len(model.classes), bytes=size,
    )
    return model


def uploaded(model: ModelVersion) -> bool:
    return (model.source or {}).get("kind") == "upload"


# ---- samples of the yaml -----------------------------------------------------------------

_SAMPLES = Path(__file__).resolve().parent.parent / "samples"
_HEADER = """\
# 모델과 함께 올리는 클래스 정보 (YOLO 데이터셋의 data.yaml과 같은 형식)
# - names: 아래에 모델의 클래스를 번호 순서대로 적습니다. 0번이 모델의 클래스 0입니다.
# - 클래스 수는 가중치(.pt)의 클래스 수와 같아야 올라갑니다.
# - 학습에 쓴 data.yaml이 있으면 그 파일을 그대로 올려도 됩니다 (train · val 같은 줄은 읽지 않습니다).
"""
SAMPLES = {
    "example": ("classes-example.yaml", "예시 (클래스 3개)"),
    "coco": ("coco80.yaml", "COCO 80 클래스 (Ultralytics 사전학습 검출 · 분할 가중치)"),
    "coco-pose": ("coco-pose.yaml", "COCO 사람 키포인트 (Ultralytics 사전학습 pose 가중치)"),
}


def sample_yaml(kind: str = "example") -> tuple[str, str]:
    """A yaml to start from: (file name, text)."""
    if kind not in SAMPLES:
        raise ServiceError(404, f"샘플은 {', '.join(SAMPLES)} 중 하나입니다.")
    file_name = SAMPLES[kind][0]
    if kind == "example":
        body = "names:\n  0: person\n  1: car\n  2: dog\n"
        return file_name, _HEADER + "# 예: 사람 · 자동차 · 개 3개를 찾는 모델\n" + body
    folder = "coco8" if kind == "coco" else "coco8-pose"
    names = parse_yolo_yaml((_SAMPLES / folder / "data.yaml").read_text(encoding="utf-8"))["names"]
    lines = "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    extra = "kpt_shape: [17, 3]  # 키포인트 17개 (x, y, 보임)\n" if kind == "coco-pose" else ""
    what = "yolo11n-pose.pt 같은 사람 키포인트" if kind == "coco-pose" else "yolo11n.pt · yolo11n-seg.pt 같은 COCO"
    return file_name, _HEADER + f"# {what} 사전학습 가중치의 클래스\n" + extra + "names:\n" + lines
