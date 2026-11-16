"""Model templates: a folder of training / evaluation / inference code plus one declaration file.

``badamoa.toml`` says what the code can do (entry points), what the user may
set (params, which become the form in the training dialog), which data format
it reads and which environment it needs. The built-in Ultralytics YOLO trainer
is a template like any other (``badamoa/templates/yolo``).

The code itself never imports BADAMOA: it gets a JSON config file and reports
through stdout lines starting with ``@@badamoa `` (see ``badamoa.templates``).
"""

from __future__ import annotations

import ast
import io
import re
import shlex
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from badamoa.core import tasks as task_names

MANIFEST = "badamoa.toml"
TASKS = ("detect", "segment", "classify")  # what the built-in templates train; a template may name any task
DATA_FORMATS = ("yolo", "yolo-seg", "imagefolder")  # coco and manifest come with the templates that need them
ENTRYPOINTS = ("train", "eval", "predict", "export")
PARAM_TYPES = ("string", "integer", "number", "boolean", "object")
# Set by BADAMOA for every run; a template may not use these as param names.
RESERVED = {
    "classes", "unlabelled", "val_fraction", "gpus", "min_vram_mb", "model_name", "run_id", "release_id", "base_model",
    "resumed_from",
}  # fmt: skip
MAX_BUNDLE_BYTES = 64 * 1024**2
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea", ".vscode", "runs", "wandb"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


class TemplateError(ValueError):
    """The declaration or the bundle is not usable; the message is shown to whoever registers it."""


class ParamSpec(BaseModel):
    type: str = "string"
    default: Any = None
    label: str = ""
    help: str = ""
    choices: list[Any] = Field(default_factory=list)
    custom: bool = False  # with choices: also allow a value typed by hand
    minimum: float | None = None
    maximum: float | None = None
    advanced: bool = False  # folded under "세부 설정" in the dialog
    group: str = ""  # heading inside "세부 설정", e.g. "최적화"
    # The code does not use this setting while another one has one of these values:
    # ``unused_when = { optimizer = ["auto"] }`` (Ultralytics picks the learning rate itself then).
    unused_when: dict[str, list[Any]] = Field(default_factory=dict)


class EnvironmentSpec(BaseModel):
    default: str = ""  # name of an environment registered on the Hub; empty = install automatically
    requirements: list[str] = Field(default_factory=list)
    python: str = "3.12"
    torch: bool = False  # pick the PyTorch build that matches the node's GPU driver


class TemplateManifest(BaseModel):
    name: str
    title: str = ""
    version: str
    description: str = ""
    task: str = "detect"
    data_format: str = "yolo"
    license: str = ""
    hidden: bool = False  # not offered in the training dialog (test helpers)
    variant: str = ""  # the param that names the model variant, e.g. "model" → yolo11n
    # The train code can pick up where an interrupted run stopped (it reads ``resume_dir`` from its config).
    resume: bool = False
    entrypoints: dict[str, str] = Field(default_factory=dict)
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)


# ---- settings declared in the code itself (``badamoa_sdk.params(...)``) ---------------

_DECLARATIONS = {"choice": "string", "text": "string", "integer": "integer", "number": "number", "boolean": "boolean"}
_OPTIONS = {"label", "help", "group", "advanced", "custom", "unused_when"}


def entry_script(command: str) -> str:
    """The Python file a command starts, e.g. ``python train.py --config {config}`` → ``train.py``."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return ""
    return next((part for part in parts if part.endswith(".py")), "")


def _train_script(manifest_text: str) -> str:
    """The train script named in a declaration file, read leniently (full checks come later)."""
    try:
        command = tomllib.loads(manifest_text).get("entrypoints", {}).get("train", "")
    except tomllib.TOMLDecodeError:
        return ""
    return entry_script(command) if isinstance(command, str) else ""


def params_from_code(source: str, file_name: str = "train.py") -> dict[str, ParamSpec]:
    """Read the settings a script declares with ``params(name=integer(…), …)`` without running it.

    Only the syntax is looked at, so the values must be written out (numbers, strings,
    lists) — that is what makes it safe to do on the Hub, where the code never runs.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise TemplateError(f"{file_name}을 읽을 수 없습니다 ({e.lineno}행): {e.msg}") from e
    found: dict[str, ParamSpec] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", getattr(node.func, "id", "")) != "params":
            continue
        for keyword in node.keywords:
            call = keyword.value
            kind = getattr(getattr(call, "func", None), "attr", getattr(getattr(call, "func", None), "id", ""))
            if keyword.arg is None or not isinstance(call, ast.Call) or kind not in _DECLARATIONS:
                continue  # not a declaration of ours: some other function that happens to be called params
            where = f"{file_name} {call.lineno}행, 설정 {keyword.arg}"
            try:
                args = [ast.literal_eval(a) for a in call.args]
                options = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg}
            except (ValueError, SyntaxError):
                raise TemplateError(f"{where}: 선언에는 값을 직접 적어야 합니다 (변수나 계산식은 쓸 수 없습니다).") from None
            spec: dict[str, Any] = {"type": _DECLARATIONS[kind], **{k: v for k, v in options.items() if k in _OPTIONS}}
            if kind == "choice":
                choices = list(args[0] if args else options.get("options", []))
                if not choices:
                    raise TemplateError(f"{where}: choice에는 고를 값의 목록이 필요합니다.")
                spec["choices"] = choices
                spec["default"] = options.get("default", args[1] if len(args) > 1 else choices[0])
            else:
                spec["default"] = options.get("default", args[0] if args else ("" if kind == "text" else None))
                spec["minimum"], spec["maximum"] = options.get("min"), options.get("max")
            try:
                found[keyword.arg] = ParamSpec(**spec)
            except ValidationError:
                raise TemplateError(f"{where}: 선언의 값이 올바르지 않습니다.") from None
    return found


def parse_manifest(text: str, code: str | None = None) -> TemplateManifest:
    """``code`` is the source of the train script; settings it declares are added to those in the file."""
    try:
        manifest = TemplateManifest(**tomllib.loads(text))
    except tomllib.TOMLDecodeError as e:
        raise TemplateError(f"{MANIFEST}을 읽을 수 없습니다: {e}") from e
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5])
        raise TemplateError(f"{MANIFEST}의 형식이 맞지 않습니다 — {problems}") from e
    if code:
        declared = params_from_code(code, entry_script(manifest.entrypoints.get("train", "")) or "train.py")
        # Code order first; what the file says about the same setting wins.
        manifest.params = {**declared, **manifest.params}
    for name, spec in manifest.params.items():
        for other in spec.unused_when:
            if other not in manifest.params:
                raise TemplateError(f"params.{name}.unused_when: {other}라는 설정이 없습니다.")
    if not _NAME_RE.match(manifest.name):
        raise TemplateError("name은 영문 소문자·숫자·`-`·`_`·`.`로 2~63자여야 합니다 (예: inhouse-rtdetr).")
    if not manifest.version.strip():
        raise TemplateError("version을 적어 주세요 (예: 1.0.0).")
    try:  # detect · segment · pose · classify (Det, Seg, Kpt, Cls), or a task a team names, such as ocr
        manifest.task = task_names.normalize(manifest.task) or "detect"
    except ValueError as e:
        raise TemplateError(f"task: {e}") from e
    if manifest.data_format not in DATA_FORMATS:
        raise TemplateError(f"data_format은 지금 {', '.join(DATA_FORMATS)}만 지원합니다.")
    unknown = set(manifest.entrypoints) - set(ENTRYPOINTS)
    if unknown:
        raise TemplateError(f"알 수 없는 진입점: {', '.join(sorted(unknown))}. {', '.join(ENTRYPOINTS)}만 쓸 수 있습니다.")
    if not manifest.entrypoints.get("train", "").strip():
        raise TemplateError("[entrypoints]에 train 명령이 필요합니다 (예: train = \"python train.py --config {config}\").")
    for kind, command in manifest.entrypoints.items():
        if "{config}" not in command:
            raise TemplateError(f"{kind} 명령에 설정 파일이 들어갈 자리 {{config}}가 없습니다.")
    for name, spec in manifest.params.items():
        if name in RESERVED:
            raise TemplateError(f"설정 이름 {name}은 BADAMOA가 쓰는 이름이라 사용할 수 없습니다.")
        if spec.type not in PARAM_TYPES:
            raise TemplateError(f"설정 {name}의 type은 {', '.join(PARAM_TYPES)} 중 하나여야 합니다.")
        try:
            if spec.default is not None:
                spec.default = _coerce(name, spec, spec.default)
        except TemplateError as e:
            raise TemplateError(f"기본값이 맞지 않습니다 — {e}") from e
    if manifest.variant and manifest.variant not in manifest.params:
        raise TemplateError(f"variant가 가리키는 설정 {manifest.variant}이 [params]에 없습니다.")
    manifest.title = manifest.title or manifest.name
    return manifest


def load_manifest(folder: Path) -> TemplateManifest:
    path = folder / MANIFEST
    if not path.is_file():
        raise TemplateError(f"{folder}에 {MANIFEST}이 없습니다. `badamoa template init`으로 뼈대를 만들 수 있습니다.")
    text = path.read_text(encoding="utf-8")
    script = folder / _train_script(text)
    return parse_manifest(text, script.read_text(encoding="utf-8") if script.is_file() else None)


def _coerce(name: str, spec: ParamSpec, value: Any) -> Any:
    try:
        if spec.type == "integer":
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            value = int(value)
        elif spec.type == "number":
            if isinstance(value, bool):
                raise ValueError
            value = float(value)
        elif spec.type == "boolean":
            if not isinstance(value, bool):
                raise ValueError
        elif spec.type == "string":
            if isinstance(value, dict | list):
                raise ValueError
            value = str(value)
        elif spec.type == "object" and not isinstance(value, dict):
            raise ValueError
    except (TypeError, ValueError):
        raise TemplateError(f"{spec.label or name}: {spec.type} 값이어야 합니다.") from None
    if spec.type in ("integer", "number"):
        if spec.minimum is not None and value < spec.minimum:
            raise TemplateError(f"{spec.label or name}: {spec.minimum:g} 이상이어야 합니다.")
        if spec.maximum is not None and value > spec.maximum:
            raise TemplateError(f"{spec.label or name}: {spec.maximum:g} 이하여야 합니다.")
    if spec.choices and not spec.custom and value not in spec.choices:
        raise TemplateError(f"{spec.label or name}: {', '.join(str(c) for c in spec.choices)} 중 하나여야 합니다.")
    return value


def resolve_params(manifest: TemplateManifest, given: dict[str, Any]) -> dict[str, Any]:
    """Check user-supplied settings against the declaration and fill in defaults."""
    unknown = set(given) - set(manifest.params)
    if unknown:
        raise TemplateError(f"{manifest.title}에 없는 설정입니다: {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for name, spec in manifest.params.items():
        value = given.get(name, spec.default)
        if value is None or value == "" and spec.type != "string":
            if spec.default is None:
                continue
            value = spec.default
        out[name] = _coerce(name, spec, value)
    return out


# ---- bundles (a template folder as one zip) ----------------------------------------


def bundle(folder: Path) -> bytes:
    """Zip a template folder, leaving out caches, virtual environments and version-control data."""
    load_manifest(folder)
    buffer = io.BytesIO()
    total = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if not path.is_file() or any(part in SKIP_DIRS for part in relative.parts) or path.suffix == ".pyc":
                continue
            total += path.stat().st_size
            if total > MAX_BUNDLE_BYTES:
                limit = MAX_BUNDLE_BYTES // 1024**2
                raise TemplateError(f"템플릿이 {limit} MB를 넘습니다. 가중치나 데이터는 템플릿에 넣지 마세요.")
            # A fixed timestamp keeps the bundle's hash the same for the same content.
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def read_bundle(data: bytes) -> TemplateManifest:
    """The declaration inside a bundle; also rejects archives that are not safe to unpack."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if sum(i.file_size for i in archive.infolist()) > MAX_BUNDLE_BYTES:
                raise TemplateError(f"템플릿을 풀면 {MAX_BUNDLE_BYTES // 1024**2} MB를 넘습니다.")
            for name in names:
                if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name:
                    raise TemplateError(f"압축 파일에 허용되지 않는 경로가 있습니다: {name}")
            if MANIFEST not in names:
                raise TemplateError(f"압축 파일의 맨 위에 {MANIFEST}이 없습니다. 템플릿 폴더의 내용을 그대로 압축하세요.")
            text = archive.read(MANIFEST).decode("utf-8")
            script = _train_script(text)
            return parse_manifest(text, archive.read(script).decode("utf-8") if script in names else None)
    except zipfile.BadZipFile as e:
        raise TemplateError("zip 파일이 아닙니다.") from e
    except UnicodeDecodeError as e:
        raise TemplateError(f"{MANIFEST}은 UTF-8이어야 합니다.") from e


def unpack(data: bytes, target: Path) -> None:
    read_bundle(data)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(target)
