"""Built-in job types shared by Hub (validation, UI, MCP) and Node (execution).

Training runs the code of a model template in that template's environment
(``badamoa.core.templates``); the other types need nothing beyond the base install.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobTypeSpec(BaseModel):
    type: str
    title: str
    description: str
    params_schema: dict[str, Any] = Field(default_factory=dict)
    gpus: int = 0  # GPUs the job needs by default
    hidden: bool = False  # started from its own screen, not the generic job dialog


_TRAIN_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "release_id": {"type": "string"},
        # Snapshots taken when the run was created, so the node needs nothing else to start:
        "template": {"type": "object"},  # id, name, version, builtin, sha256, entrypoint, data_format
        "environment": {"type": "object"},  # kind=auto: requirements, python, torch · kind=python: path
        "params": {"type": "object", "default": {}},  # the template's own settings
        # Class ids of the release's label space to train on; empty means all of them.
        "classes": {"type": "array", "default": []},
        # Images nobody labelled: leave them out, or feed them as background (the YOLO convention).
        "unlabelled": {"type": "string", "default": "exclude"},
        # Used only when the release has no val split: this share of train is held out.
        "val_fraction": {"type": "number", "minimum": 0, "maximum": 0.9, "default": 0.2},
        # Start from a registered model's weights: {"blob", "file_name"}.
        "init_weights": {"type": "object"},
        # Pick up an interrupted run: {"job_id", "blob", "file_name", "epoch"} of its last checkpoint.
        "resume": {"type": "object"},
    },
}

BUILTIN_JOB_TYPES: dict[str, JobTypeSpec] = {
    spec.type: spec
    for spec in [
        JobTypeSpec(
            type="data.sync",
            title="이미지 받기",
            description="학습할 노드가 버전의 이미지를 그 이미지가 있는 노드에서 직접 받아 둡니다(Hub를 거치지 않음).",
            hidden=True,
            params_schema={"type": "object", "properties": {"release_id": {"type": "string"}}},
        ),
        JobTypeSpec(
            type="data.check",
            title="받아 둔 이미지 확인",
            description=(
                "다른 노드에서 받아 둔 이미지가 Hub의 기록과 맞는지 보고, clean이면 남길 버전이 쓰지 않는 사본을 지웁니다."
            ),
            hidden=True,
            params_schema={"type": "object", "properties": {"clean": {"type": "boolean", "default": False}}},
        ),
        JobTypeSpec(
            type="system.info",
            title="시스템 정보",
            description="노드의 OS, CPU, 메모리, 디스크, GPU, 드라이버 정보를 수집합니다.",
            params_schema={"type": "object", "properties": {}},
        ),
        JobTypeSpec(
            type="system.gpu",
            title="GPU 점검",
            description="배정받은 GPU에서 nvidia-smi를 실행합니다. GPU 단위 배정 확인용.",
            params_schema={"type": "object", "properties": {}},
            gpus=1,
        ),
        JobTypeSpec(
            type="data.import",
            title="데이터셋 가져오기",
            description="노드에 있는 YOLO/COCO 형식 데이터셋을 복사 없이 등록합니다.",
            hidden=True,
            params_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "dataset": {"type": "string"},
                    "version": {"type": "string"},
                    "path": {"type": "string"},
                    "format": {"type": "string", "default": "auto"},
                    "label_space": {"type": "string", "default": ""},
                    "site": {"type": "string", "default": ""},
                    "camera": {"type": "string", "default": ""},
                    "name_pattern": {"type": "string", "default": ""},
                    "note": {"type": "string", "default": ""},
                    "base_release_id": {"type": "string", "default": ""},
                    "task": {"type": "string", "default": ""},
                    # an image without a label file: auto | background | unlabelled (not labelled yet)
                    "unlabelled": {"type": "string", "default": "auto"},
                    "classes": {"type": "array", "default": []},  # class names for a folder that names none
                },
            },
        ),
        JobTypeSpec(
            type="data.export",
            title="데이터셋 버전 내보내기",
            description="데이터셋 버전을 YOLO · COCO · 클래스별 폴더 형식의 zip으로 묶어 Hub에 올립니다 (내려받기용).",
            hidden=True,
            params_schema={
                "type": "object",
                "properties": {
                    "release_id": {"type": "string"},
                    "format": {"type": "string", "default": "yolo"},  # yolo | coco | imagefolder
                    "images": {"type": "boolean", "default": True},  # False: the labels only
                    "task": {"type": "string", "default": "detect"},
                },
            },
        ),
        JobTypeSpec(
            type="train.run",
            title="학습",
            description="데이터셋 버전을 학습 형식으로 풀고 템플릿의 학습 코드를 실행해 지표와 모델을 Hub에 기록합니다.",
            hidden=True,
            gpus=1,
            params_schema=_TRAIN_PARAMS,
        ),
        JobTypeSpec(
            type="model.run",
            title="모델 작업",
            description="등록된 모델을 평가하거나, 다른 형식으로 내보내거나, 올린 이미지로 테스트합니다.",
            hidden=True,
            params_schema={
                "type": "object",
                "properties": {
                    "op_id": {"type": "string"},
                    "model_id": {"type": "string"},
                    "action": {"type": "string"},  # eval | export | predict
                    "template": {"type": "object"},
                    "environment": {"type": "object"},
                    "params": {"type": "object", "default": {}},
                    "train_params": {"type": "object", "default": {}},
                    "classes": {"type": "array", "default": []},
                    "weights": {"type": "object"},
                    "release_id": {"type": "string"},
                    "split": {"type": "string"},
                    "class_ids": {"type": "array"},
                },
            },
        ),
        JobTypeSpec(
            type="env.check",
            title="환경 확인",
            description="노드에 있는 Python 환경이 실행되는지, 어떤 라이브러리가 들어 있는지 확인합니다.",
            hidden=True,
            params_schema={
                "type": "object",
                "properties": {
                    "environment_id": {"type": "string"},
                    "python": {"type": "string"},  # a server's Python
                    "kind": {"type": "string", "default": "python"},  # docker: run Python in ``image``
                    "image": {"type": "string", "default": ""},
                },
            },
        ),
        JobTypeSpec(
            type="env.prepare",
            title="환경 미리 준비",
            description="모델 종류의 학습 환경을 첫 학습 전에 미리 설치합니다.",
            hidden=True,
            params_schema={
                "type": "object",
                "properties": {
                    "template": {"type": "string"},
                    "version": {"type": "string"},
                    "environment": {"type": "object"},  # kind=auto: requirements, python, torch · docker: image
                },
            },
        ),
        JobTypeSpec(
            type="env.remove",
            title="설치한 환경 지우기",
            description="모델 종류의 학습 환경(가상환경, 또는 BADAMOA가 받은 Docker 이미지)을 서버에서 지웁니다.",
            hidden=True,
            params_schema={
                "type": "object",
                "properties": {"template": {"type": "string"}, "environment": {"type": "object"}},
            },
        ),
        JobTypeSpec(
            type="debug.sleep",
            title="테스트 대기",
            description="지정한 초 동안 1초마다 로그를 출력합니다. 로그 스트리밍·취소 테스트용.",
            params_schema={
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "minimum": 1, "maximum": 3600, "default": 10},
                    "message": {"type": "string", "default": "tick"},
                    "fail": {"type": "boolean", "default": False},
                },
            },
        ),
    ]
}


def validate_params(job_type: str, params: dict[str, Any]) -> dict[str, Any]:
    """Light validation against the declared schema (types + bounds + defaults)."""
    spec = BUILTIN_JOB_TYPES.get(job_type)
    if spec is None:
        raise ValueError(f"unknown job type: {job_type}")
    props: dict[str, Any] = spec.params_schema.get("properties", {})
    out: dict[str, Any] = {}
    for name, schema in props.items():
        value = params.get(name, schema.get("default"))
        if value is None:
            continue
        kind = schema.get("type")
        if kind in ("integer", "number"):
            try:
                value = int(value) if kind == "integer" else float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be a number") from e
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"{name} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"{name} must be <= {schema['maximum']}")
        elif kind == "string":
            value = str(value)
        elif kind == "boolean":
            value = bool(value)
        elif kind == "array" and not isinstance(value, list):
            raise ValueError(f"{name} must be a list")
        elif kind == "object" and not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        out[name] = value
    unknown = set(params) - set(props)
    if unknown:
        raise ValueError(f"unknown params for {job_type}: {sorted(unknown)}")
    return out
