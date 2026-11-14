"""Built-in job types shared by Hub (validation, UI, MCP) and Node (execution).

Plugins (trainers, labelers, …) will register more types later; Phase 0 ships
only diagnostics so the Hub ↔ Node pipeline can be exercised end to end.
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


BUILTIN_JOB_TYPES: dict[str, JobTypeSpec] = {
    spec.type: spec
    for spec in [
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
        if kind == "integer":
            value = int(value)
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"{name} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"{name} must be <= {schema['maximum']}")
        elif kind == "string":
            value = str(value)
        elif kind == "boolean":
            value = bool(value)
        out[name] = value
    unknown = set(params) - set(props)
    if unknown:
        raise ValueError(f"unknown params for {job_type}: {sorted(unknown)}")
    return out
