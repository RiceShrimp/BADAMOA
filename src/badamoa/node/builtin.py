"""Built-in job implementations, run as a subprocess:

    python -m badamoa.node.builtin <type> <params.json> <result.json>

stdout/stderr are streamed to the Hub as the job log; the JSON written to
``result.json`` becomes the job result. Plugin jobs (trainers, labelers, …)
will follow the same contract from their own virtual environments.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from badamoa.core import sysinfo


def system_info(_: dict[str, Any]) -> dict[str, Any]:
    info = sysinfo.static_info()
    print(f"host    : {info['hostname']}")
    print(f"os      : {info['os_release']} ({info['arch']})")
    print(f"cpu     : {info['cpu_count']} logical cores")
    print(f"memory  : {info['memory_total_mb']} MB")
    if not info["gpus"]:
        print("gpu     : (NVIDIA GPU를 찾지 못했습니다)")
    for gpu in info["gpus"]:
        print(f"gpu {gpu['index']}   : {gpu['name']} {gpu['memory_total_mb']} MB, driver {gpu['driver_version']}")
    metrics = sysinfo.metrics()
    print(f"disk    : {metrics['disk_free_gb']} GB free / {metrics['disk_total_gb']} GB")
    return {"info": info, "metrics": metrics}


def debug_sleep(params: dict[str, Any]) -> dict[str, Any]:
    seconds = int(params.get("seconds", 10))
    message = str(params.get("message", "tick"))
    for i in range(1, seconds + 1):
        print(f"{message} {i}/{seconds}", flush=True)
        time.sleep(1)
    if params.get("fail"):
        raise RuntimeError("fail=true 로 요청된 의도적 실패")
    return {"slept_seconds": seconds}


def system_gpu(_: dict[str, Any]) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={visible or '(설정 안 됨)'}")
    assigned = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    found = [g for g in sysinfo.gpus() if g["index"] in assigned]
    for gpu in found:
        print(
            f"gpu {gpu['index']}: {gpu['name']} · {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB · "
            f"{gpu['utilization_pct']}% · {gpu['temperature_c']}°C · {gpu['power_w']} W"
        )
    if len(found) != len(assigned) or not assigned:
        raise RuntimeError("배정받은 GPU를 찾지 못했습니다.")
    return {"assigned": assigned, "gpus": found}


def data_import(params: dict[str, Any]) -> dict[str, Any]:
    from badamoa.node import importer

    return importer.run(params)


def train(job_type: str):
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        from badamoa.node import trainer

        return trainer.run(job_type, params)

    return handler


HANDLERS = {
    "system.info": system_info,
    "system.gpu": system_gpu,
    "debug.sleep": debug_sleep,
    "data.import": data_import,
    "train.yolo": train("train.yolo"),
    "train.dummy": train("train.dummy"),
}


def main(argv: list[str]) -> int:
    job_type, params_path, result_path = argv
    handler = HANDLERS.get(job_type)
    if handler is None:
        print(f"이 노드는 작업 유형 {job_type!r} 을 지원하지 않습니다.", file=sys.stderr)
        return 2
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    try:
        result = handler(params)
    except (RuntimeError, ValueError, FileNotFoundError) as e:  # anticipated: one clear line, no traceback
        print(f"실패: {e}", file=sys.stderr)
        return 1
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
