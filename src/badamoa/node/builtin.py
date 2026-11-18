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


def data_export(params: dict[str, Any]) -> dict[str, Any]:
    from badamoa.node import exporter

    return exporter.run_download(params)


def data_sync(params: dict[str, Any]) -> dict[str, Any]:
    from badamoa.node import sync

    return sync.run(params)


def data_check(params: dict[str, Any]) -> dict[str, Any]:
    from badamoa.node import sync

    return sync.check(params)


def train_run(params: dict[str, Any]) -> dict[str, Any]:
    from badamoa.node import trainer

    return trainer.run(params)


def model_run(params: dict[str, Any]) -> dict[str, Any]:
    from badamoa.node import modelops

    return modelops.run(params)


def env_check(params: dict[str, Any]) -> dict[str, Any]:
    from badamoa.node import containers, envs

    if params.get("kind") == "docker":  # a registered Docker image: have it here, then run Python in it
        home = os.environ.get("BADAMOA_NODE_HOME")
        if not home:
            raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
        image = str(params.get("image", ""))
        image_id = containers.ensure(Path(home).resolve(), image)
        gpus = bool((containers.capability() or {}).get("gpu"))
        done = containers.run_python(image, envs.PROBE, gpus=gpus)
        if done.returncode != 0:
            raise RuntimeError(f"이미지 안의 Python이 오류로 끝났습니다: {done.stderr.strip()[-400:]}")
        found = {"ok": True, "image": image, "image_id": image_id, **json.loads(done.stdout.strip().splitlines()[-1])}
    else:
        found = envs.check(str(params.get("python", "")))
    print(f"python  : {found.get('python')} ({found.get('executable')})")
    for name, version in found.get("packages", {}).items():
        print(f"{name:<8}: {version}")
    if "cuda" in found:
        gpus = ", ".join(found.get("gpus") or []) if found["cuda"] else "PyTorch가 GPU를 쓰지 못합니다 (CPU만)"
        print(f"GPU     : {gpus}")
    return found


def env_prepare(params: dict[str, Any]) -> dict[str, Any]:
    """Build a template's environment ahead of its first training, so that one does not wait for the install."""
    from badamoa.node import containers, envs

    home = os.environ.get("BADAMOA_NODE_HOME")
    if not home:
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    started = time.monotonic()
    spec = params.get("environment") or {}
    if spec.get("kind") == "docker":  # an image: pull it now
        image_id = containers.ensure(Path(home).resolve(), str(spec.get("image", "")))
        took = time.monotonic() - started
        print(f"{params.get('template', '')} Docker 이미지 준비됨: {spec.get('image')} ({image_id[:19]}, {took:.0f}초)")
        return {"image": spec.get("image"), "image_id": image_id}
    python = envs.resolve(Path(home).resolve(), params.get("environment") or {})
    installed = envs.packages(python)
    took = time.monotonic() - started
    print(f"{params.get('template', '')} 환경 준비됨: {python} (패키지 {len(installed)}개, {took:.0f}초)")
    return {"python": str(python), "packages": len(installed)}


def env_remove(params: dict[str, Any]) -> dict[str, Any]:
    """Take a template's installed environment off this node (a virtual environment, or an image BADAMOA pulled)."""
    from badamoa.node import containers, envs

    home = os.environ.get("BADAMOA_NODE_HOME")
    if not home:
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    spec = params.get("environment") or {}
    if spec.get("kind") == "docker":
        return containers.remove(Path(home).resolve(), str(spec.get("image", "")))
    return envs.remove(Path(home).resolve(), spec)


HANDLERS = {
    "system.info": system_info,
    "system.gpu": system_gpu,
    "debug.sleep": debug_sleep,
    "data.import": data_import,
    "data.export": data_export,
    "data.sync": data_sync,
    "data.check": data_check,
    "train.run": train_run,
    "model.run": model_run,
    "env.check": env_check,
    "env.prepare": env_prepare,
    "env.remove": env_remove,
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
