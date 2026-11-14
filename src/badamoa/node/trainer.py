"""Training jobs (``train.run``): export the release, run the template's code, report to the Hub.

    1. fetch the release's manifest from the Hub and write it in the template's data format
    2. get the template's code (built-in, or downloaded from the Hub and cached by hash)
    3. get the environment's Python (built on first use, or an existing one)
    4. run the template's train command, forwarding its log and metrics as they come
    5. upload the weights and register the model

The template's code knows nothing about BADAMOA (see ``badamoa.templates``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from badamoa import templates as shipped
from badamoa.core import templates as template_format

from . import envs, exporter

EVENT_PREFIX = "@@badamoa "
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # colours and cursor moves; the job log is plain text
PROGRESS = re.compile(r"\d+%\s*[━─╸]|\dit/s|s/it")  # one redraw of a progress bar
PROGRESS_EVERY_S = 5.0
MANIFEST_PAGE = 2000


def _device() -> str:
    """The Hub exposes only the GPUs it reserved, so they are always numbered from 0 here."""
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    return ",".join(str(i) for i in range(len(visible))) if visible else "cpu"


def _manifest(client: httpx.Client, base: str) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    """The manifest header plus a lazy stream of its items (paged by item id)."""

    def page(after: int) -> dict[str, Any]:
        resp = client.get(f"{base}/manifest", params={"after": after, "limit": MANIFEST_PAGE})
        if resp.status_code != 200:
            raise RuntimeError(f"Hub가 거부했습니다 ({resp.status_code}): {resp.text[:300]}")
        return resp.json()

    first = page(0)

    def items() -> Iterator[dict[str, Any]]:
        current = first
        while True:
            yield from current["items"]
            if current.get("next") is None:
                return
            current = page(current["next"])

    return first, items()


def template_dir(client: httpx.Client, base: str, node_home: Path, template: dict[str, Any]) -> Path:
    """The folder holding the template's code on this node."""
    if template.get("builtin"):
        folder = shipped.builtin().get(template["name"])
        if folder is None:
            raise RuntimeError(f"이 노드의 BADAMOA에는 내장 템플릿 {template['name']}이 없습니다. 노드를 업데이트하세요.")
        return folder
    digest = template["sha256"]
    folder = node_home / "cache" / "templates" / digest
    if (folder / template_format.MANIFEST).is_file():
        return folder
    resp = client.get(f"{base}/template", timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"템플릿을 받지 못했습니다 ({resp.status_code}): {resp.text[:300]}")
    staging = folder.with_name(f"{digest}.{os.getpid()}.part")
    shutil.rmtree(staging, ignore_errors=True)
    template_format.unpack(resp.content, staging)
    try:
        staging.rename(folder)
    except OSError:  # another job unpacked it first
        shutil.rmtree(staging, ignore_errors=True)
    return folder


def command_for(entrypoint: str, python: Path, config_path: Path) -> list[str]:
    """``python train.py --config {config}`` → the environment's Python and the real config path."""
    parts = shlex.split(entrypoint, posix=os.name != "nt")
    parts = [part.strip('"') for part in parts]
    if parts and parts[0] in ("python", "python3"):
        parts[0] = str(python)
    return [part.replace("{config}", str(config_path)) for part in parts]


def run(params: dict[str, Any]) -> dict[str, Any]:
    hub, token, job_id, home = (
        os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID", "BADAMOA_NODE_HOME")
    )
    if not (hub and token and job_id and home):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    node_home = Path(home).resolve()
    work = node_home / "runs" / job_id
    base = f"/api/agent/jobs/{job_id}/train"
    template = params["template"]

    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:

        def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            resp = client.post(base + path, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"Hub가 거부했습니다 ({resp.status_code}): {resp.text[:300]}")
            return resp.json()

        def stage(name: str) -> None:
            post("/report", {"event": "stage", "data": {"stage": name}})

        # 1. dataset
        stage("data")
        header, items = _manifest(client, base)
        print(f"릴리스 {header['dataset']} {header['version']} 를 학습 형식으로 준비합니다: {work / 'data'}")
        options = exporter.ExportOptions(
            classes=[int(c) for c in params.get("classes") or []],
            unlabelled=params.get("unlabelled") or "exclude",
            val_fraction=float(params.get("val_fraction", 0.2)),
        )
        shutil.rmtree(work, ignore_errors=True)
        summary = exporter.build(items, header["classes"], header["locations"], work / "data", options)
        counts, left_out = summary["items"], summary["excluded"]
        print(
            f"이미지 train {counts['train']:,} · val {counts['val']:,} · test {counts['test']:,} · "
            f"박스 {summary['boxes']:,} · 클래스 {len(summary['classes'])}개"
        )
        if summary["val_source"] == "holdout":
            print(f"릴리스에 val split이 없어 train의 {options.val_fraction:.0%}를 검증용으로 떼었습니다.")
        elif summary["val_source"] == "train":
            print("검증용으로 뗄 이미지가 부족해 train으로 검증합니다. 지표가 실제보다 높게 나옵니다.")
        if any(left_out.values()):
            print(
                f"제외: 미라벨 {left_out['unlabelled']:,} · 일부 클래스만 라벨 {left_out['partial']:,} · "
                f"파일 없음 {left_out['missing']:,}"
            )
        post("/report", {"event": "export", "data": {k: v for k, v in summary.items() if k != "data_yaml"}})

        # 2–3. code and environment
        stage("env")
        code = template_dir(client, base, node_home, template)
        python = envs.resolve(node_home, params.get("environment") or {})

        # 4. train
        cache = node_home / "cache" / "template-data" / template["name"]
        cache.mkdir(parents=True, exist_ok=True)
        (node_home / "cache" / "ultralytics").mkdir(parents=True, exist_ok=True)
        config = {
            "data": summary["data_yaml"],
            "data_dir": str(work / "data"),
            "out": str(work / "out"),
            "cache": str(cache),
            "device": _device(),
            "classes": summary["classes"],
            "params": params.get("params") or {},
        }
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "NO_COLOR": "1",  # the log is plain text
            # Template code can ``import badamoa_sdk`` without shipping or installing it.
            "PYTHONPATH": os.pathsep.join(p for p in (str(shipped.SDK), os.environ.get("PYTHONPATH", "")) if p),
            # Ultralytics keeps its settings here instead of touching the user's own.
            "YOLO_CONFIG_DIR": str(node_home / "cache" / "ultralytics"),
        }
        command = command_for(template["entrypoint"], python, config_path)
        print(f"학습 시작: {template['name']} {template['version']} · 장치 {config['device']}", flush=True)
        stage("train")
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=code,
            )
        except OSError as e:
            raise RuntimeError(f"학습 명령을 실행하지 못했습니다 ({command[0]}): {e}") from e
        assert proc.stdout is not None
        outcome: dict[str, Any] | None = None
        last_progress = 0.0
        for raw in proc.stdout:
            line = ANSI.sub("", raw).rstrip()
            if not line.startswith(EVENT_PREFIX):
                if not line.strip():
                    continue
                if PROGRESS.search(line) and "100%" not in line:
                    # A bar redraws many times a second; the log keeps one state every few seconds.
                    if time.monotonic() - last_progress < PROGRESS_EVERY_S:
                        continue
                    last_progress = time.monotonic()
                print(line, flush=True)
                continue
            try:
                event = json.loads(line[len(EVENT_PREFIX) :])
            except ValueError:
                continue
            kind = event.pop("event", "")
            if kind == "result":
                outcome = event
            elif kind == "info":
                post("/report", {"event": "info", "data": event})
            elif kind == "epoch":  # one epoch finished: metrics at that epoch, and progress
                post("/report", {"event": "epoch", "data": event})
            elif kind == "log":  # metrics at any step, without touching progress
                post("/report", {"event": "epoch", "data": {"step": event.get("step", 0), "metrics": event.get("metrics")}})
            elif kind == "progress":
                post("/report", {"event": "epoch", "data": {"epoch": event.get("current"), "epochs": event.get("total")}})
        if proc.wait() != 0:
            raise RuntimeError(f"학습 프로세스가 종료 코드 {proc.returncode}로 끝났습니다.")
        if outcome is None:
            raise RuntimeError("학습 코드가 결과를 알리지 않았습니다. 끝날 때 result(또는 finish)를 호출해야 합니다.")

        # 5. weights → Hub, then register the model
        stage("upload")
        uploaded: dict[str, Any] = {}
        weights = Path(outcome["weights"]) if outcome.get("weights") else None
        if weights is not None and not weights.is_absolute():
            weights = code / weights
        if weights is not None and weights.is_file():
            print(f"가중치 업로드: {weights.name} ({weights.stat().st_size / 1024**2:.1f} MB)", flush=True)
            with weights.open("rb") as stream:
                resp = client.put(f"{base}/artifact", params={"name": weights.name}, content=stream, timeout=3600)
            if resp.status_code != 200:
                raise RuntimeError(f"가중치 업로드 실패 ({resp.status_code}): {resp.text[:300]}")
            uploaded = resp.json()
        elif weights is not None:
            raise RuntimeError(f"학습 코드가 알린 가중치 파일이 없습니다: {weights}")
        final = {"metrics": outcome.get("metrics") or {}, "classes": outcome.get("classes") or [], "weights": uploaded}
        result = post("/finish", final)
    shutil.rmtree(work, ignore_errors=True)  # the export is links and the weights are on the Hub now
    if result.get("model"):
        print(f"완료: 모델 {result['model']['name']} v{result['model']['version']} 등록")
    else:
        print("완료: 지표를 기록했습니다. 가중치를 알리지 않아 모델은 등록하지 않았습니다.")
    return result

