"""Training jobs (``train.*``): export the release, run the trainer, report to the Hub.

    1. fetch the release's manifest from the Hub and write it as a YOLO dataset
    2. make sure the plugin environment exists (first use builds it)
    3. run the trainer script in that environment, forwarding its log and metrics
    4. upload the weights and register the model

The trainer script itself knows nothing about BADAMOA (see ``badamoa.plugins``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from badamoa import plugins
from badamoa.core.jobs import BUILTIN_JOB_TYPES
from badamoa.core.plugins import PLUGINS

from . import envs, exporter

EVENT_PREFIX = "@@badamoa "
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # colours and cursor moves; the job log is plain text
PROGRESS = re.compile(r"\d+%\s*[━─╸]|\dit/s|s/it")  # one redraw of a progress bar
PROGRESS_EVERY_S = 5.0
SCRIPTS = {"train.yolo": "yolo_train.py", "train.dummy": "dummy_train.py"}
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


def run(job_type: str, params: dict[str, Any]) -> dict[str, Any]:
    hub, token, job_id, home = (
        os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID", "BADAMOA_NODE_HOME")
    )
    if not (hub and token and job_id and home):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    node_home = Path(home).resolve()
    work = node_home / "runs" / job_id
    base = f"/api/agent/jobs/{job_id}/train"

    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:

        def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            resp = client.post(base + path, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"Hub가 거부했습니다 ({resp.status_code}): {resp.text[:300]}")
            return resp.json()

        # 1. dataset
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

        # 2. environment
        spec = BUILTIN_JOB_TYPES[job_type]
        python = str(envs.ensure(node_home, PLUGINS[spec.plugin])) if spec.plugin else sys.executable

        # 3. train
        config = {
            "data": summary["data_yaml"],
            "out": str(work / "out"),
            "device": _device(),
            "classes": summary["classes"],
            **{k: params[k] for k in ("model", "epochs", "imgsz", "batch", "workers", "pretrained", "extra") if k in params},
        }
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        cache = node_home / "cache"
        (cache / "weights").mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "NO_COLOR": "1",  # the log is plain text
            "YOLO_CONFIG_DIR": str(cache / "ultralytics"),  # keep the user's own Ultralytics settings untouched
        }
        (cache / "ultralytics").mkdir(parents=True, exist_ok=True)
        script = Path(plugins.__file__).parent / SCRIPTS[job_type]
        print(f"학습 시작: {params.get('model')} · epoch {params.get('epochs')} · 장치 {config['device']}", flush=True)
        proc = subprocess.Popen(
            [python, str(script), str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cache / "weights",  # pretrained weights download here once and are reused
        )
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
            elif kind in ("info", "epoch"):
                post("/report", {"event": kind, "data": event})
        if proc.wait() != 0:
            raise RuntimeError(f"학습 프로세스가 종료 코드 {proc.returncode}로 끝났습니다.")
        if outcome is None:
            raise RuntimeError("학습 프로세스가 결과를 남기지 않았습니다.")

        # 4. weights → Hub, then register the model
        weights = Path(outcome["weights"])
        print(f"가중치 업로드: {weights.name} ({weights.stat().st_size / 1024**2:.1f} MB)", flush=True)
        with weights.open("rb") as stream:
            resp = client.put(f"{base}/artifact", params={"name": weights.name}, content=stream, timeout=3600)
        if resp.status_code != 200:
            raise RuntimeError(f"가중치 업로드 실패 ({resp.status_code}): {resp.text[:300]}")
        result = post(
            "/finish", {"metrics": outcome.get("metrics", {}), "classes": outcome.get("classes", []), "weights": resp.json()}
        )
    shutil.rmtree(work, ignore_errors=True)  # the export is links and the weights are on the Hub now
    print(f"완료: 모델 {result['model']['name']} v{result['model']['version']} 등록")
    return result
