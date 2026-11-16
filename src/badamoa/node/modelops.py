"""Model jobs (``model.run``): evaluate, export or test a registered model with its template's code.

The node fetches the weights (cached by hash), prepares what the action needs —
the release in the template's data format for ``eval``, the uploaded images for
``predict`` — runs the template's entry point in its environment, and sends the
result and any files the code recorded back to the Hub.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from badamoa import templates as shipped

from . import envs, exporter, trainer


def _fetch(client: httpx.Client, url: str, target: Path, params: dict[str, Any] | None = None) -> Path:
    if not target.is_file():
        resp = client.get(url, params=params, timeout=3600)
        if resp.status_code != 200:
            raise RuntimeError(f"Hub에서 파일을 받지 못했습니다 ({resp.status_code}): {resp.text[:200]}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}.part")
        partial.write_bytes(resp.content)
        partial.replace(target)
    return target


def run(params: dict[str, Any]) -> dict[str, Any]:
    hub, token, job_id, home = (
        os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID", "BADAMOA_NODE_HOME")
    )
    if not (hub and token and job_id and home):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    node_home = Path(home).resolve()
    work = node_home / "runs" / job_id
    base = f"/api/agent/jobs/{job_id}/model"
    template, action = params["template"], params["action"]
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)

    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:
        weights_info = params["weights"]
        cached = node_home / "cache" / "models" / f"{weights_info['blob']}{Path(weights_info['file_name']).suffix}"
        weights = _fetch(client, f"{base}/blob/weights", cached)
        cache = node_home / "cache" / "template-data" / template["name"]
        cache.mkdir(parents=True, exist_ok=True)
        (node_home / "cache" / "ultralytics").mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {
            "action": action,
            "weights": str(weights),
            "out": str(work / "out"),
            "cache": str(cache),
            "device": trainer._device(),
            "classes": params.get("classes") or [],
            "params": params.get("params") or {},
            "train_params": params.get("train_params") or {},
        }

        if action == "eval":
            header, items = trainer.manifest_stream(client, base)
            build = exporter.BUILDERS.get(template.get("data_format", "yolo"))
            if build is None:
                raise RuntimeError("이 노드의 BADAMOA는 이 데이터 형식을 지원하지 않습니다. 노드를 업데이트하세요.")
            options = exporter.ExportOptions(classes=[int(c) for c in params.get("class_ids") or []], keep_splits=True)
            summary = build(items, header["classes"], header["locations"], work / "data", options)
            split = params.get("split") or "val"
            if not summary["items"].get(split):
                raise RuntimeError(f"이 릴리스에는 {split} split의 이미지가 없습니다.")
            name = f"{header['dataset']} {header['version']}"
            print(f"릴리스 {name} 의 {split} split {summary['items'][split]:,}장으로 평가합니다.")
            config.update(data=summary["data_yaml"], data_dir=str(work / "data"), split=split)
        elif action == "predict":
            source = work / "inputs"
            for n, image in enumerate(config["params"].get("images") or []):
                _fetch(client, f"{base}/blob/input", source / image["name"], {"index": n})
            config["source"] = str(source)
            config["params"] = {k: v for k, v in config["params"].items() if k != "images"}

        code = trainer.template_dir(client, base + "/blob", node_home, template)
        python = envs.resolve(node_home, params.get("environment") or {})
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "NO_COLOR": "1",
            "PYTHONPATH": os.pathsep.join(p for p in (str(shipped.SDK), os.environ.get("PYTHONPATH", "")) if p),
            "YOLO_CONFIG_DIR": str(node_home / "cache" / "ultralytics"),
        }
        command = trainer.command_for(template["entrypoint"], python, config_path)
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
            raise RuntimeError(f"명령을 실행하지 못했습니다 ({command[0]}): {e}") from e
        assert proc.stdout is not None
        outcome: dict[str, Any] | None = None
        last_progress = 0.0
        for raw in proc.stdout:
            line = trainer.ANSI.sub("", raw).rstrip()
            if not line.startswith(trainer.EVENT_PREFIX):
                if not line.strip():
                    continue
                if trainer.PROGRESS.search(line) and "100%" not in line:
                    if time.monotonic() - last_progress < trainer.PROGRESS_EVERY_S:
                        continue
                    last_progress = time.monotonic()
                print(line, flush=True)
                continue
            try:
                event = json.loads(line[len(trainer.EVENT_PREFIX) :])
            except ValueError:
                continue
            kind = event.pop("event", "")
            if kind == "result":
                outcome = event
            elif kind == "artifact":
                path = Path(str(event.get("path", "")))
                path = path if path.is_absolute() else code / path
                if path.is_file():
                    query = {"name": str(event.get("name") or path.name)}
                    fields = ("key", "step", "caption", "media")
                    query.update({k: event[k] for k in fields if event.get(k) not in (None, "")})
                    sent = client.put(f"{base}/file", params=query, content=path.read_bytes(), timeout=3600)
                    if sent.status_code != 200:
                        raise RuntimeError(f"결과 파일을 올리지 못했습니다 ({path.name}): {sent.text[:200]}")
                    trainer.note_uploaded(work, path)
                else:
                    print(f"기록할 파일을 찾을 수 없습니다: {path}", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"프로세스가 종료 코드 {proc.returncode}로 끝났습니다.")
        if outcome is None:
            raise RuntimeError("코드가 결과를 알리지 않았습니다. 끝날 때 finish를 호출해야 합니다.")
        outcome.pop("weights", None)

        def put(path: Path, name: str) -> bool:
            with path.open("rb") as stream:
                return client.put(f"{base}/file", params={"name": name}, content=stream, timeout=3600).status_code == 200

        kept, size = trainer.keep_outputs(work, put)  # plots and tables the code wrote without recording them
        if kept:
            print(f"나머지 산출물 {kept:,}개 ({size / 1024**2:.1f} MB)를 Hub에 보관했습니다.", flush=True)
        resp = trainer.send(client.post, f"{base}/finish", json=outcome)
        if resp.status_code != 200:
            raise RuntimeError(f"Hub가 결과를 받지 않았습니다 ({resp.status_code}): {resp.text[:200]}")
    shutil.rmtree(work, ignore_errors=True)
    print("완료")
    return {"action": action, "metrics": outcome.get("metrics") or {}}
