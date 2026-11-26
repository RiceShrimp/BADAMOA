"""Image vectors (``model.embed``) for 이미지 탐색.

The Hub lists the version's images that have no vector by the model yet (an image is known by its content, so a
picture made for another version or dataset is not listed). The node reads each where it is — registered here, or
received into the cache before the job (a training's copy) — runs the model's script (``badamoa.embedders``) in its
environment, and sends each part file to the Hub as soon as the script has written it, so a stopped job loses at most
the part it was on.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from badamoa import embedders

from . import cache as image_cache
from . import envs, places, trainer

PART = 20_000  # vectors to a part file (SigLIP2 base: 30 MB)


def run(params: dict[str, Any]) -> dict[str, Any]:
    hub, token, job_id, home = (
        os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID", "BADAMOA_NODE_HOME")
    )
    if not (hub and token and job_id and home):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    node_home = Path(home).resolve()
    where = places.of(node_home)
    work = where.runs / job_id
    base = f"/api/agent/jobs/{job_id}/embed"
    model, spec = params["model"], params["spec"]
    script = embedders.script(spec["script"])
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)

    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:
        header, items = trainer.manifest_stream(client, base)
        received = Path(os.environ.get("BADAMOA_CACHE_DIR") or node_home / "cache")
        listed = work / "images.jsonl"
        count = missing = 0
        with listed.open("w", encoding="utf-8") as out:
            for item in items:
                root = header["locations"].get(str(item.get("location_id")))
                if root:
                    image = Path(root).expanduser() / item["rel_path"]
                else:  # registered on another server: the copy received by its hash (a training's)
                    image = received / image_cache.rel_of(item["sha256"], Path(item["rel_path"]).suffix)
                try:
                    found = image.is_file() and image.stat().st_size == item["bytes"]
                except OSError:
                    found = False
                if not found:
                    missing += 1
                    continue
                out.write(json.dumps({"sha256": item["sha256"], "path": str(image)}) + "\n")
                count += 1
        name = f"{header['dataset']} {header['version']}"
        skipped = f" (이 서버에서 찾을 수 없는 이미지 {missing:,}장은 건너뜀)" if missing else ""
        encoders = spec.get("export_text") or spec.get("export_vision")  # the Hub's query encoders, still to come
        if count == 0 and not encoders:
            print(f"{name}에는 벡터를 만들 이미지가 없습니다{skipped}. 이미 모두 만들었거나 이미지가 이 서버에 없습니다.")
            return {"model": model, "embedded": 0, "missing": missing}
        if count:
            print(f"{name}의 이미지 {count:,}장으로 {model} 벡터를 만듭니다{skipped}.", flush=True)
        else:
            print(f"{name}의 이미지는 모두 벡터가 있어, Hub의 검색 모델만 만듭니다.", flush=True)

        python, _, _ = trainer.ready_environment(node_home, params.get("environment") or {}, print, client)
        hf = where.find("cache", "huggingface")  # the model's checkpoint, downloaded once
        hf.mkdir(parents=True, exist_ok=True)
        config = {
            "checkpoint": spec.get("checkpoint", ""),
            "model": model,
            "dim": spec["dim"],
            "list": str(listed),
            "out": str(work / "out"),
            "part": PART,
            "export_text": bool(spec.get("export_text")),  # the first job: 글 검색's text encoder too
            "export_vision": bool(spec.get("export_vision")),  # and 이미지로 찾기's image encoder
        }
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        env = envs.code_env(
            {
                **{"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HOME": str(hf)},
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",  # Windows without symlinks: copies, said once is enough
            }
        )
        stored = 0
        sent: set[str] = set()

        def send_ready() -> None:
            nonlocal stored
            for part in sorted((work / "out").glob("part-*.bin")):
                if part.name in sent:
                    continue
                resp = client.put(f"{base}/part", content=part.read_bytes(), timeout=1800)
                if resp.status_code != 200:
                    raise RuntimeError(f"벡터를 Hub에 보내지 못했습니다 ({resp.status_code}): {resp.text[:200]}")
                stored += int(resp.json().get("stored") or 0)
                sent.add(part.name)
                part.unlink(missing_ok=True)

        proc = subprocess.Popen(
            [str(python or sys.executable), str(script), str(config_path)],
            cwd=work,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = trainer.ANSI.sub("", raw).rstrip()
            if line.strip():
                print(line, flush=True)
            send_ready()  # a part is announced by a line: send it now
        if proc.wait() != 0:
            send_ready()  # what was made before it stopped is kept
            raise RuntimeError(f"임베딩이 종료 코드 {proc.returncode}로 끝났습니다.")
        send_ready()
        text = work / "out" / "text"
        if (text / "checks.json").is_file():  # 글 검색's text encoder, made by the first job
            said: dict[str, Any] = {}
            for kind, name in (("onnx", "text.onnx"), ("tokenizer", "tokenizer.json"), ("checks", "checks.json")):
                with (text / name).open("rb") as stream:
                    resp = client.put(f"{base}/text", params={"kind": kind}, content=stream, timeout=3600)
                if resp.status_code != 200:
                    raise RuntimeError(f"글 검색 모델을 Hub에 보내지 못했습니다 ({resp.status_code}): {resp.text[:200]}")
                said = resp.json()
            if said.get("ready"):
                print("글 검색 모델을 Hub에 두었습니다. 이제 이미지 탐색에서 글로 찾을 수 있습니다.", flush=True)
            else:
                print(f"Hub가 글 검색 모델을 쓰지 못합니다: {said.get('problem') or '확인 실패'}", flush=True)
        vision = work / "out" / "vision"
        if (vision / "vision.json").is_file():  # 이미지로 찾기's image encoder
            said = {}
            for kind, name in (("onnx", "vision.onnx"), ("checks", "vision.json")):
                with (vision / name).open("rb") as stream:
                    resp = client.put(f"{base}/vision", params={"kind": kind}, content=stream, timeout=3600)
                if resp.status_code != 200:
                    raise RuntimeError(f"이미지 검색 모델을 Hub에 보내지 못했습니다 ({resp.status_code}): {resp.text[:200]}")
                said = resp.json()
            if said.get("ready"):
                print("이미지 검색 모델을 Hub에 두었습니다. 이제 이미지를 올려 찾을 수 있습니다.", flush=True)
            else:
                print(f"Hub가 이미지 검색 모델을 쓰지 못합니다: {said.get('problem') or '확인 실패'}", flush=True)
    print(f"벡터 {stored:,}개를 Hub에 두었습니다.", flush=True)
    return {"model": model, "embedded": stored, "missing": missing}
