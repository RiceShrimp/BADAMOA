"""Training jobs (``train.run``): export the release, run the template's code, report to the Hub.

    1. fetch the release's manifest from the Hub and write it in the template's data format
    2. get the template's code (built-in, or downloaded from the Hub and cached by hash)
    3. get the environment's Python (built on first use, or an existing one)
    4. run the template's train command, forwarding its log and metrics as they come
    5. upload the weights and register the model
    6. put whatever else the code left in its output folder on the Hub, then remove the folder

Nothing a run produced stays only on the server that ran it: the Hub keeps the record.

The template's code knows nothing about BADAMOA (see ``badamoa.templates``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
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
# While a run is healthy its checkpoint goes to the Hub this often (protection against a crashed
# server); when a run is stopped or fails, the Node Agent sends the newest one right away.
CHECKPOINT_EVERY_S = 300.0
MANIFEST_PAGE = 2000
# Output files are kept one by one up to this many; a job that leaves more gets the rest as one archive.
KEEP_SEPARATELY = 150
SNAPSHOT_BYTES = 64 * 1024**2  # a recorded file up to this size is read whole before it is sent
UPLOADED = "uploaded.txt"  # the files of a job's working folder that are on the Hub already


def send(call, *args: Any, **kwargs: Any) -> httpx.Response:
    """A request to the Hub that survives a dropped connection.

    Reports are minutes apart during training, and the Hub closes idle connections;
    now and then a request goes out on one that has just been closed. Trying again
    is safe because the Hub treats repeated reports as the same report.
    """
    for attempt in range(3):
        try:
            return call(*args, **kwargs)
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def last_checkpoint(job_dir: Path) -> tuple[Path, int] | None:
    """The newest checkpoint a training job recorded in its working folder, with its epoch."""
    try:
        info = json.loads((job_dir / "checkpoint.json").read_text(encoding="utf-8"))
        path = Path(info["path"])
    except (OSError, ValueError, KeyError):
        return None
    return (path, int(info.get("epoch") or 0)) if path.is_file() else None


def upload_checkpoint(client: httpx.Client, base: str, job_dir: Path) -> bool:
    """Send a job's newest checkpoint to the Hub. True when the Hub now has it."""
    found = last_checkpoint(job_dir)
    if found is None:
        return False
    path, epoch = found
    with path.open("rb") as stream:
        query = {"name": path.name, "kind": "checkpoint", "epoch": epoch}
        resp = client.put(f"{base}/artifact", params=query, content=stream, timeout=3600)
    return resp.status_code == 200


def note_uploaded(job_dir: Path, path: Path) -> None:
    with (job_dir / UPLOADED).open("a", encoding="utf-8") as f:
        f.write(f"{path.resolve()}\n")


def output_files(job_dir: Path) -> list[tuple[Path, str]]:
    """What the code left in its output folder and is not on the Hub yet, with the name to keep each under."""
    out = job_dir / "out"
    if not out.is_dir():
        return []
    try:
        sent = set((job_dir / UPLOADED).read_text(encoding="utf-8").splitlines())
    except OSError:
        sent = set()
    everything = [p for p in out.rglob("*") if p.is_file()]
    # One folder holding everything (Ultralytics writes into "train") is not part of the names.
    tops = {p.relative_to(out).parts[0] for p in everything}
    root = out / next(iter(tops)) if len(tops) == 1 and (out / next(iter(tops))).is_dir() else out
    left = [
        (p, p.relative_to(root).as_posix())
        for p in everything
        if not p.name.endswith(".part") and p.stat().st_size > 0 and str(p.resolve()) not in sent
    ]
    return sorted(left, key=lambda entry: (entry[1].count("/"), entry[1]))


def keep_outputs(job_dir: Path, put: Callable[[Path, str], bool]) -> tuple[int, int]:
    """Put the rest of a job's output folder on the Hub: (files kept, their bytes). Never raises."""
    kept = size = 0
    try:
        files = output_files(job_dir)
        for path, name in files[:KEEP_SEPARATELY]:
            if put(path, name):
                kept, size = kept + 1, size + path.stat().st_size
        rest = files[KEEP_SEPARATELY:]
        if rest:
            archive = job_dir / "outputs.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
                for path, name in rest:
                    bundle.write(path, name)
            if put(archive, archive.name):
                kept, size = kept + len(rest), size + archive.stat().st_size
    except (OSError, httpx.HTTPError) as e:
        print(f"산출물 일부를 Hub에 보관하지 못했습니다: {e}", flush=True)
    return kept, size


def put_run_file(client: httpx.Client, base: str, path: Path, name: str, **media: Any) -> bool:
    """Store a file with the training run of this job. False (and a line in the log) when the Hub refused it.

    ``media``: ``key``, ``step``, ``caption`` and ``media`` ("table") of a per-step image or table.
    """
    query = {"name": name, "kind": "file", **{k: v for k, v in media.items() if v not in (None, "")}}
    if path.stat().st_size <= SNAPSHOT_BYTES:
        # Read it first: code that rewrites the file every epoch may be writing it right now,
        # and a body that grows while it is sent breaks the upload.
        resp = client.put(f"{base}/artifact", params=query, content=path.read_bytes(), timeout=3600)
    else:
        with path.open("rb") as stream:
            resp = client.put(f"{base}/artifact", params=query, content=stream, timeout=3600)
    if resp.status_code != 200:
        print(f"파일을 기록하지 못했습니다 ({name}): {resp.text[:200]}", flush=True)
    return resp.status_code == 200


def _device() -> str:
    """The Hub exposes only the GPUs it reserved, so they are always numbered from 0 here."""
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    return ",".join(str(i) for i in range(len(visible))) if visible else "cpu"


def manifest_stream(client: httpx.Client, base: str) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    """The manifest header plus a lazy stream of its items (paged by item id)."""

    def page(after: int) -> dict[str, Any]:
        resp = send(client.get, f"{base}/manifest", params={"after": after, "limit": MANIFEST_PAGE})
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
            resp = send(client.post, base + path, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"Hub가 거부했습니다 ({resp.status_code}): {resp.text[:300]}")
            return resp.json()

        def stage(name: str) -> None:
            post("/report", {"event": "stage", "data": {"stage": name}})

        # 1. dataset
        stage("data")
        header, items = manifest_stream(client, base)
        print(f"릴리스 {header['dataset']} {header['version']} 를 학습 형식으로 준비합니다: {work / 'data'}")
        options = exporter.ExportOptions(
            classes=[int(c) for c in params.get("classes") or []],
            unlabelled=params.get("unlabelled") or "exclude",
            val_fraction=float(params.get("val_fraction", 0.2)),
        )
        shutil.rmtree(work, ignore_errors=True)
        data_format = template.get("data_format", "yolo")
        build = exporter.BUILDERS.get(data_format)
        if build is None:
            raise RuntimeError(f"이 노드의 BADAMOA는 데이터 형식 {data_format}을 지원하지 않습니다. 노드를 업데이트하세요.")
        summary = build(items, header["classes"], header["locations"], work / "data", options)
        counts, left_out = summary["items"], summary["excluded"]
        print(
            f"이미지 train {counts['train']:,} · val {counts['val']:,} · test {counts['test']:,} · "
            f"{'라벨' if data_format == 'imagefolder' else '박스'} {summary['boxes']:,} · 클래스 {len(summary['classes'])}개"
        )
        if summary["val_source"] == "holdout":
            print(f"릴리스에 val split이 없어 train의 {options.val_fraction:.0%}를 검증용으로 떼었습니다.")
        elif summary["val_source"] == "test":
            print("릴리스에 val split이 없어 test split으로 검증합니다.")
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
        installed = envs.packages(python)
        if installed:
            post("/report", {"event": "info", "data": {"packages": installed}})

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
        if params.get("init_weights"):  # continue from a registered model instead of pretrained weights
            start = params["init_weights"]
            cached = node_home / "cache" / "models" / f"{start['blob']}{Path(start['file_name']).suffix}"
            if not cached.is_file():
                resp = client.get(f"{base}/init-weights", timeout=3600)
                if resp.status_code != 200:
                    raise RuntimeError(f"시작할 모델의 가중치를 받지 못했습니다 ({resp.status_code}): {resp.text[:200]}")
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(resp.content)
            config["init_weights"] = str(cached)
            print(f"등록된 모델의 가중치에서 시작합니다: {cached.name}")
        resume = params.get("resume") or {}
        previous = node_home / "runs" / str(resume.get("job_id") or "-")
        if resume:
            local = last_checkpoint(previous)
            if local is not None:  # the interrupted run was on this server and its folder is still here
                config["resume_checkpoint"] = str(local[0])
                print(f"이 서버에 남아 있는 체크포인트(epoch {local[1]})에서 이어 갑니다.")
            else:
                resp = send(client.get, f"{base}/checkpoint", timeout=3600)
                if resp.status_code != 200:
                    raise RuntimeError(f"체크포인트를 받지 못했습니다 ({resp.status_code}): {resp.text[:200]}")
                target = work / "resume" / Path(str(resume.get("file_name") or "checkpoint")).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(resp.content)
                config["resume_checkpoint"] = str(target)
                print(f"Hub에 보관된 체크포인트(epoch {resume.get('epoch')})를 받아 이어 갑니다.")
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

        # Metrics go to the Hub at most about once a second, however often the code logs.
        pending: list[dict[str, Any]] = []
        lock = threading.Lock()
        ended = threading.Event()

        def flush() -> None:
            with lock:
                batch = list(pending)
                pending.clear()
            if batch:
                post("/report", {"event": "epoch", "data": {"batch": batch}})

        def flush_every_second() -> None:
            while not ended.wait(1.0):
                try:
                    flush()
                except (RuntimeError, httpx.HTTPError):
                    pass  # the next round, or the final flush, reports the problem

        def keep(entry: dict[str, Any]) -> None:
            with lock:
                pending.append(entry)

        def upload_file(event: dict[str, Any]) -> None:
            """A file the code recorded (a plot, sample predictions): stored with the run, never fatal."""
            path = Path(str(event.get("path", "")))
            path = path if path.is_absolute() else code / path
            if not path.is_file():
                print(f"기록할 파일을 찾을 수 없습니다: {path}", flush=True)
                return
            media = {k: event.get(k) for k in ("key", "step", "caption", "media")}
            try:
                if put_run_file(client, base, path, str(event.get("name") or path.name), **media):
                    note_uploaded(work, path)
            except (OSError, httpx.HTTPError) as e:
                print(f"파일을 기록하지 못했습니다 ({path.name}): {e}", flush=True)

        sent_checkpoint = time.monotonic()

        def note_checkpoint(event: dict[str, Any]) -> None:
            """Remember where the newest checkpoint is; now and then put a copy on the Hub."""
            nonlocal sent_checkpoint
            path = Path(str(event.get("path", "")))
            path = path if path.is_absolute() else code / path
            if not path.is_file():
                return
            # Keep our own copy: the code overwrites its file every epoch, and a run stopped in the
            # middle of that write would otherwise leave a broken checkpoint as the newest one.
            stable = work / "checkpoint" / path.name
            stable.parent.mkdir(parents=True, exist_ok=True)
            partial = stable.with_name(stable.name + ".part")
            try:
                before = path.stat()
                shutil.copyfile(path, partial)
                after = path.stat()
            except OSError:
                return
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                return  # it changed while being copied: the previous copy stays the newest good one
            partial.replace(stable)
            marker = {"path": str(stable), "epoch": int(event.get("epoch") or 0)}
            (work / "checkpoint.json").write_text(json.dumps(marker), encoding="utf-8")
            if time.monotonic() - sent_checkpoint >= CHECKPOINT_EVERY_S:
                sent_checkpoint = time.monotonic()
                try:
                    upload_checkpoint(client, base, work)
                except httpx.HTTPError:
                    pass  # the next one will do

        flusher = threading.Thread(target=flush_every_second, daemon=True)
        flusher.start()
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
                keep(event)
            elif kind == "log":  # metrics at any step, without touching progress
                keep({"step": event.get("step", 0), "metrics": event.get("metrics")})
            elif kind == "progress":
                keep({"epoch": event.get("current"), "epochs": event.get("total")})
            elif kind == "artifact":
                upload_file(event)
            elif kind == "checkpoint":
                note_checkpoint(event)
        ended.set()
        # A batch the thread is still sending must arrive before the final one and the result:
        # arriving after the result it would put the best-so-far back in place of the final metrics.
        flusher.join(timeout=60)
        flush()
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
            note_uploaded(work, weights)
        elif weights is not None:
            raise RuntimeError(f"학습 코드가 알린 가중치 파일이 없습니다: {weights}")
        # 6. everything else the code wrote (plots, tables, the last weights): the Hub keeps the whole record
        kept, size = keep_outputs(work, lambda path, name: put_run_file(client, base, path, name))
        if kept:
            print(f"나머지 산출물 {kept:,}개 ({size / 1024**2:.1f} MB)를 Hub에 보관했습니다.", flush=True)
        final = {"metrics": outcome.get("metrics") or {}, "classes": outcome.get("classes") or [], "weights": uploaded}
        result = post("/finish", final)
    shutil.rmtree(work, ignore_errors=True)  # the export is links, and everything the run produced is on the Hub now
    if params.get("resume"):
        shutil.rmtree(previous, ignore_errors=True)  # the interrupted run's leftovers are no longer needed
    if result.get("model"):
        print(f"완료: 모델 {result['model']['name']} v{result['model']['version']} 등록")
    else:
        print("완료: 지표를 기록했습니다. 가중치를 알리지 않아 모델은 등록하지 않았습니다.")
    return result

