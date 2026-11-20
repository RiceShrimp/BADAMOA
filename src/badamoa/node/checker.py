"""Trial run of a model template version (``template.check``) before anyone is offered it.

    1. draw a small dataset (``samples``) and write it in the template's data format
    2. get the code and prepare the environment exactly as a training does — what it installs
       stays for the trainings on this server
    3. run each entry point briefly with the ``[check]`` settings: train, then predict, eval and
       export as declared and not skipped

Passing means: every run ends with exit code 0 and reports its result; train names a weights file
that exists and at least one metric; predict answers for every image; eval reports a metric.
Whether the numbers are any good is not looked at — a dozen images for one epoch say nothing.

The verdict is the job's result (``ok``, the step that failed and why, what each step did); the
Hub keeps it with the template version. A template that fails makes a job that succeeded: the
job did its work, which was to find out.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from badamoa.core import tasks as task_names
from badamoa.core.templates import CONTRACT

from . import containers, exporter, places, samples, trainer

STEP_NAMES = {
    "data": "시험 데이터",
    "env": "환경",
    "train": "학습",
    "predict": "테스트",
    "autolabel": "오토라벨",
    "eval": "평가",
    "export": "내보내기",
}


class Failed(Exception):
    """A step of the trial did not do what it must; the message says why, in one sentence."""

    def __init__(self, step: str, reason: str) -> None:
        super().__init__(reason)
        self.step = step
        self.reason = reason


def _numbers(metrics: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (metrics or {}).items() if isinstance(metrics, dict) else []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:  # not NaN
            out[str(key)] = number
    return out


def _run(
    step: str,
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    deadline: float,
    job_id: str,
    in_container: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Start one entry point, forward its log, and return its result and what it reported on the way."""
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cwd,
        )
    except OSError as e:
        raise Failed(step, f"명령을 실행하지 못했습니다 ({command[0]}): {e}") from e
    timed_out = threading.Event()

    def stop() -> None:
        timed_out.set()
        proc.kill()
        if in_container:
            containers.kill(job_id)

    timer = threading.Timer(max(1.0, deadline - time.monotonic()), stop)
    timer.daemon = True
    timer.start()
    outcome: dict[str, Any] | None = None
    seen: dict[str, Any] = {"metrics": {}, "checkpoints": 0, "artifacts": []}
    last_progress = 0.0
    assert proc.stdout is not None
    try:
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
            elif kind in ("epoch", "log"):
                seen["metrics"].update(_numbers(event.get("metrics")))
            elif kind == "checkpoint":
                seen["checkpoints"] += 1
            elif kind == "artifact":
                seen["artifacts"].append(event)
        code = proc.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise Failed(step, "정해진 시간 안에 끝나지 않았습니다 ([check] timeout_min).")
    if code != 0:
        raise Failed(step, f"종료 코드 {code}로 끝났습니다. 로그에서 오류를 확인하세요.")
    if outcome is None:
        raise Failed(step, "결과를 알리지 않았습니다. 끝날 때 finish(또는 result)를 호출해야 합니다.")
    return outcome, seen


def _trial(
    client: httpx.Client | None,
    base: str,
    params: dict[str, Any],
    node_home: Path,
    work: Path,
    job_id: str,
    steps: list[dict[str, Any]],
    notes: list[str],
    facts: dict[str, Any],
    label: str = "",
) -> None:
    """``label``: the task this pass tries, for a kind of several tasks (Det, Seg …); its steps say it."""
    template = params["template"]
    spec = params.get("environment") or {}
    check = params.get("check") or {}
    data = params.get("data") or {}
    entrypoints: dict[str, str] = template.get("entrypoints") or {}
    in_container = spec.get("kind") == "docker"
    skip = set(check.get("skip") or [])

    def step(name: str, act: Callable[[], str]) -> None:
        print(f"── {label + ': ' if label else ''}{STEP_NAMES.get(name, name)} ──", flush=True)
        started = time.monotonic()
        try:
            detail = act()
        except Failed:
            raise
        except Exception as e:  # noqa: BLE001 - whatever went wrong is the reason this step failed
            raise Failed(name, f"{type(e).__name__}: {e}") from e
        done = {"step": name, "ok": True, "took_s": round(time.monotonic() - started, 1), "detail": detail}
        steps.append({**done, "task": template.get("task")} if label else done)
        print(f"통과: {detail}", flush=True)

    # 1. data
    data_format = str(template.get("data_format") or "yolo")
    kind = samples.kind_of(data_format)
    classes = (data.get("classes") or {}) if isinstance(data.get("classes"), dict) else {}
    summary: dict[str, Any] = {}

    def make_data() -> str:
        build = exporter.BUILDERS.get(data_format)
        if build is None:
            raise Failed("data", f"이 서버의 BADAMOA는 데이터 형식 {data_format}을 쓸 줄 모릅니다. 서버를 업데이트하세요.")
        count = samples.class_count(kind, int(classes.get("min") or 0), int(classes.get("max") or 0))
        train = max(8, int(data.get("min_images") or 0))
        points = int(data.get("keypoints") or 0)
        items, names, locations = samples.make(work / "sample", kind, classes=count, train=train, points=points)
        options = exporter.ExportOptions(keep_splits=True, symlinks=not (in_container and os.name == "nt"))
        summary.update(build(items, names, locations, work / "data", options))
        counts = summary["items"]
        return f"{data_format} · train {counts['train']} · val {counts['val']}장 · 클래스 {len(summary['classes'])}개"

    step("data", make_data)

    # 2. code and environment (what it installs stays: the trainings here use the same)
    found: dict[str, Any] = {}

    def make_env() -> str:
        if params.get("code"):  # a folder on this machine (``badamoa template check --run``)
            found["code"] = Path(params["code"])
        else:
            assert client is not None
            found["code"] = trainer.template_dir(client, base, node_home, template)
        python, installed, extra = trainer.ready_environment(node_home, spec, client=client)
        found["python"] = python
        facts.update(extra)
        facts["packages"] = installed
        # code the node fetched itself (in its work folder, or its home before) is the Hub's: not a server folder
        cached = [(base / "cache").resolve().as_posix().lower() for base in places.of(node_home).roots()]
        built = spec.get("kind") == "auto"  # a registered Python belongs to its server anyway
        for name, source in (extra.get("sources") or {}).items() if built else []:
            url = str(source.get("url") or "").lower()
            if source.get("kind") in ("local", "editable") and not any(c in url for c in cached):
                notes.append(
                    f"이 환경은 다른 서버에서 똑같이 만들 수 없습니다: {name} (이 서버의 폴더에서 설치). "
                    "wheel로 만들어 Hub의 사내 패키지에 올리세요."
                )
        if in_container:
            where = f"Docker {spec.get('image')}"
        else:
            where = "서버의 Python" if spec.get("kind") == "python" or not spec.get("requirements") else "자동 설치"
        return f"{where} · 패키지 {len(installed)}개"

    step("env", make_env)
    code: Path = found["code"]
    python: Path | None = found["python"]

    # 3. the entry points, against one clock (preparing the environment does not count)
    deadline = time.monotonic() + int(check.get("timeout_min") or 15) * 60
    cache = places.of(node_home).find("cache", "template-data", str(template["name"]))
    cache.mkdir(parents=True, exist_ok=True)
    # the GPU the Hub gave this trial, if it gave one; on a developer's PC (no Hub) only when the code needs one
    given = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    on_gpu = bool(given) or (client is None and bool(check.get("gpu")))
    device = trainer._device() if given else ("0" if on_gpu else "cpu")
    settings = params.get("params") or {}
    common = {"cache": str(cache), "device": device, "classes": summary["classes"], "check": True}
    if template.get("task"):  # the task of this pass: a kind of several tasks reads it to know what to train
        common["task"] = template["task"]

    def start(name: str, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Callable[[str], Path]]:
        command, env, where = trainer.launch(
            job_id=job_id,
            home=node_home,
            spec=spec,
            python=python,
            entrypoint=entrypoints[name],
            config=config,
            config_path=work / f"config-{name}.json",
            code=code,
            work=work,
            mounts=[(places.of(node_home).models, True)],
        )
        if not on_gpu and not in_container:
            env = {**env, "CUDA_VISIBLE_DEVICES": ""}  # a trial on the CPU leaves the GPUs to the trainings
        outcome, seen = _run(name, command, env, code, deadline, job_id, in_container)
        return outcome, seen, where

    trained: dict[str, Path] = {}

    def train() -> str:
        config = {
            **common,
            "data": summary["data_yaml"],
            "data_dir": str(work / "data"),
            "out": str(work / "out"),
            "params": settings,
        }
        outcome, seen, where = start("train", config)
        if not outcome.get("weights"):
            raise Failed("train", "가중치 파일을 알리지 않았습니다. 학습한 가중치를 log_model로 알리세요.")
        weights = where(str(outcome["weights"]))
        if not weights.is_file():
            raise Failed("train", f"알린 가중치 파일이 없습니다: {weights}")
        metrics = _numbers(outcome.get("metrics")) or seen["metrics"]
        if not metrics:
            raise Failed("train", "지표를 하나도 알리지 않았습니다. epoch이나 log로 지표를 하나 이상 보내세요.")
        if template.get("resume") and not seen["checkpoints"]:
            notes.append("resume = true인데 학습 중에 체크포인트를 알리지 않았습니다 (save_checkpoint).")
        kept = work / "weights" / weights.name
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(weights, kept)  # later steps get their own copy: the out folder is the code's
        trained["weights"] = kept
        facts["metrics"] = metrics
        return f"지표 {len(metrics)}개 · 가중치 {weights.name} ({kept.stat().st_size / 1024**2:.1f} MB)"

    facts["device"] = device
    step("train", train)
    after = {
        **common,
        "weights": str(trained["weights"]),
        "params": {},
        "train_params": settings,
    }

    def predict() -> str:
        source = work / "inputs"
        images = sorted((work / "sample" / "val").glob("*.png"))
        source.mkdir(parents=True, exist_ok=True)
        for image in images:
            shutil.copyfile(image, source / image.name)
        config = {**after, "action": "predict", "out": str(work / "out-predict"), "source": str(source)}
        outcome, _, _ = start("predict", config)
        answered = outcome.get("predictions")
        if not isinstance(answered, list):
            raise Failed("predict", "예측을 알리지 않았습니다. finish(predictions=[…])로 이미지마다 결과를 넘겨야 합니다.")
        named = {Path(str(p.get("image") or "")).stem for p in answered if isinstance(p, dict)}
        missing = [image.name for image in images if image.stem not in named]
        if missing:
            raise Failed("predict", f"이미지 {len(images)}장 중 {len(missing)}장에 예측이 없습니다 (예: {missing[0]}).")
        return f"이미지 {len(images)}장 모두 예측"

    def autolabel() -> str:
        """The predict code in autolabel mode, on images named by their hash as a labeling set gives them."""
        source = work / "inputs-autolabel"
        source.mkdir(parents=True, exist_ok=True)
        named = {}
        for image in sorted((work / "sample" / "val").glob("*.png")):
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            shutil.copyfile(image, source / f"{digest}.png")
            named[digest] = image.name
        config = {**after, "action": "autolabel", "out": str(work / "out-autolabel"), "source": str(source)}
        _, seen, where = start("predict", config)
        files = [where(str(a.get("path") or "")) for a in seen["artifacts"] if a.get("media") == "predictions"]
        found = [f for f in files if f.is_file()]
        if not found:
            raise Failed(
                "autolabel", "예측을 남기지 않았습니다. 오토라벨에서는 bm.log_predictions([...])로 이미지마다 남겨야 합니다."
            )
        try:
            images = json.loads(found[-1].read_text(encoding="utf-8")).get("images") or []
        except (OSError, ValueError) as e:
            raise Failed("autolabel", f"남긴 예측을 읽을 수 없습니다: {e}") from e
        given = {str(p.get("image") or "") for p in images if isinstance(p, dict) and isinstance(p.get("labels"), list)}
        missing = [name for digest, name in named.items() if digest not in given]
        if missing:
            raise Failed(
                "autolabel", f"이미지 {len(named)}장 중 {len(missing)}장에 예측 줄이 없습니다 (이미지는 sha256 이름)."
            )
        labels = sum(len(p["labels"]) for p in images if isinstance(p, dict) and isinstance(p.get("labels"), list))
        return f"이미지 {len(named)}장 모두 · 라벨 {labels}개"

    def evaluate() -> str:
        config = {
            **after,
            "action": "eval",
            "data": summary["data_yaml"],
            "data_dir": str(work / "data"),
            "split": "val",
            "out": str(work / "out-eval"),
        }
        outcome, seen, _ = start("eval", config)
        metrics = _numbers(outcome.get("metrics")) or seen["metrics"]
        if not metrics:
            raise Failed("eval", "평가 지표를 하나도 알리지 않았습니다. finish(metrics)로 넘겨야 합니다.")
        return f"지표 {len(metrics)}개"

    def export() -> str:
        outcome, _, _ = start("export", {**after, "action": "export", "out": str(work / "out-export")})
        return f"파일 {outcome['file']}" if outcome.get("file") else "끝남"

    for name, act in (("predict", predict), ("autolabel", autolabel), ("eval", evaluate), ("export", export)):
        entry = "predict" if name == "autolabel" else name
        if entry not in entrypoints:
            continue
        if name == "autolabel" and not template.get("autolabel"):
            why = "하지 않음 (autolabel을 선언하지 않음)"
            skipped = {"step": name, "ok": True, "skipped": True, "took_s": 0, "detail": why}
            steps.append({**skipped, "task": template.get("task")} if label else skipped)
            print("── 오토라벨: 하지 않음 (badamoa.toml에 autolabel = true가 없음) ──", flush=True)
        elif entry in skip:
            print(f"── {STEP_NAMES[name]}: [check] skip으로 건너뜀 ──", flush=True)
        else:
            step(name, act)


def run(params: dict[str, Any]) -> dict[str, Any]:
    hub, token, job_id, home = (
        os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID", "BADAMOA_NODE_HOME")
    )
    if not (hub and token and job_id and home):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:
        return attempt(params, Path(home).resolve(), job_id, client, f"/api/agent/jobs/{job_id}/check")


def attempt(
    params: dict[str, Any], node_home: Path, job_id: str, client: httpx.Client | None = None, base: str = ""
) -> dict[str, Any]:
    """The whole trial and its verdict. Without a ``client`` it runs on this machine alone: the code comes
    from ``params["code"]`` and nothing goes to a Hub (``badamoa template check --run``)."""
    work = places.of(node_home).runs / job_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    template = params["template"]
    print(f"{template.get('name')} {template.get('version')} 시험 실행을 시작합니다.", flush=True)
    steps: list[dict[str, Any]] = []
    notes: list[str] = []
    facts: dict[str, Any] = {}
    started = time.monotonic()
    failed: Failed | None = None
    # A kind of several tasks: each task in turn, on its own sample data, with its settings (``tasks``).
    passes = params.get("tasks") or [{}]
    try:
        for one in passes:
            label = task_names.short(one.get("task")) if len(passes) > 1 else ""
            here = params
            if one:
                template_here = {**template, "task": one["task"], "data_format": one.get("data_format") or "yolo"}
                here = {**params, "template": template_here, "params": one.get("params") or {}}
            folder = work / one["task"] if label else work
            folder.mkdir(parents=True, exist_ok=True)
            try:
                _trial(client, base, here, node_home, folder, job_id, steps, notes, facts, label)
            except Failed as e:
                if label:
                    e.reason = f"{label}: {e.reason}"
                raise
    except Failed as e:
        failed = e
    finally:
        if (params.get("environment") or {}).get("kind") == "docker":
            containers.kill(job_id)
    shutil.rmtree(work, ignore_errors=True)
    result: dict[str, Any] = {
        "ok": failed is None,
        "steps": steps,
        "failed": failed.step if failed else "",
        "reason": failed.reason if failed else "",
        "notes": notes,
        "took_s": round(time.monotonic() - started, 1),
        "environment": {k: facts[k] for k in ("image", "image_id") if facts.get(k)},
        "device": facts.get("device") or "",
        "packages": facts.get("packages") or [],
        "sources": facts.get("sources") or {},
        "contract": CONTRACT,
        "metrics": facts.get("metrics") or {},
    }
    if failed:
        print(f"실패 — {STEP_NAMES.get(failed.step, failed.step)}: {failed.reason}", flush=True)
    else:
        print(f"통과: {len(steps)}단계, {result['took_s']:.0f}초", flush=True)
    return result
