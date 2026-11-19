"""What the four tasks share: which task a run trains, its settings, and the epochs with a checkpoint after each."""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import badamoa_sdk as bm

TASKS = ("detect", "segment", "classify", "pose")
MANIFEST = Path(__file__).with_name("badamoa.toml")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#000075", "#f032e6"]


def task_of(cfg: dict[str, Any], default: str = "detect") -> str:
    """What this run trains (or what the model was trained for): the config says (BADAMOA 0.60+). Before that it
    does not, and the model's name tells: Mask R-CNN segments, Keypoint R-CNN finds keypoints."""
    task = str(cfg.get("task") or "").lower()
    if task in TASKS:
        return task
    name = str((cfg.get("train_params") or cfg.get("params") or {}).get("model") or "")
    if name.startswith("maskrcnn"):
        return "segment"
    if name.startswith("keypointrcnn"):
        return "pose"
    return default


def settings(values: dict[str, Any], task: str, manifest: Path = MANIFEST) -> bm.Params:
    """The settings of a run (or of the training of a model): what it was given, and for anything left out (a run
    outside BADAMOA, a model trained before a setting existed) what ``badamoa.toml`` declares for this task."""
    declared = tomllib.loads(manifest.read_text(encoding="utf-8")).get("params", {}) if manifest.is_file() else {}
    out = bm.Params()
    for name, spec in declared.items():
        if spec.get("tasks") and task not in spec["tasks"]:
            continue
        out[name] = (spec.get("by_task") or {}).get(task, {}).get("default", spec.get("default"))
    out.update(values)
    return out


def device_of(run) -> Any:
    import torch

    return torch.device("cpu" if run.device == "cpu" else "cuda:0")


def start_from(model, weights: Path) -> None:
    """Continue from a registered model: every layer whose shape still fits."""
    import torch

    state = torch.load(weights, map_location="cpu")
    own = model.state_dict()
    fitting = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(fitting, strict=False)
    print(f"등록된 모델에서 시작: {len(fitting)}/{len(own)}개 층을 불러왔습니다.")


def optimizer_for(hp: bm.Params, weights: list) -> Any:
    import torch

    if hp.optimizer == "AdamW":
        return torch.optim.AdamW(weights, lr=hp.lr, weight_decay=hp.weight_decay)
    return torch.optim.SGD(weights, lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)


def fit(
    run,
    hp: bm.Params,
    model,
    optimizer,
    one_epoch: Callable[[int], float],
    validate: Callable[[], dict[str, float]],
    primary: str,
    shown: Callable[[dict[str, float]], str],
) -> dict[str, float]:
    """The epochs: ``one_epoch(epoch)`` trains and returns the loss, ``validate()`` gives the metrics, the best by
    ``primary`` are kept in ``best.pt`` and loaded into ``model`` at the end. After each epoch a checkpoint to
    continue from, should the run be interrupted (``resume = true``). Returns the best epoch's metrics."""
    import torch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs) if hp.cos_lr else None
    best, best_path, best_metrics = -1.0, run.out / "best.pt", {}
    start = 1
    if run.resume_checkpoint and run.resume_checkpoint.is_file():  # continue an interrupted run
        state = torch.load(run.resume_checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start, best, best_metrics = state["epoch"] + 1, state["best"], state["best_metrics"]
        if state.get("best_model"):
            torch.save(state["best_model"], best_path)
        print(f"체크포인트에서 이어 갑니다: epoch {start}부터")
    for epoch in range(start, hp.epochs + 1):
        started = time.monotonic()
        loss = one_epoch(epoch)
        rate = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()
        metrics = {**validate(), "train_loss": loss, "lr": rate}
        bm.epoch(epoch, hp.epochs, metrics)
        print(f"epoch {epoch}/{hp.epochs} · loss {loss:.3f} · {shown(metrics)} · {time.monotonic() - started:.0f}초")
        if metrics[primary] > best:
            best, best_metrics = metrics[primary], {k: v for k, v in metrics.items() if k != "lr"}
            torch.save(model.state_dict(), best_path)

        # Everything needed to continue from here, should the run be interrupted.
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best": best,
            "best_metrics": best_metrics,
            "best_model": torch.load(best_path, map_location="cpu") if best_path.is_file() else None,
        }
        torch.save(checkpoint, run.out / "last.pt.part")
        (run.out / "last.pt.part").replace(run.out / "last.pt")  # never a half-written file under the real name
        bm.save_checkpoint(run.out / "last.pt", epoch)
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    return {k: v for k, v in best_metrics.items() if k != "lr"}


def export(run, model, example, names: dict[str, list[str]], trace: bool = False, **onnx: Any) -> None:
    """The model as ONNX or TorchScript (``run.config["format"]``) in the output folder, recorded and reported.
    ``trace``: TorchScript by tracing ``example`` (classifiers) instead of scripting (detectors, whose output depends
    on what they find)."""
    import torch

    model.cpu()
    if run.config.get("format") == "torchscript":
        exported = run.out / "model.torchscript"
        (torch.jit.trace(model, example) if trace else torch.jit.script(model)).save(str(exported))
    else:
        exported = run.out / "model.onnx"
        torch.onnx.export(model, example, str(exported), dynamo=False, **names, **onnx)
    print(f"내보낸 파일: {exported.name} ({exported.stat().st_size / 1024**2:.1f} MB)")
    bm.log_artifact(exported)
    bm.finish(file=exported.name, bytes=exported.stat().st_size)


def registered_names(data_dir: Path) -> dict[str, str]:
    """The exported images are named by their hash: the names they were registered under, to show in results."""
    import json

    try:
        return json.loads((data_dir / "files.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
