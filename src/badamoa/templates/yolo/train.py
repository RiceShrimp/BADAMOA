"""Ultralytics YOLO trainer, the ``train`` entry point of the built-in ``yolo`` template:

    python train.py --config config.json

See ``badamoa.templates`` for the contract.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

# Settings handed to Ultralytics under the same name (see badamoa.toml).
HYPERPARAMETERS = (
    "patience", "seed", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "cos_lr",
    "mosaic", "close_mosaic", "mixup", "fliplr", "flipud", "degrees", "translate", "scale", "hsv_h", "hsv_s", "hsv_v",
)  # fmt: skip

NAMES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    # segmentation models report the same four for their masks
    "metrics/precision(M)": "mask_precision",
    "metrics/recall(M)": "mask_recall",
    "metrics/mAP50(M)": "mask_map50",
    "metrics/mAP50-95(M)": "mask_map50_95",
}


def emit(event: str, **data) -> None:
    print("@@badamoa " + json.dumps({"event": event, **data}), flush=True)


def clean(metrics: dict) -> dict[str, float]:
    """Plain names and plain floats: ``train/box_loss`` → ``train_box_loss``, ``lr/pg0`` → ``lr_pg0``."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[NAMES.get(key) or key.replace("(B)", "").replace("/", "_")] = round(number, 6)
    return out


def continue_here(checkpoint: Path, out: Path) -> Path:
    """A copy of the checkpoint that writes into this run's folder.

    An Ultralytics checkpoint remembers the folder it was written in and a resumed run goes on
    writing there: a folder of another job, possibly on another server. The copy points at ``out``.
    """
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings = state.get("train_args") or {}
    before = Path(str(settings.get("save_dir") or ""))
    target = out / "train"
    settings.update(project=str(out), name="train", save_dir=str(target))
    state["train_args"] = settings
    # What the stopped run left, when its folder is still on this server: the table of its epochs and its best weights.
    (target / "weights").mkdir(parents=True, exist_ok=True)
    if before != target and (before / "weights" / "best.pt").is_file():
        shutil.copyfile(before / "weights" / "best.pt", target / "weights" / "best.pt")
    if before != target and (before / "results.csv").is_file():
        done = int(state.get("epoch", -1)) + 1  # epochs the checkpoint holds; a later row would be written twice
        rows = (before / "results.csv").read_text(encoding="utf-8").splitlines()
        kept = rows[:1] + [row for row in rows[1:] if row.split(",")[0].strip().isdigit() and int(row.split(",")[0]) <= done]
        (target / "results.csv").write_text("\n".join(kept) + "\n", encoding="utf-8")
    if not (target / "weights" / "best.pt").is_file():
        # The best weights so far are gone with that folder: the best of the epochs trained from here on counts.
        state["best_fitness"] = None
    moved = out.parent / "resume.pt"
    torch.save(state, moved)
    return moved


def main(argv: list[str]) -> int:
    cfg = json.loads(Path(argv[argv.index("--config") + 1]).read_text(encoding="utf-8"))
    params = cfg["params"]
    os.chdir(cfg["cache"])  # pretrained weights download into the working directory: keep them between runs

    import torch
    import ultralytics
    from ultralytics import YOLO

    device = cfg["device"]
    if device != "cpu" and not torch.cuda.is_available():
        print(
            "GPU를 배정받았지만 이 환경의 PyTorch가 CUDA를 쓰지 못합니다. "
            "노드의 드라이버를 확인하거나 BADAMOA_TORCH_BACKEND를 지정해 환경을 다시 만드세요.",
            file=sys.stderr,
        )
        return 3
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if device != "cpu" else []
    emit("info", ultralytics=ultralytics.__version__, torch=torch.__version__, device=device, gpus=gpus)

    weights = params["model"]
    if cfg.get("init_weights"):
        weights = cfg["init_weights"]  # continue from a registered model
    elif not params.get("pretrained", True):
        weights = Path(weights).stem + ".yaml"  # same architecture, random initialisation
    last = Path(cfg.get("resume_checkpoint") or "")
    resuming = bool(cfg.get("resume_checkpoint")) and last.is_file()
    if resuming:
        print(f"체크포인트에서 이어 갑니다: {last}")
        # The interrupted run's checkpoint carries its epoch, optimizer state and settings.
        weights = str(continue_here(last, Path(cfg["out"])))
    model = YOLO(weights)

    training_epoch = False

    def on_train_epoch_start(trainer) -> None:
        nonlocal training_epoch
        training_epoch = True

    def on_fit_epoch_end(trainer) -> None:
        nonlocal training_epoch
        if not training_epoch:  # the closing validation of best.pt fires this once more, as an extra epoch
            return
        training_epoch = False
        epoch = trainer.epoch + 1
        metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        emit("epoch", epoch=epoch, epochs=trainer.epochs, metrics=clean(metrics))

    def on_model_save(trainer) -> None:  # last.pt holds everything needed to continue: tell the node where it is
        emit("checkpoint", path=str(trainer.last), epoch=trainer.epoch + 1)

    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    results = model.train(resume=True, data=cfg["data"], device=device) if resuming else model.train(
        data=cfg["data"],
        epochs=params["epochs"],
        imgsz=params.get("imgsz", 640),
        batch=params.get("batch", 16),
        workers=params.get("workers", 4),
        device=device,
        project=cfg["out"],
        name="train",
        exist_ok=True,
        plots=True,  # curves, confusion matrix, sample batches: the node keeps the whole folder with the run
        **{name: params[name] for name in HYPERPARAMETERS if name in params},
        **(params.get("extra") or {}),
    )

    trainer = model.trainer
    best = Path(trainer.best)
    if not best.is_file():
        best = Path(trainer.last)
    classes = []
    box = getattr(results, "box", None)
    if box is not None and len(getattr(box, "ap_class_index", [])):
        names = results.names
        for i, class_id in enumerate(box.ap_class_index):
            classes.append(
                {
                    "id": int(class_id),
                    "name": str(names[int(class_id)]),
                    "precision": round(float(box.p[i]), 6),
                    "recall": round(float(box.r[i]), 6),
                    "map50": round(float(box.ap50[i]), 6),
                    "map50_95": round(float(box.ap[i]), 6),
                }
            )
            masks = getattr(results, "seg", None)
            if masks is not None and i < len(getattr(masks, "ap", [])):
                classes[-1]["mask_map50_95"] = round(float(masks.ap[i]), 6)
    summary = clean(getattr(results, "results_dict", {}) or {})
    emit("result", weights=str(best), metrics=summary, classes=classes)
    return 0


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    sys.exit(main(sys.argv))
