"""Ultralytics YOLO trainer, the ``train`` entry point of the built-in ``yolo`` template:

    python train.py --config config.json

See ``badamoa.templates`` for the contract.
"""

from __future__ import annotations

import json
import math
import os
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

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    results = model.train(
        data=cfg["data"],
        epochs=params["epochs"],
        imgsz=params.get("imgsz", 640),
        batch=params.get("batch", 16),
        workers=params.get("workers", 4),
        device=device,
        project=cfg["out"],
        name="train",
        exist_ok=True,
        plots=False,
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
