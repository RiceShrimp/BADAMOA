"""Ultralytics YOLO trainer. Runs inside the ``yolo`` plugin environment:

    python yolo_train.py config.json

See ``badamoa.plugins`` for the stdout contract.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

NAMES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
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


def main(config_path: str) -> int:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

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

    weights = cfg["model"]
    if not cfg.get("pretrained", True):
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
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg.get("workers", 4),
        device=device,
        project=cfg["out"],
        name="train",
        exist_ok=True,
        plots=False,
        **cfg.get("extra", {}),
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
    summary = clean(getattr(results, "results_dict", {}) or {})
    emit("result", weights=str(best), metrics=summary, classes=classes)
    return 0


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    sys.exit(main(sys.argv[1]))
