"""Evaluate, export and test a trained Ultralytics YOLO model: the ``eval`` / ``export`` / ``predict`` entry points."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import badamoa_sdk as bm
from train import clean  # the same metric names as during training


def evaluate(run, model) -> None:
    settings = run.train_params
    results = model.val(
        data=run.data,
        split=run.split,
        imgsz=settings.get("imgsz", 640),
        batch=settings.get("batch", 16) if settings.get("batch", 16) > 0 else 16,
        workers=0,
        device=run.device,
        project=str(run.out),
        name="val",
        exist_ok=True,
        plots=True,  # confusion matrix, PR curves, sample predictions: kept with the evaluation
    )
    per_class = []
    box = getattr(results, "box", None)
    if box is not None and len(getattr(box, "ap_class_index", [])):
        for i, class_id in enumerate(box.ap_class_index):
            per_class.append(
                {
                    "id": int(class_id),
                    "name": str(results.names[int(class_id)]),
                    "precision": round(float(box.p[i]), 6),
                    "recall": round(float(box.r[i]), 6),
                    "map50": round(float(box.ap50[i]), 6),
                    "map50_95": round(float(box.ap[i]), 6),
                }
            )
    bm.finish(clean(getattr(results, "results_dict", {}) or {}), classes=per_class)


def export(run, model_path: Path) -> None:
    from ultralytics import YOLO

    # Ultralytics writes the exported file next to the weights: work on a copy in the output folder.
    local = run.out / "model.pt"
    shutil.copy(run.weights, local)
    target = run.config.get("format", "onnx")
    exported = Path(YOLO(str(local)).export(format=target, imgsz=run.train_params.get("imgsz", 640)))
    local.unlink(missing_ok=True)  # the copy of the weights is not a result
    print(f"내보낸 파일: {exported.name} ({exported.stat().st_size / 1024**2:.1f} MB)")
    bm.log_artifact(exported)
    bm.finish(file=exported.name, bytes=exported.stat().st_size)


def predict(run, model) -> None:
    predictions = []
    results = model.predict(
        source=str(run.source),
        conf=float(run.config.get("conf", 0.25)),
        imgsz=run.train_params.get("imgsz", 640),
        device=run.device,
        save=False,
        verbose=False,
    )
    for result in results:
        name = Path(result.path).name
        boxes = []
        if result.boxes is not None:
            rows = zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xywhn.tolist(), strict=True)
            boxes = [[int(cls), round(conf, 4), *[round(v, 5) for v in xywh]] for cls, conf, xywh in rows]
        result.save(filename=str(run.out / name))  # the image with the boxes drawn on it
        bm.log_artifact(run.out / name)
        predictions.append({"image": name, "boxes": boxes})
        print(f"{name}: {len(boxes)}개 검출")
    bm.finish(predictions=predictions)


def main() -> None:
    run = bm.init()
    os.chdir(run.cache)
    action = run.cfg["action"]
    if action == "export":
        export(run, run.weights)
        return
    from ultralytics import YOLO

    model = YOLO(str(run.weights))
    if action == "eval":
        evaluate(run, model)
    else:
        predict(run, model)


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    main()
