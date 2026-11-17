"""Evaluate, export and test a trained Ultralytics YOLO model: the ``eval`` / ``export`` / ``predict`` entry points."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import badamoa_sdk as bm
from train import IMAGE_SUFFIXES, MAX_OUTLINE, SAMPLE_CONF, SAMPLE_SIDE, clean, match, truth_of

FAILURES = 48  # images with the most mistakes, kept as pictures with their boxes
COLUMNS = ["image", "truth", "predicted", "found", "missed", "extra"]


def as_labels(result) -> dict:
    """A result's predictions in the dataset's label form: [class, cx, cy, w, h, then the outline or keypoints]."""
    labels, scores = [], []
    if result.boxes is None or not len(result.boxes):
        return {"labels": labels, "scores": scores}
    masks = result.masks.xyn if result.masks is not None else None
    points = getattr(result, "keypoints", None)
    for n, (c, s, xywh) in enumerate(
        zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xywhn.tolist(), strict=True)
    ):
        rest: list[float] = []
        if masks is not None and n < len(masks) and len(masks[n]) >= 3:
            outline = masks[n][:: max(1, -(-len(masks[n]) // MAX_OUTLINE))]
            rest = [round(float(v), 5) for v in outline.reshape(-1).tolist()]
        elif points is not None and points.xyn is not None and n < len(points.xyn):
            seen = points.conf[n].tolist() if points.conf is not None else [1.0] * len(points.xyn[n])
            for (x, y), k in zip(points.xyn[n].tolist(), seen, strict=False):
                rest += [round(float(x), 5), round(float(y), 5), 2.0] if k >= 0.5 else [0.0, 0.0, 0.0]
        labels.append([int(c), *(round(float(v), 5) for v in xywh), *rest])
        scores.append(round(float(s), 4))
    return {"labels": labels, "scores": scores}


def per_image(run, model) -> None:
    """Every image of the evaluated split: true, predicted, found, missed and extra boxes (a prediction finds a
    true box of its class it overlaps by half). The ones with the most mistakes also as pictures with the boxes.
    """
    from PIL import Image, ImageOps

    data = Path(run.data).parent
    folder = data / "images" / run.split
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []
    if not images:
        return
    try:  # the exported images are named by their hash: show the names they were registered under
        registered = json.loads((data / "files.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        registered = {}
    print(f"이미지별 결과: {len(images):,}장을 확신도 {SAMPLE_CONF} 이상으로 예측합니다")
    rows, scored, predicted = [], [], []
    results = model.predict(
        [str(p) for p in images],
        stream=True,
        conf=SAMPLE_CONF,
        imgsz=run.train_params.get("imgsz", 640),
        device=run.device,
        verbose=False,
    )
    for image, result in zip(images, results, strict=True):
        truth, true_outlines = truth_of(image)
        boxes, outlines = [], []
        if result.boxes is not None and len(result.boxes):
            classes, scores = result.boxes.cls.tolist(), result.boxes.conf.tolist()
            boxes = [[int(c), s, *xywh] for c, s, xywh in zip(classes, scores, result.boxes.xywhn.tolist(), strict=True)]
            if result.masks is not None:
                for c, s, points in zip(classes, scores, result.masks.xyn, strict=False):
                    outlines.append([int(c), s, points[:: max(1, -(-len(points) // MAX_OUTLINE))].reshape(-1).tolist()])
        found, missed, extra = match(boxes, truth)
        predicted.append({"image": image.stem, **as_labels(result), "found": found, "missed": missed, "extra": extra})
        name = Path(registered.get(image.stem, image.name)).name
        rows.append([name, len(truth), len(boxes), found, missed, extra])
        if missed + extra:
            scored.append((missed + extra, image, name, boxes, truth, outlines, true_outlines, rows[-1]))
    rows.sort(key=lambda r: (-(r[4] + r[5]), r[0]))
    bm.log_table("per_image", COLUMNS, rows)
    if hasattr(bm, "log_predictions"):  # every image's predictions: the dataset screen sorts by likely wrong labels
        bm.log_predictions(predicted)

    pictures = run.out / ".badamoa" / "failures"
    pictures.mkdir(parents=True, exist_ok=True)
    cells = []
    for n, (_, image, name, boxes, truth, outlines, true_outlines, row) in enumerate(
        sorted(scored, key=lambda s: -s[0])[:FAILURES]
    ):
        copy = pictures / f"{n:03d}-{Path(name).stem[:40]}.jpg"
        with Image.open(image) as opened:
            picture = ImageOps.exif_transpose(opened).convert("RGB")
            picture.thumbnail((SAMPLE_SIDE, SAMPLE_SIDE))
            picture.save(copy, quality=85)
        cell = bm.overlay(
            copy,
            boxes=boxes,
            truth=truth,
            outlines=outlines if outlines or true_outlines else None,
            truth_outlines=true_outlines or None,
        )
        cells.append([cell, *row[1:]])
    if cells:
        bm.log_table("failures", COLUMNS, cells)
    wrong = sum(1 for r in rows if r[4] + r[5])
    print(f"틀린 것이 있는 이미지 {wrong:,}장 / {len(rows):,}장 (그림으로 남긴 것 {len(cells)}장)")


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
    try:
        per_image(run, model)
    except Exception as e:  # the metrics stand without it
        print(f"이미지별 결과를 남기지 못했습니다: {e}", flush=True)
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
