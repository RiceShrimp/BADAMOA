"""Evaluate, export, test and 오토라벨 with a trained Ultralytics YOLO model: the ``eval`` / ``export`` /
``predict`` entry points (오토라벨 is predict with ``action = "autolabel"``)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import badamoa_sdk as bm
from train import (
    IMAGE_SUFFIXES,
    MAX_OUTLINE,
    SAMPLE_CONF,
    class_folder_index,
    clean,
    match,
    task_of,
    truth_of,
)

COLUMNS = ["image", "truth", "predicted", "found", "missed", "extra"]
PREDICT_BATCH = 8


def predict_each(model, images: list[Path], **settings):
    """Each image with its result, a few at a time. Ultralytics reads a list of paths into memory and predicts it
    as one batch (a split of thousands ran the GPU out of memory): the paths go in a list file it reads as it goes,
    ``PREDICT_BATCH`` images per pass."""
    import tempfile

    by_path = {os.path.normcase(os.path.abspath(p)): p for p in images}
    with tempfile.TemporaryDirectory(prefix="badamoa-predict-") as tmp:
        listed = Path(tmp) / "images.txt"
        listed.write_text("\n".join(str(Path(p).absolute()) for p in images), encoding="utf-8")
        for result in model.predict(str(listed), stream=True, batch=PREDICT_BATCH, verbose=False, **settings):
            image = by_path.get(os.path.normcase(os.path.abspath(result.path)))
            if image is not None:
                yield image, result


def as_labels(result) -> dict:
    """A result's predictions in the dataset's label form: [class, cx, cy, w, h, then the outline or keypoints]."""
    labels, scores = [], []
    if getattr(result, "probs", None) is not None:  # classification: the class it gives the image
        return {"labels": [[int(result.probs.top1)]], "scores": [round(float(result.probs.top1conf), 4)]}
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


def per_class_image(run, model) -> None:
    """Classification: every image of the evaluated split with the class it got and the right one (its folder)."""
    folder = Path(run.data) / run.split
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []
    if not images:
        return
    try:
        registered = json.loads((Path(run.data) / "files.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        registered = {}
    print(f"이미지별 결과: {len(images):,}장을 분류합니다")
    rows, predicted = [], []
    for image, result in predict_each(model, images, imgsz=run.train_params.get("imgsz", 224), device=run.device):
        truth, guess, score = class_folder_index(image), int(result.probs.top1), float(result.probs.top1conf)
        wrong = int(guess != truth)
        predicted.append({"image": image.stem, "labels": [[guess]], "scores": [round(score, 4)],
                          "found": 1 - wrong, "missed": wrong, "extra": wrong})  # fmt: skip
        rows.append([Path(registered.get(image.stem, image.name)).name, truth, guess, round(score, 4), 1 - wrong])
    rows.sort(key=lambda r: (r[4], -r[3], r[0]))  # wrong ones first, the surest of them first
    bm.log_table("per_image", ["image", "truth", "predicted", "confidence", "correct"], rows)
    if hasattr(bm, "log_predictions"):
        bm.log_predictions(predicted)
    print(f"틀린 이미지 {sum(1 for r in rows if not r[4]):,}장 / {len(rows):,}장")


def per_image(run, model) -> None:
    """Every image of the evaluated split: true, predicted, found, missed and extra boxes (a prediction finds a
    true box of its class it overlaps by half). No pictures: 검수 shows each image's mistakes on the dataset's own
    image, from the predictions kept here.
    """
    task = task_of(run.cfg)
    if task == "classify":
        per_class_image(run, model)
        return
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
    rows, predicted = [], []
    settings = {"conf": SAMPLE_CONF, "imgsz": run.train_params.get("imgsz", 640), "device": run.device}
    for image, result in predict_each(model, images, **settings):
        truth, _ = truth_of(image, task)
        boxes = []
        if result.boxes is not None and len(result.boxes):
            classes, scores = result.boxes.cls.tolist(), result.boxes.conf.tolist()
            boxes = [[int(c), s, *xywh] for c, s, xywh in zip(classes, scores, result.boxes.xywhn.tolist(), strict=True)]
        found, missed, extra = match(boxes, truth)
        predicted.append({"image": image.stem, **as_labels(result), "found": found, "missed": missed, "extra": extra})
        name = Path(registered.get(image.stem, image.name)).name
        rows.append([name, len(truth), len(boxes), found, missed, extra])
    rows.sort(key=lambda r: (-(r[4] + r[5]), r[0]))
    bm.log_table("per_image", COLUMNS, rows)
    if hasattr(bm, "log_predictions"):  # every image's predictions: 검수 and the dataset screen draw them
        bm.log_predictions(predicted)
    wrong = sum(1 for r in rows if r[4] + r[5])
    print(f"틀린 것이 있는 이미지 {wrong:,}장 / {len(rows):,}장 (검수 탭에서 이미지별로 봅니다)")


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
        plots=False,  # no pictures: the numbers and per-class results stay, each image's mistakes are on 검수
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
            for key, part in (("mask_map50_95", "seg"), ("pose_map50_95", "pose")):
                found = getattr(results, part, None)
                if found is not None and i < len(getattr(found, "ap", [])):
                    per_class[-1][key] = round(float(found.ap[i]), 6)
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
        if getattr(result, "probs", None) is not None:  # classification: the five likeliest classes
            top = [
                [int(i), round(float(p), 4)] for i, p in zip(result.probs.top5, result.probs.top5conf.tolist(), strict=False)
            ]
            result.save(filename=str(run.out / name))
            bm.log_artifact(run.out / name)
            predictions.append({"image": name, "top": top})
            print(f"{name}: {top[0][0] if top else '-'} ({top[0][1] if top else 0:.2f})")
            continue
        boxes = []
        if result.boxes is not None:
            rows = zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xywhn.tolist(), strict=True)
            boxes = [[int(cls), round(conf, 4), *[round(v, 5) for v in xywh]] for cls, conf, xywh in rows]
        result.save(filename=str(run.out / name))  # the image with the boxes drawn on it
        bm.log_artifact(run.out / name)
        predictions.append({"image": name, "boxes": boxes})
        print(f"{name}: {len(boxes)}개 검출")
    bm.finish(predictions=predictions)


def autolabel(run, model) -> None:
    """A labeling set's images (the node named each file by its hash): every image's predictions in the
    dataset's label form, outlines and keypoints included. No pictures: there may be thousands."""
    images = sorted(p for p in Path(run.source).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    conf = float(run.config.get("conf", 0.25))
    print(f"오토라벨: {len(images):,}장을 확신도 {conf} 이상으로 예측합니다", flush=True)
    predicted = []
    settings = {"conf": conf, "imgsz": run.train_params.get("imgsz", 640), "device": run.device}
    for n, (image, result) in enumerate(predict_each(model, images, **settings), 1):
        predicted.append({"image": image.stem, **as_labels(result)})
        if n % 200 == 0 or n == len(images):
            print(f"{n:,}/{len(images):,}장", flush=True)
    bm.log_predictions(predicted)
    labels = sum(len(p["labels"]) for p in predicted)
    print(f"라벨 {labels:,}개를 그렸습니다 (예측이 없는 이미지 {sum(1 for p in predicted if not p['labels']):,}장)")
    bm.finish(images=len(predicted), labels=labels)


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
    elif action == "autolabel":
        autolabel(run, model)
    else:
        predict(run, model)


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    main()
