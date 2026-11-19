"""Ultralytics YOLO trainer, the ``train`` entry point of the built-in ``yolo`` template:

    python train.py --config config.json

See ``badamoa.templates`` for the contract.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import badamoa_sdk as bm

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
    # pose models: the same four for their keypoints (OKS)
    "metrics/precision(P)": "pose_precision",
    "metrics/recall(P)": "pose_recall",
    "metrics/mAP50(P)": "pose_map50",
    "metrics/mAP50-95(P)": "pose_map50_95",
    # classification models
    "metrics/accuracy_top1": "top1",
    "metrics/accuracy_top5": "top5",
}

TASKS = ("detect", "segment", "classify", "pose")
# The ending of an Ultralytics model name for each task: yolo11n.pt detects, yolo11n-seg.pt segments …
ENDINGS = {"detect": "", "segment": "-seg", "classify": "-cls", "pose": "-pose"}
_OFFICIAL = re.compile(r"^(yolo(?:v)?\d+[nslmx])(-seg|-cls|-pose|-obb)?\.(pt|yaml)$", re.IGNORECASE)


def task_of(cfg: dict) -> str:
    """What this run trains: the config says (BADAMOA 0.60+); a Hub before that does not, and the model's name
    tells, as Ultralytics reads it (``-seg``, ``-cls``, ``-pose``)."""
    task = str(cfg.get("task") or "")
    if task in TASKS:
        return task
    name = str((cfg.get("train_params") or cfg.get("params") or {}).get("model") or "")
    ending = _OFFICIAL.match(Path(name).name)
    return (
        {"-seg": "segment", "-cls": "classify", "-pose": "pose"}.get((ending.group(2) or "").lower(), "detect")
        if ending
        else "detect"
    )


def model_for(name: str, task: str) -> str:
    """An Ultralytics model name for this task: ``yolo11n.pt`` on segmentation data is ``yolo11n-seg.pt``.
    Names of files on the server, and names it does not know, stay as they are."""
    found = _OFFICIAL.match(str(name))
    if not found or found.group(2) == "-obb":
        return str(name)
    return f"{found.group(1)}{ENDINGS[task]}.{found.group(3)}"


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


# ---- predictions on a few validation images, epoch by epoch --------------------------------

SAMPLES_KEY = "val_predictions"
SAMPLE_CONF = 0.25
SAMPLE_SIDE = 640  # the stored copy of each image; boxes are fractions, so they fit any size
MAX_RECORDS = 100  # longer runs record every n-th epoch
MAX_OUTLINE = 200  # points kept of one predicted outline
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def sample_images(val: str | list[str] | None, count: int) -> list[Path]:
    """The same few validation images every epoch, spread over the split.

    ``val``: the validation folder(s) as the trainer resolved them (``trainer.data["val"]``).
    """
    if count <= 0 or not val:
        return []
    files: list[Path] = []
    for folder in val if isinstance(val, list) else [val]:
        where = Path(str(folder))
        if where.is_dir():
            files += sorted(p for p in where.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return files[:: max(1, len(files) // count)][:count]


def truth_of(image: Path, task: str = "detect") -> tuple[list[list[float]], list[list]]:
    """The image's label file (the Ultralytics layout: ``images/`` → ``labels/``) as boxes, and outlines for segments.
    Pose labels are a box and its keypoints: the box counts."""
    images, labels = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    label = Path(labels.join(str(image).rsplit(images, 1))).with_suffix(".txt")
    boxes: list[list[float]] = []
    outlines: list[list] = []
    if not label.is_file():
        return boxes, outlines
    for line in label.read_text(encoding="utf-8").splitlines():
        try:
            values = [float(v) for v in line.split()]
        except ValueError:
            continue
        if len(values) == 5 or (task == "pose" and len(values) > 5):
            boxes.append([int(values[0]), *values[1:5]])
        elif len(values) >= 7 and len(values) % 2 == 1:  # a polygon: class x1 y1 x2 y2 …
            xs, ys = values[1::2], values[2::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            boxes.append([int(values[0]), (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
            outlines.append([int(values[0]), values[1:]])
    return boxes, outlines


def _iou(a: list[float], b: list[float]) -> float:
    """Overlap of two ``cx, cy, w, h`` boxes."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inner = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inner
    return inner / union if union > 0 else 0.0


def match(predicted: list[list[float]], truth: list[list[float]]) -> tuple[int, int, int]:
    """Found, missed and extra boxes: a prediction finds a true box of its class it overlaps by half (IoU 0.5)."""
    left = list(range(len(truth)))
    found = 0
    for box in sorted(predicted, key=lambda b: -b[1]):
        best, at = 0.5, None
        for i in left:
            if truth[i][0] == box[0] and (overlap := _iou(box[2:6], truth[i][1:5])) >= best:
                best, at = overlap, i
        if at is not None:
            left.remove(at)
            found += 1
    return found, len(truth) - found, len(predicted) - found


def class_folder_index(image: Path) -> int:
    """A classification image's class: its folder, named ``0003_<name>`` in class order (BADAMOA's export)."""
    head = image.parent.name.split("_", 1)[0]
    return int(head) if head.isdigit() else -1


class Samples:
    """Predictions of each epoch's weights on the same validation images, as a table of pictures with boxes
    (classification: the class it gave and the right one)."""

    def __init__(self, count: int, out: Path, imgsz: int, device: str, task: str = "detect") -> None:
        self.count, self.out, self.imgsz, self.task = count, out, imgsz, task
        self.device = device.split(",")[0]  # predictions need one device
        self.picked: list[dict] | None = None
        self.off = count <= 0

    def prepare(self, val: str | list[str] | None) -> list[dict]:
        from PIL import Image, ImageOps

        if self.picked is None:
            folder = self.out / ".badamoa" / "samples"
            folder.mkdir(parents=True, exist_ok=True)
            self.picked = []
            for index, image in enumerate(sample_images(val, self.count)):
                copy = folder / f"{index:03d}-{image.stem[:40]}.jpg"
                with Image.open(image) as opened:
                    picture = ImageOps.exif_transpose(opened).convert("RGB")
                    picture.thumbnail((SAMPLE_SIDE, SAMPLE_SIDE))
                    picture.save(copy, quality=85)
                boxes, outlines = truth_of(image, self.task) if self.task != "classify" else ([], [])
                self.picked.append(
                    {"image": image, "copy": copy, "truth": boxes, "outlines": outlines, "class": class_folder_index(image)}
                )
        return self.picked

    def record(self, weights: Path, epoch: int, val: str | list[str] | None) -> None:
        if self.off or not weights.is_file():
            return
        try:
            self._record(weights, epoch, val)
        except Exception as e:  # pictures are a help, never a reason for a run to fail
            self.off = True
            print(f"검증 이미지 예측을 남기지 못해 이번 실행에서는 더 남기지 않습니다: {e}", flush=True)

    def _record(self, weights: Path, epoch: int, val: str | list[str] | None) -> None:
        import torch
        from ultralytics import YOLO

        picked = self.prepare(val)
        if not picked:
            self.off = True
            return
        model = YOLO(str(weights))
        results = model.predict(
            [str(p["image"]) for p in picked],
            imgsz=self.imgsz,
            device=self.device,
            conf=SAMPLE_CONF,
            verbose=False,
            save=False,
        )
        rows = []
        if self.task == "classify":
            for sample, result in zip(picked, results, strict=True):
                guess, score = int(result.probs.top1), float(result.probs.top1conf)
                cell = bm.overlay(sample["copy"])
                rows.append([cell, sample["class"], guess, round(score, 4), int(guess == sample["class"])])
            bm.log_table(SAMPLES_KEY, ["image", "truth", "predicted", "confidence", "correct"], rows, step=epoch)
            del model, results
            return
        for sample, result in zip(picked, results, strict=True):
            boxes, outlines = [], []
            if result.boxes is not None and len(result.boxes):
                classes, scores = result.boxes.cls.tolist(), result.boxes.conf.tolist()
                for cls, conf, xywh in zip(classes, scores, result.boxes.xywhn.tolist(), strict=True):
                    boxes.append([int(cls), conf, *xywh])
                if result.masks is not None:
                    for cls, conf, points in zip(classes, scores, result.masks.xyn, strict=False):
                        flat = points[:: max(1, -(-len(points) // MAX_OUTLINE))].reshape(-1).tolist()
                        outlines.append([int(cls), conf, flat])
            found, missed, extra = match(boxes, sample["truth"])
            cell = bm.overlay(
                sample["copy"],
                boxes=boxes,
                truth=sample["truth"],
                outlines=outlines if sample["outlines"] or outlines else None,
                truth_outlines=sample["outlines"] or None,
            )
            rows.append([cell, len(sample["truth"]), len(boxes), found, missed, extra])
        bm.log_table(SAMPLES_KEY, ["image", "truth", "predicted", "found", "missed", "extra"], rows, step=epoch)
        del model, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    bm.init(ultralytics=ultralytics.__version__, torch=torch.__version__, gpus=gpus)

    task = task_of(cfg)
    # the model list follows the dataset's task; a name of another task's model (a rerun, a search) is matched to it
    weights = model_for(params["model"], task) if cfg.get("task") in TASKS else params["model"]
    if weights != params["model"]:
        print(f"{task} 데이터라 {params['model']} 대신 {weights}로 학습합니다.", flush=True)
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
    samples = Samples(int(params.get("samples", 16)), Path(cfg["out"]), params.get("imgsz", 640), device, task)

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
        every = max(1, math.ceil(trainer.epochs / MAX_RECORDS))
        if epoch % every == 0 or epoch >= trainer.epochs or trainer.stop:
            samples.record(Path(trainer.last), epoch, (trainer.data or {}).get("val"))

    def on_model_save(trainer) -> None:  # last.pt holds everything needed to continue: tell the node where it is
        emit("checkpoint", path=str(trainer.last), epoch=trainer.epoch + 1)

    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    results = (
        model.train(resume=True, data=cfg["data"], device=device)
        if resuming
        else model.train(
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
            points = getattr(results, "pose", None)
            if points is not None and i < len(getattr(points, "ap", [])):
                classes[-1]["pose_map50_95"] = round(float(points.ap[i]), 6)
    summary = clean(getattr(results, "results_dict", {}) or {})
    emit("result", weights=str(best), metrics=summary, classes=classes)
    return 0


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    sys.exit(main(sys.argv))
