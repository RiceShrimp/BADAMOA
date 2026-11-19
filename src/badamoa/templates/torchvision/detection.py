"""Detection, instance segmentation and keypoints with torchvision's detection models, on data in the YOLO layout:
Faster R-CNN · RetinaNet · FCOS · SSDlite (``detect``), Mask R-CNN (``segment``, outlines drawn into masks) and
Keypoint R-CNN (``pose``, ``kpt_shape`` from data.yaml).

Box metrics come from torchmetrics, mask metrics too (``iou_type="segm"``); keypoint AP (OKS) from pycocotools.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import badamoa_sdk as bm
from common import COLORS, IMAGE_SUFFIXES, device_of, export, fit, optimizer_for, registered_names, start_from

# The metric each task's runs are ranked by, and the log line's numbers.
PRIMARY = {"detect": "map50_95", "segment": "mask_map50_95", "pose": "pose_map50_95"}
SHOWN = {
    "detect": [("mAP50-95", "map50_95"), ("mAP50", "map50")],
    "segment": [("mask mAP50-95", "mask_map50_95"), ("mask mAP50", "mask_map50"), ("box mAP50-95", "map50_95")],
    "pose": [("OKS mAP50-95", "pose_map50_95"), ("OKS mAP50", "pose_map50"), ("box mAP50-95", "map50_95")],
}
FAMILY = {"segment": "maskrcnn", "pose": "keypointrcnn"}  # the models that do each task beside detection
MAX_OUTLINE = 200  # points kept of one predicted outline
MATCH_CONF = 0.25  # predictions an evaluation compares with the labels, image by image
# Keypoint R-CNN gives each point the heatmap's logit at its place, not a visibility: above this it is labelled
# as seen (2), below as not there (0 0 0).
POINT_SCORE = 0.0
COLUMNS = ["image", "truth", "predicted", "found", "missed", "extra"]


# ---- data ----------------------------------------------------------------------------------


def data_yaml(run) -> Path:
    return Path(run.data) if run.data and Path(run.data).is_file() else run.data_dir / "data.yaml"


def pose_shape(path: Path) -> tuple[int, int, list[int] | None]:
    """Keypoints per object, values per keypoint (2: x y, 3: x y visibility) and the left/right swap of a flip."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    shape = data.get("kpt_shape") or []
    if len(shape) != 2:
        raise SystemExit("data.yaml에 kpt_shape가 없습니다. 키포인트(Kpt) 데이터셋으로 학습하세요.")
    flip = data.get("flip_idx")
    return int(shape[0]), int(shape[1]), [int(i) for i in flip] if flip else None


class YoloFolder:
    """``images/<split>`` + ``labels/<split>`` in the YOLO layout as torchvision detection targets: boxes, and for
    segmentation each outline drawn as a mask, for pose each box's keypoints (x, y, visibility in pixels)."""

    def __init__(
        self,
        root: Path,
        split: str,
        task: str = "detect",
        flip: float = 0.0,
        points: int = 0,
        dims: int = 3,
        flip_idx: list[int] | None = None,
    ) -> None:
        self.images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.labels = root / "labels" / split
        self.task, self.points, self.dims, self.flip_idx = task, points, dims, flip_idx
        # Flipping keypoints needs to know which point becomes which: without flip_idx they are not flipped.
        self.flip = 0.0 if task == "pose" and not flip_idx else flip

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image, ImageDraw
        from torchvision.transforms.functional import hflip, pil_to_tensor

        path = self.images[index]
        with Image.open(path) as opened:
            picture = opened.convert("RGB")
        width, height = picture.size
        boxes, labels, masks, keypoints = [], [], [], []
        label_file = self.labels / f"{path.stem}.txt"
        lines = label_file.read_text(encoding="utf-8").splitlines() if label_file.is_file() else []
        for line in lines:
            try:
                values = [float(v) for v in line.split()]
            except ValueError:
                continue
            if len(values) < 5:
                continue
            outline = None
            if self.task == "segment" and len(values) >= 7:  # class x1 y1 x2 y2 … (an outline of 3 points or more)
                outline = [(x * width, y * height) for x, y in zip(values[1::2], values[2::2], strict=False)]
                xs, ys = [x for x, _ in outline], [y for _, y in outline]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                cx, cy, w, h = values[1:5]
                x0, y0 = (cx - w / 2) * width, (cy - h / 2) * height
                x1, y1 = (cx + w / 2) * width, (cy + h / 2) * height
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            boxes.append([x0, y0, x1, y1])
            labels.append(int(values[0]) + 1)  # 0 is the background class in torchvision
            if self.task == "segment":
                mask = Image.new("L", (width, height), 0)
                ImageDraw.Draw(mask).polygon(outline or [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill=1)
                masks.append(pil_to_tensor(mask)[0])
            elif self.task == "pose":
                given = values[5 : 5 + self.points * self.dims]
                given += [0.0] * (self.points * self.dims - len(given))
                found = []
                for k in range(self.points):
                    x, y, *rest = given[k * self.dims : (k + 1) * self.dims]
                    seen = rest[0] if rest else (2.0 if x or y else 0.0)
                    found.append([x * width, y * height, seen] if seen > 0 else [0.0, 0.0, 0.0])
                keypoints.append(found)
        image = pil_to_tensor(picture).float() / 255
        target: dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if self.task == "segment":
            target["masks"] = torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
        elif self.task == "pose":
            target["keypoints"] = torch.tensor(keypoints, dtype=torch.float32).reshape(-1, self.points, 3)
        if self.flip and random.random() < self.flip:
            image = hflip(image)
            target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                points = target["keypoints"]
                seen = points[..., 2] > 0
                points[..., 0] = torch.where(seen, width - points[..., 0], points[..., 0])
                target["keypoints"] = points[:, self.flip_idx]
        return image, target


def collate(batch):
    return tuple(zip(*batch, strict=True))


# ---- the model -----------------------------------------------------------------------------


def build(name: str, task: str, classes: int, pretrained: bool, imgsz: int, points: int = 0):
    """``name`` from torchvision's detection models, with a head for our classes (and keypoints). Pretrained:
    Faster R-CNN, Mask R-CNN and Keypoint R-CNN start from the whole COCO model with the last layers replaced, the
    others from an ImageNet backbone."""
    import torchvision.models.detection as detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    factory = getattr(detection, name, None)
    family = FAMILY.get(task)  # detection takes any model but those two
    fits = name.startswith(family) if family else not any(name.startswith(other) for other in FAMILY.values())
    if factory is None or not callable(factory) or name.startswith("_") or not fits:
        kinds = {"detect": "검출", "segment": "Mask R-CNN (maskrcnn_…)", "pose": "Keypoint R-CNN (keypointrcnn_…)"}
        raise SystemExit(f"모델 {name}은(는) 이 태스크에 쓸 수 없습니다. torchvision의 {kinds[task]} 모델을 고르세요.")
    size = {} if name.startswith("ssd") else {"min_size": imgsz, "max_size": imgsz}  # SSD models have a fixed size
    extra = {"num_keypoints": points} if task == "pose" else {}
    whole = task in FAMILY or name.startswith("fasterrcnn")
    if pretrained and whole:
        # The whole model comes pretrained on COCO; only the last layers are replaced for our classes.
        model = factory(weights="DEFAULT", **size)
        heads = model.roi_heads
        heads.box_predictor = FastRCNNPredictor(heads.box_predictor.cls_score.in_features, classes + 1)
        if task == "segment":
            heads.mask_predictor = MaskRCNNPredictor(heads.mask_predictor.conv5_mask.in_channels, 256, classes + 1)
        if task == "pose" and points != heads.keypoint_predictor.kps_score_lowres.out_channels:
            # COCO's 17 points of a person: another number of points needs its own last layer.
            heads.keypoint_predictor = KeypointRCNNPredictor(heads.keypoint_predictor.kps_score_lowres.in_channels, points)
        return model
    backbone = "DEFAULT" if pretrained else None  # ImageNet backbone, new head
    return factory(weights=None, weights_backbone=backbone, num_classes=classes + 1, **size, **extra)


def points_in(state: dict) -> int:
    """Keypoints per object a trained Keypoint R-CNN finds, read from its weights."""
    weight = state.get("roi_heads.keypoint_predictor.kps_score_lowres.weight")
    return int(weight.shape[1]) if weight is not None else 0


# ---- metrics --------------------------------------------------------------------------------


class KeypointAP:
    """Keypoint AP over OKS as COCO computes it (pycocotools). The object's scale is its box area × 0.53 and each
    point's sigma COCO's for 17 points, else 1/K (as Ultralytics does)."""

    def __init__(self, points: int) -> None:
        self.points = points
        self.images: list[dict] = []
        self.truth: list[dict] = []
        self.found: list[dict] = []

    def update(self, outputs: list[dict], targets: list[dict]) -> None:
        for out, target in zip(outputs, targets, strict=True):
            image = len(self.images) + 1
            self.images.append({"id": image})
            rows = zip(target["boxes"].tolist(), target["labels"].tolist(), target["keypoints"].tolist(), strict=True)
            for (x0, y0, x1, y1), label, points in rows:
                self.truth.append(
                    {
                        "id": len(self.truth) + 1,
                        "image_id": image,
                        "category_id": int(label),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": (x1 - x0) * (y1 - y0) * 0.53,
                        "iscrowd": 0,
                        "num_keypoints": sum(1 for p in points if p[2] > 0),
                        "keypoints": [v for x, y, s in points for v in (x, y, 2 if s > 0 else 0)],
                    }
                )
            rows = zip(out["labels"].tolist(), out["scores"].tolist(), out["keypoints"].tolist(), strict=True)
            for label, score, points in rows:
                flat = [v for x, y, _ in points for v in (x, y, 1)]
                self.found.append({"image_id": image, "category_id": int(label), "score": score, "keypoints": flat})

    def compute(self, classes: int) -> tuple[float, float, list[float]]:
        """AP over OKS 0.5:0.95, AP at OKS 0.5, and AP of each class (0 for one without labelled points)."""
        if not self.found or not any(t["num_keypoints"] for t in self.truth):
            return 0.0, 0.0, [0.0] * classes
        import contextlib
        import io

        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints its own tables
            truth = COCO()
            categories = [{"id": i + 1, "name": str(i)} for i in range(classes)]
            truth.dataset = {"images": self.images, "annotations": self.truth, "categories": categories}
            truth.createIndex()
            scored = COCOeval(truth, truth.loadRes(self.found), "keypoints")
            if self.points != 17:  # COCOeval's own sigmas are those of COCO's 17 points
                scored.params.kpt_oks_sigmas = np.ones(self.points) / self.points
            scored.evaluate()
            scored.accumulate()
            scored.summarize()
        precision = scored.eval["precision"][:, :, :, 0, -1]  # OKS threshold × recall × class, every size
        per_class = []
        for k in range(classes):
            values = precision[:, :, k] if k < precision.shape[2] else np.array([])
            values = values[values > -1]
            per_class.append(float(values.mean()) if values.size else 0.0)
        return max(0.0, float(scored.stats[0])), max(0.0, float(scored.stats[1])), per_class


def evaluate(model, loader, device, task: str, names: list[str], points: int = 0, each_class=False, per_image=None):
    """The metrics of the task under their platform names, and each class's when ``each_class``;
    ``per_image(index, size, output, target)`` sees every image's predictions (in loader order) as it goes."""
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    segment = task == "segment"
    metric = MeanAveragePrecision(iou_type=("bbox", "segm") if segment else "bbox", class_metrics=each_class)
    keypoints = KeypointAP(points) if task == "pose" else None
    model.eval()
    seen = 0
    with torch.no_grad():
        for images, targets in loader:
            outputs = [{k: v.cpu() for k, v in out.items()} for out in model([image.to(device) for image in images])]
            predicted, truth = [], []
            for out, target in zip(outputs, targets, strict=True):
                predicted.append({k: out[k] for k in ("boxes", "scores", "labels")})
                truth.append({k: target[k] for k in ("boxes", "labels")})
                if segment:
                    predicted[-1]["masks"] = out["masks"][:, 0] > 0.5
                    truth[-1]["masks"] = target["masks"].bool()
            metric.update(predicted, truth)
            if keypoints is not None:
                keypoints.update(outputs, list(targets))
            if per_image is not None:
                for out, target, image in zip(outputs, targets, images, strict=True):
                    per_image(seen, (image.shape[-1], image.shape[-2]), out, target)
                    seen += 1
    scores = metric.compute()

    def number(name: str, kind: str = "bbox") -> float:
        value = scores.get(f"{kind}_{name}", scores.get(name)) if segment else scores.get(name)
        return max(0.0, float(value)) if value is not None else 0.0

    metrics = {"map50_95": number("map"), "map50": number("map_50"), "recall": number("mar_100")}
    if segment:
        metrics |= {
            "mask_map50_95": number("map", "segm"),
            "mask_map50": number("map_50", "segm"),
            "mask_recall": number("mar_100", "segm"),
        }
    per_class: dict[int, dict[str, Any]] = {}
    ids = scores["classes"].reshape(-1).tolist() if each_class else []
    for kind, key in (("bbox", "map50_95"), ("segm", "mask_map50_95")) if segment else (("bbox", "map50_95"),):
        values = scores[f"{kind}_map_per_class" if segment else "map_per_class"].reshape(-1).tolist()
        for class_id, value in zip(ids, values, strict=False):
            index = int(class_id) - 1
            if 0 <= index < len(names):
                entry = per_class.setdefault(index, {"id": index, "name": names[index]})
                entry[key] = max(0.0, float(value))
    if keypoints is not None:
        metrics["pose_map50_95"], metrics["pose_map50"], each = keypoints.compute(len(names))
        for index, value in enumerate(each if each_class else []):
            per_class.setdefault(index, {"id": index, "name": names[index]})["pose_map50_95"] = value
    return metrics, [per_class[i] for i in sorted(per_class)]


def shown(task: str):
    return lambda metrics: " · ".join(f"{label} {metrics[key]:.3f}" for label, key in SHOWN[task])


# ---- training -------------------------------------------------------------------------------


def train(run, hp: bm.Params, task: str) -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.hub.set_dir(str(run.cache / "torch"))  # pretrained weights are kept between runs
    device = device_of(run)
    points, dims, flip_idx = pose_shape(data_yaml(run)) if task == "pose" else (0, 3, None)
    shape = {"task": task, "points": points, "dims": dims, "flip_idx": flip_idx}
    val_split = "val" if (run.data_dir / "images" / "val").is_dir() else "train"
    # Workers kept between epochs: starting them again each epoch costs seconds (Windows starts a new Python).
    workers = {"num_workers": hp.workers, "persistent_workers": hp.workers > 0, "collate_fn": collate}
    train_data = YoloFolder(run.data_dir, "train", flip=hp.fliplr, **shape)
    loader = DataLoader(train_data, batch_size=hp.batch, shuffle=True, **workers)
    val_data = YoloFolder(run.data_dir, val_split, **shape)
    val = DataLoader(val_data, batch_size=hp.batch, **workers)
    sizes = f"train {len(train_data)}장 · val {len(val_data)}장"
    what = f" · 점 {points}개" if task == "pose" else ""
    print(f"{hp.model} · {task} · 클래스 {len(run.classes)}개{what} · {sizes} · 장치 {device}")
    if task == "pose" and hp.fliplr and not flip_idx:
        print("data.yaml에 좌우 짝(flip_idx)이 없어 이미지를 뒤집지 않습니다.")

    model = build(hp.model, task, len(run.classes), hp.pretrained, hp.imgsz, points)
    if run.init_weights:
        start_from(model, run.init_weights)
    model = model.to(device)
    weights = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_for(hp, weights)

    def one_epoch(epoch: int) -> float:
        model.train()
        total, batches = 0.0, 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise SystemExit(f"loss가 발산했습니다 (epoch {epoch}). 학습률을 낮춰 보세요.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weights, hp.clip)  # small batches make detectors blow up without it
            optimizer.step()
            total, batches = total + float(loss.detach()), batches + 1
        return total / max(1, batches)

    def validate() -> dict[str, float]:
        return evaluate(model, val, device, task, run.classes, points)[0]

    best = fit(run, hp, model, optimizer, one_epoch, validate, PRIMARY[task], shown(task))
    _, per_class = evaluate(model, val, device, task, run.classes, points, each_class=True)  # the best weights'
    bm.log_model(run.out / "best.pt", classes=per_class)
    bm.finish(best)


# ---- evaluate · export · test · 오토라벨 ---------------------------------------------------------


def found_in(out: dict, size: tuple[int, int], task: str, threshold: float, classes: int) -> list[dict[str, Any]]:
    """The predictions at or above ``threshold``: class, score, the box in pixels and as YOLO fractions, and the
    rest of the dataset's label: the outline (segment) or the keypoints with their visibility (pose)."""
    width, height = size
    rows = []
    for n, ((x0, y0, x1, y1), label, score) in enumerate(
        zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist(), strict=True)
    ):
        index = int(label) - 1
        if score < threshold or not 0 <= index < classes:
            continue
        box = [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height]
        row: dict[str, Any] = {"class": index, "score": float(score), "pixels": (x0, y0, x1, y1), "box": box, "rest": []}
        if task == "segment":
            row["mask"] = out["masks"][n, 0] > 0.5
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # a mask too small to have an outline: its box
            row["rest"] = outline_of(row["mask"], width, height) or [v for x, y in corners for v in (x / width, y / height)]
        elif task == "pose":
            row["points"] = []
            points = zip(out["keypoints"][n].tolist(), out["keypoints_scores"][n].tolist(), strict=True)
            for (x, y, _), point_score in points:
                visible = point_score > POINT_SCORE and 0 <= x <= width and 0 <= y <= height
                row["points"].append((x, y, visible))
                row["rest"] += [x / width, y / height, 2.0] if visible else [0.0, 0.0, 0.0]
        rows.append(row)
    return rows


def outline_of(mask, width: int, height: int) -> list[float]:
    """A predicted mask's outline as the dataset stores it (x, y fractions): its largest region's border, without
    the pixel steps (within a pixel), so that people fixing it move a few points."""
    import cv2

    contours, _ = cv2.findContours(mask.numpy().astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    border = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    if len(border) < 3:
        return []
    border = border[:: max(1, -(-len(border) // MAX_OUTLINE))]
    return [round(float(v), 5) for x, y in border.tolist() for v in (x / width, y / height)]


def as_label(row: dict[str, Any]) -> list[float]:
    return [row["class"], *(round(v, 5) for v in row["box"]), *(round(v, 5) for v in row["rest"])]


def match(rows: list[dict[str, Any]], target: dict) -> tuple[int, int, int]:
    """Found, missed and extra: a prediction finds a true box of its class it overlaps by half (best first)."""
    import torch
    from torchvision.ops import box_iou

    truth, classes = target["boxes"], target["labels"] - 1
    if not rows or not len(truth):
        return 0, len(truth), len(rows)
    overlap = box_iou(torch.tensor([r["pixels"] for r in rows], dtype=torch.float32), truth)
    overlap[torch.tensor([r["class"] for r in rows])[:, None] != classes[None, :]] = 0
    taken: set[int] = set()
    for i in sorted(range(len(rows)), key=lambda i: -rows[i]["score"]):
        free = [(float(overlap[i, j]), j) for j in range(len(truth)) if j not in taken]
        best = max(free, default=(0.0, -1))
        if best[0] >= 0.5:
            taken.add(best[1])
    return len(taken), len(truth) - len(taken), len(rows) - len(taken)


def draw(picture, rows: list[dict[str, Any]], names: list[str]):
    """The predictions on the image: masks, boxes with class and score, keypoints."""
    from PIL import ImageDraw

    masks = [r["mask"] for r in rows if "mask" in r]
    if masks:
        import torch
        from torchvision.transforms.functional import pil_to_tensor, to_pil_image
        from torchvision.utils import draw_segmentation_masks

        colors = [COLORS[r["class"] % len(COLORS)] for r in rows if "mask" in r]
        picture = to_pil_image(draw_segmentation_masks(pil_to_tensor(picture), torch.stack(masks), 0.45, colors))
    pen = ImageDraw.Draw(picture)
    for row in rows:
        color = COLORS[row["class"] % len(COLORS)]
        x0, y0, x1, y1 = row["pixels"]
        pen.rectangle((x0, y0, x1, y1), outline=color, width=3)
        title = names[row["class"]] if names[row["class"]].isascii() else f"#{row['class']}"
        pen.text((x0 + 4, max(0, y0 - 12)), f"{title} {row['score']:.2f}", fill=color)
        for x, y, visible in row.get("points", []):
            if visible:
                pen.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="white")
    return picture


def operate(run, settings: bm.Params, task: str) -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.hub.set_dir(str(run.cache / "torch"))
    device = device_of(run)
    state = torch.load(run.weights, map_location="cpu")
    points = points_in(state) if task == "pose" else 0
    model = build(str(settings.model), task, len(run.classes), pretrained=False, imgsz=int(settings.imgsz), points=points)
    model.load_state_dict(state)
    model.eval().to(device)
    action = run.cfg["action"]
    threshold = float(run.config.get("conf", 0.25))

    if action == "eval":
        dims = pose_shape(data_yaml(run))[1] if task == "pose" else 3
        data = YoloFolder(run.data_dir, run.split, task=task, points=points, dims=dims)
        loader = DataLoader(data, batch_size=int(settings.batch), collate_fn=collate)
        names = registered_names(run.data_dir)
        rows, predicted = [], []

        def per_image(index: int, size: tuple[int, int], out: dict, target: dict) -> None:
            path = data.images[index]
            found = found_in(out, size, task, MATCH_CONF, len(run.classes))
            right, missed, extra = match(found, target)
            labels, scores = [as_label(r) for r in found], [round(r["score"], 4) for r in found]
            counts = {"found": right, "missed": missed, "extra": extra}
            predicted.append({"image": path.stem, "labels": labels, "scores": scores, **counts})
            rows.append([Path(names.get(path.stem, path.name)).name, len(target["boxes"]), len(found), right, missed, extra])

        metrics, per_class = evaluate(model, loader, device, task, run.classes, points, True, per_image)
        rows.sort(key=lambda r: (-(r[4] + r[5]), r[0]))
        bm.log_table("per_image", COLUMNS, rows)
        if hasattr(bm, "log_predictions"):  # every image's predictions: the dataset screen sorts by likely wrong labels
            bm.log_predictions(predicted)
        wrong = sum(1 for r in rows if r[4] + r[5])
        print(f"틀린 것이 있는 이미지 {wrong:,}장 / {len(rows):,}장 (확신도 {MATCH_CONF} 이상의 예측으로)")
        bm.finish(metrics, classes=per_class)
    elif action == "export":
        side = 320 if str(settings.model).startswith("ssdlite") else int(settings.imgsz)
        outputs = {
            "detect": ["boxes", "labels", "scores"],
            "segment": ["boxes", "labels", "scores", "masks"],
            "pose": ["boxes", "labels", "scores", "keypoints", "keypoints_scores"],
        }[task]
        example = ([torch.rand(3, side, side)],)  # the model takes one argument: a list of images
        export(run, model, example, {"input_names": ["images"], "output_names": outputs}, opset_version=17)
    else:  # predict (test: pictures and boxes) or autolabel (a labeling set's images, named by their hash)
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        labelling = action == "autolabel"
        paths = sorted(p for p in run.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if labelling:
            print(f"오토라벨: {len(paths):,}장을 확신도 {threshold} 이상으로 예측합니다", flush=True)
        predictions, predicted = [], []
        for n, path in enumerate(paths, 1):
            with Image.open(path) as opened:
                picture = opened.convert("RGB")
            with torch.no_grad():
                out = model([(pil_to_tensor(picture).float() / 255).to(device)])[0]
            found = found_in({k: v.cpu() for k, v in out.items()}, picture.size, task, threshold, len(run.classes))
            if labelling:
                labels = [as_label(r) for r in found]
                predicted.append({"image": path.stem, "labels": labels, "scores": [round(r["score"], 4) for r in found]})
                if n % 200 == 0 or n == len(paths):
                    print(f"{n:,}/{len(paths):,}장", flush=True)
                continue
            draw(picture, found, run.classes).save(run.out / path.name)
            bm.log_artifact(run.out / path.name)
            boxes = [[r["class"], round(r["score"], 4), *(round(v, 5) for v in r["box"])] for r in found]
            predictions.append({"image": path.name, "boxes": boxes})
            print(f"{path.name}: {len(boxes)}개 검출")
        if labelling:
            bm.log_predictions(predicted)
            bm.finish(images=len(predicted), labels=sum(len(p["labels"]) for p in predicted))
        else:
            bm.finish(predictions=predictions)
