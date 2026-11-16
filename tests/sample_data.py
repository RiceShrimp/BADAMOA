"""Generate small YOLO / COCO datasets whose images show their own boxes (used by tests and demos)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ["person", "car", "dog"]
COLORS = [(214, 69, 65), (42, 120, 214), (27, 175, 122)]


def _image(path: Path, boxes: list[list[float]], size: tuple[int, int], seed: int) -> None:
    rng = random.Random(seed)
    w, h = size
    image = Image.new("RGB", size, (rng.randint(40, 90), rng.randint(60, 110), rng.randint(70, 120)))
    draw = ImageDraw.Draw(image)
    for cls, cx, cy, bw, bh in boxes:
        x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
        draw.rectangle([x0, y0, x0 + bw * w, y0 + bh * h], fill=COLORS[int(cls) % len(COLORS)])
    draw.text((6, 4), path.stem, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=85)


def _boxes(rng: random.Random, count: int) -> list[list[float]]:
    boxes = []
    for _ in range(count):
        bw, bh = round(rng.uniform(0.1, 0.3), 4), round(rng.uniform(0.15, 0.4), 4)
        cx = round(rng.uniform(bw / 2, 1 - bw / 2), 4)
        cy = round(rng.uniform(bh / 2, 1 - bh / 2), 4)
        boxes.append([rng.randrange(len(CLASSES)), cx, cy, bw, bh])
    return boxes


def make_yolo(root: Path, train: int = 6, val: int = 2, size: tuple[int, int] = (320, 240), seed: int = 0) -> Path:
    """images/{split}/*.jpg + labels/{split}/*.txt + data.yaml.

    In ``train``, the second image has an empty label file and the third has no label
    file at all; both are background (nothing to find in them).
    """
    rng = random.Random(seed)
    for split, count in (("train", train), ("val", val)):
        for i in range(count):
            name = f"{split}_{i:04d}"
            special = split == "train" and i in (1, 2)
            boxes = [] if special else _boxes(rng, rng.randint(1, 4))
            _image(root / "images" / split / f"{name}.jpg", boxes, size, seed * 1000 + i + (0 if split == "train" else 500))
            label = root / "labels" / split / f"{name}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            if not (split == "train" and i == 2):
                label.write_text("".join(f"{int(c)} {cx} {cy} {w} {h}\n" for c, cx, cy, w, h in boxes), encoding="utf-8")
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (root / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return root


def make_coco(root: Path, count: int = 4, size: tuple[int, int] = (200, 100), seed: int = 1) -> Path:
    """images/*.jpg + annotations/instances_train.json with pixel boxes."""
    rng = random.Random(seed)
    w, h = size
    images, annotations = [], []
    for i in range(count):
        boxes = _boxes(rng, 2) if i else []
        _image(root / "images" / f"coco_{i:03d}.jpg", boxes, size, seed * 1000 + i)
        images.append({"id": i + 1, "file_name": f"coco_{i:03d}.jpg", "width": w, "height": h})
        for cls, cx, cy, bw, bh in boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": i + 1,
                    "category_id": [1, 3, 18][int(cls)],
                    "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                }
            )
    categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 18, "name": "dog"}]
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "instances_train.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}), encoding="utf-8"
    )
    return root


if __name__ == "__main__":  # python tests/sample_data.py <dir> [train] [val]
    import sys

    target = Path(sys.argv[1])
    make_yolo(
        target, int(sys.argv[2]) if len(sys.argv) > 2 else 120, int(sys.argv[3]) if len(sys.argv) > 3 else 30, (640, 480)
    )
    print(f"YOLO sample dataset written to {target}")


def make_imagefolder(
    root: Path, classes: tuple[str, ...] = ("cat", "dog", "bird"), train: int = 4, held: int = 2, held_split: str = "val"
) -> Path:
    """A classification dataset: ``train/<class>/*.jpg`` and ``<held_split>/<class>/*.jpg``, each class its own colour."""
    for split, count in (("train", train), (held_split, held)):
        for index, name in enumerate(classes):
            for i in range(count):
                path = root / split / name / f"{name}_{split}_{i:03d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                shade = 60 + 20 * i + (0 if split == "train" else 7)
                Image.new("RGB", (64, 48), (shade if index == 0 else 30, shade if index == 1 else 30, shade)).save(path)
    return root
